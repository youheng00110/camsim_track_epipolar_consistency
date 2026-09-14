from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_geometry import (
    FrameSourceDataset,
    build_source_items,
    collate_frame_sources,
    load_frames_from_config,
    load_yaml,
    project_frame_boxes,
)
from matching_visualization import (
    aggregate_rank_outputs,
    build_frame_result,
    match_detections,
    prepare_evaluation_instances,
    render_pair_crops,
    render_visualization,
)
from sam31_detector import (
    MockDetector,
    Sam31Detector,
    discover_checkpoint,
    predict_with_oom_backoff,
)
from shared_box_projection import project_manifest_boxes


def initialize_runtime() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def finalize_runtime(world_size: int) -> None:
    if world_size > 1 and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def prepare_output_directory(config: dict[str, Any], rank: int) -> Path:
    output_dir = Path(config["paths"]["output_dir"]).expanduser().resolve()
    overwrite = bool(config.get("runtime", {}).get("overwrite", True))
    if rank == 0 and overwrite and output_dir.is_dir():
        shutil.rmtree(output_dir)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and distributed.is_initialized():
        distributed.barrier()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    return output_dir


def build_detector(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    backend = str(config.get("runtime", {}).get("backend", "sam3.1")).lower()
    if backend == "mock":
        detector = MockDetector(config["model"], device)
        return detector, detector.load_info
    if device.type != "cuda":
        raise RuntimeError("SAM 3.1 evaluation requires a CUDA GPU")
    if not bool(config["model"].get("save_masks", True)):
        raise ValueError(
            "model.save_masks must be true because this evaluator uses SAM masks for pixel filtering and IoU"
        )

    sam3_repo = str(config["paths"]["sam3_repo"])
    checkpoint = discover_checkpoint(
        sam3_repo,
        str(config["paths"].get("checkpoint", "")),
    )
    detector = Sam31Detector(
        config["model"],
        device,
        sam3_repo,
        checkpoint,
    )
    return detector, detector.load_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3.1 mask-IoU foreground-condition evaluator"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", choices=("sam3.1", "mock"), default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument(
        "--preview-root",
        default=None,
        help="Override paths.preview_root and recursively discover stflow_manifest.jsonl",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only discover and validate preview image paths; do not load SAM",
    )
    parser.add_argument(
        "--keep-reference-frames",
        action="store_true",
        help="Keep reference frames copied from GT in preview manifests",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.backend is not None:
        config.setdefault("runtime", {})["backend"] = args.backend
    if args.limit_frames is not None:
        config.setdefault("runtime", {})["limit_frames"] = args.limit_frames
    if args.preview_root is not None:
        config.setdefault("paths", {})["preview_root"] = args.preview_root
    if args.keep_reference_frames:
        config.setdefault("preview", {})["skip_reference_frames"] = False

    rank, world_size, _, device = initialize_runtime()
    try:
        seed = int(config.get("runtime", {}).get("seed", 3407)) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        frames = load_frames_from_config(config)

        if args.scan_only:
            source_counts: dict[str, int] = {}

            for frame in frames:
                source = str(
                    frame.get("preview_source", "frames")
                )
                source_counts[source] = (
                    source_counts.get(source, 0) + 1
                )

            if rank == 0:
                matched_box_count = sum(
                    1 for frame in frames if "boxes_3d" in frame
                )
                report = {
                    "frames": len(frames),
                    "shared_box_matched": matched_box_count,
                    "shared_box_missing": len(frames) - matched_box_count,
                    "sources": source_counts,
                    "first_image": frames[0]["rgb_path"],
                    "first_box_image": frames[0].get("box_image_path"),
                    "last_image": frames[-1]["rgb_path"],
                    "last_box_image": frames[-1].get("box_image_path"),
                }
                print(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return

        items = build_source_items(
            frames,
            config,
            rank,
            world_size,
        )
        dataset = FrameSourceDataset(items)
        output_dir = prepare_output_directory(config, rank)
        loader_workers = int(config["model"].get("loader_workers", 4))
        loader = DataLoader(
            dataset,
            batch_size=int(config["model"].get("batch_size", 2)),
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=loader_workers > 0,
            collate_fn=collate_frame_sources,
        )

        detector, model_info = build_detector(config, device)
        model_info = dict(model_info)
        model_info["backend"] = str(config["runtime"]["backend"])
        model_info["rank"] = rank
        model_info["world_size"] = world_size
        rank_output = output_dir / f"records.rank{rank:03d}.jsonl"
        visualization_counts: dict[str, int] = {}

        with rank_output.open("w", encoding="utf-8") as writer:
            progress = tqdm(loader, desc=f"rank {rank} eval", disable=rank != 0)
            for batch in progress:
                images: list[torch.Tensor] = []
                metadata: list[dict[str, Any]] = []

                for item in batch:
                    frame = item["frame"]
                    if "boxes_3d" in frame:
                        projections = project_manifest_boxes(
                            boxes_3d=frame["boxes_3d"],
                            T_lidar_to_camera=frame["T_lidar_to_camera"],
                            lidar_to_image=frame["lidar_to_image"],
                            image_width=int(item["width"]),
                            image_height=int(item["height"]),
                            annotation_config=config["annotation"],
                        )
                    else:
                        projections = project_frame_boxes(
                            frame=frame,
                            image_width=int(item["width"]),
                            image_height=int(item["height"]),
                            annotation_config=config["annotation"],
                        )

                    item["gt_projections"] = projections
                    images.append(item["image"])
                    metadata.append(item)

                batch_detections = predict_with_oom_backoff(
                    detector,
                    images,
                    metadata,
                )

                for item, raw_detections in zip(metadata, batch_detections):
                    prepared = prepare_evaluation_instances(
                        projections=item["gt_projections"],
                        detections=raw_detections,
                        image_width=int(item["width"]),
                        image_height=int(item["height"]),
                        visibility_config=config["visibility"],
                        matching_config=config["matching"],
                    )
                    matching = match_detections(
                        prepared["projections"],
                        prepared["detections"],
                        config["matching"],
                    )
                    result = build_frame_result(item, prepared, matching)
                    writer.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    writer.flush()

                    source_name = str(item["source"])
                    current_count = visualization_counts.get(source_name, 0)
                    max_visualizations = int(
                        config["visualization"].get("max_frames_per_source", 50)
                    )
                    if (
                        bool(config["visualization"].get("enabled", True))
                        and current_count < max_visualizations
                    ):
                        image_rgb = item["image"].permute(1, 2, 0).cpu().numpy()
                        frame_index = int(item["frame"]["frame_index"])
                        visual_path = (
                            output_dir
                            / "visualizations"
                            / source_name
                            / f"frame_{frame_index:06d}.jpg"
                        )
                        render_visualization(
                            image_rgb=image_rgb,
                            prepared=prepared,
                            matching=matching,
                            output_path=visual_path,
                            config=config["visualization"],
                        )
                        pair_crop_path = (
                            output_dir
                            / "pair_crops"
                            / source_name
                            / f"frame_{frame_index:06d}.jpg"
                        )
                        render_pair_crops(
                            image_rgb=image_rgb,
                            prepared=prepared,
                            matching=matching,
                            output_path=pair_crop_path,
                            config=config["visualization"],
                        )
                        visualization_counts[source_name] = current_count + 1

        if world_size > 1 and distributed.is_initialized():
            distributed.barrier()
        if rank == 0:
            summary = aggregate_rank_outputs(output_dir, world_size, model_info)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Results saved to {output_dir}")
    finally:
        finalize_runtime(world_size)


if __name__ == "__main__":
    main()