"""CPU-only real-data audit for Track-ID consistency supervision."""

import argparse
import copy
import csv
import gc
import json
import os
import shutil
import tempfile

from PIL import Image, ImageDraw
import torch

import dwm.common
from dwm.utils.track_consistency import (
    SPATIAL_PAIR,
    TEMPORAL_PAIR,
    box_dimensions_from_corners,
    dimension_log_distance,
    prepare_track_consistency_geometry,
    project_box_to_feature_mask,
    sample_track_consistency_selection,
)
from smoke_track_datasets import build_nuscenes_smoke_root


VEHICLE_CLASS_IDS = (0, 1, 2, 3, 4)


def _make_dataset(config, dataset_name, maximum_clips):
    training = config["training_dataset"]["base_dataset"]["datasets"]
    validation = config["validation_dataset"]["base_dataset"]["datasets"]
    source_configs = {
        "waymo": training[0]["bev_dataset"],
        "nuscenes": training[1]["bev_dataset"],
        "argoverse": training[2]["bev_dataset"],
        "nuplan": validation[0]["bev_dataset"],
    }
    dataset_config = copy.deepcopy(source_configs[dataset_name])
    temporary_directory = None

    if dataset_name == "nuscenes":
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="track_nuscenes_audit_"
        )
        source_root = config["global_state"]["nuscenes_fs"]["path"]
        dataset_config["dataset_name"] = build_nuscenes_smoke_root(
            source_root,
            temporary_directory.name,
        )
        dataset_config["fs"] = {
            "_class_name": "dwm.fs.dirfs.DirFileSystem",
            "path": temporary_directory.name,
        }

    if dataset_name == "argoverse":
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="track_argoverse_audit_"
        )
        mini_index = os.path.join(temporary_directory.name, "index")
        os.makedirs(mini_index)
        with open(
            dataset_config["balanced_json_path"],
            "r",
            encoding="utf-8",
        ) as file:
            all_intervals = json.load(file)
        selected_intervals = all_intervals[:maximum_clips]
        scene_names = sorted(
            {value["scene_name"] for value in selected_intervals}
        )
        for scene_name in scene_names:
            shutil.copy(
                os.path.join(
                    dataset_config["index_json_path"],
                    f"{scene_name}.info.json",
                ),
                mini_index,
            )
        mini_balance = os.path.join(
            temporary_directory.name,
            "balanced_windows.json",
        )
        with open(mini_balance, "w", encoding="utf-8") as file:
            json.dump(selected_intervals, file)
        dataset_config["index_json_path"] = mini_index
        dataset_config["balanced_json_path"] = mini_balance

    return (
        dwm.common.create_instance_from_config(dataset_config),
        temporary_directory,
    )


def _valid_endpoint(track_ids, classes, masks, time_id, view_id):
    valid = track_ids[time_id].gt(0) & masks[time_id, view_id].bool()
    vehicle = torch.zeros_like(valid)
    for class_id in VEHICLE_CLASS_IDS:
        vehicle |= classes[time_id, view_id] == class_id
    slots = (valid & vehicle).nonzero(as_tuple=False).flatten().tolist()
    return {
        int(track_ids[time_id, slot]): slot
        for slot in slots
    }


def _pair_query_count(pair, track_ids, classes, masks):
    time_a, view_a, time_b, view_b = pair
    ids_a = _valid_endpoint(
        track_ids, classes, masks, time_a, view_a
    )
    ids_b = _valid_endpoint(
        track_ids, classes, masks, time_b, view_b
    )
    if len(ids_a) < 2 or len(ids_b) < 2:
        return 0
    return 2 * len(set(ids_a).intersection(ids_b))


def _quantiles(values):
    if not values:
        return {key: None for key in ("p10", "p25", "p50", "p75", "p90", "p95")}
    tensor = torch.tensor(values, dtype=torch.float64)
    levels = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    result = torch.quantile(tensor, torch.tensor(levels, dtype=torch.float64))
    return {
        f"p{int(level * 100)}": float(value)
        for level, value in zip(levels, result)
    }


def _to_pil_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch.is_tensor(value):
        tensor = value.detach().cpu().float()
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW image, got {tuple(tensor.shape)}")
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        array = tensor.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type {type(value)}")


def _overlay_roi(image, region):
    image = image.copy().convert("RGB")
    width, height = image.size
    mask = Image.fromarray(
        (region.detach().cpu().numpy().astype("uint8") * 255)
    ).resize((width, height), Image.Resampling.NEAREST)
    red = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.blend(image, red, 0.42)
    image.paste(overlay, mask=mask)
    return image, mask


def _save_roi_case(
    output_path,
    item,
    pair,
    pair_name,
    case_index,
    track_id,
    slots,
    regions,
):
    time_a, view_a, time_b, view_b = pair
    image_a, mask_a = _overlay_roi(
        _to_pil_image(item["images"][time_a][view_a]), regions[0]
    )
    image_b, mask_b = _overlay_roi(
        _to_pil_image(item["images"][time_b][view_b]), regions[1]
    )
    display_width = 512
    display_height = 288
    image_a = image_a.resize((display_width, display_height))
    image_b = image_b.resize((display_width, display_height))
    mask_a = mask_a.resize(
        (display_width, display_height), Image.Resampling.NEAREST
    ).convert("RGB")
    mask_b = mask_b.resize(
        (display_width, display_height), Image.Resampling.NEAREST
    ).convert("RGB")
    title_height = 48
    canvas = Image.new(
        "RGB",
        (display_width * 2, title_height + display_height * 2),
        "white",
    )
    canvas.paste(image_a, (0, title_height))
    canvas.paste(image_b, (display_width, title_height))
    canvas.paste(mask_a, (0, title_height + display_height))
    canvas.paste(mask_b, (display_width, title_height + display_height))
    class_id = int(
        item["bbox_token_classes"][time_a, view_a, slots[0]]
    )
    ImageDraw.Draw(canvas).text(
        (8, 8),
        (
            f"{pair_name} track={track_id} class={class_id} "
            f"A=(t{time_a},v{view_a},s{slots[0]}) "
            f"B=(t{time_b},v{view_b},s{slots[1]}) "
            f"patches=({int(regions[0].sum())},{int(regions[1].sum())})"
        ),
        fill="black",
    )
    canvas.save(
        os.path.join(output_path, f"{pair_name}_{case_index:02d}.png")
    )


def audit_dataset(dataset_name, dataset, maximum_clips, output_root, roi_cases):
    actual_clips = min(maximum_clips, len(dataset))
    coverage_rows = []
    roi_rows = []
    hard_rows = []
    slot_findings = {
        "dataset": dataset_name,
        "clips_requested": maximum_clips,
        "clips_audited": actual_clips,
        "padding_visible_conflicts": [],
        "duplicate_id_frames": [],
        "same_id_adjacent_count": 0,
        "same_id_class_mismatch_count": 0,
        "same_id_size_distances": [],
        "shape_errors": [],
    }
    roi_saved = {SPATIAL_PAIR: 0, TEMPORAL_PAIR: 0}
    roi_output = os.path.join(output_root, "roi_debug", dataset_name)
    os.makedirs(roi_output, exist_ok=True)

    selection_config = {
        "track_consistency_loss_weight": 1.0,
        "track_consistency_vehicle_class_ids": list(VEHICLE_CLASS_IDS),
        "track_consistency_enable_spatial": True,
        "track_consistency_enable_temporal": True,
        "track_consistency_temporal_stride": 1,
        "track_consistency_max_spatial_pairs_per_sample": 1000000,
        "track_consistency_max_temporal_pairs_per_sample": 1000000,
    }

    for clip_index in range(actual_clips):
        item = dataset[clip_index]
        corners = item["bbox_token_corners"]
        classes = item["bbox_token_classes"]
        masks = item["bbox_token_masks"].bool()
        track_ids = item["bbox_token_track_ids"].long()
        time_count, view_count, slot_count = classes.shape
        expected_shapes = {
            "track_ids": (time_count, slot_count),
            "corners": (time_count, view_count, slot_count, 8, 3),
            "classes": (time_count, view_count, slot_count),
            "masks": (time_count, view_count, slot_count),
        }
        actual_shapes = {
            "track_ids": tuple(track_ids.shape),
            "corners": tuple(corners.shape),
            "classes": tuple(classes.shape),
            "masks": tuple(masks.shape),
        }
        if actual_shapes != expected_shapes:
            slot_findings["shape_errors"].append(
                {"clip": clip_index, "actual": actual_shapes}
            )

        conflicts = (
            track_ids.eq(0)[:, None].expand_as(masks) & masks
        ).nonzero(as_tuple=False)
        for time_id, view_id, slot_id in conflicts[:100].tolist():
            slot_findings["padding_visible_conflicts"].append(
                {
                    "clip": clip_index,
                    "time": time_id,
                    "view": view_id,
                    "slot": slot_id,
                }
            )
        for time_id in range(time_count):
            values = track_ids[time_id]
            values = values[values > 0]
            if int(values.unique().numel()) != int(values.numel()):
                slot_findings["duplicate_id_frames"].append(
                    {"clip": clip_index, "time": time_id}
                )

        for time_id in range(time_count - 1):
            slots_a = {
                int(value): slot
                for slot, value in enumerate(track_ids[time_id].tolist())
                if value > 0
            }
            slots_b = {
                int(value): slot
                for slot, value in enumerate(track_ids[time_id + 1].tolist())
                if value > 0
            }
            for track_id in set(slots_a).intersection(slots_b):
                slot_a = slots_a[track_id]
                slot_b = slots_b[track_id]
                slot_findings["same_id_adjacent_count"] += 1
                class_a = int(classes[time_id, 0, slot_a])
                class_b = int(classes[time_id + 1, 0, slot_b])
                if class_a != class_b:
                    slot_findings["same_id_class_mismatch_count"] += 1
                size_distance = dimension_log_distance(
                    box_dimensions_from_corners(corners[time_id, 0, slot_a]),
                    box_dimensions_from_corners(
                        corners[time_id + 1, 0, slot_b]
                    ),
                )
                slot_findings["same_id_size_distances"].append(
                    float(size_distance)
                )

        batch_keys = (
            "bbox_token_corners",
            "bbox_token_classes",
            "bbox_token_masks",
            "bbox_token_track_ids",
            "view_consistency_pair_mask",
            "camera_intrinsics",
            "camera_transforms",
            "ego_transforms",
            "reference_ego_transforms",
            "image_size",
        )
        batch = {
            key: item[key].unsqueeze(0)
            for key in batch_keys
            if key in item
        }
        selection_result = sample_track_consistency_selection(
            batch,
            selection_config,
            torch.Generator().manual_seed(clip_index),
            torch.device("cpu"),
        )
        if selection_result is None:
            selection_cpu = torch.empty(1, 0, 4, dtype=torch.long)
            pair_types = torch.empty(1, 0, dtype=torch.long)
        else:
            _, selection_cpu, pair_types = selection_result

        pair_mask = item["view_consistency_pair_mask"].bool()
        if pair_mask.ndim == 3 and pair_mask.shape[0] == 1:
            pair_mask = pair_mask[0]
        adjacent_count = 0
        for view_a in range(view_count):
            for view_b in range(view_a + 1, view_count):
                adjacent_count += int(
                    bool(pair_mask[view_a, view_b] or pair_mask[view_b, view_a])
                )
        spatial_valid = int((pair_types == SPATIAL_PAIR).sum())
        temporal_valid = int((pair_types == TEMPORAL_PAIR).sum())
        spatial_queries = 0
        temporal_queries = 0
        for pair, pair_type in zip(selection_cpu[0], pair_types[0]):
            if int(pair_type) < 0:
                continue
            query_count = _pair_query_count(
                pair.tolist(), track_ids, classes, masks
            )
            if int(pair_type) == SPATIAL_PAIR:
                spatial_queries += query_count
            else:
                temporal_queries += query_count
        coverage_rows.append(
            {
                "dataset": dataset_name,
                "clip": clip_index,
                "candidate_spatial_pairs": time_count * adjacent_count,
                "valid_spatial_pairs": spatial_valid,
                "valid_spatial_queries": spatial_queries,
                "candidate_temporal_pairs": (time_count - 1) * view_count,
                "valid_temporal_pairs": temporal_valid,
                "valid_temporal_queries": temporal_queries,
            }
        )

        intrinsics, camera_to_common, box_reference_to_common = (
            prepare_track_consistency_geometry(batch, torch.device("cpu"))
        )
        region_by_endpoint = {}
        for time_id in range(time_count):
            for view_id in range(view_count):
                endpoint = _valid_endpoint(
                    track_ids, classes, masks, time_id, view_id
                )
                dimensions = {}
                for track_id, slot_id in endpoint.items():
                    region = project_box_to_feature_mask(
                        corners[time_id, view_id, slot_id],
                        intrinsics[0, time_id, view_id],
                        camera_to_common[0, time_id, view_id],
                        box_reference_to_common[0, time_id],
                        18,
                        32,
                    )
                    region_by_endpoint[(time_id, view_id, track_id)] = region
                    roi_rows.append(
                        {
                            "dataset": dataset_name,
                            "clip": clip_index,
                            "time": time_id,
                            "view": view_id,
                            "slot": slot_id,
                            "track_id": track_id,
                            "class_id": int(classes[time_id, view_id, slot_id]),
                            "patch_count": int(region.sum()),
                        }
                    )
                    dimensions[track_id] = (
                        int(classes[time_id, view_id, slot_id]),
                        box_dimensions_from_corners(
                            corners[time_id, view_id, slot_id]
                        ),
                    )
                ids = sorted(dimensions)
                for index_a, track_a in enumerate(ids):
                    for track_b in ids[index_a + 1:]:
                        if dimensions[track_a][0] != dimensions[track_b][0]:
                            continue
                        hard_rows.append(
                            float(
                                dimension_log_distance(
                                    dimensions[track_a][1],
                                    dimensions[track_b][1],
                                )
                            )
                        )

        for pair, pair_type in zip(selection_cpu[0], pair_types[0]):
            pair_type = int(pair_type)
            if pair_type < 0 or roi_saved[pair_type] >= roi_cases:
                continue
            time_a, view_a, time_b, view_b = pair.tolist()
            endpoint_a = _valid_endpoint(
                track_ids, classes, masks, time_a, view_a
            )
            endpoint_b = _valid_endpoint(
                track_ids, classes, masks, time_b, view_b
            )
            shared_ids = sorted(set(endpoint_a).intersection(endpoint_b))
            for track_id in shared_ids:
                if roi_saved[pair_type] >= roi_cases:
                    break
                region_a = region_by_endpoint[(time_a, view_a, track_id)]
                region_b = region_by_endpoint[(time_b, view_b, track_id)]
                if int(region_a.sum()) < 2 or int(region_b.sum()) < 2:
                    continue
                _save_roi_case(
                    roi_output,
                    item,
                    pair.tolist(),
                    "spatial" if pair_type == SPATIAL_PAIR else "temporal",
                    roi_saved[pair_type],
                    track_id,
                    (endpoint_a[track_id], endpoint_b[track_id]),
                    (region_a, region_b),
                )
                roi_saved[pair_type] += 1

        print(
            f"[{dataset_name}] clip {clip_index + 1}/{actual_clips} "
            f"spatial={spatial_valid} temporal={temporal_valid}",
            flush=True,
        )
        del item, batch, intrinsics, camera_to_common, box_reference_to_common
        del region_by_endpoint
        gc.collect()

    size_distances = slot_findings.pop("same_id_size_distances")
    slot_findings["same_id_class_mismatch_rate"] = (
        slot_findings["same_id_class_mismatch_count"]
        / max(slot_findings["same_id_adjacent_count"], 1)
    )
    slot_findings["same_id_size_log_distance"] = _quantiles(size_distances)
    slot_findings["roi_visualizations"] = {
        "spatial": roi_saved[SPATIAL_PAIR],
        "temporal": roi_saved[TEMPORAL_PAIR],
    }
    hard_quantiles = _quantiles(hard_rows)
    hard_summary = {
        "dataset": dataset_name,
        "pair_count": len(hard_rows),
        **hard_quantiles,
        "fraction_le_0_2": (
            sum(value <= 0.2 for value in hard_rows) / max(len(hard_rows), 1)
        ),
    }

    with open(
        os.path.join(output_root, f"dataset_slot_audit_{dataset_name}.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(slot_findings, file, indent=2)
    for filename, rows in (
        (f"pair_coverage_{dataset_name}.csv", coverage_rows),
        (f"roi_patch_stats_{dataset_name}.csv", roi_rows),
        (f"hard_negative_size_stats_{dataset_name}.csv", [hard_summary]),
    ):
        with open(
            os.path.join(output_root, filename),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(slot_findings, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/lyh/bev_pv_epipolar_track.json",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("waymo", "nuscenes", "argoverse", "nuplan"),
    )
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--roi-cases-per-type", type=int, default=10)
    parser.add_argument(
        "--output-root",
        default="artifacts/track_consistency_cpu_audit",
    )
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES=''.")
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    dataset, temporary_directory = _make_dataset(
        config,
        args.dataset,
        args.max_clips,
    )
    audit_dataset(
        args.dataset,
        dataset,
        args.max_clips,
        args.output_root,
        args.roi_cases_per_type,
    )
    del temporary_directory


if __name__ == "__main__":
    main()
