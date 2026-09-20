import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dwm.metrics.fvd import FrechetVideoDistance


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate offline per-camera FVD from one paired fake/real manifest."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--i3d-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sequence-count", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--camera-names", type=str, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable incremental progress loading and saving.",
    )
    return parser


def load_manifest_items(manifest_path, max_videos):
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_videos is not None and len(items) >= max_videos:
                break
    return items


def parse_camera_names(camera_names):
    if camera_names is None or len(camera_names.strip()) == 0:
        return None
    return [name.strip() for name in camera_names.split(",") if len(name.strip()) > 0]


def get_camera_names(item):
    if len(item["frames"]) == 0:
        return []
    return [view["camera"] for view in item["frames"][0]["views"]]


def build_view_map(frame):
    return {view["camera"]: view for view in frame["views"]}


def load_rgb_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def resolve_path(manifest_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(manifest_dir, path)


def load_video_pair(manifest_dir, item, camera_name, sequence_count):
    frames = sorted(item["frames"], key=lambda x: int(x.get("frame_index", 0)))
    frames = frames[:sequence_count]

    fake_images = []
    real_images = []

    for frame in frames:
        view_map = build_view_map(frame)
        if camera_name not in view_map:
            raise KeyError(f"{camera_name} not found in {item.get('video_id', '')}")

        view = view_map[camera_name]
        if "real_image_path" not in view:
            raise KeyError(
                f"real_image_path missing for {item.get('video_id', '')}/{camera_name}. "
                "Re-export with --export-paired-real."
            )

        fake_images.append(load_rgb_tensor(resolve_path(manifest_dir, view["image_path"])))
        real_images.append(load_rgb_tensor(resolve_path(manifest_dir, view["real_image_path"])))

    if len(fake_images) < 10:
        raise RuntimeError(f"Too few frames for FVD: {len(fake_images)}")

    return torch.stack(fake_images, dim=0), torch.stack(real_images, dim=0)


def update_metric(metric, manifest_dir, batch_specs, sequence_count, device):
    fake_videos = []
    real_videos = []

    for item, camera_name in batch_specs:
        fake_video, real_video = load_video_pair(
            manifest_dir,
            item,
            camera_name,
            sequence_count,
        )
        fake_videos.append(fake_video)
        real_videos.append(real_video)

    fake_batch = torch.stack(fake_videos, dim=0).to(device)
    real_batch = torch.stack(real_videos, dim=0).to(device)

    with torch.no_grad():
        metric.update(real_batch, real=True)
        metric.update(fake_batch, real=False)


def write_json(output_path, data):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


FVD_STATE_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
    "fake_features_sum",
    "fake_features_cov_sum",
    "fake_features_num_samples",
)


def _fvd_signature(args, manifest_path, camera_names, num_samples):
    return {
        "manifest": str(manifest_path.resolve()),
        "i3d_checkpoint": str(Path(args.i3d_checkpoint).resolve()),
        "camera_names": list(camera_names),
        "sequence_count": args.sequence_count,
        "batch_size": args.batch_size,
        "num_samples": num_samples,
    }


def _metric_state(metric):
    # FrechetVideoDistance uses these six add_state() tensors as its complete
    # accumulated statistic. Saving them explicitly avoids relying on the
    # torchmetrics state_dict implementation to preserve custom metric state.
    return {
        name: getattr(metric, name).detach().cpu().clone()
        for name in FVD_STATE_NAMES
    }


def _restore_metric_state(metric, state):
    missing = [name for name in FVD_STATE_NAMES if name not in state]
    if missing:
        raise RuntimeError(f"FVD progress is missing metric state: {missing}")
    for name in FVD_STATE_NAMES:
        target = getattr(metric, name)
        value = state[name]
        if tuple(value.shape) != tuple(target.shape):
            raise RuntimeError(
                f"FVD metric state shape mismatch for {name}: "
                f"{tuple(value.shape)} vs {tuple(target.shape)}"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def _atomic_torch_save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_fvd_progress(path, signature):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != signature:
        raise RuntimeError(
            "FVD progress does not match the manifest/camera/eval configuration."
        )
    next_start = checkpoint.get("next_start")
    if not isinstance(next_start, int) or next_start < 0:
        raise RuntimeError(f"Invalid FVD progress next_start: {next_start!r}")
    return next_start, checkpoint.get("metric_state", {})


def main():
    args = create_parser().parse_args()

    device = torch.device(args.device)
    manifest_dir = str(Path(args.manifest).parent)
    items = load_manifest_items(args.manifest, args.max_videos)

    if len(items) == 0:
        raise RuntimeError("Empty manifest.")

    requested_cameras = parse_camera_names(args.camera_names)
    if requested_cameras is None:
        camera_names = get_camera_names(items[0])
    else:
        camera_names = requested_cameras

    sample_specs = []
    for item in items:
        for camera_name in camera_names:
            sample_specs.append((item, camera_name))

    metric = FrechetVideoDistance(
        inception_3d_checkpoint_path=args.i3d_checkpoint,
        sequence_count=args.sequence_count,
    ).to(device)
    metric.eval()

    print("[paired-fvd] videos:", len(items))
    print("[paired-fvd] cameras:", camera_names)
    print("[paired-fvd] total samples:", len(sample_specs))

    progress_path = Path(args.output).with_suffix(Path(args.output).suffix + ".progress.pt")
    signature = _fvd_signature(
        args,
        Path(args.manifest),
        camera_names,
        len(sample_specs),
    )
    next_start = 0
    if args.no_resume:
        print("[RESUME] disabled; starting fresh")
    elif progress_path.is_file():
        next_start, saved_state = _load_fvd_progress(progress_path, signature)
        if next_start > len(sample_specs):
            raise RuntimeError(
                f"FVD progress next_start {next_start} exceeds "
                f"sample count {len(sample_specs)}"
            )
        _restore_metric_state(metric, saved_state)
        print(
            f"[RESUME] completed {next_start}/{len(sample_specs)} "
            f"from {progress_path}"
        )
    else:
        print("[RESUME] starting fresh")

    for start in range(next_start, len(sample_specs), args.batch_size):
        end = min(start + args.batch_size, len(sample_specs))
        update_metric(
            metric,
            manifest_dir,
            sample_specs[start:end],
            args.sequence_count,
            device,
        )
        if not args.no_resume:
            _atomic_torch_save(
                progress_path,
                {
                    "next_start": end,
                    "metric_state": _metric_state(metric),
                    "signature": signature,
                    "manifest": signature["manifest"],
                    "camera_names": signature["camera_names"],
                    "sequence_count": signature["sequence_count"],
                    "batch_size": signature["batch_size"],
                    "num_samples": signature["num_samples"],
                },
            )
        print(f"[paired-fvd] processed samples {end}/{len(sample_specs)}", flush=True)

    with torch.no_grad():
        fvd_value = float(metric.compute().detach().cpu().item())

    result = {
        "fvd": fvd_value,
        "num_videos": len(items),
        "num_samples": len(sample_specs),
        "camera_names": camera_names,
        "sequence_count": args.sequence_count,
        "batch_size": args.batch_size,
        "manifest": args.manifest,
    }

    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
