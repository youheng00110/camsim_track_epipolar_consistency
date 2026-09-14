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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

    for start in range(0, len(sample_specs), args.batch_size):
        end = min(start + args.batch_size, len(sample_specs))
        update_metric(
            metric,
            manifest_dir,
            sample_specs[start:end],
            args.sequence_count,
            device,
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
