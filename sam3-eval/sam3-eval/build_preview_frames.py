from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Convert OpenDWM stflow_manifest.jsonl to SAM evaluation frames.jsonl"
)
parser.add_argument(
    "--manifest",
    required=True,
    help="Path to stflow_manifest.jsonl",
)
parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory used to save frames.jsonl",
)
parser.add_argument(
    "--source",
    choices=("generated", "real"),
    default="generated",
    help="Read generated images or paired real images",
)
parser.add_argument(
    "--skip-reference-frames",
    action="store_true",
    help="Skip frames marked as reference frames",
)
args = parser.parse_args()

manifest_path = Path(args.manifest).expanduser().resolve()
output_dir = Path(args.output_dir).expanduser().resolve()
preview_root = manifest_path.parent

if not manifest_path.is_file():
    raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

output_dir.mkdir(parents=True, exist_ok=True)
frames_path = output_dir / "frames.jsonl"

path_field = (
    "image_path"
    if args.source == "generated"
    else "real_image_path"
)

sample_index = 0
video_count = 0

with (
    manifest_path.open("r", encoding="utf-8") as manifest_file,
    frames_path.open("w", encoding="utf-8") as output_file,
):
    for line_index, line in enumerate(manifest_file):
        stripped = line.strip()
        if not stripped:
            continue

        video_record = json.loads(stripped)
        video_id = str(
            video_record.get("video_id", f"video_{line_index:06d}")
        )
        dataset_name = str(
            video_record.get("dataset_name", "unknown")
        )
        frame_records = video_record.get("frames", [])

        if not isinstance(frame_records, list):
            raise TypeError(
                f"frames must be a list in video {video_id}"
            )

        for frame_record in frame_records:
            time_index = int(
                frame_record.get("frame_index", 0)
            )
            views = frame_record.get("views", [])

            if not isinstance(views, list):
                raise TypeError(
                    f"views must be a list in {video_id}, t={time_index}"
                )

            for view_index, view in enumerate(views):
                if (
                    args.skip_reference_frames
                    and bool(view.get("is_reference_frame", False))
                ):
                    continue

                image_path_value = view.get(path_field)
                if not image_path_value:
                    if args.source == "real":
                        continue
                    raise KeyError(
                        f"Missing {path_field} in "
                        f"{video_id}, t={time_index}, view={view_index}"
                    )

                image_path = Path(str(image_path_value))
                if not image_path.is_absolute():
                    image_path = preview_root / image_path
                image_path = image_path.resolve()

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image does not exist: {image_path}"
                    )

                camera_name = str(
                    view.get("camera", f"CAM_{view_index:02d}")
                )

                intrinsic = view.get("K")
                if intrinsic is None:
                    intrinsic = [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]

                ego_from_camera = view.get("T_cam_to_ego")
                if ego_from_camera is None:
                    ego_from_camera = [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

                image_size = view.get("image_size", [0, 0])
                width = 0
                height = 0
                if isinstance(image_size, list) and len(image_size) >= 2:
                    width = int(round(float(image_size[0])))
                    height = int(round(float(image_size[1])))

                frame_token = (
                    f"{dataset_name}/"
                    f"{video_id}/"
                    f"t{time_index:03d}/"
                    f"{camera_name}"
                )

                output_record = {
                    "frame_index": sample_index,
                    "timestamp": float(sample_index),
                    "frame_token": frame_token,
                    "rgb_path": str(image_path),
                    "boxes": [],
                    "calibration": {
                        "intrinsic_rgb": intrinsic,
                        "T_ego_from_camera": ego_from_camera,
                        "raw_image_size": {
                            "width": width,
                            "height": height,
                        },
                    },
                }

                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sample_index += 1

        video_count += 1

print(f"Manifest: {manifest_path}")
print(f"Source: {args.source}")
print(f"Videos: {video_count}")
print(f"Images: {sample_index}")
print(f"Output: {frames_path}")
