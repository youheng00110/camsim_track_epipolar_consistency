from __future__ import annotations

import json
import math
import fnmatch
import os
from operator import itemgetter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from shared_box_projection import attach_shared_box_images, video_pose_signature


BOX_EDGE_INDICES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


class FrameSourceDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        if not self.items:
            raise ValueError("No image items were built from the input data")
        for item in self.items:
            image_path = Path(item["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image does not exist: {image_path}")

            box_image_path_value = item.get("box_image_path")
            if box_image_path_value:
                box_image_path = Path(str(box_image_path_value))
                if not box_image_path.is_file():
                    raise FileNotFoundError(
                        f"Shared box image does not exist: {box_image_path}"
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.items[index])
        image = read_image(item["image_path"], mode=ImageReadMode.RGB)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Invalid RGB tensor shape {tuple(image.shape)} for {item['image_path']}"
            )
        item["image"] = image
        item["height"] = int(image.shape[1])
        item["width"] = int(image.shape[2])

        box_image_path_value = item.get("box_image_path")
        if box_image_path_value:
            box_image = read_image(
                str(box_image_path_value),
                mode=ImageReadMode.RGB,
            )
            if box_image.ndim != 3 or box_image.shape[0] != 3:
                raise ValueError(
                    f"Invalid shared box image shape {tuple(box_image.shape)} "
                    f"for {box_image_path_value}"
                )
            if tuple(box_image.shape[-2:]) != tuple(image.shape[-2:]):
                raise ValueError(
                    "Generated image and shared box image size mismatch: "
                    f"generated={tuple(image.shape[-2:])}, "
                    f"box={tuple(box_image.shape[-2:])}, "
                    f"generated_path={item['image_path']}, "
                    f"box_path={box_image_path_value}"
                )
            item["box_image"] = box_image

        return item


def collate_frame_sources(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not batch:
        raise ValueError("The data loader produced an empty batch")
    for item in batch:
        if "image" not in item or not isinstance(item["image"], torch.Tensor):
            raise TypeError("Each batch item must contain an RGB torch.Tensor")
    return batch


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("The YAML root must be a dictionary")
    return config


def load_unpacked_frames(data_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(data_dir).expanduser().resolve()
    jsonl_path = root / "frames.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Expected unpacked JSONL at {jsonl_path}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise TypeError(
                    f"Line {line_index + 1} in {jsonl_path} is not a JSON object"
                )
            records.append(normalize_frame_record(record, root, line_index))

    records.sort(key=itemgetter("frame_index", "timestamp"))
    if not records:
        raise ValueError(f"No frames were found in {jsonl_path}")
    return records


def load_preview_frames(
    preview_root: str | Path,
    preview_config: dict[str, Any] | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(preview_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Preview root does not exist: {root}")

    settings = preview_config or {}
    manifest_glob = str(settings.get("manifest_glob", "**/stflow_manifest.jsonl"))
    skip_reference_frames = bool(settings.get("skip_reference_frames", True))
    strict_paths = bool(settings.get("strict_paths", False))
    include_methods = [str(value) for value in settings.get("include_methods", [])]
    exclude_methods = [str(value) for value in settings.get("exclude_methods", [])]

    print(
        f"[PREVIEW_SCAN_START] root={root} glob={manifest_glob}",
        flush=True,
    )

    if manifest_glob == "**/stflow_manifest.jsonl":
        manifest_paths: list[Path] = []
        prune_names = {
            "images",
            "paired_real",
            "visualizations",
            "masks",
            "__pycache__",
        }

        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = [
                name
                for name in dir_names
                if name not in prune_names and not name.startswith(".")
            ]

            if "stflow_manifest.jsonl" in file_names:
                manifest_paths.append(
                    Path(current_root) / "stflow_manifest.jsonl"
                )

        manifest_paths.sort()
    else:
        manifest_paths = sorted(
            path for path in root.glob(manifest_glob)
            if path.is_file()
        )

    if not manifest_paths:
        raise FileNotFoundError(
            f"No preview manifests matched {manifest_glob!r} under {root}"
        )

    print(
        f"[PREVIEW_MANIFESTS_FOUND] count={len(manifest_paths)}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    missing_generated = 0
    missing_real = 0
    skipped_reference = 0
    selected_manifest_count = 0

    for manifest_index, manifest_path in enumerate(manifest_paths, start=1):
        manifest_root = manifest_path.parent
        relative_parent = manifest_root.relative_to(root)
        method_name = str(relative_parent) if str(relative_parent) != "." else root.name
        method_name = method_name.replace("/", "__").replace("\\", "__")

        if include_methods and not any(
            fnmatch.fnmatch(method_name, pattern) for pattern in include_methods
        ):
            continue
        if exclude_methods and any(
            fnmatch.fnmatch(method_name, pattern) for pattern in exclude_methods
        ):
            continue

        selected_manifest_count += 1
        print(
            f"[PREVIEW_SCAN_MANIFEST] "
            f"{manifest_index}/{len(manifest_paths)} "
            f"method={method_name}",
            flush=True,
        )

        with manifest_path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                stripped = line.strip()
                if not stripped:
                    continue
                manifest_item = json.loads(stripped)
                if not isinstance(manifest_item, dict):
                    raise TypeError(
                        f"Line {line_index + 1} in {manifest_path} is not a JSON object"
                    )

                video_id = str(
                    manifest_item.get("video_id", f"video_{line_index:06d}")
                )
                dataset_name = str(manifest_item.get("dataset_name", "unknown"))
                video_signature = video_pose_signature(manifest_item)
                frame_entries = manifest_item.get("frames", [])
                if not isinstance(frame_entries, list):
                    raise TypeError(
                        f"frames must be a list in {manifest_path}, video={video_id}"
                    )

                for frame_entry in frame_entries:
                    if not isinstance(frame_entry, dict):
                        raise TypeError(
                            f"Each frame must be an object in {manifest_path}, video={video_id}"
                        )
                    time_index = int(frame_entry.get("frame_index", 0))
                    timestamp = float(frame_entry.get("timestamp", time_index))
                    views = frame_entry.get("views", [])
                    if not isinstance(views, list):
                        raise TypeError(
                            f"views must be a list in {manifest_path}, "
                            f"video={video_id}, frame={time_index}"
                        )

                    for view_index, view in enumerate(views):
                        if not isinstance(view, dict):
                            raise TypeError(
                                f"Each view must be an object in {manifest_path}, "
                                f"video={video_id}, frame={time_index}"
                            )
                        is_reference = bool(view.get("is_reference_frame", False))
                        if skip_reference_frames and is_reference:
                            skipped_reference += 1
                            continue

                        generated_value = view.get("image_path")
                        if not generated_value:
                            if strict_paths:
                                raise KeyError(
                                    f"Missing image_path in {manifest_path}, "
                                    f"video={video_id}, frame={time_index}, view={view_index}"
                                )
                            missing_generated += 1
                            continue

                        generated_path = Path(str(generated_value))
                        if not generated_path.is_absolute():
                            generated_path = manifest_root / generated_path
                        generated_path = generated_path.resolve()
                        if not generated_path.is_file():
                            if strict_paths:
                                raise FileNotFoundError(
                                    f"Generated preview image does not exist: {generated_path}"
                                )
                            missing_generated += 1
                            continue

                        real_path: Path | None = None
                        real_value = view.get("real_image_path")
                        if real_value:
                            candidate = Path(str(real_value))
                            if not candidate.is_absolute():
                                candidate = manifest_root / candidate
                            candidate = candidate.resolve()
                            if candidate.is_file():
                                real_path = candidate
                            elif strict_paths:
                                raise FileNotFoundError(
                                    f"Paired real preview image does not exist: {candidate}"
                                )
                            else:
                                missing_real += 1

                        camera_name = str(view.get("camera", f"CAM_{view_index:02d}"))
                        intrinsic = view.get(
                            "K",
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        )
                        ego_from_camera = view.get(
                            "T_cam_to_ego",
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        )
                        image_size = view.get("image_size", [0, 0])
                        width = 0
                        height = 0
                        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                            width = int(round(float(image_size[0])))
                            height = int(round(float(image_size[1])))

                        global_index = len(records)
                        frame_token = (
                            f"{method_name}/{dataset_name}/{video_id}/"
                            f"t{time_index:03d}/{camera_name}"
                        )
                        normalized = normalize_frame_record(
                            {
                                "frame_index": global_index,
                                "timestamp": timestamp,
                                "frame_token": frame_token,
                                "rgb_path": str(generated_path),
                                "boxes": [],
                                "calibration": {
                                    "intrinsic_rgb": intrinsic,
                                    "T_ego_from_camera": ego_from_camera,
                                    "raw_image_size": {
                                        "width": width,
                                        "height": height,
                                    },
                                },
                            },
                            manifest_root,
                            global_index,
                        )
                        normalized.update(
                            {
                                "preview_source": method_name,
                                "preview_generated_path": str(generated_path),
                                "preview_real_path": (
                                    str(real_path) if real_path is not None else None
                                ),
                                "preview_manifest_path": str(manifest_path),
                                "preview_dataset_name": dataset_name,
                                "preview_video_id": video_id,
                                "preview_video_signature": video_signature,
                                "preview_time_index": time_index,
                                "preview_camera_name": camera_name,
                                "preview_view_index": view_index,
                                "preview_is_reference_frame": is_reference,
                            }
                        )
                        records.append(normalized)

                        if len(records) % 10000 == 0:
                            print(
                                f"[PREVIEW_SCAN_PROGRESS] "
                                f"images={len(records)} "
                                f"missing_generated={missing_generated} "
                                f"skipped_reference={skipped_reference}",
                                flush=True,
                            )

                        if (
                            max_records is not None
                            and len(records) >= max_records
                        ):
                            print(
                                "[PREVIEW_AUTO_DISCOVERY] "
                                f"root={root} "
                                f"manifests_seen={selected_manifest_count} "
                                f"images={len(records)} "
                                f"limited_to={max_records}",
                                flush=True,
                            )
                            return records

    if selected_manifest_count == 0:
        raise ValueError(
            "Preview manifests were found, but none matched preview.include_methods "
            "and preview.exclude_methods"
        )
    if not records:
        raise ValueError(
            f"No readable preview images were found under {root}; "
            f"missing_generated={missing_generated}, skipped_reference={skipped_reference}"
        )

    print(
        "[PREVIEW_AUTO_DISCOVERY] "
        f"root={root} manifests={selected_manifest_count} images={len(records)} "
        f"skipped_reference={skipped_reference} "
        f"missing_generated={missing_generated} missing_real={missing_real}",
        flush=True,
    )
    return records


def load_frames_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config.get("paths", {})
    preview_root = paths.get("preview_root")
    limit_frames = int(config.get("runtime", {}).get("limit_frames", 0))
    max_records = limit_frames if limit_frames > 0 else None

    if preview_root:
        frames = load_preview_frames(
            preview_root,
            config.get("preview", {}),
            max_records=max_records,
        )
    else:
        data_dir_value = paths.get("data_dir")
        if not data_dir_value:
            raise KeyError(
                "paths must contain either preview_root or data_dir"
            )

        data_dir = Path(
            str(data_dir_value)
        ).expanduser().resolve()

        if (data_dir / "frames.jsonl").is_file():
            frames = load_unpacked_frames(data_dir)
        elif any(data_dir.glob("**/stflow_manifest.jsonl")):
            frames = load_preview_frames(
                data_dir,
                config.get("preview", {}),
                max_records=max_records,
            )
        else:
            raise FileNotFoundError(
                "Could not find frames.jsonl or "
                f"stflow_manifest.jsonl under {data_dir}"
            )

    shared_box_root = paths.get("shared_box_root")

    if shared_box_root:
        frames = attach_shared_box_images(
            frames=frames,
            shared_box_root=shared_box_root,
            settings=config.get("shared_box", {}),
        )

    return frames

def normalize_frame_record(
    record: dict[str, Any],
    root: Path,
    fallback_index: int,
) -> dict[str, Any]:
    frame_index = int(
        record.get(
            "frame_index",
            record.get("idx", record.get("frame_index_in_pt", fallback_index)),
        )
    )
    timestamp = float(record.get("timestamp", frame_index))
    frame_token = str(record.get("frame_token", record.get("token", frame_index)))

    rgb_path_value = record.get("rgb_path")
    if rgb_path_value is None:
        candidates = (
            root / "rgb" / f"{frame_index:06d}.png",
            root / "rgb" / f"{frame_index:06d}.jpg",
            root / "rgb" / f"{frame_index:06d}.jpeg",
            root / "rgb" / f"{frame_index:06d}.webp",
        )
        rgb_path = next((value for value in candidates if value.is_file()), candidates[0])
    else:
        rgb_path = Path(str(rgb_path_value))
        if not rgb_path.is_absolute():
            rgb_path = root / rgb_path
    if not rgb_path.is_file():
        raise FileNotFoundError(
            f"RGB image for frame {frame_index} does not exist: {rgb_path}"
        )

    boxes: list[dict[str, Any]] = []
    if isinstance(record.get("boxes"), list):
        for box_index, box in enumerate(record["boxes"]):
            boxes.append(
                {
                    "box_index": int(box.get("box_index", box_index)),
                    "center_xyz": [float(value) for value in box["center_xyz"]],
                    "size_lwh": [float(value) for value in box["size_lwh"]],
                    "yaw": float(box["yaw"]),
                    "class_name": str(box["class_name"]).upper(),
                    "instance_token": str(
                        box.get("instance_token", box.get("track_id_numeric", box_index))
                    ),
                }
            )
    else:
        bbox_rows = record.get("bbox", [])
        bbox_names = record.get("bbox_names", [])
        instance_tokens = record.get("ins_tokens", [])
        if len(bbox_rows) != len(bbox_names) or len(bbox_rows) != len(instance_tokens):
            raise ValueError(
                f"bbox, bbox_names and ins_tokens have inconsistent lengths at frame {frame_index}"
            )
        for box_index, row in enumerate(bbox_rows):
            if len(row) < 7:
                raise ValueError(
                    f"Box {box_index} at frame {frame_index} has fewer than seven values"
                )
            boxes.append(
                {
                    "box_index": box_index,
                    "center_xyz": [float(row[0]), float(row[1]), float(row[2])],
                    "size_lwh": [float(row[3]), float(row[4]), float(row[5])],
                    "yaw": float(row[6]),
                    "class_name": str(bbox_names[box_index]).upper(),
                    "instance_token": str(instance_tokens[box_index]),
                }
            )

    return {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "frame_token": frame_token,
        "rgb_path": str(rgb_path.resolve()),
        "boxes": boxes,
        "calibration": normalize_calibration(record),
        "ego_pose": record.get("ego_pose", {}),
    }


def normalize_calibration(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("calibration"), dict):
        calibration = record["calibration"]
        if "T_camera_from_ego" in calibration:
            camera_from_ego = np.asarray(
                calibration["T_camera_from_ego"],
                dtype=np.float64,
            )
        elif "T_ego_from_camera" in calibration:
            ego_from_camera = np.asarray(
                calibration["T_ego_from_camera"],
                dtype=np.float64,
            )
            camera_from_ego = np.linalg.inv(ego_from_camera)
        else:
            rotation = np.asarray(calibration["rotation_raw"], dtype=np.float64)
            translation = np.asarray(calibration["translation_raw"], dtype=np.float64)
            ego_from_camera = np.eye(4, dtype=np.float64)
            ego_from_camera[:3, :3] = rotation
            ego_from_camera[:3, 3] = translation
            camera_from_ego = np.linalg.inv(ego_from_camera)

        if "intrinsic_rgb_assuming_resize_then_center_crop" in calibration:
            intrinsic = np.asarray(
                calibration["intrinsic_rgb_assuming_resize_then_center_crop"],
                dtype=np.float64,
            )
            intrinsic_mode = "rgb"
        elif "intrinsic_rgb" in calibration:
            intrinsic = np.asarray(calibration["intrinsic_rgb"], dtype=np.float64)
            intrinsic_mode = "rgb"
        else:
            intrinsic = np.asarray(calibration["intrinsic_raw"], dtype=np.float64)
            intrinsic_mode = "raw"

        raw_size = calibration.get("raw_image_size", {})
        return {
            "intrinsic": intrinsic,
            "intrinsic_mode": intrinsic_mode,
            "raw_width": int(raw_size.get("width", calibration.get("width", 0))),
            "raw_height": int(raw_size.get("height", calibration.get("height", 0))),
            "camera_from_ego": camera_from_ego,
        }

    calibration = record.get("calib")
    if not isinstance(calibration, dict):
        raise KeyError("Each frame must contain either calibration or calib")

    intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
    rotation = np.asarray(calibration["rotation"], dtype=np.float64)
    translation = np.asarray(calibration["translation"], dtype=np.float64)
    ego_from_camera = np.eye(4, dtype=np.float64)
    ego_from_camera[:3, :3] = rotation
    ego_from_camera[:3, 3] = translation
    camera_from_ego = np.linalg.inv(ego_from_camera)
    return {
        "intrinsic": intrinsic,
        "intrinsic_mode": "raw",
        "raw_width": int(calibration["width"]),
        "raw_height": int(calibration["height"]),
        "camera_from_ego": camera_from_ego,
    }


def build_source_items(
    frames: list[dict[str, Any]],
    config: dict[str, Any],
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    sources = config.get("sources", {"real": {"type": "unpacked_rgb"}})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sources must contain at least one source")

    limit_frames = int(config.get("runtime", {}).get("limit_frames", 0))
    selected_frames = frames[:limit_frames] if limit_frames > 0 else frames
    items: list[dict[str, Any]] = []

    for frame_position, frame in enumerate(selected_frames):
        if frame_position % world_size != rank:
            continue
        real_name = Path(frame["rgb_path"]).name
        for source_name, source_config in sources.items():
            source_type = str(source_config.get("type", "image_dir"))
            if source_type == "unpacked_rgb":
                image_path = Path(frame["rgb_path"])
                effective_source_name = str(source_name)
            elif source_type == "image_dir":
                image_dir = Path(source_config["image_dir"]).expanduser().resolve()
                image_path = image_dir / real_name
                if not image_path.is_file():
                    stem = Path(real_name).stem
                    alternatives = (
                        image_dir / f"{stem}.png",
                        image_dir / f"{stem}.jpg",
                        image_dir / f"{stem}.jpeg",
                        image_dir / f"{stem}.webp",
                    )
                    image_path = next(
                        (value for value in alternatives if value.is_file()),
                        image_path,
                    )
                effective_source_name = str(source_name)
            elif source_type == "preview_generated":
                image_path = Path(frame["preview_generated_path"])
                if bool(source_config.get("group_by_manifest", True)):
                    effective_source_name = str(frame["preview_source"])
                else:
                    effective_source_name = str(source_name)
            elif source_type == "preview_real":
                real_path = frame.get("preview_real_path")
                if not real_path:
                    continue
                image_path = Path(str(real_path))
                if bool(source_config.get("group_by_manifest", False)):
                    effective_source_name = (
                        f"{source_name}__{frame['preview_source']}"
                    )
                else:
                    effective_source_name = str(source_name)
            else:
                raise ValueError(
                    f"Unsupported source type {source_type} for source {source_name}"
                )

            item = {
                "source": effective_source_name,
                "image_path": str(image_path),
                "frame": frame,
            }
            box_image_path = frame.get("box_image_path")
            if box_image_path:
                item["box_image_path"] = str(box_image_path)
            items.append(item)
    return items


def adjusted_intrinsic(
    calibration: dict[str, Any],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64).copy()
    if calibration["intrinsic_mode"] == "rgb":
        return intrinsic

    raw_width = int(calibration["raw_width"])
    raw_height = int(calibration["raw_height"])
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError("Raw calibration image size must be positive")

    scale = image_width / float(raw_width)
    resized_height = raw_height * scale
    crop_top = (resized_height - image_height) * 0.5
    intrinsic[0, 0] *= scale
    intrinsic[1, 1] *= scale
    intrinsic[0, 2] *= scale
    intrinsic[1, 2] = intrinsic[1, 2] * scale - crop_top
    return intrinsic


def create_box_corners(box: dict[str, Any]) -> np.ndarray:
    center = np.asarray(box["center_xyz"], dtype=np.float64)
    length, width, height = [float(value) for value in box["size_lwh"]]
    local = np.asarray(
        [
            [length * 0.5, width * 0.5, -height * 0.5],
            [length * 0.5, -width * 0.5, -height * 0.5],
            [-length * 0.5, -width * 0.5, -height * 0.5],
            [-length * 0.5, width * 0.5, -height * 0.5],
            [length * 0.5, width * 0.5, height * 0.5],
            [length * 0.5, -width * 0.5, height * 0.5],
            [-length * 0.5, -width * 0.5, height * 0.5],
            [-length * 0.5, width * 0.5, height * 0.5],
        ],
        dtype=np.float64,
    )

    yaw = float(box["yaw"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return local @ rotation.T + center[None, :]


def project_camera_point(point_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homogeneous = intrinsic @ point_camera
    if homogeneous[2] <= 0:
        raise ValueError("Cannot project a point with non-positive camera depth")
    return (homogeneous[:2] / homogeneous[2]).astype(np.float64)


def clip_convex_polygon_to_image(
    points: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, float, float] | None:
    raw_hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if raw_hull.shape[0] < 3:
        return None

    raw_area = float(abs(cv2.contourArea(raw_hull.astype(np.float32))))
    if raw_area <= 0:
        return None

    image_polygon = np.asarray(
        [
            [0.0, 0.0],
            [float(image_width - 1), 0.0],
            [float(image_width - 1), float(image_height - 1)],
            [0.0, float(image_height - 1)],
        ],
        dtype=np.float32,
    )
    clipped_area, clipped_polygon = cv2.intersectConvexConvex(
        raw_hull.astype(np.float32),
        image_polygon,
    )
    if clipped_polygon is None or float(clipped_area) <= 0:
        return None

    clipped = clipped_polygon.reshape(-1, 2).astype(np.float64)
    if clipped.shape[0] < 3:
        return None
    return clipped, raw_area, float(clipped_area)


def project_box(
    box: dict[str, Any],
    calibration: dict[str, Any],
    image_width: int,
    image_height: int,
    near_plane: float,
) -> dict[str, Any] | None:
    intrinsic = adjusted_intrinsic(calibration, image_width, image_height)
    camera_from_ego = np.asarray(calibration["camera_from_ego"], dtype=np.float64)
    corners_ego = create_box_corners(box)
    corners_h = np.concatenate(
        [corners_ego, np.ones((8, 1), dtype=np.float64)],
        axis=1,
    )
    corners_camera = (camera_from_ego @ corners_h.T).T[:, :3]

    center_ego = np.asarray([*box["center_xyz"], 1.0], dtype=np.float64)
    center_camera = (camera_from_ego @ center_ego)[:3]
    if center_camera[2] <= 0 and np.all(corners_camera[:, 2] < near_plane):
        return None

    projected_segments: list[list[list[float]]] = []
    projected_points: list[np.ndarray] = []
    for first_index, second_index in BOX_EDGE_INDICES:
        first = corners_camera[first_index].copy()
        second = corners_camera[second_index].copy()
        first_inside = first[2] >= near_plane
        second_inside = second[2] >= near_plane
        if not first_inside and not second_inside:
            continue
        if first_inside != second_inside:
            denominator = second[2] - first[2]
            if abs(denominator) < 1e-12:
                continue
            interpolation = (near_plane - first[2]) / denominator
            clipped = first + interpolation * (second - first)
            if not first_inside:
                first = clipped
            else:
                second = clipped

        first_pixel = project_camera_point(first, intrinsic)
        second_pixel = project_camera_point(second, intrinsic)
        projected_points.extend([first_pixel, second_pixel])
        projected_segments.append([first_pixel.tolist(), second_pixel.tolist()])

    if len(projected_points) < 3:
        return None

    polygon_result = clip_convex_polygon_to_image(
        np.asarray(projected_points, dtype=np.float64),
        image_width,
        image_height,
    )
    if polygon_result is None:
        return None
    clipped_polygon, raw_area, clipped_area = polygon_result

    x0, y0 = clipped_polygon.min(axis=0)
    x1, y1 = clipped_polygon.max(axis=0)
    visible_fraction = clipped_area / max(raw_area, 1e-9)

    center = np.asarray(box["center_xyz"], dtype=np.float64)
    height = float(box["size_lwh"][2])
    bottom_ego = np.asarray(
        [center[0], center[1], center[2] - height * 0.5, 1.0],
        dtype=np.float64,
    )
    bottom_camera = (camera_from_ego @ bottom_ego)[:3]
    if bottom_camera[2] >= near_plane:
        bottom_pixel = project_camera_point(bottom_camera, intrinsic)
        bottom_pixel[0] = np.clip(bottom_pixel[0], 0.0, float(image_width - 1))
        bottom_pixel[1] = np.clip(bottom_pixel[1], 0.0, float(image_height - 1))
    else:
        bottom_pixel = np.asarray([(x0 + x1) * 0.5, y1], dtype=np.float64)

    positive_depths = corners_camera[corners_camera[:, 2] >= near_plane, 2]
    nearest_depth = (
        float(positive_depths.min())
        if positive_depths.size > 0
        else float(max(center_camera[2], near_plane))
    )

    return {
        "gt_id": str(box["instance_token"]),
        "box_index": int(box["box_index"]),
        "class_name": str(box["class_name"]),
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "polygon_xy": clipped_polygon.tolist(),
        "bottom_center_xy": bottom_pixel.tolist(),
        "projected_height_px": float(y1 - y0),
        "projected_area_px": float(clipped_area),
        "in_frame_fraction": float(visible_fraction),
        "camera_depth_m": float(center_camera[2]),
        "nearest_camera_depth_m": nearest_depth,
        "cuboid_segments": projected_segments,
    }


def project_frame_boxes(
    frame: dict[str, Any],
    image_width: int,
    image_height: int,
    annotation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_classes = {
        str(value).upper()
        for value in annotation_config["classes"]
    }
    near_plane = float(annotation_config["near_plane"])
    min_height = float(annotation_config["min_projected_height_px"])
    min_area = float(annotation_config["min_projected_area_px"])
    min_in_frame = float(
        annotation_config.get(
            "min_in_frame_fraction",
            annotation_config.get("min_visible_fraction", 0.10),
        )
    )

    projections: list[dict[str, Any]] = []
    for box in frame["boxes"]:
        if str(box["class_name"]).upper() not in allowed_classes:
            continue
        projection = project_box(
            box=box,
            calibration=frame["calibration"],
            image_width=image_width,
            image_height=image_height,
            near_plane=near_plane,
        )
        if projection is None:
            continue
        if projection["projected_height_px"] < min_height:
            continue
        if projection["projected_area_px"] < min_area:
            continue
        if projection["in_frame_fraction"] < min_in_frame:
            continue
        projections.append(projection)
    return projections