from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _normalize_pose_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, list):
        return [_normalize_pose_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_pose_value(item)
            for key, item in sorted(value.items())
        }
    return value


def video_pose_signature(video_record: dict[str, Any]) -> str:
    frames = video_record.get("frames", [])
    poses: list[Any] = []

    for frame in frames:
        pose = frame.get("T_ego_to_world")
        if pose is None:
            poses = []
            break
        poses.append(_normalize_pose_value(pose))

    if poses:
        payload = json.dumps(
            poses,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "pose:" + hashlib.sha256(payload).hexdigest()

    dataset_name = str(video_record.get("dataset_name", "unknown"))
    video_id = str(video_record.get("video_id", "unknown"))
    return f"id:{dataset_name}:{video_id}"


def _resolve_manifest_image_path(
    manifest_path: Path,
    image_value: str,
) -> Path:
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    return image_path.resolve()


def _matrix_from_value(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1 and matrix.size in (12, 16):
        matrix = matrix.reshape(3, 4) if matrix.size == 12 else matrix.reshape(4, 4)
    if matrix.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"{name} must be 3x4 or 4x4, got {matrix.shape}")
    return matrix


def _create_lidar_box_corners(box: dict[str, Any]) -> np.ndarray:
    box7 = np.asarray(box["box7_lidar"], dtype=np.float64).reshape(-1)
    if box7.size < 7:
        raise ValueError(f"box7_lidar must contain seven values, got {box7.tolist()}")

    x, y, z, length, width, height, yaw = box7[:7].tolist()
    if not bool(box.get("z_is_center", True)):
        z += height * 0.5

    local = np.asarray(
        [
            [length * 0.5, width * 0.5, height * 0.5],
            [length * 0.5, -width * 0.5, height * 0.5],
            [-length * 0.5, -width * 0.5, height * 0.5],
            [-length * 0.5, width * 0.5, height * 0.5],
            [length * 0.5, width * 0.5, -height * 0.5],
            [length * 0.5, -width * 0.5, -height * 0.5],
            [-length * 0.5, -width * 0.5, -height * 0.5],
            [-length * 0.5, width * 0.5, -height * 0.5],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray([x, y, z], dtype=np.float64)


def _create_reference_ego_box_corners(
    box: dict[str, Any],
) -> np.ndarray:
    corners = np.asarray(
        box["corners_ref"],
        dtype=np.float64,
    )
    if corners.shape != (8, 3):
        raise ValueError(
            "corners_ref must have shape [8, 3], "
            f"got {corners.shape}"
        )
    if not np.isfinite(corners).all():
        raise ValueError(
            "corners_ref contains non-finite values"
        )
    return corners


def _canonical_vehicle_class(
    class_name: Any,
) -> str:
    name = str(class_name).strip().lower()

    if name == "car" or name.startswith("vehicle.car"):
        return "CAR"
    if name == "truck" or name.startswith("vehicle.truck"):
        return "TRUCK"
    if name == "bus" or name.startswith("vehicle.bus"):
        return "BUS"

    return str(class_name).strip().upper()


def _view_or_frame_value(
    frame: dict[str, Any],
    view: dict[str, Any],
    view_index: int,
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in view and view[key] is not None:
            return view[key]

    for key in keys:
        if key not in frame or frame[key] is None:
            continue

        value = frame[key]
        array = np.asarray(value)

        if (
            array.ndim >= 3
            and array.shape[0] > view_index
        ):
            return value[view_index]

        return value

    return None


def _resolve_projection_geometry(
    frame: dict[str, Any],
    view: dict[str, Any],
    view_index: int,
    boxes_3d: list[dict[str, Any]],
) -> tuple[Any, Any, str]:
    reference_camera_value = _view_or_frame_value(
        frame,
        view,
        view_index,
        (
            "T_reference_ego_to_camera",
            "reference_ego_to_camera",
            "T_ref_to_camera",
            "reference_to_camera",
            "box_reference_to_camera",
        ),
    )

    has_reference_boxes = any(
        isinstance(box, dict)
        and (
            "corners_ref" in box
            or str(
                box.get("coordinate_frame", "")
            ).lower()
            in {
                "reference_ego",
                "reference-ego",
                "ref_ego",
            }
        )
        for box in boxes_3d
    )

    if reference_camera_value is not None:
        camera_matrix = _matrix_from_value(
            reference_camera_value,
            "T_reference_ego_to_camera",
        )

        projection_value = _view_or_frame_value(
            frame,
            view,
            view_index,
            (
                "reference_ego_to_image",
                "reference_to_image",
                "ref_to_image",
            ),
        )

        if projection_value is None:
            intrinsic_value = _view_or_frame_value(
                frame,
                view,
                view_index,
                (
                    "K",
                    "camera_intrinsic",
                    "camera_intrinsics",
                ),
            )
            if intrinsic_value is None:
                raise KeyError(
                    "reference_ego boxes require either "
                    "reference_ego_to_image or K"
                )

            intrinsic = np.asarray(
                intrinsic_value,
                dtype=np.float64,
            )
            if intrinsic.shape == (4, 4):
                intrinsic = intrinsic[:3, :3]
            if intrinsic.shape != (3, 3):
                raise ValueError(
                    "K must have shape [3,3], "
                    f"got {intrinsic.shape}"
                )

            projection_matrix = (
                intrinsic
                @ camera_matrix[:3, :4]
            )
        else:
            projection_matrix = _matrix_from_value(
                projection_value,
                "reference_ego_to_image",
            )

        return (
            camera_matrix.tolist(),
            projection_matrix.tolist(),
            "reference_ego",
        )

    camera_value = _view_or_frame_value(
        frame,
        view,
        view_index,
        ("T_lidar_to_camera",),
    )
    projection_value = _view_or_frame_value(
        frame,
        view,
        view_index,
        ("lidar_to_image",),
    )

    if (
        camera_value is not None
        and projection_value is not None
    ):
        _matrix_from_value(
            camera_value,
            "T_lidar_to_camera",
        )
        _matrix_from_value(
            projection_value,
            "lidar_to_image",
        )
        return (
            camera_value,
            projection_value,
            "lidar",
        )

    if has_reference_boxes:
        raise KeyError(
            "Found corners_ref/reference_ego boxes, but "
            "no reference-ego -> camera transform. "
            "Expected one of "
            "T_reference_ego_to_camera, "
            "reference_ego_to_camera, "
            "T_ref_to_camera, reference_to_camera, "
            "or box_reference_to_camera."
        )

    raise KeyError(
        "Missing supported box projection geometry. "
        "Expected NuPlan "
        "T_lidar_to_camera + lidar_to_image, or "
        "nuScenes reference-ego -> camera geometry."
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def _project_lidar_point(point: np.ndarray, lidar_to_image: np.ndarray) -> np.ndarray:
    homogeneous = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    projected = lidar_to_image @ homogeneous
    denominator = float(projected[2])
    if denominator <= 1e-12:
        raise ValueError("Projected point has non-positive homogeneous depth")
    return projected[:2] / denominator


def _clip_convex_hull_to_image(
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


BOX_EDGE_INDICES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# OpenDWM nuScenes corners_ref follows
# MotionDataset.default_3dbox_corner_template ordering.
REFERENCE_EGO_BOX_EDGE_INDICES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (4, 6), (5, 7), (6, 7),
)


def project_manifest_boxes(
    boxes_3d: list[dict[str, Any]],
    T_lidar_to_camera: Any,
    lidar_to_image: Any,
    image_width: int,
    image_height: int,
    annotation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    # Argument names are kept for run_eval.py compatibility.
    # For nuScenes these carry reference_ego geometry.
    camera_matrix = _matrix_from_value(
        T_lidar_to_camera,
        "source_to_camera",
    )
    projection_matrix = _matrix_from_value(
        lidar_to_image,
        "source_to_image",
    )

    near_plane = float(
        annotation_config.get("near_plane", 0.10)
    )
    min_height = float(
        annotation_config.get("min_projected_height_px", 0.0)
    )
    min_area = float(
        annotation_config.get("min_projected_area_px", 0.0)
    )
    min_in_frame = float(
        annotation_config.get("min_in_frame_fraction", 0.0)
    )
    allowed_classes = {
        _canonical_vehicle_class(value)
        for value in annotation_config.get("classes", [])
    }

    projections: list[dict[str, Any]] = []

    for box_index, box_value in enumerate(boxes_3d):
        box = dict(box_value)

        if "corners_ref" in box:
            coordinate_frame = str(
                box.get(
                    "coordinate_frame",
                    "reference_ego",
                )
            ).lower()
            if coordinate_frame not in {
                "reference_ego",
                "reference-ego",
                "ref_ego",
            }:
                raise ValueError(
                    "corners_ref requires "
                    "coordinate_frame='reference_ego', "
                    f"got {coordinate_frame!r}"
                )

            corners_source = (
                _create_reference_ego_box_corners(box)
            )
            center_source = corners_source.mean(
                axis=0,
                keepdims=True,
            )
            edge_indices = (
                REFERENCE_EGO_BOX_EDGE_INDICES
            )
            projection_source = (
                "corners_reference_ego_manifest"
            )

        elif "box7_lidar" in box:
            coordinate_frame = str(
                box.get(
                    "coordinate_frame",
                    "lidar",
                )
            ).lower()
            if coordinate_frame != "lidar":
                raise ValueError(
                    "box7_lidar requires "
                    "coordinate_frame='lidar', "
                    f"got {coordinate_frame!r}"
                )

            corners_source = (
                _create_lidar_box_corners(box)
            )
            center_source = np.asarray(
                box["box7_lidar"][:3],
                dtype=np.float64,
            ).reshape(1, 3)
            edge_indices = BOX_EDGE_INDICES
            projection_source = (
                "box7_lidar_manifest"
            )

        else:
            raise KeyError(
                "Each box must contain either "
                "box7_lidar or corners_ref"
            )

        class_name = _canonical_vehicle_class(
            box.get("class_name", "CAR")
        )
        if (
            allowed_classes
            and class_name not in allowed_classes
        ):
            continue

        corners_camera = _transform_points(
            corners_source,
            camera_matrix,
        )
        center_camera = _transform_points(
            center_source,
            camera_matrix,
        )[0]

        projected_segments: list[
            list[list[float]]
        ] = []
        projected_points: list[np.ndarray] = []

        for first_index, second_index in edge_indices:
            first_source = corners_source[
                first_index
            ].copy()
            second_source = corners_source[
                second_index
            ].copy()

            first_depth = float(
                corners_camera[first_index, 2]
            )
            second_depth = float(
                corners_camera[second_index, 2]
            )

            first_inside = first_depth >= near_plane
            second_inside = second_depth >= near_plane

            if not first_inside and not second_inside:
                continue

            if first_inside != second_inside:
                denominator = second_depth - first_depth
                if abs(denominator) < 1e-12:
                    continue

                interpolation = (
                    near_plane - first_depth
                ) / denominator

                clipped_source = (
                    first_source
                    + interpolation
                    * (second_source - first_source)
                )

                if not first_inside:
                    first_source = clipped_source
                else:
                    second_source = clipped_source

            first_pixel = _project_lidar_point(
                first_source,
                projection_matrix,
            )
            second_pixel = _project_lidar_point(
                second_source,
                projection_matrix,
            )

            projected_points.extend(
                [first_pixel, second_pixel]
            )
            projected_segments.append(
                [
                    first_pixel.tolist(),
                    second_pixel.tolist(),
                ]
            )

        if len(projected_points) < 3:
            continue

        hull_result = _clip_convex_hull_to_image(
            np.asarray(
                projected_points,
                dtype=np.float64,
            ),
            image_width,
            image_height,
        )

        if hull_result is None:
            continue

        polygon, raw_area, clipped_area = hull_result
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)

        projected_height = float(y1 - y0)
        in_frame_fraction = float(
            clipped_area / max(raw_area, 1e-9)
        )

        if projected_height < min_height:
            continue
        if clipped_area < min_area:
            continue
        if in_frame_fraction < min_in_frame:
            continue

        positive_depths = corners_camera[
            corners_camera[:, 2] >= near_plane,
            2,
        ]
        nearest_depth = (
            float(positive_depths.min())
            if positive_depths.size > 0
            else float(
                max(
                    center_camera[2],
                    near_plane,
                )
            )
        )

        gt_id = str(
            box.get(
                "gt_id",
                box.get(
                    "instance_token",
                    box.get(
                        "annotation_token",
                        box_index,
                    ),
                ),
            )
        )

        projections.append(
            {
                "gt_id": gt_id,
                "instance_token": gt_id,
                "box_index": int(
                    box.get("box_index", box_index)
                ),
                "class_name": class_name,
                "bbox_xyxy": [
                    float(x0),
                    float(y0),
                    float(x1),
                    float(y1),
                ],
                "polygon_xy": polygon.tolist(),
                "projected_height_px": projected_height,
                "projected_area_px": float(clipped_area),
                "in_frame_fraction": in_frame_fraction,
                "camera_depth_m": float(
                    center_camera[2]
                ),
                "nearest_camera_depth_m": nearest_depth,
                "cuboid_segments": projected_segments,
                "projection_source": projection_source,
            }
        )

    return projections

def build_shared_box_index(
    shared_box_root: str | Path,
    settings: dict[str, Any] | None = None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    root = Path(shared_box_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Shared box root does not exist: {root}")

    config = settings or {}
    manifest_glob = str(config.get("manifest_glob", "**/box_manifest.jsonl"))
    manifest_paths = sorted(path for path in root.glob(manifest_glob) if path.is_file())
    if not manifest_paths:
        raise FileNotFoundError(
            f"No box manifests matched {manifest_glob!r} under {root}"
        )

    print(
        f"[SHARED_BOX_SCAN_START] root={root} manifests={len(manifest_paths)}",
        flush=True,
    )
    camera_name_map = {
        str(key): str(value)
        for key, value in config.get("camera_name_map", {}).items()
    }
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    duplicate_count = 0
    view_count = 0
    video_count = 0

    for manifest_number, manifest_path in enumerate(manifest_paths, start=1):
        print(
            f"[SHARED_BOX_MANIFEST] {manifest_number}/{len(manifest_paths)} "
            f"{manifest_path}",
            flush=True,
        )
        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                video_record = json.loads(stripped)
                if not isinstance(video_record, dict):
                    raise TypeError(
                        f"Line {line_number} in {manifest_path} is not an object"
                    )

                signature = video_pose_signature(video_record)
                video_id = str(video_record.get("video_id", "unknown"))
                frames = video_record.get("frames")
                if frames is None and "views" in video_record:
                    frames = [video_record]
                if not isinstance(frames, list):
                    raise TypeError(
                        f"frames must be a list in {manifest_path}, line={line_number}"
                    )

                for frame_position, frame in enumerate(frames):
                    if not isinstance(frame, dict):
                        raise TypeError(
                            f"Frame {frame_position} in {manifest_path} is not an object"
                        )
                    time_index = int(
                        frame.get("frame_index", frame.get("time_index", frame_position))
                    )
                    boxes_3d = frame.get("boxes_3d", video_record.get("boxes_3d", []))
                    if not isinstance(boxes_3d, list):
                        raise TypeError(
                            f"boxes_3d must be a list in {manifest_path}, frame={time_index}"
                        )
                    views = frame.get("views", [])
                    if not isinstance(views, list):
                        raise TypeError(
                            f"views must be a list in {manifest_path}, frame={time_index}"
                        )

                    for view_index, view in enumerate(views):
                        raw_camera_name = str(
                            view.get("camera", f"CAM_{view_index:02d}")
                        )
                        camera_name = camera_name_map.get(
                            raw_camera_name,
                            raw_camera_name,
                        )
                        (
                            camera_matrix,
                            projection_matrix,
                            projection_coordinate_frame,
                        ) = _resolve_projection_geometry(
                            frame,
                            view,
                            view_index,
                            boxes_3d,
                        )

                        key = (signature, time_index, camera_name)
                        item = {
                            "manifest_path": str(manifest_path),
                            "video_id": video_id,
                            "time_index": time_index,
                            "camera_name": camera_name,
                            "boxes_3d": boxes_3d,
                            # Compatibility keys consumed by run_eval.py.
                            # For nuScenes they carry reference_ego geometry.
                            "T_lidar_to_camera": camera_matrix,
                            "lidar_to_image": projection_matrix,
                            "box_coordinate_frame": projection_coordinate_frame,
                        }
                        image_value = view.get("image_path")
                        if image_value:
                            image_path = _resolve_manifest_image_path(
                                manifest_path,
                                str(image_value),
                            )
                            if image_path.is_file():
                                item["image_path"] = str(image_path)

                        if key in index:
                            duplicate_count += 1
                            continue
                        index[key] = item
                        view_count += 1
                video_count += 1

    if not index:
        raise ValueError(f"No projected box metadata were found under {root}")
    print(
        "[SHARED_BOX_INDEX] "
        f"videos={video_count} views={view_count} "
        f"duplicates={duplicate_count}",
        flush=True,
    )
    return index


def attach_shared_box_images(
    frames: list[dict[str, Any]],
    shared_box_root: str | Path,
    settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = settings or {}
    strict_match = bool(config.get("strict_match", True))
    box_index = build_shared_box_index(shared_box_root, config)
    matched_count = 0
    missing_count = 0
    attached_box_count = 0

    for frame in frames:
        signature = str(frame.get("preview_video_signature", ""))
        time_index = int(frame.get("preview_time_index", 0))
        camera_name = str(frame.get("preview_camera_name", ""))
        key = (signature, time_index, camera_name)
        box_item = box_index.get(key)

        if box_item is None:
            missing_count += 1
            if strict_match:
                raise KeyError(
                    "No shared box metadata matched generated frame: "
                    f"signature={signature}, frame={time_index}, "
                    f"camera={camera_name}, image={frame.get('rgb_path')}"
                )
            continue

        frame["boxes_3d"] = box_item["boxes_3d"]
        frame["T_lidar_to_camera"] = box_item["T_lidar_to_camera"]
        frame["lidar_to_image"] = box_item["lidar_to_image"]
        frame["box_coordinate_frame"] = box_item.get(
            "box_coordinate_frame",
            "lidar",
        )
        frame["box_manifest_path"] = box_item["manifest_path"]
        frame["box_video_id"] = box_item["video_id"]
        if box_item.get("image_path"):
            frame["box_image_path"] = box_item["image_path"]
        attached_box_count += len(box_item["boxes_3d"])
        matched_count += 1

    print(
        "[SHARED_BOX_ATTACH] "
        f"generated_frames={len(frames)} matched={matched_count} "
        f"missing={missing_count} boxes_3d={attached_box_count}",
        flush=True,
    )
    return frames


def extract_box_image_projections(
    box_image: torch.Tensor | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = config or {}

    if isinstance(box_image, torch.Tensor):
        array = box_image.detach().cpu().numpy()
        if array.ndim != 3:
            raise ValueError(
                f"Expected box image with 3 dimensions, got {array.shape}"
            )
        if array.shape[0] in (1, 3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
    else:
        array = np.asarray(box_image)

    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Invalid box image shape: {array.shape}")

    rgb = np.asarray(array[:, :, :3], dtype=np.float32)
    vehicle_colors = settings.get("vehicle_colors_rgb", [[0, 0, 255]])
    color_tolerance = float(settings.get("color_tolerance", 100.0))
    min_brightness = float(settings.get("min_brightness", 20.0))
    close_kernel = int(settings.get("close_kernel", 1))
    min_line_pixels = int(settings.get("min_line_pixels", 16))
    min_bbox_width = int(settings.get("min_bbox_width", 5))
    min_bbox_height = int(settings.get("min_bbox_height", 5))
    min_hull_area = float(settings.get("min_hull_area", 36.0))

    color_mask = np.zeros(rgb.shape[:2], dtype=bool)
    for color_value in vehicle_colors:
        color = np.asarray(color_value, dtype=np.float32).reshape(1, 1, 3)
        distance = np.linalg.norm(rgb - color, axis=2)
        color_mask |= distance <= color_tolerance

    brightness_mask = np.max(rgb, axis=2) >= min_brightness
    binary = np.logical_and(color_mask, brightness_mask).astype(np.uint8)

    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    projections: list[dict[str, Any]] = []

    for component_label in range(1, component_count):
        line_pixels = int(stats[component_label, cv2.CC_STAT_AREA])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])

        if line_pixels < min_line_pixels:
            continue
        if width < min_bbox_width or height < min_bbox_height:
            continue

        rows, columns = np.nonzero(labels == component_label)
        points = np.column_stack((columns, rows)).astype(np.float32)
        hull = cv2.convexHull(points).reshape(-1, 2)

        if hull.shape[0] < 3:
            continue

        hull_area = float(cv2.contourArea(hull.astype(np.float32)))
        if hull_area < min_hull_area:
            continue

        x0 = float(hull[:, 0].min())
        y0 = float(hull[:, 1].min())
        x1 = float(hull[:, 0].max() + 1.0)
        y1 = float(hull[:, 1].max() + 1.0)
        projection_index = len(projections)

        projections.append(
            {
                "gt_id": f"box_image_{projection_index:04d}",
                "box_index": projection_index,
                "class_name": "CAR",
                "instance_token": f"box_image_{projection_index:04d}",
                "polygon_xy": hull.astype(np.float64).tolist(),
                "bbox_xyxy": [x0, y0, x1, y1],
                "projected_height_px": float(y1 - y0),
                "projected_area_px": hull_area,
                "in_frame_fraction": 1.0,
                "camera_depth_m": float(1e9 + projection_index),
                "nearest_camera_depth_m": float(1e9 + projection_index),
                "cuboid_segments": [],
                "projection_source": "shared_box_image_convex_hull",
                "line_pixel_count": line_pixels,
            }
        )

    return projections
