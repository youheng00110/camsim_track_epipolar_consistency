from __future__ import annotations

import csv
import json
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PRIVATE_MASK_KEYS = {
    "_mask_crop",
    "_mask_origin_xy",
}


def rasterize_projection_polygon(
    polygon_xy: list[list[float]],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return mask.astype(bool)
    polygon[:, 0] = np.clip(np.rint(polygon[:, 0]), 0, image_width - 1)
    polygon[:, 1] = np.clip(np.rint(polygon[:, 1]), 0, image_height - 1)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


def analyze_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        return {
            "mask_crop": np.zeros((0, 0), dtype=bool),
            "origin_xy": [0, 0],
            "pixel_count": 0,
            "largest_component_pixels": 0,
            "component_count": 0,
            "center_xy": [0.0, 0.0],
            "bottom_center_xy": [0.0, 0.0],
            "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
        }

    x0 = int(columns.min())
    y0 = int(rows.min())
    x1 = int(columns.max()) + 1
    y1 = int(rows.max()) + 1
    crop = binary[y0:y1, x0:x1].copy()
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        crop.astype(np.uint8),
        connectivity=8,
    )
    largest_component_pixels = 0
    if component_count > 1:
        largest_component_pixels = int(stats[1:, cv2.CC_STAT_AREA].max())

    bottom_row = int(rows.max())
    bottom_columns = columns[rows >= max(0, bottom_row - 1)]
    bottom_column = float(np.median(bottom_columns)) if bottom_columns.size else float(columns.mean())

    return {
        "mask_crop": crop,
        "origin_xy": [x0, y0],
        "pixel_count": int(rows.size),
        "largest_component_pixels": largest_component_pixels,
        "component_count": max(0, int(component_count - 1)),
        "center_xy": [float(columns.mean()), float(rows.mean())],
        "bottom_center_xy": [bottom_column, float(bottom_row + 1)],
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros_like(binary, dtype=bool)
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return labels == largest_label


def prepare_evaluation_instances(
    projections: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    visibility_config: dict[str, Any],
    matching_config: dict[str, Any],
) -> dict[str, Any]:
    min_gt_visible_ratio = float(visibility_config.get("min_gt_visible_ratio", 0.15))
    min_gt_connected_pixels = int(
        visibility_config.get("min_gt_visible_connected_pixels", 1)
    )

    disable_gt_occlusion = bool(
        visibility_config.get("disable_gt_occlusion", False)
    )
    occupied = np.zeros((image_height, image_width), dtype=bool)
    prepared_by_index: dict[int, dict[str, Any]] = {}
    depth_order = sorted(
        range(len(projections)),
        key=lambda index: float(
            projections[index].get(
                "nearest_camera_depth_m",
                projections[index].get("camera_depth_m", 1e9),
            )
        ),
    )

    for projection_index in depth_order:
        projection = dict(projections[projection_index])
        amodal_mask = rasterize_projection_polygon(
            projection["polygon_xy"],
            image_width,
            image_height,
        )
        amodal_pixels = int(amodal_mask.sum())
        if disable_gt_occlusion:
            visible_mask = amodal_mask
        else:
            visible_mask = np.logical_and(
                amodal_mask,
                np.logical_not(occupied),
            )
            occupied = np.logical_or(occupied, amodal_mask)
        visible_analysis = analyze_binary_mask(visible_mask)
        visible_pixels = int(visible_analysis["pixel_count"])
        visible_ratio = visible_pixels / max(amodal_pixels, 1)

        projection["amodal_pixel_count"] = amodal_pixels
        projection["visible_pixel_count"] = visible_pixels
        projection["occlusion_visible_ratio"] = float(visible_ratio)
        projection["occluded_fraction"] = float(1.0 - visible_ratio)
        projection["visible_connected_pixel_count"] = int(
            visible_analysis["largest_component_pixels"]
        )
        projection["visible_component_count"] = int(
            visible_analysis["component_count"]
        )
        projection["visible_center_xy"] = visible_analysis["center_xy"]
        projection["visible_bottom_center_xy"] = visible_analysis[
            "bottom_center_xy"
        ]
        projection["_mask_crop"] = visible_analysis["mask_crop"]
        projection["_mask_origin_xy"] = visible_analysis["origin_xy"]

        ignore_reason = None
        if visible_pixels == 0:
            ignore_reason = "fully_occluded"
        elif visible_ratio < min_gt_visible_ratio:
            ignore_reason = "occluded"
        elif int(visible_analysis["largest_component_pixels"]) < min_gt_connected_pixels:
            ignore_reason = "visible_component_too_small"
        projection["ignore_reason"] = ignore_reason
        prepared_by_index[projection_index] = projection

    prepared_projections = [prepared_by_index[index] for index in range(len(projections))]
    evaluated_projections = [
        value for value in prepared_projections if value["ignore_reason"] is None
    ]
    ignored_projections = [
        value for value in prepared_projections if value["ignore_reason"] is not None
    ]

    # SAM prediction filtering must not depend on GT instance size.
    # A value of 0 disables non-empty mask area filtering.
    sam_min_connected_pixels = int(
        matching_config.get("min_sam_connected_pixels", 0)
    )
    if sam_min_connected_pixels < 0:
        raise ValueError(
            "matching.min_sam_connected_pixels must be non-negative"
        )

    allow_bbox_fallback = bool(
        matching_config.get("allow_bbox_mask_fallback", False)
    )
    evaluated_detections: list[dict[str, Any]] = []
    filtered_detections: list[dict[str, Any]] = []

    for detection_index, detection_value in enumerate(detections):
        detection = dict(detection_value)
        detection["raw_detection_index"] = detection_index
        raw_mask = detection.get("mask")
        mask_source = "sam_mask"

        if raw_mask is not None:
            full_mask = cv2.resize(
                np.asarray(raw_mask, dtype=np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        elif allow_bbox_fallback:
            mask_source = "bbox_fallback"
            full_mask = np.zeros((image_height, image_width), dtype=bool)
            x0, y0, x1, y1 = detection["bbox_xyxy"]
            ix0 = max(0, min(image_width, int(np.floor(x0))))
            iy0 = max(0, min(image_height, int(np.floor(y0))))
            ix1 = max(0, min(image_width, int(np.ceil(x1))))
            iy1 = max(0, min(image_height, int(np.ceil(y1))))
            if ix1 > ix0 and iy1 > iy0:
                full_mask[iy0:iy1, ix0:ix1] = True
        else:
            detection["filter_reason"] = "missing_mask"
            detection["connected_pixel_count"] = 0
            detection["mask_source"] = "missing"
            filtered_detections.append(detection)
            continue

        largest_mask = keep_largest_component(full_mask)
        mask_analysis = analyze_binary_mask(largest_mask)
        connected_pixels = int(mask_analysis["largest_component_pixels"])
        detection["connected_pixel_count"] = connected_pixels
        detection["mask_pixel_count"] = int(mask_analysis["pixel_count"])
        detection["mask_component_count"] = int(mask_analysis["component_count"])
        detection["mask_source"] = mask_source
        detection["mask_center_xy"] = mask_analysis["center_xy"]
        detection["bottom_center_xy"] = mask_analysis["bottom_center_xy"]
        detection["_mask_crop"] = mask_analysis["mask_crop"]
        detection["_mask_origin_xy"] = mask_analysis["origin_xy"]

        if connected_pixels <= 0:
            detection["filter_reason"] = "empty_mask"
            filtered_detections.append(detection)
            continue
        if sam_min_connected_pixels > 0 and connected_pixels < sam_min_connected_pixels:
            detection["filter_reason"] = "smaller_than_min_gt_connected_pixels"
            filtered_detections.append(detection)
            continue

        detection["filter_reason"] = None
        evaluated_detections.append(detection)

    return {
        "raw_projections": prepared_projections,
        "projections": evaluated_projections,
        "ignored_projections": ignored_projections,
        "raw_detections": detections,
        "detections": evaluated_detections,
        "filtered_detections": filtered_detections,
        "sam_min_connected_pixels": int(sam_min_connected_pixels),
    }


def cropped_mask_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_mask = np.asarray(first["_mask_crop"], dtype=bool)
    second_mask = np.asarray(second["_mask_crop"], dtype=bool)
    first_x, first_y = [int(value) for value in first["_mask_origin_xy"]]
    second_x, second_y = [int(value) for value in second["_mask_origin_xy"]]

    first_x1 = first_x + first_mask.shape[1]
    first_y1 = first_y + first_mask.shape[0]
    second_x1 = second_x + second_mask.shape[1]
    second_y1 = second_y + second_mask.shape[0]

    overlap_x0 = max(first_x, second_x)
    overlap_y0 = max(first_y, second_y)
    overlap_x1 = min(first_x1, second_x1)
    overlap_y1 = min(first_y1, second_y1)
    intersection = 0

    if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
        first_slice = first_mask[
            overlap_y0 - first_y : overlap_y1 - first_y,
            overlap_x0 - first_x : overlap_x1 - first_x,
        ]
        second_slice = second_mask[
            overlap_y0 - second_y : overlap_y1 - second_y,
            overlap_x0 - second_x : overlap_x1 - second_x,
        ]
        intersection = int(np.logical_and(first_slice, second_slice).sum())

    first_area = int(first_mask.sum())
    second_area = int(second_mask.sum())
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_detections(
    projections: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gt_count = len(projections)
    detection_count = len(detections)
    if gt_count == 0 or detection_count == 0:
        return {
            "matches": [],
            "unmatched_gt_indices": list(range(gt_count)),
            "unmatched_detection_indices": list(range(detection_count)),
        }

    invalid_cost = 1e6
    cost_matrix = np.full((gt_count, detection_count), invalid_cost, dtype=np.float64)
    pair_metrics: dict[tuple[int, int], dict[str, float]] = {}

    for gt_index, projection in enumerate(projections):
        gt_box = np.asarray(projection["bbox_xyxy"], dtype=np.float64)
        gt_center = np.asarray(projection["visible_center_xy"], dtype=np.float64)
        gt_bottom = np.asarray(projection["visible_bottom_center_xy"], dtype=np.float64)
        gt_height = max(1.0, float(gt_box[3] - gt_box[1]))
        gt_diagonal = max(1.0, float(np.linalg.norm(gt_box[2:4] - gt_box[0:2])))
        gt_pixels = max(1, int(projection["visible_pixel_count"]))

        for detection_index, detection in enumerate(detections):
            detection_center = np.asarray(detection["mask_center_xy"], dtype=np.float64)
            detection_bottom = np.asarray(detection["bottom_center_xy"], dtype=np.float64)
            detection_pixels = max(1, int(detection["mask_pixel_count"]))

            mask_iou = cropped_mask_iou(projection, detection)
            center_error = float(
                np.linalg.norm(gt_center - detection_center) / gt_diagonal
            )
            bottom_error = float(
                np.linalg.norm(gt_bottom - detection_bottom) / gt_height
            )
            scale_error = float(
                0.5 * abs(np.log(detection_pixels / float(gt_pixels)))
            )

            if (
                mask_iou < float(config["min_mask_iou"])
                and center_error > float(config["max_center_error_norm"])
            ):
                continue
            if scale_error > float(config["max_scale_error_log"]):
                continue

            cost = (
                float(config.get("cost_mask_iou", 0.65)) * (1.0 - mask_iou)
                + float(config.get("cost_center", 0.20)) * center_error
                + float(config.get("cost_bottom", 0.05)) * bottom_error
                + float(config.get("cost_scale", 0.10)) * scale_error
            )
            if cost > float(config["max_cost"]):
                continue

            cost_matrix[gt_index, detection_index] = cost
            pair_metrics[(gt_index, detection_index)] = {
                "cost": float(cost),
                "mask_iou": float(mask_iou),
                "center_error_norm": float(center_error),
                "bottom_error_norm": float(bottom_error),
                "scale_error_log": float(scale_error),
            }

    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    matches: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_detections: set[int] = set()

    for gt_index, detection_index in zip(
        row_indices.tolist(),
        column_indices.tolist(),
    ):
        if cost_matrix[gt_index, detection_index] >= invalid_cost:
            continue
        match = dict(pair_metrics[(gt_index, detection_index)])
        match["gt_index"] = gt_index
        match["detection_index"] = detection_index
        match["gt_id"] = projections[gt_index]["gt_id"]
        matches.append(match)
        matched_gt.add(gt_index)
        matched_detections.add(detection_index)

    return {
        "matches": matches,
        "unmatched_gt_indices": [
            index for index in range(gt_count) if index not in matched_gt
        ],
        "unmatched_detection_indices": [
            index for index in range(detection_count) if index not in matched_detections
        ],
    }


def serializable_instance(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in PRIVATE_MASK_KEYS or key == "mask":
            continue
        if isinstance(item, np.generic):
            result[key] = item.item()
        elif isinstance(item, np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    return result


def build_frame_result(
    item: dict[str, Any],
    prepared: dict[str, Any],
    matching: dict[str, Any],
) -> dict[str, Any]:
    projections = prepared["projections"]
    detections = prepared["detections"]
    gt_count = len(projections)
    detection_count = len(detections)
    matched_count = len(matching["matches"])
    recall = matched_count / gt_count if gt_count else 1.0
    precision = (
        matched_count / detection_count
        if detection_count
        else (1.0 if gt_count == 0 else 0.0)
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    result = {
        "source": item["source"],
        "frame_index": int(item["frame"]["frame_index"]),
        "frame_token": str(item["frame"]["frame_token"]),
        "timestamp": float(item["frame"]["timestamp"]),
        "image_path": str(item["image_path"]),
        "box_image_path": item.get("box_image_path"),
        "image_width": int(item["width"]),
        "image_height": int(item["height"]),
        "raw_gt_count": len(prepared["raw_projections"]),
        "ignored_occluded_gt_count": len(prepared["ignored_projections"]),
        "gt_count": gt_count,
        "raw_detection_count": len(prepared["raw_detections"]),
        "filtered_small_detection_count": len(prepared["filtered_detections"]),
        "detection_count": detection_count,
        "sam_min_connected_pixels": int(prepared["sam_min_connected_pixels"]),
        "matched_count": matched_count,
        "false_negative_count": gt_count - matched_count,
        "false_positive_count": detection_count - matched_count,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "matched_gt_ids": [str(value["gt_id"]) for value in matching["matches"]],
        "projections": [serializable_instance(value) for value in projections],
        "ignored_projections": [
            serializable_instance(value) for value in prepared["ignored_projections"]
        ],
        "detections": [serializable_instance(value) for value in detections],
        "filtered_detections": [
            serializable_instance(value) for value in prepared["filtered_detections"]
        ],
        "matching": matching,
    }
    preview_fields = {
        "preview_manifest_path": "manifest_path",
        "preview_dataset_name": "dataset_name",
        "preview_video_id": "video_id",
        "preview_time_index": "time_index",
        "preview_camera_name": "camera_name",
        "preview_view_index": "view_index",
        "preview_is_reference_frame": "is_reference_frame",
        "box_manifest_path": "box_manifest_path",
        "box_video_id": "box_video_id",
    }
    for frame_key, result_key in preview_fields.items():
        if frame_key in item["frame"]:
            result[result_key] = item["frame"][frame_key]
    return result


def blend_instance_mask(
    overlay: np.ndarray,
    instance: dict[str, Any],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    mask_crop = np.asarray(instance.get("_mask_crop", np.zeros((0, 0))), dtype=bool)
    if mask_crop.size == 0 or not mask_crop.any():
        return
    origin_x, origin_y = [int(value) for value in instance["_mask_origin_xy"]]
    height, width = mask_crop.shape
    target = overlay[origin_y : origin_y + height, origin_x : origin_x + width]
    if target.shape[:2] != mask_crop.shape:
        return
    target[mask_crop] = (
        target[mask_crop].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def render_visualization(
    image_rgb: np.ndarray,
    prepared: dict[str, Any],
    matching: dict[str, Any],
    output_path: str | Path,
    config: dict[str, Any],
) -> None:
    projections = prepared["projections"]
    detections = prepared["detections"]

    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    match_by_gt = {
        int(value["gt_index"]): (match_index, value)
        for match_index, value in enumerate(matching["matches"])
    }
    match_by_detection = {
        int(value["detection_index"]): (match_index, value)
        for match_index, value in enumerate(matching["matches"])
    }
    mask_alpha = float(config["mask_alpha"])

    if bool(config.get("draw_cuboid", True)):
        for gt_index, projection in enumerate(projections):
            color = (70, 210, 70) if gt_index in match_by_gt else (40, 40, 230)
            for segment in projection["cuboid_segments"]:
                first = tuple(int(round(value)) for value in segment[0])
                second = tuple(int(round(value)) for value in segment[1])
                cv2.line(overlay, first, second, color, 1, cv2.LINE_AA)

    for gt_index, projection in enumerate(projections):
        matched = match_by_gt.get(gt_index)
        color = (70, 210, 70) if matched is not None else (40, 40, 230)
        blend_instance_mask(overlay, projection, color, mask_alpha * 0.55)
        polygon = np.asarray(projection["polygon_xy"], dtype=np.int32)
        cv2.polylines(overlay, [polygon], True, color, 2, cv2.LINE_AA)
        x0, y0, _, _ = [int(round(value)) for value in projection["bbox_xyxy"]]
        label = f"M{matched[0]} GT" if matched is not None else f"MISS G{gt_index}"
        cv2.putText(
            overlay,
            label,
            (x0, max(18, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    for detection_index, detection in enumerate(detections):
        matched = match_by_detection.get(detection_index)
        color = (220, 150, 40) if matched is not None else (40, 150, 240)
        blend_instance_mask(overlay, detection, color, mask_alpha)
        if bool(config.get("draw_detection_box", True)):
            x0, y0, x1, y1 = [int(round(value)) for value in detection["bbox_xyxy"]]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
            label = (
                f"M{matched[0]} SAM {detection['score']:.2f}"
                if matched is not None
                else f"UNM {detection['score']:.2f}"
            )
            cv2.putText(
                overlay,
                label,
                (x0, min(overlay.shape[0] - 5, y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                color,
                1,
                cv2.LINE_AA,
            )

    for match_index, match in enumerate(matching["matches"]):
        gt_center = projections[int(match["gt_index"])]["visible_center_xy"]
        detection_center = detections[int(match["detection_index"])]["mask_center_xy"]
        first = tuple(int(round(value)) for value in gt_center)
        second = tuple(int(round(value)) for value in detection_center)
        cv2.line(overlay, first, second, (0, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(overlay, first, 3, (0, 230, 230), -1)
        cv2.putText(
            overlay,
            f"M{match_index} IoU {match['mask_iou']:.2f}",
            (first[0] + 3, first[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (0, 230, 230),
            1,
            cv2.LINE_AA,
        )

    status = (
        f"GT {len(projections)}  SAM {len(detections)}  "
        f"MATCH {len(matching['matches'])}  "
        f"MISS {len(matching['unmatched_gt_indices'])}  "
        f"UNM {len(matching['unmatched_detection_indices'])}  "
        f"MIN_PIX {prepared['sam_min_connected_pixels']}"
    )
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(1260, overlay.shape[1]), 34),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        overlay,
        status,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, int(config["image_quality"])],
    )


def render_pair_crops(
    image_rgb: np.ndarray,
    prepared: dict[str, Any],
    matching: dict[str, Any],
    output_path: str | Path,
    config: dict[str, Any],
) -> None:
    projections = prepared["projections"]
    detections = prepared["detections"]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_height, image_width = image_bgr.shape[:2]

    entries: list[dict[str, Any]] = []
    for match_index, match in enumerate(matching["matches"]):
        entries.append(
            {
                "kind": "match",
                "match_index": match_index,
                "gt": projections[int(match["gt_index"])],
                "detection": detections[int(match["detection_index"])],
                "metric": match,
            }
        )
    for gt_index in matching["unmatched_gt_indices"]:
        entries.append(
            {
                "kind": "miss",
                "gt": projections[int(gt_index)],
                "detection": None,
            }
        )
    for detection_index in matching["unmatched_detection_indices"]:
        entries.append(
            {
                "kind": "unmatched",
                "gt": None,
                "detection": detections[int(detection_index)],
            }
        )
    entries = entries[:30]
    if not entries:
        return

    tile_width = 360
    tile_height = 240
    column_count = 3
    row_count = int(np.ceil(len(entries) / column_count))
    sheet = np.zeros(
        (row_count * tile_height, column_count * tile_width, 3),
        dtype=np.uint8,
    )
    sheet[:] = 24

    for entry_index, entry in enumerate(entries):
        boxes: list[list[float]] = []
        if entry["gt"] is not None:
            boxes.append(entry["gt"]["bbox_xyxy"])
        if entry["detection"] is not None:
            boxes.append(entry["detection"]["bbox_xyxy"])
        if not boxes:
            continue

        x0 = min(value[0] for value in boxes)
        y0 = min(value[1] for value in boxes)
        x1 = max(value[2] for value in boxes)
        y1 = max(value[3] for value in boxes)
        margin_x = max(12.0, max(1.0, x1 - x0) * 0.25)
        margin_y = max(12.0, max(1.0, y1 - y0) * 0.25)
        crop_x0 = max(0, int(np.floor(x0 - margin_x)))
        crop_y0 = max(0, int(np.floor(y0 - margin_y)))
        crop_x1 = min(image_width, int(np.ceil(x1 + margin_x)))
        crop_y1 = min(image_height, int(np.ceil(y1 + margin_y)))
        crop = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        if crop.size == 0:
            continue

        if entry["gt"] is not None:
            gt_polygon = np.asarray(entry["gt"]["polygon_xy"], dtype=np.int32)
            gt_polygon[:, 0] -= crop_x0
            gt_polygon[:, 1] -= crop_y0
            cv2.polylines(crop, [gt_polygon], True, (70, 210, 70), 2, cv2.LINE_AA)
        if entry["detection"] is not None:
            detection_box = entry["detection"]["bbox_xyxy"]
            local_box = [
                int(round(detection_box[0] - crop_x0)),
                int(round(detection_box[1] - crop_y0)),
                int(round(detection_box[2] - crop_x0)),
                int(round(detection_box[3] - crop_y0)),
            ]
            cv2.rectangle(
                crop,
                (local_box[0], local_box[1]),
                (local_box[2], local_box[3]),
                (220, 150, 40),
                2,
            )

        resized = cv2.resize(
            crop,
            (tile_width, tile_height - 30),
            interpolation=cv2.INTER_LINEAR,
        )
        row = entry_index // column_count
        column = entry_index % column_count
        y_start = row * tile_height
        x_start = column * tile_width
        sheet[y_start + 30 : y_start + tile_height, x_start : x_start + tile_width] = resized

        if entry["kind"] == "match":
            metric = entry["metric"]
            label = f"MATCH  mask IoU {metric['mask_iou']:.2f}"
            color = (70, 210, 70)
        elif entry["kind"] == "miss":
            label = "MISS GT"
            color = (40, 40, 230)
        else:
            label = f"UNMATCHED SAM {entry['detection']['score']:.2f}"
            color = (40, 150, 240)

        cv2.putText(
            sheet,
            label,
            (x_start + 8, y_start + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, int(config["image_quality"])],
    )


def aggregate_rank_outputs(
    output_dir: str | Path,
    world_size: int,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    all_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        rank_path = root / f"records.rank{rank:03d}.jsonl"
        if not rank_path.is_file():
            raise FileNotFoundError(f"Missing rank output {rank_path}")
        with rank_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    all_records.append(json.loads(line))
    all_records.sort(key=itemgetter("source", "frame_index"))

    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        source_groups[str(record["source"])].append(record)

    real_matches: dict[int, set[str]] = {}
    for record in source_groups.get("real", []):
        real_matches[int(record["frame_index"])] = set(record["matched_gt_ids"])

    summary_sources: dict[str, Any] = {}
    for source_name, records in source_groups.items():
        raw_gt_count = sum(int(value["raw_gt_count"]) for value in records)
        ignored_gt_count = sum(
            int(value["ignored_occluded_gt_count"]) for value in records
        )
        gt_count = sum(int(value["gt_count"]) for value in records)
        raw_detection_count = sum(
            int(value["raw_detection_count"]) for value in records
        )
        filtered_detection_count = sum(
            int(value["filtered_small_detection_count"]) for value in records
        )
        detection_count = sum(int(value["detection_count"]) for value in records)
        matched_count = sum(int(value["matched_count"]) for value in records)
        recall = matched_count / gt_count if gt_count else 1.0
        precision = (
            matched_count / detection_count
            if detection_count
            else (1.0 if gt_count == 0 else 0.0)
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        pair_metrics = [
            match
            for value in records
            for match in value["matching"]["matches"]
        ]

        source_summary: dict[str, Any] = {
            "frames": len(records),
            "raw_gt_count": raw_gt_count,
            "ignored_occluded_gt_count": ignored_gt_count,
            "gt_count": gt_count,
            "gt_evaluation_fraction": (
                gt_count / raw_gt_count if raw_gt_count else 1.0
            ),
            "raw_detection_count": raw_detection_count,
            "filtered_small_detection_count": filtered_detection_count,
            "detection_count": detection_count,
            "matched_count": matched_count,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "mean_mask_iou": (
                float(np.mean([value["mask_iou"] for value in pair_metrics]))
                if pair_metrics
                else None
            ),
            "mean_center_error_norm": (
                float(np.mean([value["center_error_norm"] for value in pair_metrics]))
                if pair_metrics
                else None
            ),
            "mean_bottom_error_norm": (
                float(np.mean([value["bottom_error_norm"] for value in pair_metrics]))
                if pair_metrics
                else None
            ),
            "mean_scale_error_log": (
                float(np.mean([value["scale_error_log"] for value in pair_metrics]))
                if pair_metrics
                else None
            ),
        }

        if source_name != "real" and real_matches:
            calibrated_gt_count = 0
            calibrated_matched_count = 0
            for record in records:
                valid_ids = real_matches.get(int(record["frame_index"]), set())
                generated_ids = set(record["matched_gt_ids"])
                calibrated_gt_count += len(valid_ids)
                calibrated_matched_count += len(valid_ids.intersection(generated_ids))
            source_summary["real_calibrated_gt_count"] = calibrated_gt_count
            source_summary["real_calibrated_matched_count"] = calibrated_matched_count
            source_summary["real_calibrated_recall"] = (
                calibrated_matched_count / calibrated_gt_count
                if calibrated_gt_count
                else None
            )
        summary_sources[source_name] = source_summary

    summary = {"model": model_info, "sources": summary_sources}
    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    csv_fields = [
        "source",
        "frame_index",
        "raw_gt_count",
        "ignored_occluded_gt_count",
        "gt_count",
        "raw_detection_count",
        "filtered_small_detection_count",
        "detection_count",
        "sam_min_connected_pixels",
        "matched_count",
        "false_negative_count",
        "false_positive_count",
        "recall",
        "precision",
        "f1",
        "image_path",
    ]
    with (root / "per_frame.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for record in all_records:
            writer.writerow({key: record[key] for key in csv_fields})
    return summary