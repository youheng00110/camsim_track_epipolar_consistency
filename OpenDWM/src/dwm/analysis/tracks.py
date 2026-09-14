from __future__ import annotations

ENTITY_REACTOR_TRACKS_VERSION = "v19.4-view-transition-track-pool-20260817"

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class RawDatasetLocation:
    dataset: torch.utils.data.Dataset
    index: int


class NuPlanTrackResolver:
    """Recovers stable NuPlan tracks while preserving the validation dataset API."""

    def __init__(
        self,
        validation_dataset: torch.utils.data.Dataset,
        max_entities: int | None = None,
        allowed_classes: Iterable[str] = ("car", "vehicle", "truck", "bus"),
        minimum_track_length: int = 3,
    ) -> None:
        resolved_max_entities = None
        if max_entities is not None:
            resolved_max_entities = int(max_entities)
            if resolved_max_entities <= 0:
                raise ValueError("max_entities must be positive when provided")
        resolved_minimum_track_length = int(minimum_track_length)
        if resolved_minimum_track_length <= 0:
            raise ValueError("minimum_track_length must be positive")
        self.validation_dataset = validation_dataset
        self.max_entities = resolved_max_entities
        self.allowed_classes = tuple(str(name).lower() for name in allowed_classes)
        self.minimum_track_length = resolved_minimum_track_length

    def resolve(self, dataset_index: int) -> dict:
        location = self._locate_raw_dataset(self.validation_dataset, int(dataset_index))
        raw_dataset = location.dataset
        if not hasattr(raw_dataset, "items") or not hasattr(raw_dataset, "scenes"):
            raise TypeError("the unwrapped validation dataset does not expose NuPlan items/scenes")
        item = raw_dataset.items[location.index]
        if not isinstance(item, dict) or "scene" not in item or "indices" not in item:
            raise TypeError("unexpected NuPlan item format")
        scene = item["scene"]
        frame_indices = list(item["indices"])
        sequence = [raw_dataset.scenes[scene][frame_index] for frame_index in frame_indices]

        statistics: dict[str, dict] = {}
        frame_records: list[list[tuple[str, str, np.ndarray]]] = []
        for frame in sequence:
            boxes = frame.get("gt_boxes", None)
            names = frame.get("gt_names", None)
            tracks = frame.get("track_token", None)
            if tracks is None:
                tracks = frame.get("gt_track_token", None)
            boxes = list(boxes) if isinstance(boxes, (list, tuple, np.ndarray)) else []
            names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else []
            tracks = list(tracks) if isinstance(tracks, (list, tuple, np.ndarray)) else []
            current_records: list[tuple[str, str, np.ndarray]] = []
            for box_index, box in enumerate(boxes):
                if box_index >= len(tracks) or tracks[box_index] is None:
                    continue
                track_token = str(tracks[box_index])
                if not track_token:
                    continue
                class_name = str(names[box_index]) if box_index < len(names) else "unknown"
                if not self._class_is_allowed(class_name):
                    continue
                box_array = np.asarray(box[:7], dtype=np.float32).reshape(-1)
                if box_array.shape[0] != 7 or not np.isfinite(box_array).all():
                    continue
                current_records.append((track_token, class_name, box_array))
                current_statistics = statistics.setdefault(
                    track_token,
                    {
                        "count": 0,
                        "minimum_range": float("inf"),
                        "class_name": class_name,
                        "centers": [],
                        "depths": [],
                    },
                )
                current_statistics["count"] += 1
                current_statistics["minimum_range"] = min(
                    current_statistics["minimum_range"],
                    float(np.linalg.norm(box_array[:2])),
                )
                current_statistics["centers"].append(box_array[:3].copy())
                current_statistics["depths"].append(float(box_array[0]))
            frame_records.append(current_records)

        candidates = []
        for token, values in statistics.items():
            if int(values["count"]) < self.minimum_track_length:
                continue
            centers = np.asarray(values["centers"], dtype=np.float32)
            motion_span = float(np.linalg.norm(centers[-1, :2] - centers[0, :2])) if centers.shape[0] > 1 else 0.0
            spatial_span = float(np.linalg.norm(np.ptp(centers[:, :2], axis=0))) if centers.shape[0] > 1 else 0.0
            values["motion_span"] = motion_span
            values["spatial_span"] = spatial_span
            candidates.append(token)
        candidates.sort(
            key=lambda token: (
                -int(statistics[token]["count"]),
                -float(statistics[token]["spatial_span"]),
                -float(statistics[token]["motion_span"]),
                token,
            )
        )
        selected_tokens = candidates if self.max_entities is None else candidates[: self.max_entities]
        if not selected_tokens:
            raise RuntimeError(
                f"NuPlan sample {dataset_index} contains no stable allowed track with length "
                f">= {self.minimum_track_length}"
            )

        lidar_to_camera = self._build_lidar_to_camera(raw_dataset, sequence)
        slot_by_token = {token: slot for slot, token in enumerate(selected_tokens)}
        time_count = len(sequence)
        slot_count = len(selected_tokens)
        corners = np.zeros((time_count, slot_count, 8, 3), dtype=np.float32)
        boxes7 = np.zeros((time_count, slot_count, 7), dtype=np.float32)
        valid = np.zeros((time_count, slot_count), dtype=np.bool_)
        class_id = np.zeros((slot_count,), dtype=np.int64)
        track_hash = np.zeros((slot_count,), dtype=np.int64)

        for slot, token in enumerate(selected_tokens):
            class_id[slot] = self._class_to_id(statistics[token]["class_name"])
            track_hash[slot] = self._stable_hash(token)
        for time_index, records in enumerate(frame_records):
            for track_token, _, box_array in records:
                slot = slot_by_token.get(track_token)
                if slot is None:
                    continue
                corners[time_index, slot] = self._box_corners_lidar(box_array)
                boxes7[time_index, slot] = box_array
                valid[time_index, slot] = True

        return {
            "corners_lidar": torch.from_numpy(corners),
            "boxes7_lidar": torch.from_numpy(boxes7),
            "valid": torch.from_numpy(valid),
            "class_id": torch.from_numpy(class_id),
            "track_hash": torch.from_numpy(track_hash),
            "track_tokens": selected_tokens,
            "lidar_to_camera": torch.from_numpy(lidar_to_camera),
            "scene": str(scene),
            "raw_index": int(location.index),
            "frame_indices": frame_indices,
        }

    def _build_lidar_to_camera(
        self,
        raw_dataset: torch.utils.data.Dataset,
        sequence: list[dict],
    ) -> np.ndarray:
        sensor_channels = list(getattr(raw_dataset, "sensor_channels", []))
        if not sensor_channels:
            raise RuntimeError("raw NuPlan dataset does not expose sensor_channels")
        transforms = np.zeros((len(sequence), len(sensor_channels), 4, 4), dtype=np.float32)
        for time_index, frame in enumerate(sequence):
            for view_index, channel in enumerate(sensor_channels):
                cam_info = raw_dataset._get_cam_info(frame, channel) if hasattr(raw_dataset, "_get_cam_info") else None
                if cam_info is None:
                    transforms[time_index, view_index] = np.eye(4, dtype=np.float32)
                    continue
                rotation_camera_to_lidar = np.asarray(
                    cam_info["sensor2lidar_rotation"],
                    dtype=np.float32,
                ).reshape(3, 3)
                translation_camera_to_lidar = np.asarray(
                    cam_info["sensor2lidar_translation"],
                    dtype=np.float32,
                ).reshape(3)
                rotation_lidar_to_camera = rotation_camera_to_lidar.T
                translation_lidar_to_camera = -rotation_lidar_to_camera @ translation_camera_to_lidar
                current_transform = np.eye(4, dtype=np.float32)
                current_transform[:3, :3] = rotation_lidar_to_camera
                current_transform[:3, 3] = translation_lidar_to_camera
                transforms[time_index, view_index] = current_transform
        return transforms

    def _locate_raw_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        index: int,
    ) -> RawDatasetLocation:
        current_dataset = dataset
        current_index = int(index)
        visited = set()
        while True:
            identity = id(current_dataset)
            if identity in visited:
                raise RuntimeError("dataset wrapper cycle detected")
            visited.add(identity)
            if isinstance(current_dataset, torch.utils.data.ConcatDataset):
                if current_index < 0 or current_index >= len(current_dataset):
                    raise IndexError(current_index)
                child_index = int(
                    np.searchsorted(
                        np.asarray(current_dataset.cumulative_sizes),
                        current_index,
                        side="right",
                    )
                )
                previous_size = 0 if child_index == 0 else current_dataset.cumulative_sizes[child_index - 1]
                current_index -= int(previous_size)
                current_dataset = current_dataset.datasets[child_index]
                continue
            if hasattr(current_dataset, "base_dataset"):
                current_dataset = current_dataset.base_dataset
                continue
            if hasattr(current_dataset, "dataset") and not hasattr(current_dataset, "items"):
                current_dataset = current_dataset.dataset
                continue
            return RawDatasetLocation(current_dataset, current_index)

    def _class_is_allowed(self, class_name: str) -> bool:
        normalized = str(class_name).lower()
        matched = any(token in normalized for token in self.allowed_classes)
        if not normalized:
            return False
        return matched

    def _class_to_id(self, class_name: str) -> int:
        normalized = str(class_name).lower()
        if "truck" in normalized:
            class_id = 1
        elif "bus" in normalized:
            class_id = 3
        elif "motorcycle" in normalized:
            class_id = 6
        elif "bicycle" in normalized or "cyclist" in normalized or "bike" in normalized:
            class_id = 7
        elif "pedestrian" in normalized or "ped" in normalized:
            class_id = 8
        else:
            class_id = 0
        return class_id

    def _stable_hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        unsigned_value = int.from_bytes(digest, byteorder="little", signed=False)
        signed_safe_value = int(unsigned_value & ((1 << 63) - 1))
        if signed_safe_value < 0:
            raise RuntimeError("stable track hash overflow")
        return signed_safe_value

    def _box_corners_lidar(self, box7: np.ndarray) -> np.ndarray:
        x, y, z, dx, dy, dz, yaw = [float(value) for value in box7]
        local = np.asarray(
            [
                [-0.5 * dx, -0.5 * dy, -0.5 * dz],
                [-0.5 * dx, -0.5 * dy, 0.5 * dz],
                [-0.5 * dx, 0.5 * dy, -0.5 * dz],
                [-0.5 * dx, 0.5 * dy, 0.5 * dz],
                [0.5 * dx, -0.5 * dy, -0.5 * dz],
                [0.5 * dx, -0.5 * dy, 0.5 * dz],
                [0.5 * dx, 0.5 * dy, -0.5 * dz],
                [0.5 * dx, 0.5 * dy, 0.5 * dz],
            ],
            dtype=np.float32,
        )
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = np.asarray([x, y, z], dtype=np.float32)
        corners = local @ rotation.T + center[None]
        if corners.shape != (8, 3):
            raise RuntimeError("box corner construction returned an invalid shape")
        return corners
