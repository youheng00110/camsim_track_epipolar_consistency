from __future__ import annotations

import argparse
import bisect
import copy
import gc
import glob
import hashlib
import json
import math
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import einops
import numpy as np
import torch
import torch.nn.functional as F

import dwm.common
import dwm.datasets.common
from dwm.pipelines.wan.wan_dwm_utils import (
    encode_video_with_wan_vae_mode,
    normalize_image_tensor,
    select_latent_aligned_frame_indices,
)


from dwm.analysis.entity_reactor_unified_common import (
    UNIFIED_VERSION,
    DEFAULT_AUC_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_SIGMAS,
    composite_score,
    normalize_sigmas,
    save_unified_npz,
    write_scores_text,
    render_overview,
    render_causal,
)


WAN_ENTITY_ANALYSIS_VERSION = UNIFIED_VERSION + "-wan"
WAN_ENTITY_RESUME_VERSION = "v20-unified-aggregates-multinoise-20260819"
PROBE_FAMILIES = ("cross_view", "temporal", "mixed")
MODULES = ("temp", "cond", "crossview")
TARGET_RELATION = {
    "temp": "temporal",
    "cond": "cross_view",
    "crossview": "cross_view",
}


class NoCrossCameraTransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeEvent:
    family: str
    query: int
    positive: int
    negatives: np.ndarray
    query_track: int
    camera_pair: int
    time_gap: int


@dataclass
class ObservationPlan:
    observation_index: np.ndarray
    track_hash: np.ndarray
    class_id: np.ndarray
    geometry: np.ndarray
    anchor_visible: np.ndarray
    bbox_area_ratio: np.ndarray
    anchor_grid: torch.Tensor
    ring_grid: torch.Tensor
    btv_index: torch.Tensor
    crossview_mask: np.ndarray
    sample_index: int
    time_count: int
    view_count: int
    slot_count: int
    selected_track_tokens: list[str]
    metadata: dict


@dataclass
class ForwardResult:
    stage_order: list[str]
    stage_metrics: dict[str, dict[str, dict[str, float]]]
    transport_payloads: dict[str, dict[str, dict[str, np.ndarray]]]


@dataclass(frozen=True)
class ModuleTransition:
    module: str
    layer: int
    before_stage: str
    after_stage: str
    use_stage: str
    target_relation: str


class NuScenesEntityResolver:
    def __init__(
        self,
        dataset,
        max_entities: int,
        allowed_classes: Sequence[str],
        minimum_track_length: int,
        minimum_visible_anchors: int,
        minimum_area_ratio: float,
        ring_expand_ratio: float,
        temporal_downsample_factor: int,
        temporal_group_index: int,
    ) -> None:
        self.dataset = dataset
        self.max_entities = int(max_entities)
        self.allowed_classes = tuple(str(value).lower() for value in allowed_classes)
        self.minimum_track_length = int(minimum_track_length)
        self.minimum_visible_anchors = int(minimum_visible_anchors)
        self.minimum_area_ratio = float(minimum_area_ratio)
        self.ring_expand_ratio = float(ring_expand_ratio)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.temporal_group_index = int(temporal_group_index)
        if self.max_entities <= 0 or self.minimum_track_length <= 0:
            raise ValueError("entity limits must be positive")
        if self.minimum_visible_anchors < 1 or self.minimum_visible_anchors > 8:
            raise ValueError("minimum_visible_anchors must be inside [1,8]")

    def locate_raw_dataset(self, index: int):
        current = self.dataset
        current_index = int(index)
        visited = set()
        while True:
            identity = id(current)
            if identity in visited:
                raise RuntimeError("dataset wrapper cycle detected")
            visited.add(identity)
            if isinstance(current, torch.utils.data.Subset):
                current_index = int(current.indices[current_index])
                current = current.dataset
                continue
            if isinstance(current, torch.utils.data.ConcatDataset):
                dataset_position = bisect.bisect_right(current.cumulative_sizes, current_index)
                previous = 0 if dataset_position == 0 else int(current.cumulative_sizes[dataset_position - 1])
                current_index -= previous
                current = current.datasets[dataset_position]
                continue
            if hasattr(current, "base_dataset"):
                current = current.base_dataset
                continue
            if hasattr(current, "dataset") and not hasattr(current, "tables"):
                candidate = current.dataset
                if candidate is not current:
                    current = candidate
                    continue
            break
        required = ("items", "tables", "indices", "query", "query_range", "check_sensor")
        missing = [name for name in required if not hasattr(current, name)]
        if missing:
            raise TypeError(
                "WAN entity analysis requires a nuScenes MotionDataset-like raw dataset; "
                f"missing attributes {missing} on {type(current)!r}"
            )
        return current, current_index

    def class_name_and_id(self, raw_dataset, annotation: dict) -> tuple[str, int]:
        instance = raw_dataset.query(
            raw_dataset.tables,
            raw_dataset.indices,
            "instance",
            annotation["instance_token"],
        )
        category = raw_dataset.query(
            raw_dataset.tables,
            raw_dataset.indices,
            "category",
            instance["category_token"],
        )
        category_name = str(category["name"]).lower()
        coarse = "other"
        if "car" in category_name:
            coarse = "car"
        elif "truck" in category_name:
            coarse = "truck"
        elif "bus" in category_name:
            coarse = "bus"
        elif "pedestrian" in category_name or "person" in category_name:
            coarse = "pedestrian"
        elif "bicycle" in category_name or "bike" in category_name:
            coarse = "bicycle"
        elif "motorcycle" in category_name:
            coarse = "motorcycle"
        elif category_name.startswith("vehicle"):
            coarse = "vehicle"
        allowed = any(
            token == coarse or token in category_name or category_name.startswith(token)
            for token in self.allowed_classes
        )
        if not allowed:
            return coarse, -1
        canonical = {
            "car": 1,
            "truck": 2,
            "bus": 3,
            "vehicle": 4,
            "pedestrian": 5,
            "bicycle": 6,
            "motorcycle": 7,
        }
        return coarse, int(canonical.get(coarse, 100))

    def annotation_world_anchors(self, raw_dataset, annotation: dict) -> np.ndarray:
        if hasattr(raw_dataset, "default_3dbox_corner_template"):
            template = np.asarray(raw_dataset.default_3dbox_corner_template, dtype=np.float32)
        else:
            template = np.asarray(
                [
                    [-0.5, -0.5, -0.5, 1.0],
                    [-0.5, -0.5, 0.5, 1.0],
                    [-0.5, 0.5, -0.5, 1.0],
                    [-0.5, 0.5, 0.5, 1.0],
                    [0.5, -0.5, -0.5, 1.0],
                    [0.5, -0.5, 0.5, 1.0],
                    [0.5, 0.5, -0.5, 1.0],
                    [0.5, 0.5, 0.5, 1.0],
                ],
                dtype=np.float32,
            )
        if template.shape != (8, 4):
            template = template.reshape(8, 4)
        size = np.asarray(
            [annotation["size"][1], annotation["size"][0], annotation["size"][2]],
            dtype=np.float32,
        )
        local = template.copy()
        local[:, :3] *= size[None]
        world_from_annotation = dwm.datasets.common.get_transform(
            annotation["rotation"],
            annotation["translation"],
        ).astype(np.float32)
        corners_world = (world_from_annotation @ local.T).T[:, :3]
        center_world = corners_world.mean(axis=0, keepdims=True)
        return np.concatenate([corners_world, center_world], axis=0).astype(np.float32)

    def camera_projection(self, raw_dataset, camera_sample_data: dict, anchors_world: np.ndarray):
        calibrated_sensor = raw_dataset.query(
            raw_dataset.tables,
            raw_dataset.indices,
            "calibrated_sensor",
            camera_sample_data["calibrated_sensor_token"],
        )
        intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"],
            calibrated_sensor["translation"],
        ).astype(np.float32)
        world_from_ego = dwm.datasets.common.get_transform(
            camera_sample_data["rotation"],
            camera_sample_data["translation"],
        ).astype(np.float32)
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera).astype(np.float32)
        anchors_h = np.concatenate(
            [anchors_world, np.ones((anchors_world.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        camera_xyz = (camera_from_world @ anchors_h.T).T[:, :3]
        projected = (intrinsic @ camera_xyz.T).T
        depth = camera_xyz[:, 2]
        safe_depth = np.maximum(projected[:, 2], 1e-6)
        pixel_u = projected[:, 0] / safe_depth
        pixel_v = projected[:, 1] / safe_depth
        width = float(camera_sample_data["width"])
        height = float(camera_sample_data["height"])
        visible = (
            (depth > 0.10)
            & (pixel_u >= 0.0)
            & (pixel_u < width)
            & (pixel_v >= 0.0)
            & (pixel_v < height)
        )
        return pixel_u, pixel_v, depth, visible, width, height

    def build_observation_plan(
        self,
        sample_index: int,
        target_time_count: int,
        crossview_mask: np.ndarray,
    ) -> ObservationPlan:
        raw_dataset, raw_index = self.locate_raw_dataset(sample_index)
        item = raw_dataset.items[raw_index]
        dense_segment = [
            [
                raw_dataset.query(raw_dataset.tables, raw_dataset.indices, "sample_data", token)
                for token in frame_tokens
            ]
            for frame_tokens in item["segment"]
        ]
        camera_frames = [
            [
                sample_data
                for sample_data in frame_items
                if raw_dataset.check_sensor(
                    raw_dataset.tables,
                    raw_dataset.indices,
                    sample_data,
                    modality="camera",
                )
            ]
            for frame_items in dense_segment
        ]
        if not camera_frames or not camera_frames[0]:
            raise RuntimeError("sample contains no camera frames")
        view_count = len(camera_frames[0])
        if any(len(frame) != view_count for frame in camera_frames):
            raise RuntimeError("camera view count changes inside one clip")
        temporal_indices = select_latent_aligned_frame_indices(
            dense_t=len(camera_frames),
            target_t=int(target_time_count),
            device=torch.device("cpu"),
            temporal_downsample_factor=self.temporal_downsample_factor,
            group_index=self.temporal_group_index,
        ).tolist()
        selected_camera_frames = [camera_frames[int(index)] for index in temporal_indices]
        records_by_time: list[dict[str, tuple[int, dict, np.ndarray]]] = []
        track_counts: dict[str, int] = {}
        track_classes: dict[str, int] = {}
        track_distance: dict[str, float] = {}
        for camera_row in selected_camera_frames:
            sample_token = camera_row[0]["sample_token"]
            annotations = raw_dataset.query_range(
                raw_dataset.tables,
                raw_dataset.indices,
                "sample_annotation",
                sample_token,
                column_name="sample_token",
            )
            current: dict[str, tuple[int, dict, np.ndarray]] = {}
            for annotation in annotations:
                class_name, class_id = self.class_name_and_id(raw_dataset, annotation)
                if class_id < 0:
                    continue
                token = str(annotation["instance_token"])
                anchors_world = self.annotation_world_anchors(raw_dataset, annotation)
                center = anchors_world[-1]
                ego_from_world = np.linalg.inv(
                    dwm.datasets.common.get_transform(
                        camera_row[0]["rotation"],
                        camera_row[0]["translation"],
                    ).astype(np.float32)
                )
                center_h = np.concatenate([center, np.ones((1,), dtype=np.float32)])
                center_ego = (ego_from_world @ center_h)[:3]
                distance = float(np.linalg.norm(center_ego[:2]))
                current[token] = (class_id, annotation, anchors_world)
                track_counts[token] = track_counts.get(token, 0) + 1
                track_classes[token] = class_id
                previous_distance = track_distance.get(token, float("inf"))
                track_distance[token] = min(previous_distance, distance)
            records_by_time.append(current)
        eligible = [
            token
            for token, count in track_counts.items()
            if int(count) >= self.minimum_track_length
        ]
        eligible.sort(
            key=lambda token: (
                -int(track_counts[token]),
                float(track_distance.get(token, float("inf"))),
                token,
            )
        )
        selected_tokens = eligible[: self.max_entities]
        if not selected_tokens:
            raise NoCrossCameraTransitionError("clip has no sufficiently long allowed entity track")
        slot_by_token = {token: slot for slot, token in enumerate(selected_tokens)}
        observation_index = []
        track_hash = []
        class_ids = []
        geometries = []
        anchor_visibilities = []
        area_ratios = []
        anchor_grids = []
        ring_grids = []
        btv_indices = []
        for time_index, (camera_row, record_map) in enumerate(zip(selected_camera_frames, records_by_time)):
            for token in selected_tokens:
                record = record_map.get(token)
                if record is None:
                    continue
                class_id, _, anchors_world = record
                slot = slot_by_token[token]
                for view_index, camera_sample_data in enumerate(camera_row):
                    pixel_u, pixel_v, depth, visible, width, height = self.camera_projection(
                        raw_dataset,
                        camera_sample_data,
                        anchors_world,
                    )
                    visible_corner_count = int(visible[:8].sum())
                    if visible_corner_count < self.minimum_visible_anchors or not bool(visible[-1]):
                        continue
                    positive = depth[:8] > 0.10
                    if not bool(np.any(positive)):
                        continue
                    u0 = max(float(np.min(pixel_u[:8][positive])), 0.0)
                    v0 = max(float(np.min(pixel_v[:8][positive])), 0.0)
                    u1 = min(float(np.max(pixel_u[:8][positive])), width - 1.0)
                    v1 = min(float(np.max(pixel_v[:8][positive])), height - 1.0)
                    area = max(u1 - u0, 0.0) * max(v1 - v0, 0.0)
                    area_ratio = area / max(width * height, 1.0)
                    if not np.isfinite(area_ratio) or area_ratio < self.minimum_area_ratio:
                        continue
                    norm_u = (pixel_u + 0.5) / max(width, 1.0)
                    norm_v = (pixel_v + 0.5) / max(height, 1.0)
                    geometry = np.stack(
                        [norm_u, norm_v, np.log(np.maximum(depth, 1e-4))],
                        axis=-1,
                    ).astype(np.float32)
                    anchor_grid = np.stack(
                        [norm_u * 2.0 - 1.0, norm_v * 2.0 - 1.0],
                        axis=-1,
                    ).astype(np.float32)
                    box_width = max(u1 - u0, 1.0)
                    box_height = max(v1 - v0, 1.0)
                    ring_u0 = max(u0 - self.ring_expand_ratio * box_width, 0.0)
                    ring_v0 = max(v0 - self.ring_expand_ratio * box_height, 0.0)
                    ring_u1 = min(u1 + self.ring_expand_ratio * box_width, width - 1.0)
                    ring_v1 = min(v1 + self.ring_expand_ratio * box_height, height - 1.0)
                    ring_um = 0.5 * (ring_u0 + ring_u1)
                    ring_vm = 0.5 * (ring_v0 + ring_v1)
                    ring_u = np.asarray(
                        [ring_u0, ring_um, ring_u1, ring_u1, ring_u1, ring_um, ring_u0, ring_u0],
                        dtype=np.float32,
                    )
                    ring_v = np.asarray(
                        [ring_v0, ring_v0, ring_v0, ring_vm, ring_v1, ring_v1, ring_v1, ring_vm],
                        dtype=np.float32,
                    )
                    ring_grid = np.stack(
                        [
                            (ring_u + 0.5) / max(width, 1.0) * 2.0 - 1.0,
                            (ring_v + 0.5) / max(height, 1.0) * 2.0 - 1.0,
                        ],
                        axis=-1,
                    ).astype(np.float32)
                    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                    stable_hash = int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF
                    observation_index.append((time_index, view_index, slot))
                    track_hash.append(stable_hash)
                    class_ids.append(int(class_id))
                    geometries.append(geometry)
                    anchor_visibilities.append(visible.astype(np.bool_))
                    area_ratios.append(float(area_ratio))
                    anchor_grids.append(anchor_grid)
                    ring_grids.append(ring_grid)
                    btv_indices.append(view_index * int(target_time_count) + time_index)
        if not observation_index:
            raise NoCrossCameraTransitionError("clip has no valid projected entity observation")
        crossview_mask = np.asarray(crossview_mask, dtype=np.bool_)
        if crossview_mask.ndim == 3:
            crossview_mask = crossview_mask[0]
        if crossview_mask.ndim == 4 and crossview_mask.shape[1] == 1:
            crossview_mask = crossview_mask[0, 0]
        if crossview_mask.shape != (view_count, view_count):
            raise ValueError(
                f"crossview mask shape {crossview_mask.shape} does not match view_count={view_count}"
            )
        return ObservationPlan(
            observation_index=np.asarray(observation_index, dtype=np.int64),
            track_hash=np.asarray(track_hash, dtype=np.int64),
            class_id=np.asarray(class_ids, dtype=np.int64),
            geometry=np.stack(geometries, axis=0).astype(np.float32),
            anchor_visible=np.stack(anchor_visibilities, axis=0).astype(np.bool_),
            bbox_area_ratio=np.asarray(area_ratios, dtype=np.float32),
            anchor_grid=torch.from_numpy(np.stack(anchor_grids, axis=0)),
            ring_grid=torch.from_numpy(np.stack(ring_grids, axis=0)),
            btv_index=torch.tensor(btv_indices, dtype=torch.long),
            crossview_mask=crossview_mask,
            sample_index=int(sample_index),
            time_count=int(target_time_count),
            view_count=int(view_count),
            slot_count=len(selected_tokens),
            selected_track_tokens=list(selected_tokens),
            metadata={
                "raw_index": int(raw_index),
                "dense_time_count": int(len(camera_frames)),
                "latent_time_count": int(target_time_count),
                "temporal_indices": [int(value) for value in temporal_indices],
                "selected_track_count": int(len(selected_tokens)),
            },
        )


def select_track_balanced_events(
    events: Sequence[ProbeEvent],
    maximum_events_per_track: int,
    maximum_total_events: int,
) -> list[ProbeEvent]:
    events_by_track: dict[int, list[ProbeEvent]] = {}
    for event in events:
        events_by_track.setdefault(int(event.query_track), []).append(event)
    capped_by_track: dict[int, list[ProbeEvent]] = {}
    for track_id in sorted(events_by_track):
        current_events = sorted(
            events_by_track[track_id],
            key=lambda event: (
                int(event.camera_pair),
                abs(int(event.time_gap)),
                int(event.query),
                int(event.positive),
            ),
        )
        selected = []
        used_pairs = set()
        for event in current_events:
            pair_key = int(event.camera_pair)
            if pair_key in used_pairs:
                continue
            selected.append(event)
            used_pairs.add(pair_key)
            if len(selected) >= int(maximum_events_per_track):
                break
        if len(selected) < int(maximum_events_per_track):
            selected_ids = {(int(event.query), int(event.positive)) for event in selected}
            for event in current_events:
                event_id = (int(event.query), int(event.positive))
                if event_id in selected_ids:
                    continue
                selected.append(event)
                selected_ids.add(event_id)
                if len(selected) >= int(maximum_events_per_track):
                    break
        if selected:
            capped_by_track[int(track_id)] = selected
    output = []
    offsets = {track_id: 0 for track_id in capped_by_track}
    track_ids = sorted(capped_by_track)
    while len(output) < int(maximum_total_events):
        added = False
        for track_id in track_ids:
            current_events = capped_by_track[track_id]
            current_offset = offsets[track_id]
            if current_offset >= len(current_events):
                continue
            output.append(current_events[current_offset])
            offsets[track_id] = current_offset + 1
            added = True
            if len(output) >= int(maximum_total_events):
                break
        if not added:
            break
    return output


def select_hard_negatives(
    plan: ObservationPlan,
    positive_row: int,
    target_time: int,
    target_view: int,
    maximum_negatives: int,
) -> np.ndarray:
    observation_index = plan.observation_index
    track_hash = plan.track_hash
    class_id = plan.class_id
    geometry = plan.geometry[:, -1]
    area = plan.bbox_area_ratio
    positive_track = track_hash[positive_row]
    positive_class = class_id[positive_row]
    different_track = track_hash != positive_track
    same_view = observation_index[:, 1] == int(target_view)
    exact_time = observation_index[:, 0] == int(target_time)
    nearby_time = np.abs(observation_index[:, 0] - int(target_time)) <= 1
    same_class = class_id == positive_class
    candidate_masks = (
        different_track & same_view & exact_time & same_class,
        different_track & same_view & nearby_time & same_class,
        different_track & same_view & same_class,
    )
    candidate_rows = np.empty(0, dtype=np.int64)
    for candidate_mask in candidate_masks:
        current_rows = np.flatnonzero(candidate_mask)
        if current_rows.size >= 2:
            candidate_rows = current_rows
            break
        if candidate_rows.size == 0 and current_rows.size > 0:
            candidate_rows = current_rows
    if candidate_rows.size == 0:
        return candidate_rows
    target_geometry = geometry[positive_row]
    target_area = float(area[positive_row])
    geometry_delta = geometry[candidate_rows] - target_geometry[None]
    geometry_distance = (
        1.5 * geometry_delta[:, 0] ** 2
        + 1.5 * geometry_delta[:, 1] ** 2
        + 0.5 * geometry_delta[:, 2] ** 2
    )
    area_distance = np.abs(
        np.log(np.maximum(area[candidate_rows], 1e-8))
        - np.log(max(target_area, 1e-8))
    )
    ranking_distance = geometry_distance + 0.15 * area_distance
    order = np.argsort(ranking_distance, kind="stable")
    return candidate_rows[order[: int(maximum_negatives)]].astype(np.int64, copy=False)


def build_probe_events(
    plan: ObservationPlan,
    maximum_events_per_family: int,
    maximum_negatives: int,
    minimum_temporal_gap: Optional[int] = None,
) -> dict[str, list[ProbeEvent]]:
    observation_index = plan.observation_index
    track_hash = plan.track_hash
    crossview_mask = plan.crossview_mask
    temporal_gap = int(
        minimum_temporal_gap
        or max(2, round(max(int(plan.time_count) - 1, 1) * 0.20))
    )
    rows_by_track: dict[int, list[int]] = {}
    for row, current_track in enumerate(track_hash.tolist()):
        rows_by_track.setdefault(int(current_track), []).append(int(row))
    events: dict[str, list[ProbeEvent]] = {family: [] for family in PROBE_FAMILIES}
    handoff_count = 0
    for current_track in sorted(rows_by_track):
        track_rows = rows_by_track[current_track]
        rows_by_view: dict[int, list[int]] = {}
        for row in track_rows:
            view_id = int(observation_index[row, 1])
            rows_by_view.setdefault(view_id, []).append(row)
        available_views = sorted(rows_by_view)
        for source_position, first_view in enumerate(available_views):
            first_rows = rows_by_view[first_view]
            first_times = np.asarray(
                [int(observation_index[row, 0]) for row in first_rows],
                dtype=np.int64,
            )
            for second_view in available_views[source_position + 1 :]:
                if not bool(crossview_mask[first_view, second_view] or crossview_mask[second_view, first_view]):
                    continue
                second_rows = rows_by_view[second_view]
                second_times = np.asarray(
                    [int(observation_index[row, 0]) for row in second_rows],
                    dtype=np.int64,
                )
                if first_times.size == 0 or second_times.size == 0:
                    continue
                first_start, first_end = int(first_times.min()), int(first_times.max())
                second_start, second_end = int(second_times.min()), int(second_times.max())
                if first_start < second_start and first_end < second_end:
                    source_view, target_view = first_view, second_view
                    source_rows, source_times = first_rows, first_times
                    target_rows, target_times = second_rows, second_times
                    target_start, source_end = second_start, first_end
                elif second_start < first_start and second_end < first_end:
                    source_view, target_view = second_view, first_view
                    source_rows, source_times = second_rows, second_times
                    target_rows, target_times = first_rows, first_times
                    target_start, source_end = first_start, second_end
                else:
                    continue
                source_candidates = [
                    row
                    for row, time_value in zip(source_rows, source_times.tolist())
                    if int(time_value) < int(target_start)
                ]
                target_candidates = [
                    row
                    for row, time_value in zip(target_rows, target_times.tolist())
                    if int(time_value) > int(source_end)
                ]
                if not source_candidates or not target_candidates:
                    continue
                query_row = max(source_candidates, key=lambda row: int(observation_index[row, 0]))
                positive_row = min(target_candidates, key=lambda row: int(observation_index[row, 0]))
                query_time = int(observation_index[query_row, 0])
                target_time = int(observation_index[positive_row, 0])
                negatives = select_hard_negatives(
                    plan,
                    positive_row,
                    target_time,
                    target_view,
                    maximum_negatives,
                )
                if negatives.size == 0:
                    continue
                handoff_count += 1
                events["cross_view"].append(
                    ProbeEvent(
                        family="cross_view",
                        query=int(query_row),
                        positive=int(positive_row),
                        negatives=negatives,
                        query_track=int(current_track),
                        camera_pair=int(source_view * 100 + target_view),
                        time_gap=int(target_time - query_time),
                    )
                )
        for query_row in track_rows:
            query_time = int(observation_index[query_row, 0])
            query_view = int(observation_index[query_row, 1])
            temporal_targets = [
                row
                for row in track_rows
                if int(observation_index[row, 1]) == query_view
                and abs(int(observation_index[row, 0]) - query_time) >= temporal_gap
            ]
            temporal_targets.sort(
                key=lambda row: abs(int(observation_index[row, 0]) - query_time),
                reverse=True,
            )
            for positive_row in temporal_targets[:2]:
                target_time = int(observation_index[positive_row, 0])
                negatives = select_hard_negatives(
                    plan,
                    positive_row,
                    target_time,
                    query_view,
                    maximum_negatives,
                )
                if negatives.size == 0:
                    continue
                events["temporal"].append(
                    ProbeEvent(
                        family="temporal",
                        query=int(query_row),
                        positive=int(positive_row),
                        negatives=negatives,
                        query_track=int(current_track),
                        camera_pair=int(query_view * 100 + query_view),
                        time_gap=int(target_time - query_time),
                    )
                )
            mixed_targets = [
                row
                for row in track_rows
                if int(observation_index[row, 1]) != query_view
                and abs(int(observation_index[row, 0]) - query_time) >= temporal_gap
            ]
            mixed_targets.sort(
                key=lambda row: abs(int(observation_index[row, 0]) - query_time),
                reverse=True,
            )
            accepted_mixed = 0
            for positive_row in mixed_targets:
                target_time = int(observation_index[positive_row, 0])
                target_view = int(observation_index[positive_row, 1])
                if not bool(crossview_mask[query_view, target_view] or crossview_mask[target_view, query_view]):
                    continue
                negatives = select_hard_negatives(
                    plan,
                    positive_row,
                    target_time,
                    target_view,
                    maximum_negatives,
                )
                if negatives.size == 0:
                    continue
                events["mixed"].append(
                    ProbeEvent(
                        family="mixed",
                        query=int(query_row),
                        positive=int(positive_row),
                        negatives=negatives,
                        query_track=int(current_track),
                        camera_pair=int(query_view * 100 + target_view),
                        time_gap=int(target_time - query_time),
                    )
                )
                accepted_mixed += 1
                if accepted_mixed >= 2:
                    break
    if handoff_count == 0:
        raise NoCrossCameraTransitionError(
            "no strict temporal camera handoff has a same-class hard negative"
        )
    per_track_limits = {"cross_view": 4, "temporal": 2, "mixed": 2}
    for family in PROBE_FAMILIES:
        events[family] = select_track_balanced_events(
            events[family],
            maximum_events_per_track=per_track_limits[family],
            maximum_total_events=int(maximum_events_per_family),
        )
    return events


class StreamingEntityRecorder:
    def __init__(
        self,
        transport_stage_names: set[str],
        transport_events_per_sample: int,
        auc_weight: float = DEFAULT_AUC_WEIGHT,
        margin_scale: float = DEFAULT_MARGIN_SCALE,
    ) -> None:
        self.transport_stage_names = set(str(name) for name in transport_stage_names)
        self.transport_events_per_sample = int(transport_events_per_sample)
        self.auc_weight = float(auc_weight)
        self.margin_scale = float(margin_scale)
        self.plan: Optional[ObservationPlan] = None
        self.events: Optional[dict[str, list[ProbeEvent]]] = None
        self.batch_size = 0
        self.time_count = 0
        self.view_count = 0
        self.patch_height = 0
        self.patch_width = 0
        self.device = torch.device("cpu")
        self.anchor_grid_device: Optional[torch.Tensor] = None
        self.ring_grid_device: Optional[torch.Tensor] = None
        self.btv_index_device: Optional[torch.Tensor] = None
        self.anchor_visible_device: Optional[torch.Tensor] = None
        self.geometry_vector_device: Optional[torch.Tensor] = None
        self.event_tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.current_stage_order: list[str] = []
        self.current_stage_metrics: dict[str, dict[str, dict[str, float]]] = {}
        self.current_transport: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    def configure(
        self,
        plan: ObservationPlan,
        events: dict[str, list[ProbeEvent]],
        batch_size: int,
        patch_height: int,
        patch_width: int,
        device: torch.device,
    ) -> None:
        self.plan = plan
        self.events = events
        self.batch_size = int(batch_size)
        self.time_count = int(plan.time_count)
        self.view_count = int(plan.view_count)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.device = torch.device(device)
        self.anchor_grid_device = plan.anchor_grid.to(device=self.device, dtype=torch.float32)
        self.ring_grid_device = plan.ring_grid.to(device=self.device, dtype=torch.float32)
        self.btv_index_device = plan.btv_index.to(device=self.device, dtype=torch.long)
        self.anchor_visible_device = torch.from_numpy(plan.anchor_visible).to(
            device=self.device,
            dtype=torch.bool,
        )
        geometry = torch.from_numpy(plan.geometry).to(device=self.device, dtype=torch.float32)
        center = geometry[:, -1]
        relative = geometry[:, :-1] - center[:, None]
        corner_visible = self.anchor_visible_device[:, :-1] & self.anchor_visible_device[:, -1:]
        masked_relative = relative * corner_visible.unsqueeze(-1).to(relative.dtype)
        self.geometry_vector_device = torch.cat(
            [center, masked_relative.reshape(masked_relative.shape[0], -1)],
            dim=1,
        )
        self.event_tensors.clear()
        for family in PROBE_FAMILIES:
            family_events = events.get(family, [])
            if not family_events:
                self.event_tensors[family] = {}
                continue
            maximum_negative_count = max(int(event.negatives.size) for event in family_events)
            query = torch.tensor(
                [event.query for event in family_events],
                dtype=torch.long,
                device=self.device,
            )
            positive = torch.tensor(
                [event.positive for event in family_events],
                dtype=torch.long,
                device=self.device,
            )
            negatives = torch.zeros(
                len(family_events),
                maximum_negative_count,
                dtype=torch.long,
                device=self.device,
            )
            negative_mask = torch.zeros(
                len(family_events),
                maximum_negative_count,
                dtype=torch.bool,
                device=self.device,
            )
            query_track = torch.tensor(
                [event.query_track for event in family_events],
                dtype=torch.long,
                device=self.device,
            )
            camera_pair = torch.tensor(
                [event.camera_pair for event in family_events],
                dtype=torch.long,
                device=self.device,
            )
            for event_index, event in enumerate(family_events):
                count = int(event.negatives.size)
                negatives[event_index, :count] = torch.from_numpy(event.negatives).to(
                    device=self.device,
                    dtype=torch.long,
                )
                negative_mask[event_index, :count] = True
            self.event_tensors[family] = {
                "query": query,
                "positive": positive,
                "negatives": negatives,
                "negative_mask": negative_mask,
                "query_track": query_track,
                "camera_pair": camera_pair,
            }

    def clear_sample(self) -> None:
        self.plan = None
        self.events = None
        self.anchor_grid_device = None
        self.ring_grid_device = None
        self.btv_index_device = None
        self.anchor_visible_device = None
        self.geometry_vector_device = None
        self.event_tensors.clear()
        self.current_stage_order.clear()
        self.current_stage_metrics.clear()
        self.current_transport.clear()

    def begin_forward(self) -> None:
        if self.plan is None or self.anchor_grid_device is None:
            raise RuntimeError("recorder must be configured before forward")
        self.current_stage_order = []
        self.current_stage_metrics = {}
        self.current_transport = {}

    def normalize_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_bv = int(self.batch_size * self.view_count)
        expected_tokens = int(self.time_count * self.patch_height * self.patch_width)
        if hidden_states.ndim == 5:
            if hidden_states.shape[0] != expected_bv:
                raise ValueError(
                    f"5D hidden batch-view {hidden_states.shape[0]} does not match {expected_bv}"
                )
            normalized = einops.rearrange(
                hidden_states,
                "bv c t h w -> bv (t h w) c",
            )
        elif hidden_states.ndim == 4:
            if hidden_states.shape[0] != expected_bv:
                raise ValueError(
                    f"4D hidden batch-view {hidden_states.shape[0]} does not match {expected_bv}"
                )
            normalized = hidden_states.flatten(1, 2)
        elif hidden_states.ndim == 3:
            normalized = hidden_states
        else:
            raise ValueError(f"unsupported Wan hidden shape {tuple(hidden_states.shape)}")
        if normalized.shape[0] != expected_bv or normalized.shape[1] != expected_tokens:
            raise ValueError(
                f"Wan hidden shape {tuple(normalized.shape)} does not match "
                f"BV={expected_bv}, tokens={expected_tokens}"
            )
        return normalized

    def sample_entity_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.plan is None:
            raise RuntimeError("observation plan is missing")
        if self.anchor_grid_device is None or self.ring_grid_device is None:
            raise RuntimeError("sampling grids are missing")
        if self.btv_index_device is None:
            raise RuntimeError("BTV indices are missing")
        hidden_states = self.normalize_hidden(hidden_states)
        channel_count = int(hidden_states.shape[-1])
        normalized = F.layer_norm(hidden_states.float(), (channel_count,))
        feature_map = einops.rearrange(
            normalized,
            "(b v) (t h w) c -> (b v t) c h w",
            b=self.batch_size,
            v=self.view_count,
            t=self.time_count,
            h=self.patch_height,
            w=self.patch_width,
        )
        observation_count = int(self.btv_index_device.numel())
        anchor_count = int(self.anchor_grid_device.shape[1])
        ring_count = int(self.ring_grid_device.shape[1])
        features = torch.empty(
            observation_count,
            anchor_count,
            channel_count,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        unique_btv = torch.unique(self.btv_index_device, sorted=True)
        for current_btv in unique_btv.tolist():
            rows = torch.nonzero(
                self.btv_index_device == int(current_btv),
                as_tuple=False,
            ).flatten()
            anchor_grid = self.anchor_grid_device.index_select(0, rows).reshape(
                1,
                rows.numel() * anchor_count,
                1,
                2,
            )
            ring_grid = self.ring_grid_device.index_select(0, rows).reshape(
                1,
                rows.numel() * ring_count,
                1,
                2,
            )
            current_map = feature_map[int(current_btv) : int(current_btv) + 1]
            sampled_anchor = F.grid_sample(
                current_map,
                anchor_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_ring = F.grid_sample(
                current_map,
                ring_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_anchor = sampled_anchor[..., 0].transpose(1, 2).reshape(
                rows.numel(),
                anchor_count,
                channel_count,
            )
            sampled_ring = sampled_ring[..., 0].transpose(1, 2).reshape(
                rows.numel(),
                ring_count,
                channel_count,
            )
            local_background = sampled_ring.mean(dim=1, keepdim=True)
            features.index_copy_(0, rows, sampled_anchor - local_background)
        return features

    def prepare_encodings(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.anchor_visible_device is None:
            raise RuntimeError("anchor visibility is missing")
        visibility = self.anchor_visible_device
        visible_weights = visibility.unsqueeze(-1).to(features.dtype)
        visible_count = visible_weights.sum(dim=1).clamp(min=1.0)
        pooled = (features * visible_weights).sum(dim=1) / visible_count
        pooled = F.normalize(pooled, dim=1, eps=1e-10)
        center = F.normalize(features[:, -1], dim=1, eps=1e-10)
        relative = features[:, :-1] - features[:, -1:].expand(-1, features.shape[1] - 1, -1)
        relative = F.normalize(relative, dim=-1, eps=1e-10)
        corner_visible = visibility[:, :-1] & visibility[:, -1:]
        return {
            "pooled": pooled,
            "center": center,
            "relative": relative,
            "corner_visible": corner_visible,
        }

    def structured_similarity(
        self,
        encodings: dict[str, torch.Tensor],
        query_rows: torch.Tensor,
        candidate_rows: torch.Tensor,
    ) -> torch.Tensor:
        center = encodings["center"]
        relative = encodings["relative"]
        corner_visible = encodings["corner_visible"]
        query_center = center.index_select(0, query_rows)
        candidate_center = center[candidate_rows]
        center_score = torch.einsum("emd,ed->em", candidate_center, query_center)
        query_relative = relative.index_select(0, query_rows)
        candidate_relative = relative[candidate_rows]
        per_corner = torch.einsum("emad,ead->ema", candidate_relative, query_relative)
        common_visible = corner_visible[candidate_rows] & corner_visible.index_select(0, query_rows).unsqueeze(1)
        common_count = common_visible.sum(dim=2)
        relation_score = (
            (per_corner * common_visible.to(per_corner.dtype)).sum(dim=2)
            / common_count.clamp(min=1).to(per_corner.dtype)
        )
        relation_score = torch.where(common_count >= 2, relation_score, center_score)
        combined = 0.35 * center_score + 0.65 * relation_score
        return combined.clamp(-1.0, 1.0)

    def evaluate_family(
        self,
        family: str,
        encodings: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        tensors = self.event_tensors.get(family, {})
        if not tensors:
            return {
                "pooled_score": float("nan"),
                "structured_score": float("nan"),
                "pooled_auc": float("nan"),
                "structured_auc": float("nan"),
                "pooled_margin": float("nan"),
                "structured_margin": float("nan"),
                "pooled_top1": float("nan"),
                "structured_top1": float("nan"),
                "event_count": 0.0,
            }
        query_rows = tensors["query"]
        positive_rows = tensors["positive"]
        negative_rows = tensors["negatives"]
        negative_mask = tensors["negative_mask"]
        pooled = encodings["pooled"]
        pooled_query = pooled.index_select(0, query_rows)
        pooled_positive = torch.einsum(
            "ed,ed->e",
            pooled.index_select(0, positive_rows),
            pooled_query,
        )
        pooled_negative = torch.einsum(
            "emd,ed->em",
            pooled[negative_rows],
            pooled_query,
        )
        structured_positive = self.structured_similarity(
            encodings,
            query_rows,
            positive_rows.unsqueeze(1),
        )[:, 0]
        structured_negative = self.structured_similarity(
            encodings,
            query_rows,
            negative_rows,
        )
        valid_count = negative_mask.sum(dim=1).clamp(min=1)
        pooled_auc_values = (
            ((pooled_positive.unsqueeze(1) > pooled_negative) & negative_mask).sum(dim=1).float()
            + 0.5
            * ((pooled_positive.unsqueeze(1) == pooled_negative) & negative_mask).sum(dim=1).float()
        ) / valid_count.float()
        structured_auc_values = (
            ((structured_positive.unsqueeze(1) > structured_negative) & negative_mask).sum(dim=1).float()
            + 0.5
            * ((structured_positive.unsqueeze(1) == structured_negative) & negative_mask).sum(dim=1).float()
        ) / valid_count.float()
        negative_infinity = torch.full_like(pooled_negative, -torch.inf)
        pooled_negative_max = torch.where(negative_mask, pooled_negative, negative_infinity).max(dim=1).values
        structured_negative_max = torch.where(
            negative_mask,
            structured_negative,
            torch.full_like(structured_negative, -torch.inf),
        ).max(dim=1).values
        pooled_margins = pooled_positive - pooled_negative_max
        structured_margins = structured_positive - structured_negative_max
        pooled_auc = pooled_auc_values.mean()
        structured_auc = structured_auc_values.mean()
        pooled_margin = pooled_margins.mean()
        structured_margin = structured_margins.mean()
        pooled_score = torch.as_tensor(
            composite_score(
                float(pooled_auc.item()),
                float(pooled_margin.item()),
                self.auc_weight,
                self.margin_scale,
            ),
            device=pooled_auc.device,
            dtype=pooled_auc.dtype,
        )
        structured_score = torch.as_tensor(
            composite_score(
                float(structured_auc.item()),
                float(structured_margin.item()),
                self.auc_weight,
                self.margin_scale,
            ),
            device=structured_auc.device,
            dtype=structured_auc.dtype,
        )
        return {
            "pooled_score": float(pooled_score.clamp(-1.0, 1.0).item()),
            "structured_score": float(structured_score.clamp(-1.0, 1.0).item()),
            "pooled_auc": float(pooled_auc.item()),
            "structured_auc": float(structured_auc.item()),
            "pooled_margin": float(pooled_margin.item()),
            "structured_margin": float(structured_margin.item()),
            "pooled_top1": float((pooled_margins >= 0.0).float().mean().item()),
            "structured_top1": float((structured_margins >= 0.0).float().mean().item()),
            "event_count": float(query_rows.numel()),
        }

    def make_transport_payload(
        self,
        family: str,
        encodings: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        tensors = self.event_tensors.get(family, {})
        feature_dim = int(encodings["pooled"].shape[1])
        geometry_dim = int(self.geometry_vector_device.shape[1]) if self.geometry_vector_device is not None else 1
        if not tensors:
            return {
                "feature_delta": np.empty((0, feature_dim), dtype=np.float32),
                "geometry_delta": np.empty((0, geometry_dim), dtype=np.float32),
                "sample_id": np.empty((0,), dtype=np.int64),
                "track_id": np.empty((0,), dtype=np.int64),
                "camera_pair": np.empty((0,), dtype=np.int64),
            }
        if self.geometry_vector_device is None or self.plan is None:
            raise RuntimeError("geometry vector is unavailable")
        query = tensors["query"]
        positive = tensors["positive"]
        count = int(query.numel())
        if self.transport_events_per_sample > 0 and count > self.transport_events_per_sample:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                20260819 + int(self.plan.sample_index) * 1009 + PROBE_FAMILIES.index(family) * 37
            )
            selected_cpu = torch.randperm(count, generator=generator)[: self.transport_events_per_sample]
            selected = selected_cpu.to(device=self.device)
            query = query.index_select(0, selected)
            positive = positive.index_select(0, selected)
            query_track = tensors["query_track"].index_select(0, selected)
            camera_pair = tensors["camera_pair"].index_select(0, selected)
        else:
            query_track = tensors["query_track"]
            camera_pair = tensors["camera_pair"]
        pooled = encodings["pooled"]
        feature_delta = pooled.index_select(0, positive) - pooled.index_select(0, query)
        feature_delta = F.normalize(feature_delta, dim=1, eps=1e-10)
        geometry_delta = self.geometry_vector_device.index_select(0, positive) - self.geometry_vector_device.index_select(0, query)
        event_count = int(query.numel())
        return {
            "feature_delta": feature_delta.detach().float().cpu().numpy().astype(np.float32, copy=False),
            "geometry_delta": geometry_delta.detach().float().cpu().numpy().astype(np.float32, copy=False),
            "sample_id": np.full(event_count, int(self.plan.sample_index), dtype=np.int64),
            "track_id": query_track.detach().cpu().numpy().astype(np.int64, copy=False),
            "camera_pair": camera_pair.detach().cpu().numpy().astype(np.int64, copy=False),
        }

    def capture(self, stage_name: str, hidden_states: torch.Tensor) -> None:
        if stage_name in self.current_stage_metrics:
            return
        features = self.sample_entity_features(hidden_states)
        encodings = self.prepare_encodings(features)
        stage_metrics = {
            family: self.evaluate_family(family, encodings)
            for family in PROBE_FAMILIES
        }
        self.current_stage_order.append(str(stage_name))
        self.current_stage_metrics[str(stage_name)] = stage_metrics
        if str(stage_name) in self.transport_stage_names:
            self.current_transport[str(stage_name)] = {
                family: self.make_transport_payload(family, encodings)
                for family in PROBE_FAMILIES
            }
        del features, encodings

    def end_forward(self) -> ForwardResult:
        return ForwardResult(
            stage_order=list(self.current_stage_order),
            stage_metrics=copy.deepcopy(self.current_stage_metrics),
            transport_payloads=copy.deepcopy(self.current_transport),
        )



class _WanPatchEmbeddingHook:
    def __init__(self, controller) -> None:
        self.controller = controller

    def __call__(self, module, args, kwargs, output):
        del module, args, kwargs
        controller = self.controller
        if not controller.active:
            return None
        stem = controller.recorder.normalize_hidden(output)
        controller.recorder.capture("stem", stem)
        controller.last_hidden = stem
        return None


class _WanBlockPreHook:
    def __init__(self, controller, layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, kwargs):
        del module
        controller = self.controller
        layer_index = self.layer_index
        if not controller.active:
            return None
        if not args:
            raise RuntimeError(f"Wan block L{layer_index:02d} received no positional hidden state")
        incoming = args[0]
        previous = controller.last_hidden
        if previous is None:
            raise RuntimeError(f"Wan block L{layer_index:02d} ran before stem capture")
        incoming_normalized = controller.recorder.normalize_hidden(incoming)
        previous_normalized = controller.recorder.normalize_hidden(previous)
        if controller.key_layer(layer_index):
            controller.recorder.capture(f"L{layer_index:02d}.in", previous_normalized)
        if layer_index in controller.condition_layers:
            incoming_normalized = controller.gate_residual(
                "cond",
                layer_index,
                previous_normalized,
                incoming_normalized,
            )
            if controller.key_layer(layer_index):
                controller.recorder.capture(f"L{layer_index:02d}.cond", incoming_normalized)
        controller.block_inputs[layer_index] = incoming_normalized
        temb = kwargs.get("temb")
        if not isinstance(temb, torch.Tensor):
            raise RuntimeError(f"Wan block L{layer_index:02d} did not receive tensor temb")
        controller.block_temb[layer_index] = temb
        new_args = list(args)
        new_args[0] = incoming_normalized
        return tuple(new_args), kwargs


class _WanSelfAttentionHook:
    def __init__(self, controller, layer_index: int, raw_block) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)
        self.raw_block = raw_block

    def __call__(self, module, args, kwargs, output):
        del module, args, kwargs
        controller = self.controller
        layer_index = self.layer_index
        if not controller.active:
            return None
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Wan self-attention L{layer_index:02d} output is not a tensor")
        alpha = controller.gate_alpha("temp", layer_index)
        gated_output = output if alpha == 1.0 else output * alpha
        if not controller.key_layer(layer_index):
            return gated_output
        block_input = controller.block_inputs.get(layer_index)
        temb = controller.block_temb.get(layer_index)
        if block_input is None or temb is None:
            raise RuntimeError(f"missing block state for temp capture L{layer_index:02d}")
        if temb.ndim != 4:
            raise ValueError(
                f"temp capture expects temb [BV,T,6,C], got {tuple(temb.shape)}"
            )
        batch_size, token_count, hidden_dim = block_input.shape
        frame_count = int(temb.shape[1])
        if token_count % frame_count != 0:
            raise ValueError(
                f"hidden tokens {token_count} are not divisible by frame_count={frame_count}"
            )
        spatial_count = token_count // frame_count
        scale_shift_table = self.raw_block.scale_shift_table.to(
            device=temb.device,
            dtype=torch.float32,
        )
        if scale_shift_table.ndim == 2:
            scale_shift_table = scale_shift_table.view(
                1,
                1,
                scale_shift_table.shape[0],
                scale_shift_table.shape[1],
            )
        elif scale_shift_table.ndim == 3:
            scale_shift_table = scale_shift_table.unsqueeze(1)
        else:
            raise ValueError(
                f"unsupported scale_shift_table shape {tuple(scale_shift_table.shape)}"
            )
        gate_msa = (scale_shift_table + temb.float()).chunk(6, dim=2)[2].squeeze(2)
        post_self = (
            block_input.float().reshape(
                batch_size,
                frame_count,
                spatial_count,
                hidden_dim,
            )
            + gated_output.float().reshape(
                batch_size,
                frame_count,
                spatial_count,
                hidden_dim,
            )
            * gate_msa.unsqueeze(2)
        ).reshape(batch_size, token_count, hidden_dim).type_as(block_input)
        controller.recorder.capture(f"L{layer_index:02d}.temp", post_self)
        return gated_output


class _WanBlockPostHook:
    def __init__(self, controller, layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, kwargs, output):
        del module, args, kwargs
        controller = self.controller
        layer_index = self.layer_index
        if not controller.active:
            return None
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Wan block L{layer_index:02d} output is not a tensor")
        normalized = controller.recorder.normalize_hidden(output)
        if controller.key_layer(layer_index):
            controller.recorder.capture(f"L{layer_index:02d}.base", normalized)
        controller.last_hidden = normalized
        return output


class _WanCrossviewPreHook:
    def __init__(self, controller, layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, kwargs):
        del module
        controller = self.controller
        layer_index = self.layer_index
        if not controller.active:
            return None
        if "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        elif args:
            hidden = args[0]
        else:
            raise RuntimeError(f"crossview L{layer_index:02d} has no hidden_states input")
        controller.crossview_inputs[layer_index] = controller.recorder.normalize_hidden(hidden)
        return None


class _WanCrossviewPostHook:
    def __init__(self, controller, layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, kwargs, output):
        del module, args, kwargs
        controller = self.controller
        layer_index = self.layer_index
        if not controller.active:
            return None
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"crossview L{layer_index:02d} output is not a tensor")
        base = controller.crossview_inputs.get(layer_index)
        output_normalized = controller.recorder.normalize_hidden(output)
        gated = controller.gate_residual(
            "crossview",
            layer_index,
            base,
            output_normalized,
        )
        if controller.key_layer(layer_index):
            controller.recorder.capture(f"L{layer_index:02d}.crossview", gated)
        controller.last_hidden = gated
        return gated


class _WanFinalPreHook:
    def __init__(self, controller) -> None:
        self.controller = controller

    def __call__(self, module, args, kwargs):
        del module, kwargs
        controller = self.controller
        if not controller.active:
            return None
        if not args:
            raise RuntimeError("Wan norm_out received no input")
        final_hidden = controller.recorder.normalize_hidden(args[0])
        controller.recorder.capture("final", final_hidden)
        return None


class WanEntityReactorController:
    def __init__(
        self,
        pipeline,
        key_layers: Sequence[int],
        causal_layers: Sequence[int],
        transport_events_per_sample: int,
        auc_weight: float = DEFAULT_AUC_WEIGHT,
        margin_scale: float = DEFAULT_MARGIN_SCALE,
    ) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.num_layers = len(self.model.blocks)
        self.condition_layers = tuple(
            sorted(int(value) for value in self.model.condition_layer_to_adapter_index.keys())
        )
        self.crossview_layers = tuple(
            sorted(int(value) for value in self.model.crossview_layer_to_index.keys())
        )
        self.key_layers = tuple(sorted(set(int(value) for value in key_layers)))
        self.causal_layers = tuple(sorted(set(int(value) for value in causal_layers)))
        invalid_key = [value for value in self.key_layers if value < 0 or value >= self.num_layers]
        invalid_causal = [value for value in self.causal_layers if value < 0 or value >= self.num_layers]
        if invalid_key or invalid_causal:
            raise ValueError(
                f"layer index out of range, key={invalid_key}, causal={invalid_causal}, num_layers={self.num_layers}"
            )
        self.stage_order_template = self.build_stage_order()
        self.transport_stage_names = self.build_transport_stage_names()
        self.transitions = self.build_module_transitions()
        self.recorder = StreamingEntityRecorder(
            transport_stage_names=self.transport_stage_names,
            transport_events_per_sample=transport_events_per_sample,
            auc_weight=float(auc_weight),
            margin_scale=float(margin_scale),
        )
        self.active = False
        self.gate_spec: dict[tuple[str, int], float] = {}
        self.last_hidden: Optional[torch.Tensor] = None
        self.block_inputs: dict[int, torch.Tensor] = {}
        self.block_temb: dict[int, torch.Tensor] = {}
        self.crossview_inputs: dict[int, torch.Tensor] = {}
        self.handles = []
        self.latest_result: Optional[ForwardResult] = None
        self.attach()

    def unwrap_module(self, module):
        current = module
        visited = set()
        while hasattr(current, "_fsdp_wrapped_module"):
            identity = id(current)
            if identity in visited:
                break
            visited.add(identity)
            candidate = current._fsdp_wrapped_module
            if candidate is current:
                break
            current = candidate
        return current

    def build_stage_order(self) -> list[str]:
        stages = ["stem"]
        condition_set = set(self.condition_layers)
        crossview_set = set(self.crossview_layers)
        for layer in self.key_layers:
            stages.append(f"L{layer:02d}.in")
            if layer in condition_set:
                stages.append(f"L{layer:02d}.cond")
            stages.append(f"L{layer:02d}.temp")
            stages.append(f"L{layer:02d}.base")
            if layer in crossview_set:
                stages.append(f"L{layer:02d}.crossview")
        stages.append("final")
        return stages

    def build_transport_stage_names(self) -> set[str]:
        output_stages = set()
        crossview_set = set(self.crossview_layers)
        for layer in self.key_layers:
            if layer in crossview_set:
                output_stages.add(f"L{layer:02d}.crossview")
            else:
                output_stages.add(f"L{layer:02d}.base")
        output_stages.add("final")
        return output_stages

    def build_module_transitions(self) -> list[ModuleTransition]:
        transitions = []
        condition_set = set(self.condition_layers)
        crossview_set = set(self.crossview_layers)
        key_layer_positions = {layer: position for position, layer in enumerate(self.key_layers)}
        for layer in self.causal_layers:
            if layer not in key_layer_positions:
                continue
            if layer in condition_set:
                transitions.append(
                    ModuleTransition(
                        module="cond",
                        layer=int(layer),
                        before_stage=f"L{layer:02d}.in",
                        after_stage=f"L{layer:02d}.cond",
                        use_stage=f"L{layer:02d}.temp",
                        target_relation="cross_view",
                    )
                )
            temp_before = f"L{layer:02d}.cond" if layer in condition_set else f"L{layer:02d}.in"
            transitions.append(
                ModuleTransition(
                    module="temp",
                    layer=int(layer),
                    before_stage=temp_before,
                    after_stage=f"L{layer:02d}.temp",
                    use_stage=f"L{layer:02d}.base",
                    target_relation="temporal",
                )
            )
            if layer in crossview_set:
                position = key_layer_positions[layer]
                if position + 1 < len(self.key_layers):
                    next_layer = self.key_layers[position + 1]
                    use_stage = f"L{next_layer:02d}.in"
                else:
                    use_stage = "final"
                transitions.append(
                    ModuleTransition(
                        module="crossview",
                        layer=int(layer),
                        before_stage=f"L{layer:02d}.base",
                        after_stage=f"L{layer:02d}.crossview",
                        use_stage=use_stage,
                        target_relation="cross_view",
                    )
                )
        module_rank = {name: index for index, name in enumerate(MODULES)}
        transitions.sort(key=lambda item: (item.layer, module_rank[item.module]))
        return transitions

    def gate_alpha(self, module_name: str, layer_index: int) -> float:
        return float(self.gate_spec.get((str(module_name), int(layer_index)), 1.0))

    def gate_residual(
        self,
        module_name: str,
        layer_index: int,
        module_input: Optional[torch.Tensor],
        module_output: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.gate_alpha(module_name, layer_index)
        if alpha == 1.0 or module_input is None:
            return module_output
        if module_input.shape != module_output.shape:
            raise RuntimeError(
                f"cannot gate {module_name}@L{layer_index:02d}, input={tuple(module_input.shape)}, "
                f"output={tuple(module_output.shape)}"
            )
        return module_input + alpha * (module_output - module_input)

    def key_layer(self, layer_index: int) -> bool:
        return int(layer_index) in self.key_layers

    def attach(self) -> None:
        self.handles.append(
            self.model.patch_embedding.register_forward_hook(
                _WanPatchEmbeddingHook(self),
                with_kwargs=True,
            )
        )
        for layer_index, wrapper_block in enumerate(self.model.blocks):
            self.handles.append(
                wrapper_block.register_forward_pre_hook(
                    _WanBlockPreHook(self, layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                wrapper_block.register_forward_hook(
                    _WanBlockPostHook(self, layer_index),
                    with_kwargs=True,
                )
            )
            raw_block = self.unwrap_module(wrapper_block)
            if not hasattr(raw_block, "attn1") or not hasattr(raw_block, "scale_shift_table"):
                raise TypeError(
                    f"Wan block L{layer_index:02d} is missing attn1/scale_shift_table after FSDP unwrap"
                )
            self.handles.append(
                raw_block.attn1.register_forward_hook(
                    _WanSelfAttentionHook(self, layer_index, raw_block),
                    with_kwargs=True,
                )
            )
        for cross_index, wrapper_cross in enumerate(self.model.crossview_modules):
            layer_index = int(self.crossview_layers[cross_index])
            self.handles.append(
                wrapper_cross.register_forward_pre_hook(
                    _WanCrossviewPreHook(self, layer_index),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                wrapper_cross.register_forward_hook(
                    _WanCrossviewPostHook(self, layer_index),
                    with_kwargs=True,
                )
            )
        self.handles.append(
            self.model.norm_out.register_forward_pre_hook(
                _WanFinalPreHook(self),
                with_kwargs=True,
            )
        )

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def configure_sample(
        self,
        plan: ObservationPlan,
        events: dict[str, list[ProbeEvent]],
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> None:
        patch_size_t, patch_size_h, patch_size_w = self.model.config.patch_size
        if int(patch_size_t) != 1:
            raise ValueError(f"analysis expects Wan patch_size_t=1, got {patch_size_t}")
        patch_height = int(latent_height) // int(patch_size_h)
        patch_width = int(latent_width) // int(patch_size_w)
        self.recorder.configure(
            plan=plan,
            events=events,
            batch_size=int(batch_size),
            patch_height=patch_height,
            patch_width=patch_width,
            device=self.pipeline.device,
        )

    def begin_forward(self, gate_spec: Optional[dict[tuple[str, int], float]] = None) -> None:
        self.active = True
        self.gate_spec = dict(gate_spec or {})
        self.last_hidden = None
        self.block_inputs.clear()
        self.block_temb.clear()
        self.crossview_inputs.clear()
        self.latest_result = None
        self.recorder.begin_forward()

    def end_forward(self) -> ForwardResult:
        self.active = False
        result = self.recorder.end_forward()
        missing = [name for name in self.stage_order_template if name not in result.stage_metrics]
        if missing:
            raise RuntimeError("WAN entity hooks missed expected stages " + ",".join(missing))
        result.stage_order = [
            name for name in self.stage_order_template if name in result.stage_metrics
        ]
        self.latest_result = result
        self.last_hidden = None
        self.block_inputs.clear()
        self.block_temb.clear()
        self.crossview_inputs.clear()
        return result

    def abort_forward(self) -> None:
        self.active = False
        self.last_hidden = None
        self.block_inputs.clear()
        self.block_temb.clear()
        self.crossview_inputs.clear()
        self.latest_result = None


class WanControlledInputBuilder:
    def __init__(self, pipeline, seed: int) -> None:
        self.pipeline = pipeline
        self.seed = int(seed)

    def encode_real_video(self, batch: dict) -> torch.Tensor:
        images = normalize_image_tensor(batch["vae_images"].to(self.pipeline.device))
        latents = encode_video_with_wan_vae_mode(self.pipeline.vae, images)
        return latents.to(device=self.pipeline.device, dtype=self.pipeline.model_dtype)

    def make_noise(self, latent_shape: Sequence[int], sample_index: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(sample_index) * 1009)
        noise = torch.randn(tuple(int(value) for value in latent_shape), generator=generator, dtype=torch.float32)
        return noise.to(device=self.pipeline.device, dtype=self.pipeline.model_dtype)

    def timestep_for_sigma(self, sigma: float) -> torch.Tensor:
        scheduler = self.pipeline.train_scheduler
        scheduler_sigmas = scheduler.sigmas.detach().float().cpu()
        scheduler_timesteps = scheduler.timesteps.detach().float().cpu()
        usable_count = min(int(scheduler_sigmas.shape[0]), int(scheduler_timesteps.shape[0]))
        if usable_count <= 0:
            raise RuntimeError("Wan scheduler has no sigma/timestep entries")
        distance = torch.abs(scheduler_sigmas[:usable_count] - float(sigma))
        index = int(torch.argmin(distance).item())
        return scheduler_timesteps[index].to(device=self.pipeline.device, dtype=torch.float32)

    def prepare_conditions(self, batch: dict, latents: torch.Tensor) -> dict:
        disable_temporal = torch.zeros(
            int(latents.shape[0]),
            dtype=torch.bool,
            device="cpu",
        )
        return self.pipeline.get_conditions(
            latent_shape=latents.shape,
            batch=batch,
            disable_temporal=disable_temporal,
            do_classifier_free_guidance=False,
        )

    def run_forward(
        self,
        controller: WanEntityReactorController,
        latents: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
        model_conditions: dict,
        gate_spec: Optional[dict[tuple[str, int], float]] = None,
    ) -> ForwardResult:
        noisy_latents = float(sigma) * noise + (1.0 - float(sigma)) * latents
        model_input = einops.rearrange(
            noisy_latents,
            "b t v c h w -> (b v) c t h w",
        ).to(device=self.pipeline.device, dtype=self.pipeline.model_dtype)
        timestep = self.timestep_for_sigma(float(sigma))
        timestep_tokens = timestep.expand(
            int(latents.shape[0]) * int(latents.shape[2]),
            int(latents.shape[1]),
        ).contiguous()
        controller.begin_forward(gate_spec=gate_spec)
        try:
            with torch.no_grad(), self.pipeline.get_autocast_context():
                model_output = self.pipeline.model_wrapper(
                    hidden_states=model_input,
                    timestep=timestep_tokens,
                    return_dict=False,
                    **model_conditions,
                )
            result = controller.end_forward()
        except Exception:
            controller.abort_forward()
            raise
        del model_output, model_input, noisy_latents, timestep_tokens
        return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--causal-sigma", type=float, default=0.6, help="Compatibility-only. Unified v20 runs causal intervention at every --sigmas value.")
    parser.add_argument("--causal-sample-count", type=int, default=8, help="Causal clips. 0 means all accepted baseline clips.")
    parser.add_argument("--key-layers", type=int, nargs="*", default=None)
    parser.add_argument("--causal-layers", type=int, nargs="*", default=None)
    parser.add_argument("--max-entities", type=int, default=32)
    parser.add_argument("--minimum-track-length", type=int, default=3)
    parser.add_argument("--minimum-visible-anchors", type=int, default=5)
    parser.add_argument("--minimum-area-ratio", type=float, default=5e-5)
    parser.add_argument("--ring-expand-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-negatives", type=int, default=6)
    parser.add_argument("--maximum-events-per-family", type=int, default=256)
    parser.add_argument("--minimum-temporal-gap", type=int, default=None)
    parser.add_argument(
        "--entity-classes",
        nargs="+",
        default=("car", "vehicle", "truck", "bus"),
    )
    parser.add_argument("--transport-events-per-sample", type=int, default=1)
    parser.add_argument("--transport-reservoir-max", type=int, default=24)
    parser.add_argument("--analysis-seed", type=int, default=3107)
    parser.add_argument("--auc-weight", type=float, default=DEFAULT_AUC_WEIGHT)
    parser.add_argument("--margin-scale", type=float, default=DEFAULT_MARGIN_SCALE)
    parser.add_argument("--keep-resume", action="store_true")
    parser.add_argument("--no-transport", action="store_true")
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=None,
        help="Compatibility-only no-op. All quantitative analysis uses the full hidden dimension.",
    )
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=None,
        help="Compatibility-only no-op. No random projection is used.",
    )
    return parser


def rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def distributed_barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def setup_distributed(config: dict) -> torch.device:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend=config["ddp_backend"])
        return device
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def initialize_global_state(config: dict) -> None:
    dwm.common.global_state.clear()
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)


def checkpoint_sort_value(path: Path) -> tuple[int, str]:
    numeric_tokens = re.findall(r"\d+", path.stem)
    numeric_value = int(numeric_tokens[-1]) if numeric_tokens else -1
    return numeric_value, str(path.resolve())


def checkpoint_output_name(checkpoint_path: Path) -> str:
    numeric_tokens = re.findall(r"\d+", checkpoint_path.stem)
    if numeric_tokens:
        return f"checkpoint-{int(numeric_tokens[-1])}"
    return checkpoint_path.stem


def resolve_checkpoint_paths(config: dict, requested: Optional[str], analyze_all: bool) -> list[Path]:
    source = requested
    if source is None:
        source = config.get("pipeline", {}).get("model_checkpoint_path")
    if source is None or str(source).strip() == "":
        raise ValueError("checkpoint path is missing")
    source_text = os.path.expanduser(os.path.expandvars(str(source)))
    wildcard = any(mark in source_text for mark in ("*", "?", "["))
    candidates = []
    if wildcard:
        candidates = [Path(value) for value in glob.glob(source_text)]
    else:
        source_path = Path(source_text)
        if source_path.is_file():
            candidates = [source_path]
        elif source_path.is_dir():
            search_root = source_path / "checkpoints" if (source_path / "checkpoints").is_dir() else source_path
            candidates = list(search_root.glob("*.pth")) + list(search_root.glob("*.safetensors"))
        else:
            raise FileNotFoundError(f"checkpoint source does not exist {source_path}")
    candidates = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=checkpoint_sort_value,
    )
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found from {source_text}")
    return candidates if analyze_all else [candidates[-1]]


def instantiate_pipeline(
    base_config: dict,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
):
    config = copy.deepcopy(base_config)
    config.pop("optimizer", None)
    config.pop("lr_scheduler", None)
    config["pipeline"]["model_checkpoint_path"] = str(checkpoint_path)
    config["pipeline"]["metrics"] = {}
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=str(output_path),
        config=config,
        device=device,
    )
    pipeline.model.eval()
    pipeline.model_wrapper.eval()
    if hasattr(pipeline.model, "disable_gradient_checkpointing"):
        pipeline.model.disable_gradient_checkpointing()
    return pipeline, config


def instantiate_validation_dataset(base_config: dict):
    return dwm.common.create_instance_from_config(base_config["validation_dataset"])


def instantiate_validation_collate(base_config: dict):
    collate_config = copy.deepcopy(base_config.get("validation_dataloader", {}).get("collate_fn"))
    if collate_config is None:
        return None
    return dwm.common.create_instance_from_config(collate_config)


def collate_one(dataset, collate_fn, sample_index: int) -> dict:
    sample = dataset[int(sample_index)]
    if collate_fn is None:
        return torch.utils.data.default_collate([sample])
    return collate_fn([sample])


def extract_crossview_mask(batch: dict) -> np.ndarray:
    value = batch.get("crossview_mask")
    if value is None:
        raise KeyError("analysis requires crossview_mask in validation samples")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.bool_, copy=False)
    return np.asarray(value, dtype=np.bool_)


def choose_key_layers(model, requested: Optional[Sequence[int]]) -> list[int]:
    if requested:
        return sorted(set(int(value) for value in requested))
    condition_layers = set(int(value) for value in model.condition_layer_to_adapter_index.keys())
    crossview_layers = set(int(value) for value in model.crossview_layer_to_index.keys())
    selected = condition_layers | crossview_layers | {len(model.blocks) - 1}
    return sorted(selected)


def choose_causal_layers(model, requested: Optional[Sequence[int]], key_layers: Sequence[int]) -> list[int]:
    if requested:
        selected = sorted(set(int(value) for value in requested))
    else:
        selected = sorted(int(value) for value in model.crossview_layer_to_index.keys())
    key_set = set(int(value) for value in key_layers)
    return [value for value in selected if value in key_set]


def sigma_key(value: float) -> str:
    return f"{float(value):.6f}"


def closest_sigma(sigmas: Sequence[float], target: float) -> float:
    return float(min((float(value) for value in sigmas), key=lambda value: abs(value - float(target))))


def forward_result_to_payload(result: ForwardResult) -> dict:
    return {
        "stage_order": list(result.stage_order),
        "stage_metrics": copy.deepcopy(result.stage_metrics),
    }


def forward_result_from_payload(payload: dict) -> ForwardResult:
    return ForwardResult(
        stage_order=[str(value) for value in payload["stage_order"]],
        stage_metrics=copy.deepcopy(payload["stage_metrics"]),
        transport_payloads={},
    )


def empty_transport_reservoir(feature_dim: int = 0, geometry_dim: int = 0) -> dict[str, np.ndarray]:
    return {
        "feature_delta": np.empty((0, int(feature_dim)), dtype=np.float32),
        "geometry_delta": np.empty((0, int(geometry_dim)), dtype=np.float32),
        "sample_id": np.empty((0,), dtype=np.int64),
        "track_id": np.empty((0,), dtype=np.int64),
        "camera_pair": np.empty((0,), dtype=np.int64),
    }


def merge_transport_reservoir(
    current: Optional[dict[str, np.ndarray]],
    incoming: dict[str, np.ndarray],
    maximum_events: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if incoming["feature_delta"].shape[0] == 0:
        return current if current is not None else empty_transport_reservoir(
            incoming["feature_delta"].shape[1],
            incoming["geometry_delta"].shape[1],
        )
    if current is None or current["feature_delta"].shape[0] == 0:
        merged = {key: np.asarray(value).copy() for key, value in incoming.items()}
    else:
        merged = {
            key: np.concatenate([current[key], incoming[key]], axis=0)
            for key in incoming
        }
    event_count = int(merged["feature_delta"].shape[0])
    if event_count <= int(maximum_events):
        return merged
    generator = np.random.default_rng(int(seed))
    selected = np.sort(
        generator.choice(event_count, size=int(maximum_events), replace=False)
    )
    return {key: value[selected] for key, value in merged.items()}


def transport_knn_alignment(
    payload: dict[str, np.ndarray],
    neighbor_count: int = 5,
) -> float:
    feature_delta = payload["feature_delta"].astype(np.float64, copy=False)
    geometry_delta = payload["geometry_delta"].astype(np.float64, copy=False)
    sample_id = payload["sample_id"].astype(np.int64, copy=False)
    track_id = payload["track_id"].astype(np.int64, copy=False)
    event_count = int(feature_delta.shape[0])
    if event_count < max(12, int(neighbor_count) + 2):
        return float("nan")
    feature_delta = feature_delta / np.maximum(
        np.linalg.norm(feature_delta, axis=1, keepdims=True),
        1e-10,
    )
    geometry_centered = geometry_delta - np.nanmean(geometry_delta, axis=0, keepdims=True)
    geometry_scale = np.nanstd(geometry_centered, axis=0, keepdims=True)
    geometry_centered = geometry_centered / np.where(geometry_scale < 1e-5, 1.0, geometry_scale)
    feature_similarity = feature_delta @ feature_delta.T
    geometry_norm = np.sum(geometry_centered * geometry_centered, axis=1, keepdims=True)
    geometry_distance = geometry_norm + geometry_norm.T - 2.0 * geometry_centered @ geometry_centered.T
    geometry_distance = np.maximum(geometry_distance, 0.0)
    scores = []
    for event_index in range(event_count):
        candidate_mask = (sample_id != sample_id[event_index]) & (track_id != track_id[event_index])
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size < max(3, int(neighbor_count)):
            continue
        current_k = min(int(neighbor_count), int(candidate_indices.size))
        geometry_values = geometry_distance[event_index, candidate_indices]
        feature_values = feature_similarity[event_index, candidate_indices]
        geometry_local = np.argpartition(geometry_values, current_k - 1)[:current_k]
        feature_local = np.argpartition(-feature_values, current_k - 1)[:current_k]
        geometry_neighbors = candidate_indices[geometry_local]
        feature_neighbors = candidate_indices[feature_local]
        overlap = np.intersect1d(
            geometry_neighbors,
            feature_neighbors,
            assume_unique=False,
        ).size / float(current_k)
        chance = current_k / float(candidate_indices.size)
        normalized = (overlap - chance) / max(1.0 - chance, 1e-8)
        scores.append(float(normalized))
    if not scores:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def bootstrap_mean(values: Sequence[float], seed: int) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(array.mean())
    if count == 1:
        return mean, mean, mean, count
    generator = np.random.default_rng(int(seed))
    bootstrap_indices = generator.integers(0, count, size=(512, count))
    bootstrap_means = array[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    return mean, float(low), float(high), count


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def make_resume_signature(
    args,
    checkpoint_path: Path,
    key_layers: Sequence[int],
    causal_layers: Sequence[int],
) -> dict:
    return {
        "analysis_version": WAN_ENTITY_ANALYSIS_VERSION,
        "resume_version": WAN_ENTITY_RESUME_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sigmas": [float(value) for value in args.sigmas],
                "key_layers": [int(value) for value in key_layers],
        "causal_layers": [int(value) for value in causal_layers],
        "sample_count": int(args.sample_count),
        "sample_stride": int(args.sample_stride),
        "start_index": int(args.start_index),
        "sample_indices": None if not args.sample_indices else [int(value) for value in args.sample_indices],
        "entity_classes": [str(value) for value in args.entity_classes],
        "causal_sample_count": int(args.causal_sample_count),
        "auc_weight": float(args.auc_weight),
        "margin_scale": float(args.margin_scale),
        "max_entities": int(args.max_entities),
        "minimum_track_length": int(args.minimum_track_length),
        "minimum_visible_anchors": int(args.minimum_visible_anchors),
        "minimum_area_ratio": float(args.minimum_area_ratio),
        "ring_expand_ratio": float(args.ring_expand_ratio),
        "maximum_negatives": int(args.maximum_negatives),
        "maximum_events_per_family": int(args.maximum_events_per_family),
        "minimum_temporal_gap": args.minimum_temporal_gap,
        "transport_events_per_sample": int(args.transport_events_per_sample),
        "transport_reservoir_max": int(args.transport_reservoir_max),
        "no_transport": bool(args.no_transport),
        "no_causal": bool(args.no_causal),
        "strict_negative_tiers": 3,
        "full_hidden_dimension": True,
    }


def initial_candidate_state(args, dataset_length: int) -> tuple[list[int], int, int]:
    if int(args.sample_stride) <= 0:
        raise ValueError("sample_stride must be positive")
    if args.sample_indices:
        candidate_indices = [int(value) for value in args.sample_indices]
        target_count = len(candidate_indices)
        if target_count <= 0:
            raise ValueError("sample_indices is empty")
        next_backfill = max(candidate_indices) + int(args.sample_stride)
    else:
        if int(args.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        target_count = int(args.sample_count)
        candidate_indices = [
            int(args.start_index) + position * int(args.sample_stride)
            for position in range(target_count)
        ]
        next_backfill = int(args.start_index) + target_count * int(args.sample_stride)
    invalid = [value for value in candidate_indices if value < 0 or value >= int(dataset_length)]
    if invalid:
        raise IndexError(
            f"initial sample index outside dataset, first invalid={invalid[0]}, dataset_length={dataset_length}"
        )
    return candidate_indices, int(next_backfill), int(target_count)


def new_resume_state(
    signature: dict,
    candidate_indices: list[int],
    next_backfill: int,
    target_count: int,
) -> dict:
    return {
        "signature": copy.deepcopy(signature),
        "status": "running",
        "candidate_indices": [int(value) for value in candidate_indices],
        "candidate_position": 0,
        "next_backfill_index": int(next_backfill),
        "target_count": int(target_count),
        "accepted_samples": [],
        "rejected_samples": {},
        "baselines": {},
        "transport": {},
        "interventions": {},
    }


def load_resume_state(path: Path, signature: dict) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("rb") as file:
        payload = pickle.load(file)
    if payload.get("signature") != signature:
        raise RuntimeError(
            f"resume state {path} belongs to another analysis configuration; move/delete _resume first"
        )
    return payload


def commit_resume_state(path: Path, state: dict) -> None:
    if rank_zero():
        atomic_pickle(path, state)
    distributed_barrier()


def ensure_next_candidate(state: dict, dataset_length: int, sample_stride: int) -> int:
    position = int(state["candidate_position"])
    candidates = state["candidate_indices"]
    if position < len(candidates):
        return int(candidates[position])
    next_index = int(state["next_backfill_index"])
    if next_index >= int(dataset_length):
        raise RuntimeError(
            f"ran out of validation data while backfilling valid clips, next={next_index}, length={dataset_length}"
        )
    candidates.append(next_index)
    state["next_backfill_index"] = next_index + int(sample_stride)
    return next_index


def event_family_is_valid(events: dict[str, list[ProbeEvent]]) -> bool:
    return all(len(events.get(family, [])) > 0 for family in PROBE_FAMILIES)


def baseline_payload(state: dict, sigma: float, sample_index: int) -> Optional[dict]:
    return state.get("baselines", {}).get(sigma_key(sigma), {}).get(int(sample_index))


def store_baseline_payload(state: dict, sigma: float, sample_index: int, result: ForwardResult) -> None:
    state.setdefault("baselines", {}).setdefault(sigma_key(sigma), {})[int(sample_index)] = (
        forward_result_to_payload(result)
    )


def merge_result_transport(
    state: dict,
    sigma: float,
    result: ForwardResult,
    maximum_events: int,
    seed: int,
) -> None:
    sigma_store = state.setdefault("transport", {}).setdefault(sigma_key(sigma), {})
    for stage_name, family_payloads in result.transport_payloads.items():
        payload = family_payloads.get("cross_view")
        if payload is None or payload["feature_delta"].shape[0] == 0:
            continue
        current = sigma_store.get(stage_name)
        sigma_store[stage_name] = merge_transport_reservoir(
            current,
            payload,
            maximum_events=int(maximum_events),
            seed=int(seed) + int(round(float(sigma) * 1000)) * 1009 + sum(ord(ch) for ch in stage_name),
        )


def metric_value(result: ForwardResult, stage: str, family: str, metric: str = "structured_score") -> float:
    stage_metrics = result.stage_metrics.get(stage)
    if stage_metrics is None:
        return float("nan")
    family_metrics = stage_metrics.get(family)
    if family_metrics is None:
        return float("nan")
    return float(family_metrics.get(metric, float("nan")))


def intervention_label(transition: ModuleTransition) -> str:
    return f"{transition.module}_L{int(transition.layer):02d}"


def summarize_intervention(
    transition: ModuleTransition,
    baseline: ForwardResult,
    ablated: ForwardResult,
    sigma: float,
) -> dict:
    relation_summary = {}
    for family in PROBE_FAMILIES:
        baseline_use = metric_value(baseline, transition.use_stage, family)
        ablated_use = metric_value(ablated, transition.use_stage, family)
        baseline_final = metric_value(baseline, "final", family)
        ablated_final = metric_value(ablated, "final", family)
        relation_summary[family] = {
            "use": float(baseline_use - ablated_use)
            if np.isfinite(baseline_use) and np.isfinite(ablated_use)
            else float("nan"),
            "retain": float(baseline_final - ablated_final)
            if np.isfinite(baseline_final) and np.isfinite(ablated_final)
            else float("nan"),
        }
    baseline_before = metric_value(
        baseline,
        transition.before_stage,
        transition.target_relation,
    )
    baseline_after = metric_value(
        baseline,
        transition.after_stage,
        transition.target_relation,
    )
    return {
        "label": intervention_label(transition),
        "module": transition.module,
        "layer": int(transition.layer),
        "sigma": float(sigma),
        "target_relation": transition.target_relation,
        "before_stage": transition.before_stage,
        "after_stage": transition.after_stage,
        "use_stage": transition.use_stage,
        "write": float(baseline_after - baseline_before)
        if np.isfinite(baseline_before) and np.isfinite(baseline_after)
        else float("nan"),
        "relations": relation_summary,
    }


def aggregate_baselines(
    state: dict,
    sigmas: Sequence[float],
    stage_order: Sequence[str],
    seed: int,
) -> dict:
    aggregate = {}
    accepted = [int(value) for value in state["accepted_samples"]]
    for sigma_position, sigma in enumerate(sigmas):
        sigma_results = state["baselines"].get(sigma_key(sigma), {})
        sigma_summary = {}
        for stage_position, stage in enumerate(stage_order):
            stage_summary = {}
            for family_position, family in enumerate(PROBE_FAMILIES):
                family_summary = {}
                for metric_position, metric in enumerate(
                    ("structured_score", "structured_auc", "structured_margin", "structured_top1")
                ):
                    values = []
                    for sample_index in accepted:
                        payload = sigma_results.get(sample_index)
                        if payload is None:
                            continue
                        result = forward_result_from_payload(payload)
                        values.append(metric_value(result, stage, family, metric))
                    summary = bootstrap_mean(
                        values,
                        int(seed)
                        + sigma_position * 100003
                        + stage_position * 1009
                        + family_position * 53
                        + metric_position * 7,
                    )
                    family_summary[metric] = {
                        "mean": summary[0],
                        "low": summary[1],
                        "high": summary[2],
                        "count": summary[3],
                    }
                stage_summary[family] = family_summary
            sigma_summary[stage] = stage_summary
        aggregate[sigma_key(sigma)] = sigma_summary
    return aggregate


def aggregate_interventions(state: dict, transitions: Sequence[ModuleTransition], seed: int) -> list[dict]:
    output = []
    for transition_position, transition in enumerate(transitions):
        label = intervention_label(transition)
        per_sample = state.get("interventions", {}).get(label, {})
        if not per_sample:
            continue
        entry = {
            "label": label,
            "module": transition.module,
            "layer": int(transition.layer),
            "target_relation": transition.target_relation,
        }
        writes = [float(value.get("write", float("nan"))) for value in per_sample.values()]
        write_summary = bootstrap_mean(writes, int(seed) + transition_position * 1009)
        entry["write"] = {
            "mean": write_summary[0],
            "low": write_summary[1],
            "high": write_summary[2],
            "count": write_summary[3],
        }
        entry["relations"] = {}
        for family_position, family in enumerate(PROBE_FAMILIES):
            uses = [
                float(value.get("relations", {}).get(family, {}).get("use", float("nan")))
                for value in per_sample.values()
            ]
            retains = [
                float(value.get("relations", {}).get(family, {}).get("retain", float("nan")))
                for value in per_sample.values()
            ]
            use_summary = bootstrap_mean(
                uses,
                int(seed) + transition_position * 1013 + family_position * 37,
            )
            retain_summary = bootstrap_mean(
                retains,
                int(seed) + transition_position * 1019 + family_position * 41,
            )
            entry["relations"][family] = {
                "use": {
                    "mean": use_summary[0],
                    "low": use_summary[1],
                    "high": use_summary[2],
                    "count": use_summary[3],
                },
                "retain": {
                    "mean": retain_summary[0],
                    "low": retain_summary[1],
                    "high": retain_summary[2],
                    "count": retain_summary[3],
                },
            }
        output.append(entry)
    return output


def aggregate_transport(state: dict, sigmas: Sequence[float], stage_order: Sequence[str]) -> dict:
    output = {}
    for sigma in sigmas:
        sigma_store = state.get("transport", {}).get(sigma_key(sigma), {})
        stage_values = {}
        for stage in stage_order:
            payload = sigma_store.get(stage)
            if payload is None:
                stage_values[stage] = float("nan")
                continue
            stage_values[stage] = transport_knn_alignment(payload)
        output[sigma_key(sigma)] = stage_values
    return output


def layer_output_stages(controller: WanEntityReactorController) -> list[str]:
    crossview_set = set(controller.crossview_layers)
    output = []
    for layer in controller.key_layers:
        if layer in crossview_set:
            output.append(f"L{layer:02d}.crossview")
        else:
            output.append(f"L{layer:02d}.base")
    output.append("final")
    return output


def build_plot_cache(
    state: dict,
    args,
    controller: WanEntityReactorController,
    checkpoint_path: Path,
) -> dict:
    output_stages = layer_output_stages(controller)
    baseline_aggregate = aggregate_baselines(
        state,
        args.sigmas,
        output_stages,
        int(args.analysis_seed),
    )
    interventions = aggregate_interventions(
        state,
        controller.transitions,
        int(args.analysis_seed),
    )
    transport = {} if args.no_transport else aggregate_transport(
        state,
        args.sigmas,
        output_stages,
    )
    return {
        "version": WAN_ENTITY_ANALYSIS_VERSION,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "accepted_samples": [int(value) for value in state["accepted_samples"]],
        "rejected_samples": copy.deepcopy(state["rejected_samples"]),
        "sigmas": [float(value) for value in args.sigmas],
        "causal_sigma": closest_sigma(args.sigmas, args.causal_sigma),
        "key_layers": [int(value) for value in controller.key_layers],
        "causal_layers": [int(value) for value in controller.causal_layers],
        "condition_layers": [int(value) for value in controller.condition_layers],
        "crossview_layers": [int(value) for value in controller.crossview_layers],
        "output_stages": output_stages,
        "baseline": baseline_aggregate,
        "transport": transport,
        "interventions": interventions,
        "analysis_policy": {
            "hidden_dimension": "full",
            "hidden_dtype_for_metrics": "fp32",
            "hard_negative_tiers": [
                "same-view exact-time same-class",
                "same-view nearby-time same-class",
                "same-view same-class",
            ],
            "cross_class_fallback": False,
            "invalid_clip_backfill": True,
            "temp_definition": "Wan attn1 self-attention branch only",
            "cond_definition": "condition-image residual plus per-layer camera residual before Wan block",
            "crossview_definition": "WanCrossviewBlock attention plus mixer",
        },
    }


def write_plot_cache(cache: dict, run_root: Path) -> tuple[Path, Path]:
    cache_path = run_root / "wan_entity_relation_plot_data.pkl"
    manifest_path = run_root / "wan_entity_relation_manifest.json"
    atomic_pickle(cache_path, cache)
    manifest = {
        "version": cache["version"],
        "checkpoint": cache["checkpoint"],
        "accepted_sample_count": len(cache["accepted_samples"]),
        "accepted_samples": cache["accepted_samples"],
        "sigmas": cache["sigmas"],
        "key_layers": cache["key_layers"],
        "causal_layers": cache["causal_layers"],
        "analysis_policy": cache["analysis_policy"],
        "plot_cache": cache_path.name,
    }
    atomic_json(manifest_path, manifest)
    return cache_path, manifest_path


def render_atlas(cache: dict, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    output_stages = cache["output_stages"]
    sigmas = [float(value) for value in cache["sigmas"]]
    relation_matrices = {}
    for family in PROBE_FAMILIES:
        matrix = np.full((len(sigmas), len(output_stages)), np.nan, dtype=np.float32)
        for sigma_index, sigma in enumerate(sigmas):
            sigma_summary = cache["baseline"][sigma_key(sigma)]
            for stage_index, stage in enumerate(output_stages):
                matrix[sigma_index, stage_index] = float(
                    sigma_summary[stage][family]["structured_score"]["mean"]
                )
        relation_matrices[family] = matrix
    transport_matrix = np.full((len(sigmas), len(output_stages)), np.nan, dtype=np.float32)
    for sigma_index, sigma in enumerate(sigmas):
        sigma_transport = cache.get("transport", {}).get(sigma_key(sigma), {})
        for stage_index, stage in enumerate(output_stages):
            transport_matrix[sigma_index, stage_index] = float(
                sigma_transport.get(stage, float("nan"))
            )
    causal_sigma = float(cache["causal_sigma"])
    causal_sigma_index = int(np.argmin(np.abs(np.asarray(sigmas) - causal_sigma)))
    x = np.arange(len(output_stages))
    figure = plt.figure(figsize=(17.0, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    heatmap_titles = {
        "cross_view": "Cross-view entity relation",
        "temporal": "Temporal entity relation",
        "mixed": "View + time entity relation",
    }
    for panel_index, family in enumerate(PROBE_FAMILIES):
        axis = figure.add_subplot(grid[0, panel_index])
        image = axis.imshow(relation_matrices[family], aspect="auto", vmin=-1.0, vmax=1.0)
        axis.set_title(heatmap_titles[family])
        axis.set_yticks(np.arange(len(sigmas)))
        axis.set_yticklabels([f"{value:.1f}" for value in sigmas])
        axis.set_xticks(x)
        axis.set_xticklabels(output_stages, rotation=75, ha="right", fontsize=7)
        axis.set_ylabel("noise sigma")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    relation_axis = figure.add_subplot(grid[1, 0])
    for family in PROBE_FAMILIES:
        relation_axis.plot(
            x,
            relation_matrices[family][causal_sigma_index],
            marker="o",
            label=family,
        )
    relation_axis.set_xticks(x)
    relation_axis.set_xticklabels(output_stages, rotation=75, ha="right", fontsize=7)
    relation_axis.set_ylim(-1.05, 1.05)
    relation_axis.set_title(f"Relation profile at sigma {sigmas[causal_sigma_index]:.1f}")
    relation_axis.legend(fontsize=8)
    transport_axis = figure.add_subplot(grid[1, 1])
    transport_axis.plot(
        x,
        transport_matrix[causal_sigma_index],
        marker="o",
    )
    transport_axis.set_xticks(x)
    transport_axis.set_xticklabels(output_stages, rotation=75, ha="right", fontsize=7)
    transport_axis.set_title("Cross-view projected-geometry transport")
    intervention_axis = figure.add_subplot(grid[1, 2])
    interventions = cache.get("interventions", [])
    labels = [entry["label"] for entry in interventions]
    use_values = []
    for entry in interventions:
        target_relation = str(entry["target_relation"])
        use_values.append(
            float(entry["relations"][target_relation]["use"]["mean"])
        )
    if labels:
        positions = np.arange(len(labels))
        intervention_axis.bar(positions, use_values)
        intervention_axis.set_xticks(positions)
        intervention_axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    intervention_axis.axhline(0.0, linewidth=0.8)
    intervention_axis.set_title("Causal USE by temp / cond / crossview")
    figure.suptitle("WAN Entity Relation Mechanism Atlas", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_baseline_collection(
    args,
    dataset,
    collate_fn,
    resolver: NuScenesEntityResolver,
    pipeline,
    input_builder: WanControlledInputBuilder,
    controller: WanEntityReactorController,
    state: dict,
    resume_path: Path,
) -> None:
    dataset_length = len(dataset)
    while len(state["accepted_samples"]) < int(state["target_count"]):
        sample_index = ensure_next_candidate(
            state,
            dataset_length,
            int(args.sample_stride),
        )
        if rank_zero():
            print(
                f"[WanEntity] candidate={sample_index} accepted={len(state['accepted_samples'])}/{state['target_count']}",
                flush=True,
            )
        try:
            batch = collate_one(dataset, collate_fn, sample_index)
            frame_count = int(batch["vae_images"].shape[1])
            latent_time_count = int(pipeline.get_latent_sequence_length(frame_count))
            crossview_mask = extract_crossview_mask(batch)
            plan = resolver.build_observation_plan(
                sample_index=sample_index,
                target_time_count=latent_time_count,
                crossview_mask=crossview_mask,
            )
            events = build_probe_events(
                plan,
                maximum_events_per_family=int(args.maximum_events_per_family),
                maximum_negatives=int(args.maximum_negatives),
                minimum_temporal_gap=args.minimum_temporal_gap,
            )
            if not event_family_is_valid(events):
                raise NoCrossCameraTransitionError(
                    "one or more probe families has no strict same-class event"
                )
        except (NoCrossCameraTransitionError, ValueError, KeyError, RuntimeError) as error:
            state["rejected_samples"][int(sample_index)] = str(error)
            state["candidate_position"] = int(state["candidate_position"]) + 1
            commit_resume_state(resume_path, state)
            if rank_zero():
                print(f"[WanEntity] reject sample={sample_index} reason={error}", flush=True)
            continue
        latents = input_builder.encode_real_video(batch)
        noise = input_builder.make_noise(latents.shape, sample_index)
        model_conditions = input_builder.prepare_conditions(batch, latents)
        controller.configure_sample(
            plan=plan,
            events=events,
            batch_size=int(latents.shape[0]),
            latent_height=int(latents.shape[-2]),
            latent_width=int(latents.shape[-1]),
        )
        sample_results = {}
        try:
            for sigma in args.sigmas:
                result = input_builder.run_forward(
                    controller,
                    latents,
                    noise,
                    float(sigma),
                    model_conditions,
                    gate_spec=None,
                )
                sample_results[float(sigma)] = result
                if rank_zero():
                    print(
                        f"[WanEntity] baseline sample={sample_index} sigma={float(sigma):.2f} full-D",
                        flush=True,
                    )
        except Exception:
            controller.recorder.clear_sample()
            raise
        for sigma, result in sample_results.items():
            store_baseline_payload(state, sigma, sample_index, result)
            if not args.no_transport:
                merge_result_transport(
                    state,
                    sigma,
                    result,
                    int(args.transport_reservoir_max),
                    int(args.analysis_seed),
                )
        state["accepted_samples"].append(int(sample_index))
        state["candidate_position"] = int(state["candidate_position"]) + 1
        commit_resume_state(resume_path, state)
        controller.recorder.clear_sample()
        del batch, latents, noise, model_conditions, plan, events, sample_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rank_zero():
            print(
                f"[WanEntity] accept sample={sample_index} accepted={len(state['accepted_samples'])}/{state['target_count']} resume-committed",
                flush=True,
            )


def run_causal_collection(
    args,
    dataset,
    collate_fn,
    resolver: NuScenesEntityResolver,
    pipeline,
    input_builder: WanControlledInputBuilder,
    controller: WanEntityReactorController,
    state: dict,
    resume_path: Path,
) -> None:
    if args.no_causal:
        return
    causal_limit = int(args.causal_sample_count)
    causal_samples = [
        int(value)
        for value in (
            state["accepted_samples"]
            if causal_limit == 0
            else state["accepted_samples"][:causal_limit]
        )
    ]
    for sample_index in causal_samples:
        required = []
        for sigma in args.sigmas:
            for transition in controller.transitions:
                storage_label = f"{sigma_key(float(sigma))}::{intervention_label(transition)}"
                if int(sample_index) not in state.get("interventions", {}).get(storage_label, {}):
                    required.append((float(sigma), transition, storage_label))
        if not required:
            continue
        batch = collate_one(dataset, collate_fn, sample_index)
        frame_count = int(batch["vae_images"].shape[1])
        latent_time_count = int(pipeline.get_latent_sequence_length(frame_count))
        plan = resolver.build_observation_plan(
            sample_index=sample_index,
            target_time_count=latent_time_count,
            crossview_mask=extract_crossview_mask(batch),
        )
        events = build_probe_events(
            plan,
            maximum_events_per_family=int(args.maximum_events_per_family),
            maximum_negatives=int(args.maximum_negatives),
            minimum_temporal_gap=args.minimum_temporal_gap,
        )
        latents = input_builder.encode_real_video(batch)
        noise = input_builder.make_noise(latents.shape, sample_index)
        model_conditions = input_builder.prepare_conditions(batch, latents)
        controller.configure_sample(
            plan=plan,
            events=events,
            batch_size=int(latents.shape[0]),
            latent_height=int(latents.shape[-2]),
            latent_width=int(latents.shape[-1]),
        )
        for sigma, transition, storage_label in required:
            baseline_raw = baseline_payload(state, sigma, sample_index)
            if baseline_raw is None:
                raise RuntimeError(
                    f"causal sample {sample_index} has no baseline at sigma={sigma}"
                )
            baseline_result = forward_result_from_payload(baseline_raw)
            gate_spec = {(transition.module, int(transition.layer)): 0.0}
            ablated = input_builder.run_forward(
                controller,
                latents,
                noise,
                sigma,
                model_conditions,
                gate_spec=gate_spec,
            )
            summary = summarize_intervention(
                transition,
                baseline_result,
                ablated,
                sigma,
            )
            state.setdefault("interventions", {}).setdefault(storage_label, {})[
                int(sample_index)
            ] = summary
            commit_resume_state(resume_path, state)
            if rank_zero():
                print(
                    f"[UnifiedWan] causal sigma={sigma:.1f} sample={sample_index} "
                    f"{intervention_label(transition)} resume-committed",
                    flush=True,
                )
        controller.recorder.clear_sample()
        del batch, plan, events, latents, noise, model_conditions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_unified_cache_wan(
    state: dict,
    args,
    controller: WanEntityReactorController,
    checkpoint_path: Path,
) -> dict:
    output_stages = layer_output_stages(controller)
    aggregate = aggregate_baselines(
        state,
        args.sigmas,
        output_stages,
        int(args.analysis_seed),
    )
    baseline = {}
    metric_map = {
        "score": "structured_score",
        "auc": "structured_auc",
        "margin": "structured_margin",
        "top1": "structured_top1",
    }
    for family in PROBE_FAMILIES:
        baseline[family] = {}
        for public_metric, internal_metric in metric_map.items():
            stat_payload = {
                "mean": np.full((len(args.sigmas), len(output_stages)), np.nan, dtype=np.float32),
                "low": np.full((len(args.sigmas), len(output_stages)), np.nan, dtype=np.float32),
                "high": np.full((len(args.sigmas), len(output_stages)), np.nan, dtype=np.float32),
                "count": np.zeros((len(args.sigmas), len(output_stages)), dtype=np.int16),
            }
            for sigma_pos, sigma in enumerate(args.sigmas):
                sigma_summary = aggregate[sigma_key(float(sigma))]
                for stage_pos, stage in enumerate(output_stages):
                    summary = sigma_summary[stage][family][internal_metric]
                    for stat in ("mean", "low", "high"):
                        stat_payload[stat][sigma_pos, stage_pos] = float(summary[stat])
                    stat_payload["count"][sigma_pos, stage_pos] = int(summary["count"])
            baseline[family][public_metric] = stat_payload

    modules = list(MODULES)
    layers = sorted({int(t.layer) for t in controller.transitions})
    use = np.full((len(args.sigmas), len(layers), len(modules)), np.nan, dtype=np.float32)
    retain = np.full_like(use, np.nan)
    write = np.full_like(use, np.nan)
    count = np.zeros(use.shape, dtype=np.int16)
    layer_pos = {value: idx for idx, value in enumerate(layers)}
    module_pos = {value: idx for idx, value in enumerate(modules)}
    for sigma_pos, sigma in enumerate(args.sigmas):
        for transition_pos, transition in enumerate(controller.transitions):
            storage_label = f"{sigma_key(float(sigma))}::{intervention_label(transition)}"
            per_sample = state.get("interventions", {}).get(storage_label, {})
            if not per_sample:
                continue
            family = transition.target_relation
            use_values = [
                float(item.get("relations", {}).get(family, {}).get("use", float("nan")))
                for item in per_sample.values()
            ]
            retain_values = [
                float(item.get("relations", {}).get(family, {}).get("retain", float("nan")))
                for item in per_sample.values()
            ]
            write_values = [float(item.get("write", float("nan"))) for item in per_sample.values()]
            use_summary = bootstrap_mean(
                use_values,
                int(args.analysis_seed) + sigma_pos * 100003 + transition_pos * 1009,
            )
            retain_summary = bootstrap_mean(
                retain_values,
                int(args.analysis_seed) + sigma_pos * 100019 + transition_pos * 1013,
            )
            write_summary = bootstrap_mean(
                write_values,
                int(args.analysis_seed) + sigma_pos * 100043 + transition_pos * 1019,
            )
            index = (sigma_pos, layer_pos[int(transition.layer)], module_pos[str(transition.module)])
            use[index] = use_summary[0]
            retain[index] = retain_summary[0]
            write[index] = write_summary[0]
            count[index] = use_summary[3]

    causal_limit = int(args.causal_sample_count)
    causal_sample_indices = (
        list(state["accepted_samples"])
        if causal_limit == 0
        else list(state["accepted_samples"][:causal_limit])
    ) if not args.no_causal else []
    stage_depths = []
    for stage in output_stages:
        match = re.search(r"L(\d+)", stage)
        if match:
            stage_depths.append((int(match.group(1)) + 1) / max(len(controller.model.blocks), 1))
        elif stage == "final":
            stage_depths.append(1.0)
        else:
            stage_depths.append(0.0)
    return {
        "version": WAN_ENTITY_ANALYSIS_VERSION,
        "backend": "wan",
        "sigmas": [float(value) for value in args.sigmas],
        "stage_names": output_stages,
        "stage_depths": stage_depths,
        "sample_indices": [int(value) for value in state["accepted_samples"]],
        "causal_sample_indices": [int(value) for value in causal_sample_indices],
        "baseline": baseline,
        "causal": {
            "modules": modules,
            "layers": layers,
            "use": use,
            "retain": retain,
            "write": write,
            "count": count,
        },
        "metadata": {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "auc_weight": float(args.auc_weight),
            "margin_scale": float(args.margin_scale),
            "hidden_dimension": "full",
            "hidden_dtype_for_metrics": "fp32",
            "hard_negative_policy": "strict same-class only",
            "temp_definition": "Wan attn1 self-attention branch only",
            "cond_definition": "condition-image residual plus per-layer camera residual before Wan block",
            "crossview_definition": "WanCrossviewBlock attention plus mixer",
            "score_formula": "w*(2*AUC-1)+(1-w)*tanh(margin/scale)",
        },
    }

def run_checkpoint(
    args,
    base_config: dict,
    dataset,
    collate_fn,
    checkpoint_path: Path,
    device: torch.device,
) -> Path:
    run_root = args.output_path / checkpoint_output_name(checkpoint_path)
    if rank_zero():
        run_root.mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    pipeline, runtime_config = instantiate_pipeline(
        base_config,
        checkpoint_path,
        run_root,
        device,
    )
    model_class_name = type(pipeline.model).__name__
    if model_class_name != "WanCrossviewConditionModel":
        raise TypeError(
            f"this analyzer expects WanCrossviewConditionModel, got {model_class_name}"
        )
    key_layers = choose_key_layers(pipeline.model, args.key_layers)
    causal_layers = choose_causal_layers(pipeline.model, args.causal_layers, key_layers)
    controller = WanEntityReactorController(
        pipeline,
        key_layers=key_layers,
        causal_layers=causal_layers,
        transport_events_per_sample=int(args.transport_events_per_sample),
        auc_weight=float(args.auc_weight),
        margin_scale=float(args.margin_scale),
    )
    input_builder = WanControlledInputBuilder(pipeline, int(args.analysis_seed))
    common_training = runtime_config["pipeline"].get("training_config", {})
    resolver = NuScenesEntityResolver(
        dataset=dataset,
        max_entities=int(args.max_entities),
        allowed_classes=args.entity_classes,
        minimum_track_length=int(args.minimum_track_length),
        minimum_visible_anchors=int(args.minimum_visible_anchors),
        minimum_area_ratio=float(args.minimum_area_ratio),
        ring_expand_ratio=float(args.ring_expand_ratio),
        temporal_downsample_factor=int(
            common_training.get("condition_temporal_downsample_factor", 4)
        ),
        temporal_group_index=int(
            common_training.get("condition_temporal_group_index", 3)
        ),
    )
    resume_root = run_root / "_resume"
    resume_path = resume_root / "state.pkl"
    signature = make_resume_signature(args, checkpoint_path, key_layers, causal_layers)
    state = load_resume_state(resume_path, signature)
    if state is None:
        candidates, next_backfill, target_count = initial_candidate_state(
            args,
            len(dataset),
        )
        state = new_resume_state(
            signature,
            candidates,
            next_backfill,
            target_count,
        )
        commit_resume_state(resume_path, state)
    if rank_zero():
        print(
            f"[WanEntity] checkpoint={checkpoint_path} model={model_class_name}",
            flush=True,
        )
        print(
            f"[WanEntity] modules=temp(attn1), cond(pre-block residual), crossview(WanCrossviewBlock)",
            flush=True,
        )
        print(
            f"[WanEntity] key_layers={key_layers} causal_layers={causal_layers} resume_accepted={len(state['accepted_samples'])}",
            flush=True,
        )
    try:
        run_baseline_collection(
            args,
            dataset,
            collate_fn,
            resolver,
            pipeline,
            input_builder,
            controller,
            state,
            resume_path,
        )
        run_causal_collection(
            args,
            dataset,
            collate_fn,
            resolver,
            pipeline,
            input_builder,
            controller,
            state,
            resume_path,
        )
        cache = build_unified_cache_wan(
            state,
            args,
            controller,
            checkpoint_path,
        )
        if rank_zero():
            cache_path = save_unified_npz(
                cache,
                run_root / "entity_relation_unified_data.npz",
            )
            text_path = write_scores_text(
                cache,
                run_root / "entity_relation_scores.txt",
            )
            overview_path = run_root / "entity_relation_overview_clean.png"
            causal_path = run_root / "entity_relation_causal_use.png"
            if not args.no_render:
                render_overview(cache, overview_path)
                if not args.no_causal:
                    render_causal(cache, causal_path)
            manifest = {
                "version": WAN_ENTITY_ANALYSIS_VERSION,
                "backend": "wan",
                "cache": cache_path.name,
                "scores_text": text_path.name,
                "overview_figure": overview_path.name if not args.no_render else None,
                "causal_figure": causal_path.name if (not args.no_render and not args.no_causal) else None,
                "sigmas": [float(value) for value in args.sigmas],
                "sample_indices": cache["sample_indices"],
                "causal_sample_indices": cache["causal_sample_indices"],
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "auc_weight": float(args.auc_weight),
                "margin_scale": float(args.margin_scale),
            }
            atomic_json(run_root / "entity_relation_manifest.json", manifest)
            print(f"[UnifiedWan] overview={overview_path}", flush=True)
            if not args.no_causal:
                print(f"[UnifiedWan] causal={causal_path}", flush=True)
            print(f"[UnifiedWan] scores={text_path}", flush=True)
            print(f"[UnifiedWan] cache={cache_path}", flush=True)
            print(f"[UnifiedWan] accepted={cache['sample_indices']}", flush=True)
            state["status"] = "complete"
            if resume_root.is_dir() and not args.keep_resume:
                shutil.rmtree(resume_root)
        distributed_barrier()
        return run_root
    finally:
        controller.detach()
        controller.recorder.clear_sample()
        if getattr(pipeline, "summary", None) is not None:
            pipeline.summary.close()
        del controller, input_builder, resolver, pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None) -> None:
    args = create_parser().parse_args(argv)
    args.sigmas = normalize_sigmas(args.sigmas)
    if not 0.0 <= float(args.auc_weight) <= 1.0:
        raise ValueError("--auc-weight must be in [0,1]")
    if float(args.margin_scale) <= 0.0:
        raise ValueError("--margin-scale must be positive")
    base_config = json.loads(args.config_path.read_text(encoding="utf-8"))
    device = setup_distributed(base_config)
    initialize_global_state(base_config)
    args.output_path.mkdir(parents=True, exist_ok=True)
    dataset = instantiate_validation_dataset(base_config)
    collate_fn = instantiate_validation_collate(base_config)
    checkpoint_paths = resolve_checkpoint_paths(
        base_config,
        args.checkpoint,
        args.all_checkpoints,
    )
    try:
        for checkpoint_path in checkpoint_paths:
            run_checkpoint(
                args,
                base_config,
                dataset,
                collate_fn,
                checkpoint_path,
                device,
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
