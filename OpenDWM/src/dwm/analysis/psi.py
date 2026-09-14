from __future__ import annotations

ENTITY_REACTOR_PSI_VERSION = "v19.7.2-fullD-strict-sameclass-probe-20260819"
ENTITY_REACTOR_PLOT_CACHE_VERSION = "v5-fullD-strict-sameclass-final-only-20260819"

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


PROBE_FAMILIES = ("cross_view", "temporal", "mixed")
MODULE_ORDER = ("camera", "condition", "temporal", "crossview")
TARGET_RELATION = {
    "camera": "cross_view",
    "condition": "cross_view",
    "temporal": "temporal",
    "crossview": "cross_view",
}


@dataclass(frozen=True)
class ProbeEvent:
    family: str
    query: int
    positive: int
    negatives: np.ndarray
    query_track: int
    camera_pair: int
    time_gap: int


@dataclass(frozen=True)
class StageDescriptor:
    name: str
    layer: int
    module: str
    depth: float
    short_label: str


@dataclass
class CaptureProbeResult:
    capture_path: Path
    sample_index: int
    stage_names: list[str]
    metrics: dict[str, np.ndarray]
    event_counts: dict[str, int]


@dataclass(frozen=True)
class ModuleTransition:
    layer: int
    module: str
    before_index: int
    after_index: int


def load_capture(path: Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        features = np.asarray(data["features"])
        geometry = data["geometry"].astype(np.float32)
        anchor_visible = (
            data["anchor_visible"].astype(np.bool_)
            if "anchor_visible" in data
            else np.ones(geometry.shape[:2], dtype=np.bool_)
        )
        bbox_area_ratio = (
            data["bbox_area_ratio"].astype(np.float32)
            if "bbox_area_ratio" in data
            else np.ones(geometry.shape[0], dtype=np.float32)
        )
        crossview_mask = data["crossview_mask"].astype(np.bool_)
        if crossview_mask.ndim == 3:
            crossview_mask = crossview_mask[0]
        capture = {
            "path": path,
            "stage_names": data["stage_names"].astype(str).tolist(),
            "features": features,
            "geometry": geometry,
            "anchor_visible": anchor_visible,
            "bbox_area_ratio": bbox_area_ratio,
            "observation_index": data["observation_index"].astype(np.int64),
            "track_hash": data["track_hash"].astype(np.int64),
            "class_id": data["class_id"].astype(np.int64),
            "crossview_mask": crossview_mask,
            "metadata": metadata,
        }
    if features.ndim != 4 or geometry.ndim != 3:
        raise ValueError(f"unsupported capture tensor layout in {path}")
    if features.shape[1] != geometry.shape[0] or features.shape[2] != geometry.shape[1]:
        raise ValueError(f"feature and geometry observations do not match in {path}")
    return capture


def discover_baseline_capture_files(input_root: Path) -> list[Path]:
    input_root = Path(input_root)
    resolved_root = input_root.resolve() if input_root.exists() else input_root
    if input_root.is_file() and input_root.suffix == ".npz":
        files = [input_root]
    else:
        files = sorted(input_root.rglob("capture.npz"))
        files = [path for path in files if "intervention" not in path.parts]
    files = [Path(path) for path in files if Path(path).is_file()]
    if not files:
        raise RuntimeError(f"no baseline capture.npz found under {resolved_root}")
    return files


def discover_intervention_capture_files(input_root: Path) -> list[Path]:
    input_root = Path(input_root)
    files = sorted(input_root.rglob("*.npz"))
    intervention_files = [path for path in files if "intervention" in path.parts]
    valid_files = []
    for path in intervention_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "metadata" not in data.files:
                    continue
                metadata = json.loads(str(data["metadata"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
            continue
        if str(metadata.get("intervention_label", "")).strip():
            valid_files.append(path)
    return valid_files


def group_capture_paths_by_sigma(paths: Iterable[Path]) -> dict[float, list[Path]]:
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("capture path list is empty")
    indexed_groups: dict[float, list[tuple[int, Path]]] = {}
    for path in path_list:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
        sigma = float(metadata.get("sigma", -1.0))
        sample_index = int(metadata.get("sample_index", 0))
        indexed_groups.setdefault(sigma, []).append((sample_index, Path(path)))
    groups = {
        sigma: [path for _, path in sorted(indexed_paths, key=lambda item: item[0])]
        for sigma, indexed_paths in indexed_groups.items()
    }
    ordered_groups = dict(sorted(groups.items(), key=lambda item: item[0]))
    return ordered_groups


def describe_stages(stage_names: Sequence[str], num_layers: int) -> list[StageDescriptor]:
    descriptors: list[StageDescriptor] = []
    safe_layer_count = max(int(num_layers), 1)
    module_labels = {
        "in": ("input", "I"),
        "cond": ("condition", "C"),
        "bev": ("condition", "C"),
        "base": ("base", "B"),
        "temp": ("temporal", "T"),
        "view": ("crossview", "V"),
    }
    for stage_name in stage_names:
        if stage_name == "stem":
            descriptors.append(StageDescriptor(stage_name, -1, "stem", 0.0, "stem"))
            continue
        if stage_name == "final":
            descriptors.append(StageDescriptor(stage_name, safe_layer_count, "final", 1.0, "final"))
            continue
        layer_text, suffix = stage_name.split(".", 1)
        layer_index = int(layer_text[1:])
        module, short_code = module_labels.get(suffix, (suffix, suffix[:1].upper()))
        depth = float(layer_index + 1) / float(safe_layer_count)
        descriptors.append(
            StageDescriptor(
                name=stage_name,
                layer=layer_index,
                module=module,
                depth=depth,
                short_label=f"{layer_index:02d}{short_code}",
            )
        )
    return descriptors


def layer_output_indices(descriptors: Sequence[StageDescriptor]) -> list[int]:
    layer_to_index: dict[int, int] = {}
    final_index = None
    for stage_index, descriptor in enumerate(descriptors):
        if descriptor.module == "final":
            final_index = stage_index
            continue
        if descriptor.layer < 0 or descriptor.module == "input":
            continue
        layer_to_index[int(descriptor.layer)] = stage_index
    output_indices = [layer_to_index[layer] for layer in sorted(layer_to_index)]
    if final_index is not None:
        output_indices.append(int(final_index))
    if not output_indices:
        raise RuntimeError("no structural layer outputs were found in capture stages")
    return output_indices


def select_track_balanced_events(
    events: Sequence[ProbeEvent],
    maximum_events_per_track: int,
    maximum_total_events: int,
) -> list[ProbeEvent]:
    if maximum_events_per_track <= 0 or maximum_total_events <= 0:
        raise ValueError("event limits must be positive")
    events_by_track: dict[int, list[ProbeEvent]] = {}
    for event in events:
        events_by_track.setdefault(int(event.query_track), []).append(event)

    capped_by_track: dict[int, list[ProbeEvent]] = {}
    for track_id in sorted(events_by_track):
        current_events = events_by_track[track_id]
        current_events = sorted(
            current_events,
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
            if len(selected) >= maximum_events_per_track:
                break
        if len(selected) < maximum_events_per_track:
            selected_ids = {(int(event.query), int(event.positive)) for event in selected}
            for event in current_events:
                event_id = (int(event.query), int(event.positive))
                if event_id in selected_ids:
                    continue
                selected.append(event)
                selected_ids.add(event_id)
                if len(selected) >= maximum_events_per_track:
                    break
        if selected:
            capped_by_track[int(track_id)] = selected

    output = []
    offsets = {track_id: 0 for track_id in capped_by_track}
    track_ids = sorted(capped_by_track)
    while len(output) < maximum_total_events:
        added = False
        for track_id in track_ids:
            current_events = capped_by_track[track_id]
            current_offset = offsets[track_id]
            if current_offset >= len(current_events):
                continue
            output.append(current_events[current_offset])
            offsets[track_id] = current_offset + 1
            added = True
            if len(output) >= maximum_total_events:
                break
        if not added:
            break
    return output


def select_hard_negatives(
    capture: dict,
    positive_row: int,
    target_time: int,
    target_view: int,
    maximum_negatives: int,
) -> np.ndarray:
    observation_index = capture["observation_index"]
    track_hash = capture["track_hash"]
    class_id = capture["class_id"]
    geometry = capture["geometry"][:, -1]
    area = capture["bbox_area_ratio"]
    positive_track = track_hash[positive_row]
    positive_class = class_id[positive_row]

    different_track = track_hash != positive_track
    same_view = observation_index[:, 2] == int(target_view)
    exact_time = observation_index[:, 1] == int(target_time)
    nearby_time = np.abs(observation_index[:, 1] - int(target_time)) <= 1
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
    capture: dict,
    maximum_events_per_family: int = 256,
    maximum_negatives: int = 6,
    minimum_temporal_gap: Optional[int] = None,
) -> dict[str, list[ProbeEvent]]:
    observation_index = capture["observation_index"]
    track_hash = capture["track_hash"]
    crossview_mask = capture["crossview_mask"]
    metadata = capture["metadata"]
    sequence_length = int(metadata.get("sequence_length", observation_index[:, 1].max() + 1))
    temporal_gap = int(minimum_temporal_gap or max(2, round(max(sequence_length - 1, 1) * 0.20)))

    rows_by_track: dict[int, list[int]] = {}
    for row, index_values in enumerate(observation_index.tolist()):
        current_track = int(track_hash[row])
        rows_by_track.setdefault(current_track, []).append(row)

    sampling_summary = metadata.get("cross_camera_sampling", {})
    transition_track_hashes = {
        int(value)
        for value in sampling_summary.get("selected_transition_track_hashes", [])
    }
    if not transition_track_hashes:
        raise RuntimeError(
            "capture has no temporal camera-handoff track list; rerun capture with v19 code"
        )

    events: dict[str, list[ProbeEvent]] = {family: [] for family in PROBE_FAMILIES}
    for current_track, track_rows in rows_by_track.items():
        if int(current_track) not in transition_track_hashes:
            continue
        track_rows = sorted(
            track_rows,
            key=lambda row: (
                int(observation_index[row, 1]),
                int(observation_index[row, 2]),
            ),
        )
        rows_by_view: dict[int, list[int]] = {}
        for row in track_rows:
            view_index = int(observation_index[row, 2])
            rows_by_view.setdefault(view_index, []).append(row)

        available_views = sorted(rows_by_view)
        for source_position, first_view in enumerate(available_views):
            first_rows = rows_by_view[first_view]
            first_times = np.asarray([int(observation_index[row, 1]) for row in first_rows], dtype=np.int64)
            for second_view in available_views[source_position + 1:]:
                if not bool(crossview_mask[first_view, second_view] or crossview_mask[second_view, first_view]):
                    continue
                second_rows = rows_by_view[second_view]
                second_times = np.asarray([int(observation_index[row, 1]) for row in second_rows], dtype=np.int64)
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
                query_row = max(source_candidates, key=lambda row: int(observation_index[row, 1]))
                positive_row = min(target_candidates, key=lambda row: int(observation_index[row, 1]))
                query_time = int(observation_index[query_row, 1])
                target_time = int(observation_index[positive_row, 1])
                negatives = select_hard_negatives(
                    capture,
                    positive_row,
                    target_time,
                    target_view,
                    maximum_negatives,
                )
                if negatives.size > 0:
                    events["cross_view"].append(
                        ProbeEvent(
                            "cross_view",
                            query_row,
                            positive_row,
                            negatives,
                            current_track,
                            source_view * 100 + target_view,
                            target_time - query_time,
                        )
                    )

        for query_row in track_rows:
            query_time = int(observation_index[query_row, 1])
            query_view = int(observation_index[query_row, 2])
            temporal_targets = [
                row
                for row in track_rows
                if int(observation_index[row, 2]) == query_view
                and abs(int(observation_index[row, 1]) - query_time) >= temporal_gap
            ]
            temporal_targets.sort(
                key=lambda row: abs(int(observation_index[row, 1]) - query_time),
                reverse=True,
            )
            for positive_row in temporal_targets[:2]:
                target_time = int(observation_index[positive_row, 1])
                negatives = select_hard_negatives(
                    capture,
                    positive_row,
                    target_time,
                    query_view,
                    maximum_negatives,
                )
                if negatives.size > 0:
                    events["temporal"].append(
                        ProbeEvent(
                            "temporal",
                            query_row,
                            positive_row,
                            negatives,
                            current_track,
                            query_view * 100 + query_view,
                            target_time - query_time,
                        )
                    )

            mixed_targets = [
                row
                for row in track_rows
                if int(observation_index[row, 2]) != query_view
                and abs(int(observation_index[row, 1]) - query_time) >= temporal_gap
            ]
            mixed_targets.sort(
                key=lambda row: abs(int(observation_index[row, 1]) - query_time),
                reverse=True,
            )
            accepted_mixed = 0
            for positive_row in mixed_targets:
                target_time = int(observation_index[positive_row, 1])
                target_view = int(observation_index[positive_row, 2])
                if not bool(crossview_mask[query_view, target_view] or crossview_mask[target_view, query_view]):
                    continue
                negatives = select_hard_negatives(
                    capture,
                    positive_row,
                    target_time,
                    target_view,
                    maximum_negatives,
                )
                if negatives.size == 0:
                    continue
                events["mixed"].append(
                    ProbeEvent(
                        "mixed",
                        query_row,
                        positive_row,
                        negatives,
                        current_track,
                        query_view * 100 + target_view,
                        target_time - query_time,
                    )
                )
                accepted_mixed += 1
                if accepted_mixed >= 2:
                    break

    per_track_limits = {
        "cross_view": 4,
        "temporal": 2,
        "mixed": 2,
    }
    for family in PROBE_FAMILIES:
        events[family] = select_track_balanced_events(
            events[family],
            maximum_events_per_track=per_track_limits[family],
            maximum_total_events=int(maximum_events_per_family),
        )
    return events


def prepare_feature_encodings(
    stage_features: np.ndarray,
    anchor_visible: np.ndarray,
) -> dict[str, np.ndarray]:
    features = stage_features.astype(np.float32, copy=False)
    visibility = anchor_visible.astype(np.bool_, copy=False)
    visible_weights = visibility[..., None].astype(np.float32)
    visible_count = np.maximum(visible_weights.sum(axis=1), 1.0)
    pooled = (features * visible_weights).sum(axis=1) / visible_count
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-10)

    center = features[:, -1]
    center_normalized = center / np.maximum(np.linalg.norm(center, axis=1, keepdims=True), 1e-10)
    relative = features[:, :-1] - center[:, None]
    relative_normalized = relative / np.maximum(np.linalg.norm(relative, axis=-1, keepdims=True), 1e-10)
    corner_visible = visibility[:, :-1] & visibility[:, -1:]
    masked_relative = relative_normalized * corner_visible[..., None].astype(np.float32)
    structured_vector = np.concatenate(
        [center_normalized, masked_relative.reshape(masked_relative.shape[0], -1)],
        axis=1,
    )
    structured_vector = structured_vector / np.maximum(
        np.linalg.norm(structured_vector, axis=1, keepdims=True),
        1e-10,
    )
    return {
        "pooled": pooled,
        "center": center_normalized,
        "relative": relative_normalized,
        "corner_visible": corner_visible,
        "structured_vector": structured_vector,
    }


def prepare_geometry_vectors(
    geometry: np.ndarray,
    anchor_visible: np.ndarray,
) -> np.ndarray:
    geometry = geometry.astype(np.float32, copy=False)
    visibility = anchor_visible.astype(np.bool_, copy=False)
    center = geometry[:, -1]
    relative = geometry[:, :-1] - center[:, None]
    corner_visible = visibility[:, :-1] & visibility[:, -1:]
    masked_relative = relative * corner_visible[..., None].astype(np.float32)
    geometry_vector = np.concatenate(
        [center, masked_relative.reshape(masked_relative.shape[0], -1)],
        axis=1,
    )
    if geometry_vector.ndim != 2 or geometry_vector.shape[0] != geometry.shape[0]:
        raise RuntimeError("geometry vector construction failed")
    return geometry_vector


def prepare_event_arrays(
    events: Sequence[ProbeEvent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_count = len(events)
    maximum_negative_count = max(int(event.negatives.size) for event in events)
    query_rows = np.empty(event_count, dtype=np.int64)
    positive_rows = np.empty(event_count, dtype=np.int64)
    negative_rows = np.zeros((event_count, maximum_negative_count), dtype=np.int64)
    negative_mask = np.zeros((event_count, maximum_negative_count), dtype=np.bool_)
    for event_index, event in enumerate(events):
        current_count = int(event.negatives.size)
        query_rows[event_index] = int(event.query)
        positive_rows[event_index] = int(event.positive)
        negative_rows[event_index, :current_count] = event.negatives
        negative_mask[event_index, :current_count] = True
    return query_rows, positive_rows, negative_rows, negative_mask


def structured_pair_similarity(
    encodings: dict[str, np.ndarray],
    query_rows: np.ndarray,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    center = encodings["center"]
    relative = encodings["relative"]
    corner_visible = encodings["corner_visible"]
    query_center = center[query_rows]
    candidate_center = center[candidate_rows]
    center_score = np.einsum("emd,ed->em", candidate_center, query_center, optimize=True)

    query_relative = relative[query_rows]
    candidate_relative = relative[candidate_rows]
    per_corner = np.einsum("emad,ead->ema", candidate_relative, query_relative, optimize=True)
    common_visible = corner_visible[candidate_rows] & corner_visible[query_rows][:, None]
    common_count = common_visible.sum(axis=2)
    relation_score = (
        (per_corner * common_visible.astype(np.float32)).sum(axis=2)
        / np.maximum(common_count, 1)
    )
    relation_score = np.where(common_count >= 2, relation_score, center_score)
    combined = 0.35 * center_score + 0.65 * relation_score
    return np.clip(combined, -1.0, 1.0)


def evaluate_event_family(
    events: Sequence[ProbeEvent],
    encodings: dict[str, np.ndarray],
) -> dict[str, float]:
    empty_result = {
        "pooled_score": float("nan"),
        "structured_score": float("nan"),
        "pooled_auc": float("nan"),
        "structured_auc": float("nan"),
        "pooled_margin": float("nan"),
        "structured_margin": float("nan"),
        "pooled_top1": float("nan"),
        "structured_top1": float("nan"),
    }
    if not events:
        return empty_result

    query_rows, positive_rows, negative_rows, negative_mask = prepare_event_arrays(events)
    pooled_vectors = encodings["pooled"]
    pooled_query = pooled_vectors[query_rows]
    pooled_positive = np.einsum("ed,ed->e", pooled_vectors[positive_rows], pooled_query, optimize=True)
    pooled_negative = np.einsum("emd,ed->em", pooled_vectors[negative_rows], pooled_query, optimize=True)
    structured_positive = structured_pair_similarity(encodings, query_rows, positive_rows[:, None])[:, 0]
    structured_negative = structured_pair_similarity(encodings, query_rows, negative_rows)

    valid_count = np.maximum(negative_mask.sum(axis=1), 1)
    pooled_auc_values = (
        ((pooled_positive[:, None] > pooled_negative) & negative_mask).sum(axis=1)
        + 0.5 * ((pooled_positive[:, None] == pooled_negative) & negative_mask).sum(axis=1)
    ) / valid_count
    structured_auc_values = (
        ((structured_positive[:, None] > structured_negative) & negative_mask).sum(axis=1)
        + 0.5 * ((structured_positive[:, None] == structured_negative) & negative_mask).sum(axis=1)
    ) / valid_count

    pooled_negative_max = np.where(negative_mask, pooled_negative, -np.inf).max(axis=1)
    structured_negative_max = np.where(negative_mask, structured_negative, -np.inf).max(axis=1)
    pooled_margins = pooled_positive - pooled_negative_max
    structured_margins = structured_positive - structured_negative_max
    pooled_auc = float(np.mean(pooled_auc_values))
    structured_auc = float(np.mean(structured_auc_values))
    pooled_margin = float(np.mean(pooled_margins))
    structured_margin = float(np.mean(structured_margins))
    pooled_score = 0.65 * (2.0 * pooled_auc - 1.0) + 0.35 * np.tanh(pooled_margin / 0.12)
    structured_score = 0.65 * (2.0 * structured_auc - 1.0) + 0.35 * np.tanh(structured_margin / 0.12)
    return {
        "pooled_score": float(np.clip(pooled_score, -1.0, 1.0)),
        "structured_score": float(np.clip(structured_score, -1.0, 1.0)),
        "pooled_auc": pooled_auc,
        "structured_auc": structured_auc,
        "pooled_margin": pooled_margin,
        "structured_margin": structured_margin,
        "pooled_top1": float(np.mean(pooled_margins >= 0.0)),
        "structured_top1": float(np.mean(structured_margins >= 0.0)),
    }


def make_transport_payload(
    capture: dict,
    events: Sequence[ProbeEvent],
    encodings: dict[str, np.ndarray],
    geometry_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    if not events:
        feature_dim = int(encodings["structured_vector"].shape[1])
        geometry_dim = int(geometry_vectors.shape[1])
        return {
            "feature_delta": np.empty((0, feature_dim), dtype=np.float32),
            "geometry_delta": np.empty((0, geometry_dim), dtype=np.float32),
            "sample_id": np.empty(0, dtype=np.int64),
            "track_id": np.empty(0, dtype=np.int64),
            "camera_pair": np.empty(0, dtype=np.int64),
        }
    query_rows = np.asarray([event.query for event in events], dtype=np.int64)
    positive_rows = np.asarray([event.positive for event in events], dtype=np.int64)
    feature_delta = encodings["structured_vector"][positive_rows] - encodings["structured_vector"][query_rows]
    geometry_delta = geometry_vectors[positive_rows] - geometry_vectors[query_rows]
    feature_delta = feature_delta / np.maximum(np.linalg.norm(feature_delta, axis=1, keepdims=True), 1e-10)
    sample_index = int(capture["metadata"].get("sample_index", 0))
    return {
        "feature_delta": feature_delta.astype(np.float32, copy=False),
        "geometry_delta": geometry_delta.astype(np.float32, copy=False),
        "sample_id": np.full(len(events), sample_index, dtype=np.int64),
        "track_id": np.asarray([event.query_track for event in events], dtype=np.int64),
        "camera_pair": np.asarray([event.camera_pair for event in events], dtype=np.int64),
    }


def evaluate_capture(
    capture: dict,
    transport_stage_indices: Optional[set[int]] = None,
) -> tuple[CaptureProbeResult, dict[str, list[Optional[dict[str, np.ndarray]]]]]:
    events = build_probe_events(capture)
    stage_names = list(capture["stage_names"])
    selected_transport_stages = set(range(len(stage_names))) if transport_stage_indices is None else set(transport_stage_indices)
    metric_names = []
    for family in PROBE_FAMILIES:
        metric_names.extend(
            [
                f"{family}_pooled",
                f"{family}_structured",
                f"{family}_gap",
                f"{family}_pooled_auc",
                f"{family}_structured_auc",
                f"{family}_pooled_top1",
                f"{family}_structured_top1",
            ]
        )
    metrics = {name: np.full(len(stage_names), np.nan, dtype=np.float64) for name in metric_names}
    geometry_vectors = prepare_geometry_vectors(capture["geometry"], capture["anchor_visible"])
    transport_payloads = {family: [] for family in PROBE_FAMILIES}

    for stage_index in range(len(stage_names)):
        encodings = prepare_feature_encodings(
            capture["features"][stage_index],
            capture["anchor_visible"],
        )
        for family in PROBE_FAMILIES:
            family_metrics = evaluate_event_family(events[family], encodings)
            pooled_value = family_metrics["pooled_score"]
            structured_value = family_metrics["structured_score"]
            metrics[f"{family}_pooled"][stage_index] = pooled_value
            metrics[f"{family}_structured"][stage_index] = structured_value
            metrics[f"{family}_gap"][stage_index] = structured_value - pooled_value
            metrics[f"{family}_pooled_auc"][stage_index] = family_metrics["pooled_auc"]
            metrics[f"{family}_structured_auc"][stage_index] = family_metrics["structured_auc"]
            metrics[f"{family}_pooled_top1"][stage_index] = family_metrics["pooled_top1"]
            metrics[f"{family}_structured_top1"][stage_index] = family_metrics["structured_top1"]
            if stage_index in selected_transport_stages:
                transport_payloads[family].append(
                    make_transport_payload(
                        capture,
                        events[family],
                        encodings,
                        geometry_vectors,
                    )
                )
            else:
                transport_payloads[family].append(None)

    result = CaptureProbeResult(
        capture_path=capture["path"],
        sample_index=int(capture["metadata"].get("sample_index", 0)),
        stage_names=stage_names,
        metrics=metrics,
        event_counts={family: len(events[family]) for family in PROBE_FAMILIES},
    )
    return result, transport_payloads


def bootstrap_interval(
    values: np.ndarray,
    seed: int,
    iterations: int = 400,
) -> tuple[float, float, float, int]:
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    count = int(finite.size)
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean_value = float(finite.mean())
    if count == 1:
        return mean_value, mean_value, mean_value, count
    generator = np.random.default_rng(int(seed))
    sampled_indices = generator.integers(0, count, size=(int(iterations), count))
    bootstrap_means = finite[sampled_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean_value, float(low), float(high), count


def aggregate_metric_cube(values: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    if values.ndim != 3:
        raise ValueError("metric cube must be [sigma,capture,stage]")
    sigma_count, _, stage_count = values.shape
    mean = np.full((sigma_count, stage_count), np.nan, dtype=np.float64)
    low = np.full_like(mean, np.nan)
    high = np.full_like(mean, np.nan)
    count = np.zeros_like(mean, dtype=np.int32)
    for sigma_index in range(sigma_count):
        for stage_index in range(stage_count):
            summary = bootstrap_interval(
                values[sigma_index, :, stage_index],
                seed=seed + sigma_index * 1009 + stage_index * 37,
            )
            mean[sigma_index, stage_index] = summary[0]
            low[sigma_index, stage_index] = summary[1]
            high[sigma_index, stage_index] = summary[2]
            count[sigma_index, stage_index] = summary[3]
    return {"mean": mean, "low": low, "high": high, "count": count}


def concatenate_transport_payloads(payloads: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    valid_payloads = [payload for payload in payloads if payload["feature_delta"].shape[0] > 0]
    keys = ("feature_delta", "geometry_delta", "sample_id", "track_id", "camera_pair")
    if not valid_payloads:
        empty_result = {
            "feature_delta": np.empty((0, 1), dtype=np.float32),
            "geometry_delta": np.empty((0, 1), dtype=np.float32),
            "sample_id": np.empty(0, dtype=np.int64),
            "track_id": np.empty(0, dtype=np.int64),
            "camera_pair": np.empty(0, dtype=np.int64),
        }
        return empty_result
    combined = {key: np.concatenate([payload[key] for payload in valid_payloads], axis=0) for key in keys}
    event_count = combined["feature_delta"].shape[0]
    if any(combined[key].shape[0] != event_count for key in keys):
        raise RuntimeError("transport payload arrays have inconsistent event counts")
    return combined



def merge_transport_reservoir(
    existing: Optional[dict[str, np.ndarray]],
    incoming: dict[str, np.ndarray],
    maximum_events: int,
    seed: int,
) -> Optional[dict[str, np.ndarray]]:
    incoming_count = int(incoming["feature_delta"].shape[0])
    if incoming_count == 0:
        return existing
    sample_id = int(incoming["sample_id"][0]) if incoming["sample_id"].size else 0
    generator = np.random.default_rng(int(seed) + sample_id * 1000003)
    incoming_priority = generator.random(incoming_count).astype(np.float64)
    incoming_payload = {
        "feature_delta": incoming["feature_delta"].astype(np.float32, copy=False),
        "geometry_delta": incoming["geometry_delta"].astype(np.float32, copy=False),
        "sample_id": incoming["sample_id"].astype(np.int64, copy=False),
        "track_id": incoming["track_id"].astype(np.int64, copy=False),
        "camera_pair": incoming["camera_pair"].astype(np.int64, copy=False),
        "priority": incoming_priority,
    }
    if existing is None:
        combined = incoming_payload
    else:
        combined = {
            key: np.concatenate([existing[key], incoming_payload[key]], axis=0)
            for key in incoming_payload
        }
    event_count = int(combined["priority"].shape[0])
    if event_count <= int(maximum_events):
        return combined
    selected = np.argpartition(combined["priority"], int(maximum_events) - 1)[: int(maximum_events)]
    selected = selected[np.argsort(combined["priority"][selected], kind="stable")]
    return {key: value[selected] for key, value in combined.items()}

def transport_knn_alignment(
    payload: dict[str, np.ndarray],
    seed: int,
    maximum_events: int = 768,
    neighbor_count: int = 5,
) -> dict[str, float]:
    feature_delta = payload["feature_delta"].astype(np.float64, copy=False)
    geometry_delta = payload["geometry_delta"].astype(np.float64, copy=False)
    sample_id = payload["sample_id"].astype(np.int64, copy=False)
    track_id = payload["track_id"].astype(np.int64, copy=False)
    event_count = int(feature_delta.shape[0])
    if event_count < max(12, neighbor_count + 2):
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "count": 0}

    generator = np.random.default_rng(int(seed))
    if event_count > maximum_events:
        selected = np.sort(generator.choice(event_count, size=int(maximum_events), replace=False))
        feature_delta = feature_delta[selected]
        geometry_delta = geometry_delta[selected]
        sample_id = sample_id[selected]
        track_id = track_id[selected]
        event_count = int(selected.size)

    feature_delta = feature_delta / np.maximum(np.linalg.norm(feature_delta, axis=1, keepdims=True), 1e-10)
    geometry_centered = geometry_delta - np.nanmean(geometry_delta, axis=0, keepdims=True)
    geometry_scale = np.nanstd(geometry_centered, axis=0, keepdims=True)
    geometry_centered = geometry_centered / np.where(geometry_scale < 1e-5, 1.0, geometry_scale)
    feature_similarity = feature_delta @ feature_delta.T
    geometry_norm = np.sum(geometry_centered * geometry_centered, axis=1, keepdims=True)
    geometry_distance = geometry_norm + geometry_norm.T - 2.0 * geometry_centered @ geometry_centered.T
    geometry_distance = np.maximum(geometry_distance, 0.0)

    event_scores = np.full(event_count, np.nan, dtype=np.float64)
    for event_index in range(event_count):
        candidate_mask = (sample_id != sample_id[event_index]) & (track_id != track_id[event_index])
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size < max(3, neighbor_count):
            continue
        current_k = min(int(neighbor_count), int(candidate_indices.size))
        geometry_values = geometry_distance[event_index, candidate_indices]
        feature_values = feature_similarity[event_index, candidate_indices]
        geometry_partition = np.argpartition(geometry_values, current_k - 1)[:current_k]
        feature_partition = np.argpartition(-feature_values, current_k - 1)[:current_k]
        geometry_neighbors = candidate_indices[geometry_partition]
        feature_neighbors = candidate_indices[feature_partition]
        overlap = np.intersect1d(geometry_neighbors, feature_neighbors, assume_unique=False).size / float(current_k)
        chance = current_k / float(candidate_indices.size)
        normalized_overlap = (overlap - chance) / max(1.0 - chance, 1e-8)
        event_scores[event_index] = float(np.clip(normalized_overlap, -1.0, 1.0))

    sample_scores = []
    for current_sample in np.unique(sample_id):
        current_values = event_scores[sample_id == current_sample]
        current_values = current_values[np.isfinite(current_values)]
        if current_values.size > 0:
            sample_scores.append(float(current_values.mean()))
    summary = bootstrap_interval(np.asarray(sample_scores, dtype=np.float64), seed=seed + 911)
    return {"mean": summary[0], "low": summary[1], "high": summary[2], "count": summary[3]}


def build_module_transitions(descriptors: Sequence[StageDescriptor]) -> list[ModuleTransition]:
    descriptor_list = list(descriptors)
    if not descriptor_list:
        raise ValueError("stage descriptor list is empty")
    transitions: list[ModuleTransition] = []
    allowed_modules = {"condition", "temporal", "crossview"}
    for stage_index, descriptor in enumerate(descriptor_list):
        if descriptor.module not in allowed_modules:
            continue
        before_index = stage_index - 1
        if before_index < 0:
            continue
        before_descriptor = descriptor_list[before_index]
        if before_descriptor.layer > descriptor.layer:
            continue
        transitions.append(
            ModuleTransition(
                layer=int(descriptor.layer),
                module=str(descriptor.module),
                before_index=int(before_index),
                after_index=int(stage_index),
            )
        )
    transitions.sort(key=lambda item: (item.layer, item.after_index))
    return transitions


def analyze_interventions(
    run_root: Path,
    atlas: dict,
    seed: int,
) -> list[dict]:
    intervention_files = discover_intervention_capture_files(run_root)
    if not intervention_files:
        return []
    descriptors: list[StageDescriptor] = atlas["stage_descriptors"]
    stage_names = list(atlas["stage_names"])
    sigmas = np.asarray(atlas["sigmas"], dtype=np.float64)
    sample_indices = list(atlas["sample_indices"])
    sample_position = {int(sample_index): position for position, sample_index in enumerate(sample_indices)}
    output_indices = layer_output_indices(descriptors)
    transitions = build_module_transitions(descriptors)
    transition_lookup = {(transition.module, transition.layer): transition for transition in transitions}

    grouped: dict[str, list[Path]] = {}
    metadata_by_label: dict[str, dict] = {}
    for path in intervention_files:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
        label = str(metadata.get("intervention_label", "")).strip()
        if not label:
            continue
        grouped.setdefault(label, []).append(path)
        metadata_by_label[label] = metadata

    intervention_results = []
    for label_index, label in enumerate(sorted(grouped)):
        metadata = metadata_by_label[label]
        module = str(metadata.get("intervention_module", ""))
        layer = int(metadata.get("intervention_layer", -1))
        sigma = float(metadata.get("sigma", metadata.get("intervention_sigma", 0.6)))
        target_relation = TARGET_RELATION.get(module, "cross_view")
        transition = transition_lookup.get((module, layer))
        if module != "camera" and transition is None:
            continue
        sigma_index = int(np.argmin(np.abs(sigmas - sigma)))
        if abs(float(sigmas[sigma_index]) - sigma) > 1e-4:
            continue

        downstream_candidates = [
            index
            for index in output_indices
            if descriptors[index].layer > layer or descriptors[index].module == "final"
        ]
        next_stage_index = downstream_candidates[0] if downstream_candidates else output_indices[-1]
        final_stage_index = output_indices[-1]
        use_stage_name = stage_names[next_stage_index]
        final_stage_name = stage_names[final_stage_index]
        per_relation_use = {family: [] for family in PROBE_FAMILIES}
        per_relation_retain = {family: [] for family in PROBE_FAMILIES}
        intervention_capture_paths = []
        baseline_capture_paths = []

        for path in sorted(grouped[label]):
            intervention_capture = load_capture(path)
            sample_index = int(intervention_capture["metadata"].get("sample_index", -1))
            if sample_index not in sample_position:
                continue
            intervention_result, _ = evaluate_capture(intervention_capture, transport_stage_indices=set())
            intervention_stage_position = {
                name: index for index, name in enumerate(intervention_result.stage_names)
            }
            if use_stage_name not in intervention_stage_position or final_stage_name not in intervention_stage_position:
                continue
            intervention_next_index = intervention_stage_position[use_stage_name]
            intervention_final_index = intervention_stage_position[final_stage_name]
            sample_pos = sample_position[sample_index]
            for family in PROBE_FAMILIES:
                metric_name = f"{family}_structured"
                baseline_cube = atlas["metric_cubes"][metric_name]
                baseline_next = baseline_cube[sigma_index, sample_pos, next_stage_index]
                baseline_final = baseline_cube[sigma_index, sample_pos, final_stage_index]
                ablated_next = intervention_result.metrics[metric_name][intervention_next_index]
                ablated_final = intervention_result.metrics[metric_name][intervention_final_index]
                if np.isfinite(baseline_next) and np.isfinite(ablated_next):
                    per_relation_use[family].append(float(baseline_next - ablated_next))
                if np.isfinite(baseline_final) and np.isfinite(ablated_final):
                    per_relation_retain[family].append(float(baseline_final - ablated_final))
            intervention_capture_paths.append(path)
            sigma_root = None
            for parent in path.parents:
                if parent.name.startswith("sigma_"):
                    sigma_root = parent
                    break
            if sigma_root is not None:
                baseline_path = sigma_root / "capture.npz"
                if baseline_path.is_file():
                    baseline_capture_paths.append(baseline_path)

        relation_summaries = {}
        for family_index, family in enumerate(PROBE_FAMILIES):
            use_summary = bootstrap_interval(
                np.asarray(per_relation_use[family], dtype=np.float64),
                seed + label_index * 1009 + family_index * 37,
            )
            retain_summary = bootstrap_interval(
                np.asarray(per_relation_retain[family], dtype=np.float64),
                seed + label_index * 1013 + family_index * 41,
            )
            relation_summaries[family] = {
                "use": use_summary[0],
                "use_low": use_summary[1],
                "use_high": use_summary[2],
                "retain": retain_summary[0],
                "retain_low": retain_summary[1],
                "retain_high": retain_summary[2],
                "sample_count": use_summary[3],
            }

        if transition is None:
            write_summary = (float("nan"), float("nan"), float("nan"), 0)
            before_stage_name = "camera_embedding"
            after_stage_name = f"L{layer:02d}.camera_consumer"
        else:
            target_metric = atlas["metric_cubes"][f"{target_relation}_structured"]
            baseline_write = target_metric[:, :, transition.after_index] - target_metric[:, :, transition.before_index]
            write_summary = bootstrap_interval(
                baseline_write[sigma_index],
                seed + 70001 + label_index * 53,
            )
            before_stage_name = stage_names[transition.before_index]
            after_stage_name = stage_names[transition.after_index]

        baseline_transport = compute_transport_for_capture_group(
            baseline_capture_paths,
            use_stage_name,
            target_relation,
            seed + label_index * 109 + 11,
        )
        intervention_transport = compute_transport_for_capture_group(
            intervention_capture_paths,
            use_stage_name,
            target_relation,
            seed + label_index * 109 + 17,
        )
        transport_use = float("nan")
        if np.isfinite(baseline_transport["mean"]) and np.isfinite(intervention_transport["mean"]):
            transport_use = float(baseline_transport["mean"] - intervention_transport["mean"])

        intervention_results.append(
            {
                "label": label,
                "module": module,
                "layer": layer,
                "sigma": sigma,
                "target_relation": target_relation,
                "before_stage": before_stage_name,
                "after_stage": after_stage_name,
                "use_stage": stage_names[next_stage_index],
                "final_stage": stage_names[final_stage_index],
                "write": write_summary[0],
                "write_low": write_summary[1],
                "write_high": write_summary[2],
                "target_use": relation_summaries[target_relation]["use"],
                "target_use_low": relation_summaries[target_relation]["use_low"],
                "target_use_high": relation_summaries[target_relation]["use_high"],
                "target_retain": relation_summaries[target_relation]["retain"],
                "mixed_use": relation_summaries["mixed"]["use"],
                "mixed_retain": relation_summaries["mixed"]["retain"],
                "transport_use": transport_use,
                "transport_baseline": baseline_transport["mean"],
                "transport_ablated": intervention_transport["mean"],
                "sample_count": relation_summaries[target_relation]["sample_count"],
            }
        )
    return intervention_results


def compute_transport_for_capture_group(
    capture_paths: Sequence[Path],
    stage_name: str,
    family: str,
    seed: int,
) -> dict[str, float]:
    unique_paths = sorted(set(Path(path) for path in capture_paths))
    if family not in PROBE_FAMILIES:
        raise ValueError(f"unknown transport family {family}")
    payloads = []
    for path in unique_paths:
        capture = load_capture(path)
        events = build_probe_events(capture)
        stage_position = {name: index for index, name in enumerate(capture["stage_names"])}
        if str(stage_name) not in stage_position:
            continue
        stage_index = stage_position[str(stage_name)]
        encodings = prepare_feature_encodings(
            capture["features"][stage_index],
            capture["anchor_visible"],
        )
        geometry_vectors = prepare_geometry_vectors(capture["geometry"], capture["anchor_visible"])
        payloads.append(
            make_transport_payload(
                capture,
                events.get(family, []),
                encodings,
                geometry_vectors,
            )
        )
    combined = concatenate_transport_payloads(payloads)
    summary = transport_knn_alignment(combined, seed=seed)
    return summary


def analyze_binding_run(run_root: Path, seed: int = 20260816) -> dict:
    run_root = Path(run_root)
    paths_by_sigma = group_capture_paths_by_sigma(discover_baseline_capture_files(run_root))
    sigmas = sorted(paths_by_sigma)
    first_capture = load_capture(paths_by_sigma[sigmas[0]][0])
    stage_names = list(first_capture["stage_names"])
    num_layers = int(first_capture["metadata"].get("num_layers", 24))
    descriptors = describe_stages(stage_names, num_layers)
    sample_indices = sorted(
        {
            int(load_capture(path)["metadata"].get("sample_index", 0))
            for sigma_paths in paths_by_sigma.values()
            for path in sigma_paths
        }
    )
    sample_position = {sample_index: position for position, sample_index in enumerate(sample_indices)}
    output_stage_indices = set(layer_output_indices(descriptors))

    all_metric_names: set[str] = set()
    metric_cubes: dict[str, np.ndarray] = {}
    event_counts = {family: [] for family in PROBE_FAMILIES}
    transport_summary = {
        family: {
            "mean": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "low": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "high": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "count": np.zeros((len(sigmas), len(stage_names)), dtype=np.int32),
        }
        for family in PROBE_FAMILIES
    }
    event_count_matrix = {
        family: np.full((len(sigmas), len(sample_indices)), -1, dtype=np.int32)
        for family in PROBE_FAMILIES
    }

    for sigma_index, sigma in enumerate(sigmas):
        transport_accumulator = {
            family: [[] for _ in stage_names]
            for family in PROBE_FAMILIES
        }
        for capture_path in paths_by_sigma[sigma]:
            capture = load_capture(capture_path)
            if list(capture["stage_names"]) != stage_names:
                raise ValueError("all captures in one run must share stage order")
            result, transport_payloads = evaluate_capture(capture, transport_stage_indices=output_stage_indices)
            all_metric_names.update(result.metrics)
            sample_pos = sample_position[result.sample_index]
            for metric_name, metric_values in result.metrics.items():
                if metric_name not in metric_cubes:
                    metric_cubes[metric_name] = np.full(
                        (len(sigmas), len(sample_indices), len(stage_names)),
                        np.nan,
                        dtype=np.float64,
                    )
                metric_cubes[metric_name][sigma_index, sample_pos] = metric_values
            for family in PROBE_FAMILIES:
                event_counts[family].append(int(result.event_counts[family]))
                event_count_matrix[family][sigma_index, sample_pos] = int(result.event_counts[family])
                for stage_index, payload in enumerate(transport_payloads[family]):
                    if payload is not None:
                        transport_accumulator[family][stage_index].append(payload)

        for family_index, family in enumerate(PROBE_FAMILIES):
            for stage_index in sorted(output_stage_indices):
                combined = concatenate_transport_payloads(transport_accumulator[family][stage_index])
                summary = transport_knn_alignment(
                    combined,
                    seed=seed + sigma_index * 10007 + family_index * 911 + stage_index * 43,
                )
                for key in ("mean", "low", "high", "count"):
                    transport_summary[family][key][sigma_index, stage_index] = summary[key]

    metric_summary = {
        metric_name: aggregate_metric_cube(
            cube,
            seed=seed + sum(ord(character) for character in metric_name),
        )
        for metric_name, cube in metric_cubes.items()
    }
    atlas = {
        "version": ENTITY_REACTOR_PSI_VERSION,
        "run_root": str(run_root),
        "metadata": dict(first_capture["metadata"]),
        "sigmas": np.asarray(sigmas, dtype=np.float64),
        "stage_names": stage_names,
        "stage_descriptors": descriptors,
        "sample_indices": sample_indices,
        "metrics": metric_summary,
        "metric_cubes": metric_cubes,
        "transport": transport_summary,
        "event_counts": {
            family: {
                "mean": float(np.mean(values)) if values else 0.0,
                "minimum": int(np.min(values)) if values else 0,
                "maximum": int(np.max(values)) if values else 0,
            }
            for family, values in event_counts.items()
        },
        "event_count_matrix": event_count_matrix,
        "capture_count": max(len(paths_by_sigma[sigma]) for sigma in sigmas),
    }
    atlas["interventions"] = analyze_interventions(run_root, atlas, seed + 500003)
    atlas["signature"] = compute_design_signature(atlas)
    return atlas


def compute_design_signature(atlas: dict) -> dict:
    sigmas = np.asarray(atlas["sigmas"], dtype=np.float64)
    descriptors: list[StageDescriptor] = atlas["stage_descriptors"]
    output_indices = layer_output_indices(descriptors)
    output_depths = np.asarray([descriptors[index].depth for index in output_indices], dtype=np.float64)
    diagnostic_sigma_index = int(np.argmin(np.abs(sigmas - 0.60)))
    low_noise_index = int(np.argmin(np.abs(sigmas - 0.30)))
    cv_summary = atlas["metrics"]["cross_view_structured"]
    mixed_summary = atlas["metrics"]["mixed_structured"]
    cv_curve = cv_summary["mean"][diagnostic_sigma_index, output_indices]
    cv_low = cv_summary["low"][diagnostic_sigma_index, output_indices]
    mixed_curve = mixed_summary["mean"][diagnostic_sigma_index, output_indices]
    transport_curve = atlas["transport"]["cross_view"]["mean"][diagnostic_sigma_index, output_indices]
    low_noise_cv_curve = cv_summary["mean"][low_noise_index, output_indices]

    finite_cv = cv_curve[np.isfinite(cv_curve)]
    peak_cv = float(np.max(finite_cv)) if finite_cv.size else float("nan")
    peak_index = int(np.nanargmax(cv_curve)) if np.isfinite(cv_curve).any() else 0
    peak_stage = descriptors[output_indices[peak_index]].name
    formation_threshold = max(0.08, 0.50 * max(peak_cv, 0.0))
    formation_depth = 1.0
    formation_stage = "unresolved"
    for curve_index, stage_index in enumerate(output_indices):
        value = cv_curve[curve_index]
        confidence_low = cv_low[curve_index]
        if np.isfinite(value) and value >= formation_threshold and np.isfinite(confidence_low) and confidence_low > 0.0:
            formation_depth = float(output_depths[curve_index])
            formation_stage = descriptors[stage_index].name
            break

    final_cv = float(low_noise_cv_curve[-1]) if np.isfinite(low_noise_cv_curve[-1]) else float("nan")
    low_noise_peak = float(np.nanmax(low_noise_cv_curve)) if np.isfinite(low_noise_cv_curve).any() else float("nan")
    late_retention = 0.0
    if np.isfinite(final_cv) and np.isfinite(low_noise_peak) and low_noise_peak > 0.05:
        late_retention = float(np.clip(final_cv / low_noise_peak, 0.0, 1.25))
    mixed_peak = float(np.nanmax(mixed_curve)) if np.isfinite(mixed_curve).any() else float("nan")
    transport_peak = float(np.nanmax(transport_curve)) if np.isfinite(transport_curve).any() else float("nan")

    intervention_by_module = {
        module: {"target_use": [], "mixed_use": [], "transport_use": []}
        for module in MODULE_ORDER
    }
    strongest_necessary = ("none", -1, float("-inf"))
    strongest_writer = ("none", -1, float("-inf"))
    for result in atlas.get("interventions", []):
        module = str(result["module"])
        if module not in intervention_by_module:
            continue
        for key in ("target_use", "mixed_use", "transport_use"):
            value = float(result.get(key, float("nan")))
            if np.isfinite(value):
                intervention_by_module[module][key].append(value)
        write_value = float(result.get("write", float("nan")))
        use_value = float(result.get("target_use", float("nan")))
        if np.isfinite(write_value) and write_value > strongest_writer[2]:
            strongest_writer = (module, int(result["layer"]), write_value)
        if np.isfinite(use_value) and use_value > strongest_necessary[2]:
            strongest_necessary = (module, int(result["layer"]), use_value)

    causal_summary = {}
    for module in MODULE_ORDER:
        causal_summary[module] = {}
        for key, values in intervention_by_module[module].items():
            causal_summary[module][key] = float(np.nanmax(values)) if values else float("nan")

    if not np.isfinite(peak_cv) or peak_cv < 0.08:
        pattern = "NO STABLE CROSS-VIEW BINDING"
    elif formation_depth > 0.72:
        pattern = "LATE FORMATION"
    elif late_retention < 0.35:
        pattern = "MID-LAYER BINDING / LATE LOSS"
    elif mixed_peak < 0.08:
        pattern = "VIEW-ONLY BINDING"
    elif transport_peak < 0.05:
        pattern = "IDENTITY WITHOUT TRANSPORT"
    else:
        pattern = "STABLE RELATION ORGANIZATION"

    writer = "none"
    if strongest_writer[0] != "none" and strongest_writer[2] > 0.01:
        writer = f"{strongest_writer[0]}@L{strongest_writer[1]:02d}"
    necessary = "unresolved"
    if strongest_necessary[0] != "none" and strongest_necessary[2] > 0.01:
        necessary = f"{strongest_necessary[0]}@L{strongest_necessary[1]:02d}"

    return {
        "setting_name": str(atlas["metadata"].get("setting_name", atlas["metadata"].get("model_name", "unknown"))),
        "model_name": str(atlas["metadata"].get("model_name", "unknown")),
        "checkpoint_name": str(atlas["metadata"].get("checkpoint_name", "unknown")),
        "pattern": pattern,
        "writer": writer,
        "necessary_module": necessary,
        "peak_cross_view_binding": peak_cv,
        "peak_cross_view_stage": peak_stage,
        "formation_depth": formation_depth,
        "formation_stage": formation_stage,
        "low_noise_final_cross_view": final_cv,
        "late_retention": late_retention,
        "peak_mixed_binding": mixed_peak,
        "peak_cross_view_transport": transport_peak,
        "diagnostic_sigma": float(sigmas[diagnostic_sigma_index]),
        "low_noise_sigma": float(sigmas[low_noise_index]),
        "causal_summary": causal_summary,
        "event_counts": atlas["event_counts"],
        "intervention_count": len(atlas.get("interventions", [])),
    }




def summarize_runtime_interventions(
    atlas: dict,
    runtime_interventions: dict[str, dict],
    seed: int,
) -> list[dict]:
    if not runtime_interventions:
        return []
    descriptors: list[StageDescriptor] = atlas["stage_descriptors"]
    stage_names = list(atlas["stage_names"])
    sigmas = np.asarray(atlas["sigmas"], dtype=np.float64)
    sample_indices = list(atlas["sample_indices"])
    sample_position = {int(sample_index): position for position, sample_index in enumerate(sample_indices)}
    output_indices = layer_output_indices(descriptors)
    transitions = build_module_transitions(descriptors)
    transition_lookup = {(transition.module, transition.layer): transition for transition in transitions}
    intervention_results = []
    for label_index, label in enumerate(sorted(runtime_interventions)):
        payload = runtime_interventions[label]
        metadata = dict(payload.get("metadata", {}))
        result_by_sample = dict(payload.get("results", {}))
        module = str(metadata.get("intervention_module", ""))
        layer = int(metadata.get("intervention_layer", -1))
        sigma = float(metadata.get("sigma", metadata.get("intervention_sigma", 0.6)))
        target_relation = TARGET_RELATION.get(module, "cross_view")
        transition = transition_lookup.get((module, layer))
        if module != "camera" and transition is None:
            continue
        sigma_index = int(np.argmin(np.abs(sigmas - sigma)))
        downstream = [
            index
            for index in output_indices
            if descriptors[index].module == "final" or descriptors[index].layer > layer
        ]
        next_stage_index = downstream[0] if downstream else output_indices[-1]
        final_stage_index = output_indices[-1]
        use_stage_name = stage_names[next_stage_index]
        final_stage_name = stage_names[final_stage_index]
        per_relation_use = {family: [] for family in PROBE_FAMILIES}
        per_relation_retain = {family: [] for family in PROBE_FAMILIES}
        for sample_index, intervention_result in sorted(result_by_sample.items()):
            sample_index = int(sample_index)
            if sample_index not in sample_position:
                continue
            stage_position = {name: index for index, name in enumerate(intervention_result.stage_names)}
            if use_stage_name not in stage_position or final_stage_name not in stage_position:
                continue
            sample_pos = sample_position[sample_index]
            intervention_next = stage_position[use_stage_name]
            intervention_final = stage_position[final_stage_name]
            for family in PROBE_FAMILIES:
                metric_name = f"{family}_structured"
                baseline_cube = atlas["metric_cubes"][metric_name]
                baseline_next = baseline_cube[sigma_index, sample_pos, next_stage_index]
                baseline_final = baseline_cube[sigma_index, sample_pos, final_stage_index]
                ablated_next = intervention_result.metrics[metric_name][intervention_next]
                ablated_final = intervention_result.metrics[metric_name][intervention_final]
                if np.isfinite(baseline_next) and np.isfinite(ablated_next):
                    per_relation_use[family].append(float(baseline_next - ablated_next))
                if np.isfinite(baseline_final) and np.isfinite(ablated_final):
                    per_relation_retain[family].append(float(baseline_final - ablated_final))
        relation_summaries = {}
        for family_index, family in enumerate(PROBE_FAMILIES):
            use_summary = bootstrap_interval(
                np.asarray(per_relation_use[family], dtype=np.float64),
                seed + label_index * 1009 + family_index * 37,
            )
            retain_summary = bootstrap_interval(
                np.asarray(per_relation_retain[family], dtype=np.float64),
                seed + label_index * 1013 + family_index * 41,
            )
            relation_summaries[family] = {
                "use": use_summary[0],
                "use_low": use_summary[1],
                "use_high": use_summary[2],
                "retain": retain_summary[0],
                "retain_low": retain_summary[1],
                "retain_high": retain_summary[2],
                "sample_count": use_summary[3],
            }
        if transition is None:
            write_summary = (float("nan"), float("nan"), float("nan"), 0)
            before_stage_name = "camera_embedding"
            after_stage_name = f"L{layer:02d}.camera_consumer"
        else:
            target_metric = atlas["metric_cubes"][f"{target_relation}_structured"]
            baseline_write = target_metric[:, :, transition.after_index] - target_metric[:, :, transition.before_index]
            write_summary = bootstrap_interval(
                baseline_write[sigma_index],
                seed + 70001 + label_index * 53,
            )
            before_stage_name = stage_names[transition.before_index]
            after_stage_name = stage_names[transition.after_index]
        intervention_results.append(
            {
                "label": label,
                "module": module,
                "layer": layer,
                "sigma": sigma,
                "target_relation": target_relation,
                "before_stage": before_stage_name,
                "after_stage": after_stage_name,
                "use_stage": use_stage_name,
                "final_stage": final_stage_name,
                "write": write_summary[0],
                "write_low": write_summary[1],
                "write_high": write_summary[2],
                "target_use": relation_summaries[target_relation]["use"],
                "target_use_low": relation_summaries[target_relation]["use_low"],
                "target_use_high": relation_summaries[target_relation]["use_high"],
                "target_retain": relation_summaries[target_relation]["retain"],
                "mixed_use": relation_summaries["mixed"]["use"],
                "mixed_retain": relation_summaries["mixed"]["retain"],
                "transport_use": float("nan"),
                "transport_baseline": float("nan"),
                "transport_ablated": float("nan"),
                "sample_count": relation_summaries[target_relation]["sample_count"],
            }
        )
    return intervention_results


def build_runtime_atlas(
    baseline_results: dict[float, dict[int, CaptureProbeResult]],
    transport_reservoirs: dict[tuple[float, int], dict[str, np.ndarray]],
    runtime_interventions: dict[str, dict],
    metadata: dict,
    num_layers: int,
    seed: int,
) -> dict:
    if not baseline_results:
        raise RuntimeError("runtime analysis has no baseline results")
    sigmas = sorted(float(value) for value in baseline_results)
    sample_sets = [set(int(index) for index in baseline_results[sigma]) for sigma in sigmas]
    sample_indices = sorted(set.intersection(*sample_sets)) if sample_sets else []
    if not sample_indices:
        raise RuntimeError("no sample completed every requested sigma")
    first_result = baseline_results[sigmas[0]][sample_indices[0]]
    stage_names = list(first_result.stage_names)
    descriptors = describe_stages(stage_names, int(num_layers))
    output_indices = set(layer_output_indices(descriptors))
    all_metric_names = sorted(first_result.metrics)
    metric_cubes = {
        metric_name: np.full(
            (len(sigmas), len(sample_indices), len(stage_names)),
            np.nan,
            dtype=np.float64,
        )
        for metric_name in all_metric_names
    }
    event_counts = {family: [] for family in PROBE_FAMILIES}
    event_count_matrix = {
        family: np.full((len(sigmas), len(sample_indices)), -1, dtype=np.int32)
        for family in PROBE_FAMILIES
    }
    for sigma_index, sigma in enumerate(sigmas):
        for sample_pos, sample_index in enumerate(sample_indices):
            result = baseline_results[sigma][sample_index]
            if list(result.stage_names) != stage_names:
                raise RuntimeError("runtime captures do not share one stage order")
            for metric_name, values in result.metrics.items():
                metric_cubes[metric_name][sigma_index, sample_pos] = values
            for family in PROBE_FAMILIES:
                count = int(result.event_counts[family])
                event_counts[family].append(count)
                event_count_matrix[family][sigma_index, sample_pos] = count
    metric_summary = {
        metric_name: aggregate_metric_cube(
            cube,
            seed=seed + sum(ord(character) for character in metric_name),
        )
        for metric_name, cube in metric_cubes.items()
    }
    transport_summary = {
        family: {
            "mean": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "low": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "high": np.full((len(sigmas), len(stage_names)), np.nan, dtype=np.float64),
            "count": np.zeros((len(sigmas), len(stage_names)), dtype=np.int32),
        }
        for family in PROBE_FAMILIES
    }
    for sigma_index, sigma in enumerate(sigmas):
        for stage_index in sorted(output_indices):
            payload = transport_reservoirs.get((float(sigma), int(stage_index)))
            if payload is None:
                continue
            summary = transport_knn_alignment(
                payload,
                seed=seed + sigma_index * 10007 + stage_index * 43,
                maximum_events=int(payload["feature_delta"].shape[0]),
            )
            for key in ("mean", "low", "high", "count"):
                transport_summary["cross_view"][key][sigma_index, stage_index] = summary[key]
    atlas = {
        "version": ENTITY_REACTOR_PSI_VERSION,
        "run_root": "<runtime-only>",
        "metadata": dict(metadata),
        "sigmas": np.asarray(sigmas, dtype=np.float64),
        "stage_names": stage_names,
        "stage_descriptors": descriptors,
        "sample_indices": sample_indices,
        "metrics": metric_summary,
        "metric_cubes": metric_cubes,
        "transport": transport_summary,
        "event_counts": {
            family: {
                "mean": float(np.mean(values)) if values else 0.0,
                "minimum": int(np.min(values)) if values else 0,
                "maximum": int(np.max(values)) if values else 0,
            }
            for family, values in event_counts.items()
        },
        "event_count_matrix": event_count_matrix,
        "capture_count": len(sample_indices),
    }
    atlas["interventions"] = summarize_runtime_interventions(
        atlas,
        runtime_interventions,
        seed + 500003,
    )
    atlas["signature"] = compute_design_signature(atlas)
    return atlas

def write_plot_data_cache(atlas: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "relation_plot_data.npz"
    manifest_path = output_dir / "relation_plot_manifest.json"
    descriptors: list[StageDescriptor] = atlas["stage_descriptors"]
    output_indices = layer_output_indices(descriptors)
    sigmas = np.asarray(atlas["sigmas"], dtype=np.float32)

    output_stage_names = np.asarray(
        [atlas["stage_names"][index] for index in output_indices],
        dtype="U64",
    )
    output_stage_depths = np.asarray(
        [descriptors[index].depth for index in output_indices],
        dtype=np.float32,
    )

    payload: dict[str, np.ndarray] = {
        "plot_cache_version": np.asarray(ENTITY_REACTOR_PLOT_CACHE_VERSION),
        "psi_version": np.asarray(str(atlas.get("version", ENTITY_REACTOR_PSI_VERSION))),
        "sigmas": sigmas,
        "stage_names": output_stage_names,
        "stage_depths": output_stage_depths,
        "sample_indices": np.asarray(atlas.get("sample_indices", []), dtype=np.int32),
        "metadata_json": np.asarray(
            json.dumps(atlas["metadata"], ensure_ascii=False, sort_keys=True)
        ),
        "signature_json": np.asarray(
            json.dumps(atlas["signature"], ensure_ascii=False, sort_keys=True)
        ),
    }

    metric_names = (
        "cross_view_structured",
        "temporal_structured",
        "mixed_structured",
    )
    for metric_name in metric_names:
        summary = atlas["metrics"][metric_name]
        payload[f"metric__{metric_name}__mean"] = np.asarray(
            summary["mean"][:, output_indices], dtype=np.float32
        )
        payload[f"metric__{metric_name}__low"] = np.asarray(
            summary["low"][:, output_indices], dtype=np.float32
        )

    cross_view_transport = atlas["transport"]["cross_view"]
    payload["transport__cross_view__mean"] = np.asarray(
        cross_view_transport["mean"][:, output_indices], dtype=np.float32
    )
    payload["transport__cross_view__low"] = np.asarray(
        cross_view_transport["low"][:, output_indices], dtype=np.float32
    )

    interventions = list(atlas.get("interventions", []))
    causal_layers = sorted(
        {
            int(result["layer"])
            for result in interventions
            if str(result.get("module", "")) in MODULE_ORDER
            and int(result.get("layer", -1)) >= 0
        }
    )
    causal_use = np.full(
        (len(causal_layers), len(MODULE_ORDER)),
        np.nan,
        dtype=np.float32,
    )
    causal_low = np.full_like(causal_use, np.nan)
    causal_layer_index = {layer: index for index, layer in enumerate(causal_layers)}
    causal_module_index = {module: index for index, module in enumerate(MODULE_ORDER)}
    for result in interventions:
        module = str(result.get("module", ""))
        layer = int(result.get("layer", -1))
        if module not in causal_module_index or layer not in causal_layer_index:
            continue
        row = causal_layer_index[layer]
        column = causal_module_index[module]
        causal_use[row, column] = float(result.get("target_use", float("nan")))
        causal_low[row, column] = float(result.get("target_use_low", float("nan")))

    payload["causal_layers"] = np.asarray(causal_layers, dtype=np.int16)
    payload["causal_modules"] = np.asarray(MODULE_ORDER, dtype="U16")
    payload["causal_use"] = causal_use
    payload["causal_use_low"] = causal_low

    np.savez_compressed(plot_path, **payload)

    sample_count = max(1, len(atlas.get("sample_indices", [])))
    cache_bytes = int(plot_path.stat().st_size)
    bytes_per_sample_equivalent = float(cache_bytes) / float(sample_count)
    per_sample_budget_bytes = 10 * 1024 * 1024
    if bytes_per_sample_equivalent > per_sample_budget_bytes:
        plot_path.unlink(missing_ok=True)
        raise RuntimeError(
            "relation_plot_data.npz exceeds the visualization-cache budget: "
            f"{bytes_per_sample_equivalent / (1024 * 1024):.2f} MiB/sample > 10 MiB/sample"
        )

    manifest = {
        "plot_cache_version": ENTITY_REACTOR_PLOT_CACHE_VERSION,
        "psi_version": str(atlas.get("version", ENTITY_REACTOR_PSI_VERSION)),
        "setting_name": str(atlas["signature"].get("setting_name", "unknown")),
        "checkpoint_name": str(atlas["signature"].get("checkpoint_name", "unknown")),
        "plot_data_file": plot_path.name,
        "sample_count": int(len(atlas.get("sample_indices", []))),
        "cache_bytes": cache_bytes,
        "cache_mib": cache_bytes / float(1024 * 1024),
        "bytes_per_sample_equivalent": bytes_per_sample_equivalent,
        "mib_per_sample_equivalent": bytes_per_sample_equivalent / float(1024 * 1024),
        "per_sample_budget_mib": 10.0,
        "camera_is_separate_module": True,
        "module_order": list(MODULE_ORDER),
        "feature_dimension_policy": "full_hidden_dimension",
        "hard_negative_policy": "same_view_same_class_only",
        "accepted_sample_indices": [int(value) for value in atlas.get("sample_indices", [])],
        "persistent_intermediate_features": False,
        "saved_for_replot": [
            "layer×noise relation means and bootstrap lower bounds",
            "cross-view transport means and lower bounds",
            "Camera/Condition/Temporal/Cross-view causal USE matrix",
            "compact signature and metadata",
        ],
        "not_saved_in_plot_cache": [
            "per-sample feature tensors",
            "per-sample metric cubes",
            "raw intervention features",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plot_path, manifest_path

def write_binding_tables(atlas: dict, output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "relation_stage_metrics.csv"
    transport_path = output_dir / "relation_transport_metrics.csv"
    intervention_path = output_dir / "relation_interventions.csv"
    signature_path = output_dir / "design_signature.json"
    profile_path = output_dir / "relation_design_profile.json"
    sigmas = np.asarray(atlas["sigmas"], dtype=np.float64)
    descriptors: list[StageDescriptor] = atlas["stage_descriptors"]

    metric_names = (
        "cross_view_structured",
        "cross_view_pooled",
        "temporal_structured",
        "mixed_structured",
        "cross_view_gap",
    )
    with metric_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        header = ["sigma", "stage", "layer", "module", "depth"]
        for metric_name in metric_names:
            header.extend([metric_name, f"{metric_name}_ci_low", f"{metric_name}_ci_high", f"{metric_name}_n"])
        writer.writerow(header)
        for sigma_index, sigma in enumerate(sigmas):
            for stage_index, descriptor in enumerate(descriptors):
                row = [float(sigma), descriptor.name, int(descriptor.layer), descriptor.module, float(descriptor.depth)]
                for metric_name in metric_names:
                    summary = atlas["metrics"][metric_name]
                    row.extend(
                        [
                            float(summary["mean"][sigma_index, stage_index]),
                            float(summary["low"][sigma_index, stage_index]),
                            float(summary["high"][sigma_index, stage_index]),
                            int(summary["count"][sigma_index, stage_index]),
                        ]
                    )
                writer.writerow(row)

    with transport_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sigma", "stage", "layer", "module", "family", "transport_alignment", "ci_low", "ci_high", "sample_count"])
        for family in PROBE_FAMILIES:
            summary = atlas["transport"][family]
            for sigma_index, sigma in enumerate(sigmas):
                for stage_index, descriptor in enumerate(descriptors):
                    writer.writerow(
                        [
                            float(sigma),
                            descriptor.name,
                            int(descriptor.layer),
                            descriptor.module,
                            family,
                            float(summary["mean"][sigma_index, stage_index]),
                            float(summary["low"][sigma_index, stage_index]),
                            float(summary["high"][sigma_index, stage_index]),
                            int(summary["count"][sigma_index, stage_index]),
                        ]
                    )

    interventions = atlas.get("interventions", [])
    with intervention_path.open("w", encoding="utf-8", newline="") as file:
        if interventions:
            writer = csv.DictWriter(file, fieldnames=list(interventions[0].keys()))
            writer.writeheader()
            writer.writerows(interventions)
        else:
            file.write("label,module,layer\n")

    signature_path.write_text(json.dumps(atlas["signature"], ensure_ascii=False, indent=2), encoding="utf-8")
    output_indices = layer_output_indices(descriptors)
    diagnostic_sigma_index = int(np.argmin(np.abs(sigmas - 0.60)))
    low_noise_index = int(np.argmin(np.abs(sigmas - 0.30)))
    profile = {
        "version": ENTITY_REACTOR_PSI_VERSION,
        "setting_name": atlas["signature"]["setting_name"],
        "checkpoint_name": atlas["signature"]["checkpoint_name"],
        "depth": [float(descriptors[index].depth) for index in output_indices],
        "stage": [descriptors[index].name for index in output_indices],
        "cross_view_binding": atlas["metrics"]["cross_view_structured"]["mean"][diagnostic_sigma_index, output_indices].tolist(),
        "mixed_binding": atlas["metrics"]["mixed_structured"]["mean"][diagnostic_sigma_index, output_indices].tolist(),
        "temporal_binding": atlas["metrics"]["temporal_structured"]["mean"][diagnostic_sigma_index, output_indices].tolist(),
        "cross_view_transport": atlas["transport"]["cross_view"]["mean"][diagnostic_sigma_index, output_indices].tolist(),
        "low_noise_cross_view": atlas["metrics"]["cross_view_structured"]["mean"][low_noise_index, output_indices].tolist(),
        "causal_summary": atlas["signature"]["causal_summary"],
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plot_data_cache(atlas, output_dir)
    return metric_path, transport_path, intervention_path, signature_path, profile_path


def group_captures_by_run(paths: Iterable[Path]) -> dict[tuple[str, str, float], list[dict]]:
    path_list = [Path(path) for path in paths]
    if not path_list:
        return {}
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for path in path_list:
        capture = load_capture(path)
        metadata = capture["metadata"]
        key = (
            str(metadata.get("model_name", "unknown")),
            str(metadata.get("checkpoint_name", "unknown")),
            float(metadata.get("sigma", -1.0)),
        )
        groups.setdefault(key, []).append(capture)
    for captures in groups.values():
        captures.sort(key=lambda capture: int(capture["metadata"].get("sample_index", 0)))
    return groups
