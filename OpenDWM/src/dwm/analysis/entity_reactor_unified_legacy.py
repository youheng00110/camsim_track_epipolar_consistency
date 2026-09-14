from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

import dwm.common
from dwm.analysis.entity_reactor import (
    EntityReactorController,
    NoCrossCameraTransitionError,
    entity_reactor_forward,
)
from dwm.analysis.psi import (
    MODULE_ORDER,
    CaptureProbeResult,
    build_probe_events,
    build_runtime_atlas,
    evaluate_capture,
)
from dwm.analysis.tracks import NuPlanTrackResolver
from dwm.analyze_entity_reactor import (
    ControlledModelInputBuilder,
    _resolve_intervention_save_stages,
    _validate_pv_conditions,
    checkpoint_output_name,
    detect_model_kind,
    initialize_global_state,
    instantiate_pipeline,
    resolve_checkpoint_paths,
    setup_distributed,
    summarize_design_config,
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

import dwm.analysis.psi as psi_module

LEGACY_UNIFIED_VERSION = UNIFIED_VERSION + "-legacy"
MODULE_LABELS = ("camera", "condition", "temporal", "crossview")
_SCORE_AUC_WEIGHT = DEFAULT_AUC_WEIGHT
_SCORE_MARGIN_SCALE = DEFAULT_MARGIN_SCALE




def _weighted_evaluate_event_family(events, encodings):
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
    query_rows, positive_rows, negative_rows, negative_mask = psi_module.prepare_event_arrays(events)
    pooled_vectors = encodings["pooled"]
    pooled_query = pooled_vectors[query_rows]
    pooled_positive = np.einsum("ed,ed->e", pooled_vectors[positive_rows], pooled_query, optimize=True)
    pooled_negative = np.einsum("emd,ed->em", pooled_vectors[negative_rows], pooled_query, optimize=True)
    structured_positive = psi_module.structured_pair_similarity(encodings, query_rows, positive_rows[:, None])[:, 0]
    structured_negative = psi_module.structured_pair_similarity(encodings, query_rows, negative_rows)
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
    pooled_score = float(composite_score(pooled_auc, pooled_margin, _SCORE_AUC_WEIGHT, _SCORE_MARGIN_SCALE))
    structured_score = float(composite_score(structured_auc, structured_margin, _SCORE_AUC_WEIGHT, _SCORE_MARGIN_SCALE))
    return {
        "pooled_score": pooled_score,
        "structured_score": structured_score,
        "pooled_auc": pooled_auc,
        "structured_auc": structured_auc,
        "pooled_margin": pooled_margin,
        "structured_margin": structured_margin,
        "pooled_top1": float(np.mean(pooled_margins >= 0.0)),
        "structured_top1": float(np.mean(structured_margins >= 0.0)),
    }


def _weighted_evaluate_capture(capture, transport_stage_indices=None):
    events = psi_module.build_probe_events(capture)
    stage_names = list(capture["stage_names"])
    selected_transport_stages = set(range(len(stage_names))) if transport_stage_indices is None else set(transport_stage_indices)
    metric_names = []
    for family in psi_module.PROBE_FAMILIES:
        metric_names.extend([
            f"{family}_pooled",
            f"{family}_structured",
            f"{family}_gap",
            f"{family}_pooled_auc",
            f"{family}_structured_auc",
            f"{family}_pooled_margin",
            f"{family}_structured_margin",
            f"{family}_pooled_top1",
            f"{family}_structured_top1",
        ])
    metrics = {name: np.full(len(stage_names), np.nan, dtype=np.float64) for name in metric_names}
    geometry_vectors = psi_module.prepare_geometry_vectors(capture["geometry"], capture["anchor_visible"])
    transport_payloads = {family: [] for family in psi_module.PROBE_FAMILIES}
    for stage_index in range(len(stage_names)):
        encodings = psi_module.prepare_feature_encodings(capture["features"][stage_index], capture["anchor_visible"])
        for family in psi_module.PROBE_FAMILIES:
            family_metrics = _weighted_evaluate_event_family(events[family], encodings)
            pooled_value = family_metrics["pooled_score"]
            structured_value = family_metrics["structured_score"]
            metrics[f"{family}_pooled"][stage_index] = pooled_value
            metrics[f"{family}_structured"][stage_index] = structured_value
            metrics[f"{family}_gap"][stage_index] = structured_value - pooled_value
            metrics[f"{family}_pooled_auc"][stage_index] = family_metrics["pooled_auc"]
            metrics[f"{family}_structured_auc"][stage_index] = family_metrics["structured_auc"]
            metrics[f"{family}_pooled_margin"][stage_index] = family_metrics["pooled_margin"]
            metrics[f"{family}_structured_margin"][stage_index] = family_metrics["structured_margin"]
            metrics[f"{family}_pooled_top1"][stage_index] = family_metrics["pooled_top1"]
            metrics[f"{family}_structured_top1"][stage_index] = family_metrics["structured_top1"]
            if stage_index in selected_transport_stages:
                transport_payloads[family].append(
                    psi_module.make_transport_payload(capture, events[family], encodings, geometry_vectors)
                )
            else:
                transport_payloads[family].append(None)
    result = CaptureProbeResult(
        capture_path=capture["path"],
        sample_index=int(capture["metadata"].get("sample_index", 0)),
        stage_names=stage_names,
        metrics=metrics,
        event_counts={family: len(events[family]) for family in psi_module.PROBE_FAMILIES},
    )
    return result, transport_payloads


def _bootstrap_matrix_from_atlas(atlas, output_indices):
    sigmas = [float(value) for value in atlas["sigmas"]]
    families = ("cross_view", "temporal", "mixed")
    metrics = ("score", "auc", "margin", "top1")
    baseline = {}
    metric_name_map = {
        "score": "{family}_structured",
        "auc": "{family}_structured_auc",
        "margin": "{family}_structured_margin",
        "top1": "{family}_structured_top1",
    }
    for family in families:
        baseline[family] = {}
        for metric in metrics:
            source = atlas["metrics"][metric_name_map[metric].format(family=family)]
            baseline[family][metric] = {
                stat: np.asarray(source[stat][:, output_indices], dtype=np.float32 if stat != "count" else np.int16)
                for stat in ("mean", "low", "high", "count")
            }
    return sigmas, baseline


def _causal_cache_from_atlas(atlas, sigmas):
    interventions = list(atlas.get("interventions", []))
    modules = [name for name in MODULE_ORDER]
    layers = sorted({int(item["layer"]) for item in interventions if int(item.get("layer", -1)) >= 0})
    use = np.full((len(sigmas), len(layers), len(modules)), np.nan, dtype=np.float32)
    retain = np.full_like(use, np.nan)
    write = np.full_like(use, np.nan)
    count = np.zeros(use.shape, dtype=np.int16)
    sigma_pos = {round(float(value), 8): idx for idx, value in enumerate(sigmas)}
    layer_pos = {int(value): idx for idx, value in enumerate(layers)}
    module_pos = {str(value): idx for idx, value in enumerate(modules)}
    for item in interventions:
        sigma = round(float(item.get("sigma", 0.6)), 8)
        module = str(item.get("module", ""))
        layer = int(item.get("layer", -1))
        if sigma not in sigma_pos or module not in module_pos or layer not in layer_pos:
            continue
        index = (sigma_pos[sigma], layer_pos[layer], module_pos[module])
        use[index] = float(item.get("target_use", float("nan")))
        retain[index] = float(item.get("target_retain", float("nan")))
        write[index] = float(item.get("write", float("nan")))
        count[index] = int(item.get("sample_count", 0))
    return {"modules": modules, "layers": layers, "use": use, "retain": retain, "write": write, "count": count}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified legacy Entity Reactor analysis with full-D strict probing and multi-noise causal intervention.")
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help=(
            "Checkpoint path. If omitted and --reference-run-root contains "
            "checkpoint_path metadata, that checkpoint is reused. Otherwise "
            "pipeline.model_checkpoint_path from the config is used."
        ),
    )
    parser.add_argument(
        "--reference-run-root",
        default=None,
        type=Path,
        help=(
            "Old completed non-WAN Entity Reactor run root, experiment root, "
            "or relation_plot_data.npz. Its accepted sample_indices are reused "
            "so this quick test uses clips already known to be valid."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of valid clips for Figure 1 / baseline representation state.",
    )
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="Noise levels. Default keeps the requested 0.9, 0.6, 0.3 order.",
    )
    parser.add_argument("--analysis-seed", type=int, default=3107)
    parser.add_argument("--causal-sample-count", type=int, default=8, help="Number of accepted baseline clips used for causal Figure 2; 0 means all.")
    parser.add_argument("--auc-weight", type=float, default=DEFAULT_AUC_WEIGHT, help="Composite score AUC weight. Default 0.55; margin weight is 1-AUC weight.")
    parser.add_argument("--margin-scale", type=float, default=DEFAULT_MARGIN_SCALE, help="Hard-negative margin tanh scale.")
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-entities", type=int, default=32)
    parser.add_argument("--minimum-track-length", type=int, default=3)
    parser.add_argument(
        "--entity-classes",
        nargs="+",
        default=("car", "vehicle", "truck", "bus"),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of key layers. Default reproduces all old Figure-2 causal layers.",
    )
    parser.add_argument(
        "--capture-pattern",
        action="append",
        default=None,
        help="Optional Entity Reactor capture pattern; normally leave unset.",
    )
    parser.add_argument(
        "--keep-resume",
        action="store_true",
        help="Keep _resume statistics after successful completion. Raw hidden features are never saved.",
    )
    return parser


def resolve_reference_plot_cache(reference: Optional[Path]) -> Optional[Path]:
    if reference is None:
        return None
    reference = Path(reference).expanduser().resolve()
    if reference.is_file():
        if reference.name != "relation_plot_data.npz":
            raise ValueError(
                "--reference-run-root file must be relation_plot_data.npz, got "
                f"{reference}"
            )
        return reference
    if not reference.is_dir():
        raise FileNotFoundError(f"reference run does not exist: {reference}")
    direct = reference / "relation_plot_data.npz"
    if direct.is_file():
        return direct
    candidates = sorted(
        reference.glob("**/relation_plot_data.npz"),
        key=lambda path: str(path),
    )
    if not candidates:
        raise FileNotFoundError(
            f"no relation_plot_data.npz was found under {reference}"
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "reference path contains multiple completed checkpoint runs; "
            "point --reference-run-root to the exact checkpoint root. Found: "
            + ", ".join(str(path.parent) for path in candidates[:8])
        )
    return candidates[0]


def load_reference_info(plot_cache: Optional[Path]) -> dict:
    if plot_cache is None:
        return {}
    with np.load(plot_cache, allow_pickle=False) as data:
        sample_indices = (
            np.asarray(data["sample_indices"], dtype=np.int64).tolist()
            if "sample_indices" in data.files
            else []
        )
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        signature = {}
        if "signature_json" in data.files:
            signature = json.loads(str(data["signature_json"].item()))
    return {
        "plot_cache": str(plot_cache),
        "sample_indices": [int(value) for value in sample_indices],
        "metadata": metadata,
        "signature": signature,
    }


def resolve_checkpoint(
    config: dict,
    requested: Optional[str],
    reference_info: dict,
) -> Path:
    source = requested
    if source is None:
        source = reference_info.get("metadata", {}).get("checkpoint_path")
    if source is None:
        source = reference_info.get("signature", {}).get("checkpoint_path")
    paths = resolve_checkpoint_paths(config, source, False)
    if len(paths) != 1:
        raise RuntimeError(f"quick probe expects exactly one checkpoint, got {paths}")
    return Path(paths[0])


def resolve_initial_samples(
    args,
    reference_info: dict,
    dataset_length: int,
) -> list[int]:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    reference_samples = [
        int(value) for value in reference_info.get("sample_indices", [])
    ]
    if reference_samples:
        selected = reference_samples[: int(args.sample_count)]
        if len(selected) < int(args.sample_count):
            print(
                f"[CausalNoiseProbe] reference has only {len(selected)} samples; using all of them",
                flush=True,
            )
        return selected
    selected = [
        int(args.start_index) + position * int(args.sample_stride)
        for position in range(int(args.sample_count))
    ]
    invalid = [index for index in selected if index < 0 or index >= dataset_length]
    if invalid:
        raise IndexError(
            f"initial sample indices exceed validation dataset: {invalid[:8]} length={dataset_length}"
        )
    return selected


def build_intervention_plan(
    controller: EntityReactorController,
    requested_layers: Optional[list[int]],
) -> list[dict]:
    layer_filter = None
    if requested_layers:
        layer_filter = {int(value) for value in requested_layers}
    module_layers = (
        ("camera", controller.camera_layers, "cam"),
        ("condition", controller.condition_layers, "cond"),
        ("temporal", controller.temporal_layers, "temp"),
        ("crossview", controller.crossview_layers, "view"),
    )
    plan = []
    seen = set()
    for module_name, layer_values, suffix in module_layers:
        for layer_index in sorted(set(int(value) for value in layer_values)):
            if layer_index not in controller.key_layers:
                continue
            if layer_filter is not None and layer_index not in layer_filter:
                continue
            key = (module_name, layer_index)
            if key in seen:
                continue
            seen.add(key)
            plan.append(
                {
                    "module": module_name,
                    "layer": layer_index,
                    "pattern": f"L{layer_index:02d}.{suffix}",
                    "label": f"{module_name}_L{layer_index:02d}",
                }
            )
    if not plan:
        raise RuntimeError("no causal intervention site was found")
    return plan


def result_to_payload(result: CaptureProbeResult) -> dict:
    return {
        "sample_index": int(result.sample_index),
        "stage_names": [str(value) for value in result.stage_names],
        "metrics": {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in result.metrics.items()
        },
        "event_counts": {
            str(name): int(value) for name, value in result.event_counts.items()
        },
    }


def result_from_payload(payload: dict) -> CaptureProbeResult:
    return CaptureProbeResult(
        capture_path=Path("<causal-noise-probe-resume>"),
        sample_index=int(payload["sample_index"]),
        stage_names=[str(value) for value in payload["stage_names"]],
        metrics={
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in payload["metrics"].items()
        },
        event_counts={
            str(name): int(value)
            for name, value in payload["event_counts"].items()
        },
    )


def write_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def read_pickle(path: Path) -> Optional[dict]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def make_signature(
    checkpoint_path: Path,
    sigmas: list[float],
    sample_indices: list[int],
    intervention_plan: list[dict],
    args,
) -> dict:
    return {
        "version": LEGACY_UNIFIED_VERSION,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "sigmas": [float(value) for value in sigmas],
        "sample_indices": [int(value) for value in sample_indices],
        "analysis_seed": int(args.analysis_seed),
        "interventions": [str(value["label"]) for value in intervention_plan],
        "entity_classes": [str(value) for value in args.entity_classes],
        "minimum_track_length": int(args.minimum_track_length),
        "auc_weight": float(args.auc_weight),
        "margin_scale": float(args.margin_scale),
        "causal_sample_count": int(args.causal_sample_count),
        "no_causal": bool(args.no_causal),
    }


def resume_path(
    resume_root: Path,
    sigma: float,
    sample_index: int,
    label: str,
) -> Path:
    sigma_name = f"sigma_{int(round(float(sigma) * 1000)):04d}"
    return (
        resume_root
        / sigma_name
        / f"sample_{int(sample_index):05d}"
        / f"{label}.pkl"
    )


def save_result_resume(
    path: Path,
    signature: dict,
    role: str,
    sigma: float,
    sample_index: int,
    result: CaptureProbeResult,
    metadata: Optional[dict] = None,
) -> None:
    write_pickle_atomic(
        path,
        {
            "signature": dict(signature),
            "role": str(role),
            "sigma": float(sigma),
            "sample_index": int(sample_index),
            "metadata": dict(metadata or {}),
            "result": result_to_payload(result),
        },
    )


def load_result_resume(
    path: Path,
    signature: dict,
    expected_role: str,
    sigma: float,
    sample_index: int,
) -> Optional[tuple[CaptureProbeResult, dict]]:
    payload = read_pickle(path)
    if payload is None:
        return None
    if payload.get("signature") != signature:
        return None
    if str(payload.get("role", "")) != str(expected_role):
        return None
    if abs(float(payload.get("sigma", -999.0)) - float(sigma)) > 1e-8:
        return None
    if int(payload.get("sample_index", -1)) != int(sample_index):
        return None
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        return None
    return result_from_payload(result_payload), dict(payload.get("metadata", {}))


def check_sample_eligibility(
    controller: EntityReactorController,
    batch: dict,
    track_data: dict,
    token_height: int,
    token_width: int,
    metadata: dict,
) -> Optional[str]:
    try:
        controller.recorder.configure(
            batch,
            track_data,
            token_height=token_height,
            token_width=token_width,
            metadata=metadata,
        )
        capture = controller.recorder.build_probe_capture()
        events = build_probe_events(capture)
    except NoCrossCameraTransitionError as error:
        return str(error)
    if len(events["cross_view"]) == 0:
        return "no cross-view event with strict same-class negative"
    return None


def run_baseline(
    controller: EntityReactorController,
    pipeline,
    input_builder: ControlledModelInputBuilder,
    batch: dict,
    track_data: dict,
    latents: torch.Tensor,
    fixed_noise: torch.Tensor,
    model_conditions: dict,
    token_height: int,
    token_width: int,
    sample_index: int,
    sigma: float,
    metadata: dict,
) -> CaptureProbeResult:
    controller.gate_spec = {}
    current_metadata = dict(metadata)
    current_metadata.update(
        {
            "sigma": float(sigma),
            "capture_role": "baseline",
            "storage_policy": "memory_fullD_resume_statistics_only",
        }
    )
    controller.configure_capture(
        batch,
        track_data,
        token_height,
        token_width,
        Path("<causal-noise-probe-baseline>"),
        metadata=current_metadata,
    )
    noisy_latents = float(sigma) * fixed_noise + (1.0 - float(sigma)) * latents
    timestep_value = input_builder.timestep_for_sigma(float(sigma))
    timesteps = timestep_value.expand(*noisy_latents.shape[:3]).clone()
    with torch.no_grad():
        output = entity_reactor_forward(
            pipeline,
            noisy_latents.to(pipeline.model_dtype),
            timesteps,
            model_conditions,
        )
    del output, noisy_latents, timesteps
    capture = controller.latest_capture
    if capture is None:
        raise RuntimeError("baseline forward produced no Entity Reactor snapshot")
    result, _ = _weighted_evaluate_capture(capture, transport_stage_indices=set())
    controller.latest_capture = None
    controller.recorder.stage_features.clear()
    return result


def run_intervention(
    controller: EntityReactorController,
    pipeline,
    input_builder: ControlledModelInputBuilder,
    batch: dict,
    track_data: dict,
    latents: torch.Tensor,
    fixed_noise: torch.Tensor,
    model_conditions: dict,
    token_height: int,
    token_width: int,
    sigma: float,
    intervention: dict,
    baseline_stage_names: list[str],
    metadata: dict,
) -> tuple[CaptureProbeResult, dict]:
    save_stage_names = _resolve_intervention_save_stages(
        baseline_stage_names,
        intervention,
        controller.num_layers,
    )
    current_metadata = dict(metadata)
    current_metadata.update(
        {
            "sigma": float(sigma),
            "capture_role": "intervention",
            "storage_policy": "memory_fullD_resume_statistics_only",
            "save_stage_names": save_stage_names,
            "intervention_label": str(intervention["label"]),
            "intervention_module": str(intervention["module"]),
            "intervention_layer": int(intervention["layer"]),
            "intervention_pattern": str(intervention["pattern"]),
            "intervention_sigma": float(sigma),
        }
    )
    controller.gate_spec = {str(intervention["pattern"]): 0.0}
    controller.configure_capture(
        batch,
        track_data,
        token_height,
        token_width,
        Path("<causal-noise-probe-intervention>"),
        metadata=current_metadata,
    )
    noisy_latents = float(sigma) * fixed_noise + (1.0 - float(sigma)) * latents
    timestep_value = input_builder.timestep_for_sigma(float(sigma))
    timesteps = timestep_value.expand(*noisy_latents.shape[:3]).clone()
    try:
        with torch.no_grad():
            output = entity_reactor_forward(
                pipeline,
                noisy_latents.to(pipeline.model_dtype),
                timesteps,
                model_conditions,
            )
        del output
        capture = controller.latest_capture
        if capture is None:
            raise RuntimeError("intervention forward produced no Entity Reactor snapshot")
        result, _ = _weighted_evaluate_capture(capture, transport_stage_indices=set())
        controller.latest_capture = None
        controller.recorder.stage_features.clear()
        return result, current_metadata
    finally:
        controller.gate_spec = {}
        del noisy_latents, timesteps


def causal_matrix_from_atlas(atlas: dict) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    interventions = list(atlas.get("interventions", []))
    layers = sorted(
        {
            int(item["layer"])
            for item in interventions
            if str(item.get("module", "")) in MODULE_ORDER
            and int(item.get("layer", -1)) >= 0
        }
    )
    matrix = np.full((len(layers), len(MODULE_ORDER)), np.nan, dtype=np.float64)
    low = np.full_like(matrix, np.nan)
    counts = np.zeros_like(matrix, dtype=np.int32)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    module_index = {module: index for index, module in enumerate(MODULE_ORDER)}
    for item in interventions:
        module = str(item.get("module", ""))
        layer = int(item.get("layer", -1))
        if module not in module_index or layer not in layer_index:
            continue
        row = layer_index[layer]
        column = module_index[module]
        matrix[row, column] = float(item.get("target_use", np.nan))
        low[row, column] = float(item.get("target_use_low", np.nan))
        counts[row, column] = int(item.get("sample_count", 0))
    return layers, matrix, low, counts


def align_causal_matrices(
    sigma_results: dict[float, tuple[list[int], np.ndarray, np.ndarray, np.ndarray]],
    sigmas: list[float],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    all_layers = sorted(
        {
            int(layer)
            for sigma in sigmas
            for layer in sigma_results[float(sigma)][0]
        }
    )
    use = np.full(
        (len(sigmas), len(all_layers), len(MODULE_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    low = np.full_like(use, np.nan)
    counts = np.zeros_like(use, dtype=np.int32)
    layer_to_row = {layer: row for row, layer in enumerate(all_layers)}
    for sigma_pos, sigma in enumerate(sigmas):
        layers, matrix, matrix_low, matrix_counts = sigma_results[float(sigma)]
        for source_row, layer in enumerate(layers):
            target_row = layer_to_row[int(layer)]
            use[sigma_pos, target_row] = matrix[source_row]
            low[sigma_pos, target_row] = matrix_low[source_row]
            counts[sigma_pos, target_row] = matrix_counts[source_row]
    return all_layers, use, low, counts


def draw_causal_noise_probe(
    output_path: Path,
    sigmas: list[float],
    layers: list[int],
    use: np.ndarray,
    low: np.ndarray,
    setting_name: str,
    checkpoint_name: str,
    sample_count: int,
) -> Path:
    panel_count = len(sigmas)
    figure_height = max(5.3, 0.62 * len(layers) + 2.5)
    figure_width = max(12.0, 5.0 * panel_count)
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    # Keep the exact old Figure-2 causal scale so the three noise panels and
    # the previous sigma=0.6 figure are visually comparable. Values outside
    # the range are still printed numerically in the cells.
    limit = 0.25
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    for sigma_pos, sigma in enumerate(sigmas):
        axis = axes[0, sigma_pos]
        matrix = use[sigma_pos]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
        )
        axis.set_title(f"σ = {sigma:.1f}", fontsize=13, fontweight="bold")
        axis.set_xticks(np.arange(len(MODULE_ORDER)))
        axis.set_xticklabels(MODULE_LABELS, rotation=20, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(layers)))
        axis.set_yticklabels([f"L{layer:02d}" for layer in layers], fontsize=8)
        axis.set_xlabel("module", fontsize=9)
        if sigma_pos == 0:
            axis.set_ylabel("layer", fontsize=9)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    axis.text(
                        column,
                        row,
                        f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                lower_bound = low[sigma_pos, row, column]
                if np.isfinite(lower_bound) and lower_bound > 0.0:
                    rectangle = plt.Rectangle(
                        (column - 0.47, row - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        linewidth=1.3,
                        edgecolor="black",
                    )
                    axis.add_patch(rectangle)
    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("baseline − gated  (downstream USE)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Entity Relation Causal USE · Noise Probe",
        fontsize=20,
        fontweight="bold",
        x=0.04,
        y=0.985,
        ha="left",
    )
    figure.text(
        0.04,
        0.945,
        f"{setting_name} · {checkpoint_name} · same {sample_count} clips and same fixed noise seeds",
        fontsize=9,
        alpha=0.72,
    )
    figure.text(
        0.04,
        0.020,
        "Same definition as the old Figure 2: Camera/Condition/Cross-view → cross-view identity; Temporal → temporal identity. "
        "Positive means removing that single site hurts the next structural stage.",
        fontsize=8,
        alpha=0.72,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.93,
        top=0.89,
        bottom=0.12,
        wspace=0.28,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_final_cache(
    output_path: Path,
    sigmas: list[float],
    layers: list[int],
    use: np.ndarray,
    low: np.ndarray,
    counts: np.ndarray,
    sample_indices: list[int],
    metadata: dict,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.asarray(LEGACY_UNIFIED_VERSION),
        sigmas=np.asarray(sigmas, dtype=np.float32),
        causal_layers=np.asarray(layers, dtype=np.int16),
        causal_modules=np.asarray(MODULE_ORDER, dtype="U16"),
        causal_use=np.asarray(use, dtype=np.float32),
        causal_use_low=np.asarray(low, dtype=np.float32),
        causal_sample_count=np.asarray(counts, dtype=np.int16),
        sample_indices=np.asarray(sample_indices, dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
    )
    return output_path


def run_probe(args) -> tuple[Path, Path]:
    base_config = json.loads(args.config_path.read_text(encoding="utf-8"))
    reference_plot_cache = resolve_reference_plot_cache(args.reference_run_root)
    reference_info = load_reference_info(reference_plot_cache)
    checkpoint_path = resolve_checkpoint(
        base_config,
        args.checkpoint,
        reference_info,
    )
    sigmas = normalize_sigmas(args.sigmas)
    if not 0.0 <= float(args.auc_weight) <= 1.0:
        raise ValueError("--auc-weight must be inside [0,1]")
    if float(args.margin_scale) <= 0.0:
        raise ValueError("--margin-scale must be positive")
    global _SCORE_AUC_WEIGHT, _SCORE_MARGIN_SCALE
    _SCORE_AUC_WEIGHT = float(args.auc_weight)
    _SCORE_MARGIN_SCALE = float(args.margin_scale)

    device = setup_distributed(base_config)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        raise RuntimeError("causal-noise quick probe must run with one process")
    initialize_global_state(base_config)
    validation_dataset = dwm.common.create_instance_from_config(
        base_config["validation_dataset"]
    )
    track_resolver = NuPlanTrackResolver(
        validation_dataset,
        max_entities=None,
        allowed_classes=args.entity_classes,
        minimum_track_length=args.minimum_track_length,
    )
    initial_samples = resolve_initial_samples(
        args,
        reference_info,
        len(validation_dataset),
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    pipeline, config = instantiate_pipeline(
        base_config,
        checkpoint_path,
        args.output_path,
        device,
    )
    model_kind = detect_model_kind(config)
    input_builder = ControlledModelInputBuilder(
        pipeline,
        model_kind,
        int(args.analysis_seed),
    )
    controller = EntityReactorController(
        capture_patterns=args.capture_pattern,
        gate_spec={},
        projection_dim=64,
        projection_seed=20260815,
        max_cross_camera_tracks=int(args.max_entities),
    )
    controller.attach(pipeline)
    intervention_plan = build_intervention_plan(controller, args.layers)

    checkpoint_name = checkpoint_output_name(checkpoint_path)
    setting_name = args.output_path.name
    design_summary = summarize_design_config(config, setting_name)
    run_root = args.output_path / checkpoint_name
    run_root.mkdir(parents=True, exist_ok=True)
    selected_samples = list(initial_samples)
    target_count = len(initial_samples)
    seen_samples = set(selected_samples)
    next_backfill = (
        int(args.start_index) + int(args.sample_count) * int(args.sample_stride)
    )
    using_reference_samples = bool(reference_info.get("sample_indices"))
    signature = make_signature(
        checkpoint_path,
        sigmas,
        selected_samples,
        intervention_plan,
        args,
    )
    resume_root = run_root / "_resume"

    print(
        f"[CausalNoiseProbe] {LEGACY_UNIFIED_VERSION}",
        flush=True,
    )
    print(
        f"[CausalNoiseProbe] checkpoint={checkpoint_path}",
        flush=True,
    )
    print(
        f"[CausalNoiseProbe] sigmas={sigmas} samples={selected_samples}",
        flush=True,
    )
    print(
        "[UnifiedLegacy] full-D baseline + multi-noise causal; raw full-D features are never persisted",
        flush=True,
    )

    baseline_results: dict[float, dict[int, CaptureProbeResult]] = {
        sigma: {} for sigma in sigmas
    }
    intervention_results: dict[float, dict[str, dict]] = {
        sigma: {} for sigma in sigmas
    }
    accepted_samples: list[int] = []
    sample_position = 0

    try:
        while len(accepted_samples) < target_count:
            if sample_position >= len(selected_samples):
                if using_reference_samples:
                    raise RuntimeError(
                        "a reference-run sample unexpectedly became invalid; "
                        "increase --sample-count only after checking the dataset/config is unchanged"
                    )
                candidate = int(next_backfill)
                while candidate in seen_samples:
                    candidate += int(args.sample_stride)
                if candidate >= len(validation_dataset):
                    raise RuntimeError(
                        "validation dataset ended before enough valid clips were found"
                    )
                selected_samples.append(candidate)
                seen_samples.add(candidate)
                next_backfill = candidate + int(args.sample_stride)
            sample_index = int(selected_samples[sample_position])
            sample_position += 1
            print(
                f"[CausalNoiseProbe] sample={sample_index} accepted={len(accepted_samples)}/{target_count}",
                flush=True,
            )

            item = validation_dataset[sample_index]
            batch = torch.utils.data.default_collate([item])
            if "clip_text" in item:
                batch["clip_text"] = [item["clip_text"]]
            track_data = track_resolver.resolve(sample_index)
            if "lidar_to_camera" not in batch:
                batch["lidar_to_camera"] = track_data["lidar_to_camera"].unsqueeze(0)
            with torch.no_grad():
                latents = input_builder.encode_real_video(batch)
                fixed_noise = input_builder.make_noise(latents.shape, sample_index)
                model_conditions = input_builder.prepare_conditions(batch, latents)
            if model_kind == "pv":
                _validate_pv_conditions(latents, model_conditions)

            patch_size = int(pipeline.model.config.patch_size)
            token_height = int(latents.shape[-2] // patch_size)
            token_width = int(latents.shape[-1] // patch_size)
            sample_metadata = {
                "model_name": config["pipeline"]["model"]["_class_name"].split(".")[-1],
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "sample_index": int(sample_index),
                "sequence_length": int(latents.shape[1]),
                "view_count": int(latents.shape[2]),
                "scene": track_data.get("scene", ""),
                "track_tokens": track_data.get("track_tokens", []),
                "causal_noise_probe_version": LEGACY_UNIFIED_VERSION,
            }
            sample_metadata.update(design_summary)

            rejected_reason = check_sample_eligibility(
                controller,
                batch,
                track_data,
                token_height,
                token_width,
                sample_metadata,
            )
            if rejected_reason is not None:
                print(
                    f"[CausalNoiseProbe] reject sample={sample_index}: {rejected_reason}",
                    flush=True,
                )
                if using_reference_samples:
                    raise RuntimeError(
                        f"reference sample {sample_index} is no longer eligible: {rejected_reason}"
                    )
                del batch, latents, fixed_noise, model_conditions, track_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for sigma in sigmas:
                baseline_file = resume_path(
                    resume_root,
                    sigma,
                    sample_index,
                    "baseline",
                )
                loaded_baseline = load_result_resume(
                    baseline_file,
                    signature,
                    "baseline",
                    sigma,
                    sample_index,
                )
                if loaded_baseline is None:
                    baseline_result = run_baseline(
                        controller,
                        pipeline,
                        input_builder,
                        batch,
                        track_data,
                        latents,
                        fixed_noise,
                        model_conditions,
                        token_height,
                        token_width,
                        sample_index,
                        sigma,
                        sample_metadata,
                    )
                    save_result_resume(
                        baseline_file,
                        signature,
                        "baseline",
                        sigma,
                        sample_index,
                        baseline_result,
                    )
                    print(
                        f"[CausalNoiseProbe] baseline sample={sample_index} sigma={sigma:.1f} committed",
                        flush=True,
                    )
                else:
                    baseline_result = loaded_baseline[0]
                    print(
                        f"[CausalNoiseProbe] resume baseline sample={sample_index} sigma={sigma:.1f}",
                        flush=True,
                    )
                baseline_results[float(sigma)][sample_index] = baseline_result

                causal_limit = int(args.causal_sample_count)
                do_causal_for_sample = (
                    not args.no_causal
                    and (causal_limit == 0 or len(accepted_samples) < causal_limit)
                )
                if do_causal_for_sample:
                    for intervention in intervention_plan:
                        label = str(intervention["label"])
                        intervention_file = resume_path(
                            resume_root,
                            sigma,
                            sample_index,
                            label,
                        )
                        loaded_intervention = load_result_resume(
                            intervention_file,
                            signature,
                            "intervention",
                            sigma,
                            sample_index,
                        )
                        if loaded_intervention is None:
                            intervention_result, intervention_metadata = run_intervention(
                                controller,
                                pipeline,
                                input_builder,
                                batch,
                                track_data,
                                latents,
                                fixed_noise,
                                model_conditions,
                                token_height,
                                token_width,
                                sigma,
                                intervention,
                                list(baseline_result.stage_names),
                                sample_metadata,
                            )
                            save_result_resume(
                                intervention_file,
                                signature,
                                "intervention",
                                sigma,
                                sample_index,
                                intervention_result,
                                metadata=intervention_metadata,
                            )
                            print(
                                f"[UnifiedLegacy] sigma={sigma:.1f} sample={sample_index} {label} committed",
                                flush=True,
                            )
                        else:
                            intervention_result, intervention_metadata = loaded_intervention
                        entry = intervention_results[float(sigma)].setdefault(
                            label,
                            {
                                "metadata": dict(intervention_metadata),
                                "results": {},
                            },
                        )
                        entry["results"][sample_index] = intervention_result

            accepted_samples.append(sample_index)
            del batch, latents, fixed_noise, model_conditions, track_data
            controller.latest_capture = None
            controller.recorder.stage_features.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        baseline_atlas = build_runtime_atlas(
            baseline_results,
            {},
            {},
            metadata={
                **design_summary,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "version": LEGACY_UNIFIED_VERSION,
                "auc_weight": float(args.auc_weight),
                "margin_scale": float(args.margin_scale),
            },
            num_layers=controller.num_layers,
            seed=int(args.analysis_seed),
        )
        output_indices = psi_module.layer_output_indices(baseline_atlas["stage_descriptors"])
        sigmas_from_atlas, baseline_cache = _bootstrap_matrix_from_atlas(
            baseline_atlas,
            output_indices,
        )
        requested_sigmas = [float(value) for value in sigmas]
        if sigmas_from_atlas != requested_sigmas:
            source_pos = {round(float(value), 8): idx for idx, value in enumerate(sigmas_from_atlas)}
            order = [source_pos[round(float(value), 8)] for value in requested_sigmas]
            for family_payload in baseline_cache.values():
                for metric_payload in family_payload.values():
                    for stat_name, matrix in list(metric_payload.items()):
                        metric_payload[stat_name] = np.asarray(matrix)[order]
            sigmas_from_atlas = requested_sigmas
        all_interventions = []
        if not args.no_causal:
            for sigma in sigmas:
                sigma_atlas = build_runtime_atlas(
                    {float(sigma): baseline_results[float(sigma)]},
                    {},
                    intervention_results[float(sigma)],
                    metadata={
                        **design_summary,
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_path": str(checkpoint_path),
                        "version": LEGACY_UNIFIED_VERSION,
                        "sigma": float(sigma),
                    },
                    num_layers=controller.num_layers,
                    seed=int(args.analysis_seed) + int(round(float(sigma) * 1000)),
                )
                all_interventions.extend(sigma_atlas.get("interventions", []))
        causal_cache = _causal_cache_from_atlas(
            {"interventions": all_interventions},
            sigmas_from_atlas,
        )
        causal_limit = int(args.causal_sample_count)
        causal_sample_indices = (
            list(accepted_samples)
            if causal_limit == 0
            else list(accepted_samples[:causal_limit])
        ) if not args.no_causal else []
        unified_cache = {
            "version": LEGACY_UNIFIED_VERSION,
            "backend": "legacy",
            "sigmas": sigmas_from_atlas,
            "stage_names": [baseline_atlas["stage_names"][idx] for idx in output_indices],
            "stage_depths": [baseline_atlas["stage_descriptors"][idx].depth for idx in output_indices],
            "sample_indices": list(accepted_samples),
            "causal_sample_indices": causal_sample_indices,
            "baseline": baseline_cache,
            "causal": causal_cache,
            "metadata": {
                **design_summary,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "auc_weight": float(args.auc_weight),
                "margin_scale": float(args.margin_scale),
                "hidden_dimension": "full",
                "hard_negative_policy": "strict same-class only",
                "score_formula": "w*(2*AUC-1)+(1-w)*tanh(margin/scale)",
            },
        }
        cache_path = save_unified_npz(
            unified_cache,
            run_root / "entity_relation_unified_data.npz",
        )
        text_path = write_scores_text(
            unified_cache,
            run_root / "entity_relation_scores.txt",
        )
        overview_path = run_root / "entity_relation_overview_clean.png"
        causal_path = run_root / "entity_relation_causal_use.png"
        if not args.no_render:
            render_overview(unified_cache, overview_path)
            if not args.no_causal:
                render_causal(unified_cache, causal_path)
        manifest = {
            "version": LEGACY_UNIFIED_VERSION,
            "backend": "legacy",
            "cache": cache_path.name,
            "scores_text": text_path.name,
            "overview_figure": overview_path.name if not args.no_render else None,
            "causal_figure": causal_path.name if (not args.no_render and not args.no_causal) else None,
            "sigmas": sigmas_from_atlas,
            "sample_indices": accepted_samples,
            "causal_sample_indices": causal_sample_indices,
            "checkpoint_path": str(checkpoint_path),
            "auc_weight": float(args.auc_weight),
            "margin_scale": float(args.margin_scale),
        }
        (run_root / "entity_relation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[UnifiedLegacy] overview -> {overview_path}", flush=True)
        if not args.no_causal:
            print(f"[UnifiedLegacy] causal   -> {causal_path}", flush=True)
        print(f"[UnifiedLegacy] scores   -> {text_path}", flush=True)
        print(f"[UnifiedLegacy] cache    -> {cache_path}", flush=True)
        if resume_root.is_dir() and not args.keep_resume:
            shutil.rmtree(resume_root)
        return overview_path, cache_path
    finally:
        controller.gate_spec = {}
        controller.detach()
        del controller, input_builder, pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None) -> None:
    args = create_parser().parse_args(argv)
    run_probe(args)


if __name__ == "__main__":
    main()
