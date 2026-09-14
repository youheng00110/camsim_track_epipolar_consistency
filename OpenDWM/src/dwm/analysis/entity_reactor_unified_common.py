from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

UNIFIED_VERSION = "v20.0-unified-score55-multinoise-20260819"
DEFAULT_SIGMAS = (0.9, 0.6, 0.3)
DEFAULT_AUC_WEIGHT = 0.55
DEFAULT_MARGIN_SCALE = 0.12


def composite_score(auc, margin, auc_weight: float = DEFAULT_AUC_WEIGHT, margin_scale: float = DEFAULT_MARGIN_SCALE):
    auc_weight = float(auc_weight)
    margin_weight = 1.0 - auc_weight
    if not 0.0 <= auc_weight <= 1.0:
        raise ValueError(f"auc_weight must be in [0,1], got {auc_weight}")
    if float(margin_scale) <= 0.0:
        raise ValueError(f"margin_scale must be positive, got {margin_scale}")
    return np.clip(
        auc_weight * (2.0 * np.asarray(auc) - 1.0)
        + margin_weight * np.tanh(np.asarray(margin) / float(margin_scale)),
        -1.0,
        1.0,
    )


def normalize_sigmas(values: Iterable[float]) -> list[float]:
    output = []
    seen = set()
    for value in values:
        sigma = float(value)
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"sigma must be in [0,1], got {sigma}")
        key = round(sigma, 8)
        if key in seen:
            continue
        seen.add(key)
        output.append(sigma)
    if not output:
        raise ValueError("at least one sigma is required")
    return output


def sigma_key(value: float) -> str:
    return f"sigma_{int(round(float(value) * 1000)):04d}"


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_unified_npz(cache: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = cache["baseline"]
    causal = cache["causal"]
    payload = {
        "unified_version": np.asarray(str(cache.get("version", UNIFIED_VERSION))),
        "backend": np.asarray(str(cache.get("backend", "unknown"))),
        "sigmas": np.asarray(cache["sigmas"], dtype=np.float32),
        "stage_names": np.asarray(cache["stage_names"], dtype="U64"),
        "stage_depths": np.asarray(cache.get("stage_depths", []), dtype=np.float32),
        "sample_indices": np.asarray(cache.get("sample_indices", []), dtype=np.int32),
        "causal_sample_indices": np.asarray(cache.get("causal_sample_indices", []), dtype=np.int32),
        "causal_layers": np.asarray(causal.get("layers", []), dtype=np.int16),
        "causal_modules": np.asarray(causal.get("modules", []), dtype="U16"),
        "causal_use": np.asarray(causal.get("use", []), dtype=np.float32),
        "causal_retain": np.asarray(causal.get("retain", []), dtype=np.float32),
        "causal_write": np.asarray(causal.get("write", []), dtype=np.float32),
        "causal_count": np.asarray(causal.get("count", []), dtype=np.int16),
        "metadata_json": np.asarray(json.dumps(cache.get("metadata", {}), ensure_ascii=False, sort_keys=True)),
    }
    for family, family_payload in baseline.items():
        for metric, metric_payload in family_payload.items():
            for stat_name in ("mean", "low", "high", "count"):
                if stat_name in metric_payload:
                    dtype = np.int16 if stat_name == "count" else np.float32
                    payload[f"baseline__{family}__{metric}__{stat_name}"] = np.asarray(
                        metric_payload[stat_name], dtype=dtype
                    )
    np.savez_compressed(output_path, **payload)
    return output_path


def write_scores_text(cache: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache.get("metadata", {})
    lines = []
    lines.append(f"Entity Reactor Unified Scores  {cache.get('version', UNIFIED_VERSION)}")
    lines.append(f"backend = {cache.get('backend', 'unknown')}")
    lines.append(f"checkpoint = {metadata.get('checkpoint_path', metadata.get('checkpoint', 'unknown'))}")
    lines.append(f"score = auc_weight*(2*AUC-1) + (1-auc_weight)*tanh(margin/margin_scale)")
    lines.append(f"auc_weight = {float(metadata.get('auc_weight', DEFAULT_AUC_WEIGHT)):.4f}")
    lines.append(f"margin_weight = {1.0 - float(metadata.get('auc_weight', DEFAULT_AUC_WEIGHT)):.4f}")
    lines.append(f"margin_scale = {float(metadata.get('margin_scale', DEFAULT_MARGIN_SCALE)):.6f}")
    lines.append("structured_similarity = 0.35*center_cosine + 0.65*corner_relation_cosine")
    lines.append("hard_negatives = same-view same-class only; priority exact-time -> nearby-time -> same-class")
    lines.append(f"sigmas = {', '.join(f'{float(v):.3f}' for v in cache['sigmas'])}")
    lines.append(f"baseline_samples = {len(cache.get('sample_indices', []))}")
    lines.append(f"causal_samples = {len(cache.get('causal_sample_indices', []))}")
    lines.append("")
    lines.append("[BASELINE REPRESENTATION STATE]")
    lines.append("sigma\tstage\tfamily\tscore\tauc\tmargin\ttop1\tcount")
    stages = list(cache["stage_names"])
    for sigma_pos, sigma in enumerate(cache["sigmas"]):
        for stage_pos, stage in enumerate(stages):
            for family in ("cross_view", "temporal", "mixed"):
                family_payload = cache["baseline"].get(family, {})
                values = []
                for metric in ("score", "auc", "margin", "top1"):
                    matrix = family_payload.get(metric, {}).get("mean")
                    value = float(matrix[sigma_pos, stage_pos]) if matrix is not None else float("nan")
                    values.append(value)
                count_matrix = family_payload.get("score", {}).get("count")
                count = int(count_matrix[sigma_pos, stage_pos]) if count_matrix is not None else 0
                lines.append(
                    f"{float(sigma):.3f}\t{stage}\t{family}\t"
                    f"{values[0]:.6f}\t{values[1]:.6f}\t{values[2]:.6f}\t{values[3]:.6f}\t{count}"
                )
    lines.append("")
    lines.append("[CAUSAL SINGLE-SITE EFFECT]")
    lines.append("sigma\tlayer\tmodule\tUSE\tRETAIN\tWRITE\tcount")
    causal = cache["causal"]
    use = np.asarray(causal.get("use", []))
    retain = np.asarray(causal.get("retain", []))
    write = np.asarray(causal.get("write", []))
    count = np.asarray(causal.get("count", []))
    layers = list(causal.get("layers", []))
    modules = list(causal.get("modules", []))
    if use.ndim == 3:
        for sigma_pos, sigma in enumerate(cache["sigmas"]):
            for layer_pos, layer in enumerate(layers):
                for module_pos, module in enumerate(modules):
                    u = float(use[sigma_pos, layer_pos, module_pos])
                    r = float(retain[sigma_pos, layer_pos, module_pos]) if retain.size else float("nan")
                    w = float(write[sigma_pos, layer_pos, module_pos]) if write.size else float("nan")
                    n = int(count[sigma_pos, layer_pos, module_pos]) if count.size else 0
                    lines.append(f"{float(sigma):.3f}\t{int(layer)}\t{module}\t{u:.6f}\t{r:.6f}\t{w:.6f}\t{n}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def render_overview(cache: dict, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    sigmas = list(cache["sigmas"])
    stages = list(cache["stage_names"])
    families = ("cross_view", "temporal", "mixed")
    titles = {
        "cross_view": "Cross-view entity relation",
        "temporal": "Temporal entity relation",
        "mixed": "View + time entity relation",
    }
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    for axis, family in zip(np.ravel(axes), families):
        matrix = np.asarray(cache["baseline"][family]["score"]["mean"], dtype=np.float32)
        image = axis.imshow(matrix, aspect="auto", vmin=-1.0, vmax=1.0)
        axis.set_title(titles[family])
        axis.set_yticks(np.arange(len(sigmas)))
        axis.set_yticklabels([f"{float(value):.1f}" for value in sigmas])
        axis.set_xticks(np.arange(len(stages)))
        axis.set_xticklabels(stages, rotation=75, ha="right", fontsize=7)
        axis.set_ylabel("noise sigma")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Entity relation representation state", fontsize=15)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_causal(cache: dict, output_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sigmas = list(cache["sigmas"])
    causal = cache["causal"]
    layers = list(causal.get("layers", []))
    modules = list(causal.get("modules", []))
    use = np.asarray(causal.get("use", []), dtype=np.float32)
    panel_count = max(len(sigmas), 1)
    figure, axes = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 6.0), squeeze=False, constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
    image = None
    for sigma_pos, sigma in enumerate(sigmas):
        axis = axes[0, sigma_pos]
        matrix = use[sigma_pos] if use.ndim == 3 else np.empty((0, 0), dtype=np.float32)
        if matrix.size:
            image = axis.imshow(matrix, aspect="auto", norm=norm)
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = float(matrix[row, col])
                    if np.isfinite(value):
                        axis.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=8)
        axis.set_title(f"σ = {float(sigma):.1f}")
        axis.set_xticks(np.arange(len(modules)))
        axis.set_xticklabels(modules, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(layers)))
        axis.set_yticklabels([f"L{int(layer):02d}" for layer in layers])
        if sigma_pos == 0:
            axis.set_ylabel("intervention layer")
        axis.set_xlabel("single-site module ablation")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Causal USE  baseline score − ablated score")
    figure.suptitle("Entity relation causal USE across noise levels", fontsize=15)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path
