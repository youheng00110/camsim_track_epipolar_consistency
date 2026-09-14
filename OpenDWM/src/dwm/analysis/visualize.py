from __future__ import annotations

ENTITY_REACTOR_VIS_VERSION = "v19.7-camera-separated-two-figures-20260817"

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from dwm.analysis.psi import (
    ENTITY_REACTOR_PLOT_CACHE_VERSION,
    analyze_binding_run,
    write_binding_tables,
)

MODULE_ORDER = ("camera", "condition", "temporal", "crossview")
MODULE_LABELS = ("Camera", "Condition", "Temporal", "Cross-view")


def safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def plot_cache_path(run_root: Path) -> Path:
    return Path(run_root) / "relation_plot_data.npz"


def plot_manifest_path(run_root: Path) -> Path:
    return Path(run_root) / "relation_plot_manifest.json"


def _read_cache_version(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "plot_cache_version" not in data:
                return ""
            return str(data["plot_cache_version"].item())
    except (OSError, ValueError, KeyError, EOFError):
        return ""


def ensure_cached_analysis(run_root: Path) -> None:
    run_root = Path(run_root)
    cache_path = plot_cache_path(run_root)
    manifest_path = plot_manifest_path(run_root)
    all_capture_files = [path for path in run_root.rglob("*.npz") if path.is_file() and path.name != cache_path.name]
    baseline_capture_files = [
        path
        for path in all_capture_files
        if path.name == "capture.npz" and "intervention" not in path.parts
    ]

    cache_ready = cache_path.is_file() and cache_path.stat().st_size > 0
    cache_version_ok = cache_ready and _read_cache_version(cache_path) == ENTITY_REACTOR_PLOT_CACHE_VERSION
    cache_fresh = False
    if cache_version_ok:
        if not all_capture_files:
            cache_fresh = True
        else:
            newest_capture = max(path.stat().st_mtime_ns for path in all_capture_files)
            cache_fresh = cache_path.stat().st_mtime_ns >= newest_capture

    if cache_version_ok and cache_fresh:
        print(f"[EntityReactor] compact plot cache -> {cache_path}", flush=True)
        return

    if not baseline_capture_files:
        reason = "missing" if not cache_ready else "stale/incompatible"
        raise RuntimeError(
            f"relation_plot_data.npz is {reason} under {run_root}, and no baseline capture.npz "
            "is available to rebuild it."
        )

    reason = "missing" if not cache_ready else "stale/incompatible"
    print(
        f"[EntityReactor] compact plot cache {reason} -> recomputing metrics from "
        f"{len(baseline_capture_files)} baseline captures",
        flush=True,
    )
    atlas = analyze_binding_run(run_root)
    write_binding_tables(atlas, run_root)
    if not cache_path.is_file() or _read_cache_version(cache_path) != ENTITY_REACTOR_PLOT_CACHE_VERSION:
        raise RuntimeError(f"failed to rebuild compatible plot cache under {run_root}")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_mib = safe_float(manifest.get("cache_mib"))
            per_sample_mib = safe_float(manifest.get("mib_per_sample_equivalent"))
            print(
                f"[EntityReactor] plot cache size={cache_mib:.3f} MiB, "
                f"equivalent={per_sample_mib:.4f} MiB/sample",
                flush=True,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def discover_run_roots(input_root: Path) -> list[Path]:
    input_root = Path(input_root)
    roots = set()
    for cache_path in input_root.rglob("relation_plot_data.npz"):
        roots.add(cache_path.parent)
    for capture_path in input_root.rglob("capture.npz"):
        if "intervention" in capture_path.parts:
            continue
        sample_root = capture_path.parent.parent
        if sample_root.name.startswith("sample_"):
            roots.add(sample_root.parent)
    unique = sorted(path for path in roots if path.is_dir())
    if not unique:
        raise RuntimeError(
            f"No Entity Reactor run found under {input_root}. Expected relation_plot_data.npz "
            "or baseline sample_*/sigma_*/capture.npz files."
        )
    return unique


def _decode_json_scalar(value: np.ndarray) -> dict:
    text = str(value.item())
    result = json.loads(text)
    if not isinstance(result, dict):
        raise RuntimeError("cached JSON scalar must contain an object")
    return result


def load_cached_run(run_root: Path) -> dict:
    run_root = Path(run_root)
    ensure_cached_analysis(run_root)
    cache_path = plot_cache_path(run_root)
    with np.load(cache_path, allow_pickle=False) as data:
        required = {
            "sigmas",
            "stage_names",
            "stage_depths",
            "signature_json",
            "metric__cross_view_structured__mean",
            "metric__cross_view_structured__low",
            "metric__temporal_structured__mean",
            "metric__temporal_structured__low",
            "metric__mixed_structured__mean",
            "metric__mixed_structured__low",
            "transport__cross_view__mean",
            "transport__cross_view__low",
            "causal_layers",
            "causal_modules",
            "causal_use",
            "causal_use_low",
        }
        missing = sorted(required.difference(set(data.files)))
        if missing:
            raise RuntimeError(f"compact plot cache is missing arrays {missing}")

        stage_names = data["stage_names"].astype(str).tolist()
        labels = []
        for stage in stage_names:
            if str(stage).lower() == "final":
                labels.append("Final")
            else:
                prefix = str(stage).split(".", 1)[0]
                labels.append(prefix if prefix.startswith("L") else str(stage))

        causal_modules = data["causal_modules"].astype(str).tolist()
        causal_matrix = np.asarray(data["causal_use"], dtype=np.float64)
        causal_low = np.asarray(data["causal_use_low"], dtype=np.float64)
        if causal_modules != list(MODULE_ORDER):
            reordered = np.full((causal_matrix.shape[0], len(MODULE_ORDER)), np.nan, dtype=np.float64)
            reordered_low = np.full_like(reordered, np.nan)
            index_by_module = {module: index for index, module in enumerate(causal_modules)}
            for target_index, module in enumerate(MODULE_ORDER):
                source_index = index_by_module.get(module, None)
                if source_index is None:
                    continue
                reordered[:, target_index] = causal_matrix[:, source_index]
                reordered_low[:, target_index] = causal_low[:, source_index]
            causal_matrix = reordered
            causal_low = reordered_low

        return {
            "run_root": run_root,
            "signature": _decode_json_scalar(data["signature_json"]),
            "sigmas": np.asarray(data["sigmas"], dtype=np.float64),
            "stage_names": stage_names,
            "stage_labels": labels,
            "stage_depths": np.asarray(data["stage_depths"], dtype=np.float64),
            "cross_view_mean": np.asarray(data["metric__cross_view_structured__mean"], dtype=np.float64),
            "cross_view_low": np.asarray(data["metric__cross_view_structured__low"], dtype=np.float64),
            "temporal_mean": np.asarray(data["metric__temporal_structured__mean"], dtype=np.float64),
            "temporal_low": np.asarray(data["metric__temporal_structured__low"], dtype=np.float64),
            "mixed_mean": np.asarray(data["metric__mixed_structured__mean"], dtype=np.float64),
            "mixed_low": np.asarray(data["metric__mixed_structured__low"], dtype=np.float64),
            "transport_mean": np.asarray(data["transport__cross_view__mean"], dtype=np.float64),
            "transport_low": np.asarray(data["transport__cross_view__low"], dtype=np.float64),
            "causal_layers": np.asarray(data["causal_layers"], dtype=np.int64).tolist(),
            "causal_use": causal_matrix,
            "causal_use_low": causal_low,
        }


def draw_heatmap(
    axis,
    values: np.ndarray,
    low: Optional[np.ndarray],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    color_label: str,
) -> None:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_xticks(np.arange(len(x_labels)))
    axis.set_xticklabels(list(x_labels), fontsize=8)
    axis.set_yticks(np.arange(len(y_labels)))
    axis.set_yticklabels(list(y_labels), fontsize=8)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            if low is not None:
                lower_bound = low[row_index, column_index]
                if np.isfinite(lower_bound) and lower_bound > 0.0:
                    rectangle = plt.Rectangle(
                        (column_index - 0.47, row_index - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        linewidth=1.4,
                        edgecolor="black",
                    )
                    axis.add_patch(rectangle)

    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label(color_label, fontsize=8)


def draw_profile(axis, run_data: dict) -> None:
    sigmas = run_data["sigmas"]
    sigma_index = int(np.argmin(np.abs(sigmas - 0.60)))
    depth = run_data["stage_depths"]
    curves = (
        (run_data["cross_view_mean"][sigma_index], "Cross-view transition identity", "tab:blue", "o", "-"),
        (run_data["temporal_mean"][sigma_index], "Temporal identity", "tab:orange", "o", "-"),
        (run_data["mixed_mean"][sigma_index], "View + time identity", "tab:green", "o", "-"),
        (run_data["transport_mean"][sigma_index], "Cross-view transport", "tab:red", "s", "--"),
    )
    axis.axhline(0.0, linewidth=0.8, alpha=0.40)
    for values, label, color, marker, line_style in curves:
        if values.size != depth.size:
            continue
        axis.plot(
            depth,
            values,
            linewidth=2.0,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=label,
        )
    axis.set_title("E  Relation profile at σ ≈ 0.6", loc="left", fontsize=12, fontweight="bold")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.30, 0.80)
    axis.set_xlabel("normalized network depth", fontsize=8)
    axis.set_ylabel("probe score", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=7, loc="best")


def draw_summary(axis, signature: dict) -> None:
    axis.axis("off")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.text(0.0, 0.96, "Mechanism summary", fontsize=12, fontweight="bold", va="top")
    axis.text(0.0, 0.82, str(signature.get("pattern", "unknown")), fontsize=14, fontweight="bold", va="top")
    rows = (
        ("Immediate writer", signature.get("writer", "none")),
        ("Downstream-needed", signature.get("necessary_module", "none")),
        ("Peak cross-view", signature.get("peak_cross_view_binding", float("nan"))),
        ("Formation depth", signature.get("formation_depth", float("nan"))),
        ("Low-noise final", signature.get("low_noise_final_cross_view", float("nan"))),
        ("Peak mixed", signature.get("peak_mixed_binding", float("nan"))),
        ("Peak transport", signature.get("peak_cross_view_transport", float("nan"))),
    )
    start_y = 0.68
    for index, item in enumerate(rows):
        label, value = item
        y = start_y - index * 0.09
        axis.text(0.0, y, label, fontsize=8, va="center")
        if isinstance(value, str):
            display = value
        else:
            number = safe_float(value)
            display = "n/a" if not np.isfinite(number) else f"{number:.2f}"
        axis.text(0.98, y, display, fontsize=9, fontweight="bold", ha="right", va="center")
        axis.plot([0.0, 0.98], [y - 0.04, y - 0.04], linewidth=0.5, alpha=0.20)


def render_overview(run_data: dict, output_path: Path) -> Path:
    sigmas = run_data["sigmas"]
    x_labels = run_data["stage_labels"]
    y_labels = [f"σ {value:.1f}" for value in sigmas]

    figure = plt.figure(figsize=(16.5, 11.0), constrained_layout=False)
    grid = GridSpec(
        3,
        2,
        figure=figure,
        height_ratios=(1.0, 1.0, 0.92),
        width_ratios=(1.0, 1.0),
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.07,
        hspace=0.38,
        wspace=0.24,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axis_e = figure.add_subplot(grid[2, 0])
    axis_f = figure.add_subplot(grid[2, 1])

    draw_heatmap(
        axis_a,
        run_data["cross_view_mean"],
        run_data["cross_view_low"],
        x_labels,
        y_labels,
        "A  Cross-view transition identity",
        -0.25,
        0.75,
        "RdYlGn",
        "identity separation",
    )
    draw_heatmap(
        axis_b,
        run_data["temporal_mean"],
        run_data["temporal_low"],
        x_labels,
        y_labels,
        "B  Same entity · different time",
        -0.25,
        0.75,
        "RdYlGn",
        "identity separation",
    )
    draw_heatmap(
        axis_c,
        run_data["mixed_mean"],
        run_data["mixed_low"],
        x_labels,
        y_labels,
        "C  Same entity · view + time change",
        -0.25,
        0.75,
        "RdYlGn",
        "identity separation",
    )
    draw_heatmap(
        axis_d,
        run_data["transport_mean"],
        run_data["transport_low"],
        x_labels,
        y_labels,
        "D  Cross-view geometry transport",
        -0.10,
        0.45,
        "BrBG",
        "kNN alignment",
    )
    draw_profile(axis_e, run_data)
    draw_summary(axis_f, run_data["signature"])

    signature = run_data["signature"]
    figure.suptitle("Entity Relation Mechanism Atlas", fontsize=22, fontweight="bold", x=0.06, y=0.965, ha="left")
    figure.text(
        0.06,
        0.925,
        f"{signature.get('setting_name', 'unknown')}  ·  checkpoint {signature.get('checkpoint_name', 'unknown')}  ·  compact cached analysis",
        fontsize=9,
        alpha=0.70,
    )
    figure.text(
        0.06,
        0.03,
        "A–C show relation decodability across layer × noise. D shows geometry-consistent transport. Camera causality is separated from Cross-view in the second figure.",
        fontsize=8,
        alpha=0.64,
    )
    figure.text(0.98, 0.03, ENTITY_REACTOR_VIS_VERSION, fontsize=7, alpha=0.45, ha="right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_causal_use(run_data: dict, output_path: Path) -> Path:
    matrix = run_data["causal_use"]
    low = run_data["causal_use_low"]
    layers = run_data["causal_layers"]
    if matrix.shape != (len(layers), len(MODULE_ORDER)):
        raise RuntimeError(
            f"causal matrix shape {matrix.shape} does not match layers={len(layers)} modules={len(MODULE_ORDER)}"
        )

    figure_height = max(5.3, 0.62 * len(layers) + 2.5)
    figure = plt.figure(figsize=(10.2, figure_height), constrained_layout=False)
    axis = figure.add_axes([0.14, 0.16, 0.72, 0.70])
    draw_heatmap(
        axis,
        matrix,
        low,
        MODULE_LABELS,
        [f"L{layer:02d}" for layer in layers],
        "Downstream USE of intended relation",
        -0.25,
        0.25,
        "RdBu_r",
        "baseline − gated",
    )
    axis.set_xlabel("module", fontsize=9)
    axis.set_ylabel("layer", fontsize=9)

    signature = run_data["signature"]
    figure.suptitle("Entity Relation Causal USE", fontsize=20, fontweight="bold", x=0.10, y=0.97, ha="left")
    figure.text(
        0.10,
        0.92,
        f"{signature.get('setting_name', 'unknown')}  ·  checkpoint {signature.get('checkpoint_name', 'unknown')}  ·  compact cached interventions",
        fontsize=9,
        alpha=0.70,
    )
    figure.text(
        0.10,
        0.070,
        "Target relation  Camera/Condition/Cross-view → cross-view identity   ·   Temporal → temporal identity",
        fontsize=8.5,
        alpha=0.72,
    )
    figure.text(
        0.10,
        0.040,
        "Camera gate removes only the camera embedding at that consumer layer; the Cross-view/Temporal module itself stays enabled. Positive means gating hurts the next structural stage.",
        fontsize=8.2,
        alpha=0.72,
    )
    figure.text(0.98, 0.040, ENTITY_REACTOR_VIS_VERSION, fontsize=7, alpha=0.45, ha="right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_model_reactor(
    run_root: Path,
    output_dir: Optional[Path] = None,
    preferred_sigma: float = 0.6,
) -> tuple[Path, Path]:
    del preferred_sigma
    run_data = load_cached_run(run_root)
    if output_dir is None:
        output_dir = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    causal_path = render_causal_use(run_data, output_dir / "entity_relation_causal_use.png")
    overview_path = render_overview(run_data, output_dir / "entity_relation_overview_clean.png")
    return causal_path, overview_path


def render_design_comparison(
    run_roots: Sequence[Path],
    output_path: Path,
) -> Optional[Path]:
    del run_roots
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return None
