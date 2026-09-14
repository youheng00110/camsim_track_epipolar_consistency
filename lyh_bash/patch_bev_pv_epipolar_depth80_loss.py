#!/usr/bin/env python3
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/camsim/OpenDWM"
)

UTILS = ROOT / "src/dwm/utils/view_consistency.py"
CONFIG_DIR = ROOT / "configs/lyh"

CONFIGS = [
    (CONFIG_DIR / "bev_pv_epipolar.json", False),
    (CONFIG_DIR / "PV_track_train_epipolar.json", False),
    (CONFIG_DIR / "bev_pv_epipolar_debug.json", True),
    (CONFIG_DIR / "PV_track_train_epipolar_debug.json", True),
]

HELPER = r"""

def build_depth_limited_epipolar_candidate_mask(
    query_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    camera_intrinsics_a_norm: torch.Tensor,
    camera_intrinsics_b_norm: torch.Tensor,
    camera2referego_a: torch.Tensor,
    camera2referego_b: torch.Tensor,
    feature_height: int,
    feature_width: int,
    min_depth: float,
    max_depth: float,
    epipolar_band_width: float,
    fundamental_matrix: torch.Tensor,
) -> torch.Tensor:
    # Return [Q,N] finite epipolar-segment candidates.
    #
    # The source query patch defines a camera-A ray. Restrict that ray to
    # [min_depth, max_depth] meters, transform it to camera B, clip it to
    # points in front of B, and project the resulting finite 3D interval.
    # Target patches are valid positives only when they lie near both the
    # finite projected segment and the ordinary epipolar line.

    min_depth = float(min_depth)
    max_depth = float(max_depth)
    if min_depth < 0.0:
        raise ValueError(
            f"view-consistency min depth must be >= 0, got {min_depth}."
        )
    if max_depth <= min_depth:
        raise ValueError(
            "view-consistency max depth must be greater than min depth, "
            f"got {min_depth}..{max_depth}."
        )

    intrinsics_a = scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_a_norm,
        feature_height,
        feature_width,
    ).to(
        device=query_coordinates.device,
        dtype=torch.float32,
    )
    intrinsics_b = scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_b_norm,
        feature_height,
        feature_width,
    ).to(
        device=query_coordinates.device,
        dtype=torch.float32,
    )

    query_coordinates = query_coordinates.float()
    target_coordinates = target_coordinates.float()

    ray_a = torch.linalg.solve(
        intrinsics_a,
        query_coordinates.transpose(0, 1),
    ).transpose(0, 1)
    ray_a = torch.nn.functional.normalize(
        ray_a,
        dim=1,
    )

    camera_b_from_camera_a = (
        torch.linalg.inv(camera2referego_b.float())
        @ camera2referego_a.float()
    )
    rotation = camera_b_from_camera_a[:3, :3]
    translation = camera_b_from_camera_a[:3, 3]

    direction_b = torch.matmul(
        ray_a,
        rotation.transpose(0, 1),
    )
    z_slope = direction_b[:, 2]
    z_offset = translation[2]

    depth_lo = torch.full_like(z_slope, min_depth)
    depth_hi = torch.full_like(z_slope, max_depth)

    z_epsilon = 1e-4
    slope_epsilon = 1e-8

    increasing = z_slope > slope_epsilon
    decreasing = z_slope < -slope_epsilon
    parallel = ~(increasing | decreasing)

    crossing_depth = torch.zeros_like(z_slope)
    nonparallel = ~parallel
    crossing_depth[nonparallel] = (
        z_epsilon - z_offset
    ) / z_slope[nonparallel]

    depth_lo = torch.where(
        increasing,
        torch.maximum(depth_lo, crossing_depth + 1e-4),
        depth_lo,
    )
    depth_hi = torch.where(
        decreasing,
        torch.minimum(depth_hi, crossing_depth - 1e-4),
        depth_hi,
    )

    valid_interval = depth_hi > depth_lo
    if float(z_offset) <= z_epsilon:
        valid_interval = valid_interval & ~parallel

    point_lo = (
        depth_lo[:, None] * direction_b
        + translation[None, :]
    )
    point_hi = (
        depth_hi[:, None] * direction_b
        + translation[None, :]
    )

    projected_lo_h = torch.matmul(
        point_lo,
        intrinsics_b.transpose(0, 1),
    )
    projected_hi_h = torch.matmul(
        point_hi,
        intrinsics_b.transpose(0, 1),
    )
    projected_lo = (
        projected_lo_h[:, :2]
        / projected_lo_h[:, 2:3].clamp_min(z_epsilon)
    )
    projected_hi = (
        projected_hi_h[:, :2]
        / projected_hi_h[:, 2:3].clamp_min(z_epsilon)
    )

    target_xy = target_coordinates[:, :2]
    segment = projected_hi - projected_lo
    segment_length_sq = segment.square().sum(dim=1)

    relative = (
        target_xy.unsqueeze(0)
        - projected_lo.unsqueeze(1)
    )
    projection_ratio = (
        relative
        * segment.unsqueeze(1)
    ).sum(dim=2) / segment_length_sq.clamp_min(1e-8).unsqueeze(1)
    projection_ratio = projection_ratio.clamp(0.0, 1.0)

    closest = (
        projected_lo.unsqueeze(1)
        + projection_ratio.unsqueeze(2)
        * segment.unsqueeze(1)
    )
    segment_distance = torch.linalg.vector_norm(
        target_xy.unsqueeze(0) - closest,
        dim=2,
    )

    target_lines = torch.matmul(
        fundamental_matrix.float(),
        query_coordinates.transpose(0, 1),
    ).transpose(0, 1)
    epipolar_numerator = torch.abs(
        torch.matmul(
            target_coordinates,
            target_lines.transpose(0, 1),
        ).transpose(0, 1)
    )
    epipolar_denominator = torch.sqrt(
        target_lines[:, 0].square()
        + target_lines[:, 1].square()
    ).clamp(min=1e-6)
    epipolar_distance = (
        epipolar_numerator
        / epipolar_denominator.unsqueeze(1)
    )

    candidate_mask = (
        valid_interval.unsqueeze(1)
        & torch.isfinite(segment_distance)
        & segment_distance.le(float(epipolar_band_width))
        & epipolar_distance.le(float(epipolar_band_width))
    )
    return candidate_mask
"""

OLD_SEMANTIC_SIGNATURE = """def semantic_epipolar_rank_direction(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    query_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_valid_mask: torch.Tensor,
    fundamental_matrix: torch.Tensor,
    epipolar_band_width: float,
    negative_band_scale: float,
    margin: float,
    temperature: float,
    query_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
"""

NEW_SEMANTIC_SIGNATURE = """def semantic_epipolar_rank_direction(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    query_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_valid_mask: torch.Tensor,
    fundamental_matrix: torch.Tensor,
    epipolar_band_width: float,
    negative_band_scale: float,
    margin: float,
    temperature: float,
    query_weights: Optional[torch.Tensor] = None,
    positive_candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
"""

OLD_POSITIVE = """    positive_mask = target_valid_mask & epipolar_distance.le(
        float(epipolar_band_width)
    )
"""

NEW_POSITIVE = """    positive_mask = target_valid_mask & epipolar_distance.le(
        float(epipolar_band_width)
    )
    if positive_candidate_mask is not None:
        positive_candidate_mask = positive_candidate_mask.to(
            device=positive_mask.device,
            dtype=torch.bool,
        )
        if positive_candidate_mask.shape != positive_mask.shape:
            raise ValueError(
                "positive_candidate_mask must match [Q,N] positive shape, "
                f"got {tuple(positive_candidate_mask.shape)} vs "
                f"{tuple(positive_mask.shape)}."
            )
        positive_mask = positive_mask & positive_candidate_mask
"""

OLD_CONFIG_READ = """    lower_half_start_ratio = float(
        training_config.get(
            "view_consistency_lower_half_start_ratio",
            0.5,
        )
    )

    batch_size, pair_count = selection.shape[:2]
"""

NEW_CONFIG_READ = """    lower_half_start_ratio = float(
        training_config.get(
            "view_consistency_lower_half_start_ratio",
            0.5,
        )
    )
    depth_range_enabled = bool(
        training_config.get(
            "view_consistency_depth_range_enabled",
            False,
        )
    )
    min_depth = float(
        training_config.get(
            "view_consistency_min_depth",
            0.1,
        )
    )
    max_depth = float(
        training_config.get(
            "view_consistency_max_depth",
            80.0,
        )
    )
    if depth_range_enabled:
        if min_depth < 0.0 or max_depth <= min_depth:
            raise ValueError(
                "Invalid view-consistency depth range: "
                f"{min_depth}..{max_depth}."
            )

    batch_size, pair_count = selection.shape[:2]
"""

OLD_TARGETS = """    target_indices = lower_half_mask_flat.nonzero(
        as_tuple=False
    ).flatten()
    if target_indices.numel() < min_patches:
        return projected_features.sum() * 0.0

    target_coordinates = build_patch_homogeneous_coordinates(
        target_indices,
        feature_width,
    )
"""

NEW_TARGETS = """    if depth_range_enabled:
        # Query sampling still uses the lower half, but the correspondence can
        # appear anywhere in the other camera. Search the complete target grid
        # and let finite epipolar geometry restrict positive candidates.
        target_indices = torch.arange(
            feature_height * feature_width,
            device=device,
            dtype=torch.long,
        )
    else:
        target_indices = lower_half_mask_flat.nonzero(
            as_tuple=False
        ).flatten()

    if target_indices.numel() < min_patches:
        return projected_features.sum() * 0.0

    target_coordinates = build_patch_homogeneous_coordinates(
        target_indices,
        feature_width,
    )
"""

OLD_AFTER_QUERY_COORDS = """            query_coordinates_b = build_patch_homogeneous_coordinates(
                query_indices_b,
                feature_width,
            )

            loss_a_to_b = semantic_epipolar_rank_direction(
"""

NEW_AFTER_QUERY_COORDS = """            query_coordinates_b = build_patch_homogeneous_coordinates(
                query_indices_b,
                feature_width,
            )

            positive_candidate_mask_a_to_b = None
            positive_candidate_mask_b_to_a = None
            if depth_range_enabled:
                positive_candidate_mask_a_to_b = (
                    build_depth_limited_epipolar_candidate_mask(
                        query_coordinates=query_coordinates_a,
                        target_coordinates=target_coordinates,
                        camera_intrinsics_a_norm=camera_intrinsics_norm[
                            batch_index,
                            time_a,
                            view_a,
                        ],
                        camera_intrinsics_b_norm=camera_intrinsics_norm[
                            batch_index,
                            time_b,
                            view_b,
                        ],
                        camera2referego_a=camera2referego[
                            batch_index,
                            time_a,
                            view_a,
                        ],
                        camera2referego_b=camera2referego[
                            batch_index,
                            time_b,
                            view_b,
                        ],
                        feature_height=feature_height,
                        feature_width=feature_width,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        epipolar_band_width=epipolar_band_width,
                        fundamental_matrix=fundamental_matrix,
                    )
                )
                positive_candidate_mask_b_to_a = (
                    build_depth_limited_epipolar_candidate_mask(
                        query_coordinates=query_coordinates_b,
                        target_coordinates=target_coordinates,
                        camera_intrinsics_a_norm=camera_intrinsics_norm[
                            batch_index,
                            time_b,
                            view_b,
                        ],
                        camera_intrinsics_b_norm=camera_intrinsics_norm[
                            batch_index,
                            time_a,
                            view_a,
                        ],
                        camera2referego_a=camera2referego[
                            batch_index,
                            time_b,
                            view_b,
                        ],
                        camera2referego_b=camera2referego[
                            batch_index,
                            time_a,
                            view_a,
                        ],
                        feature_height=feature_height,
                        feature_width=feature_width,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        epipolar_band_width=epipolar_band_width,
                        fundamental_matrix=fundamental_matrix.transpose(0, 1),
                    )
                )

            loss_a_to_b = semantic_epipolar_rank_direction(
"""

OLD_CALL_A = """                temperature=temperature,
                query_weights=query_weights_a,
            )
"""

NEW_CALL_A = """                temperature=temperature,
                query_weights=query_weights_a,
                positive_candidate_mask=positive_candidate_mask_a_to_b,
            )
"""

OLD_CALL_B = """                temperature=temperature,
                query_weights=query_weights_b,
            )
"""

NEW_CALL_B = """                temperature=temperature,
                query_weights=query_weights_b,
                positive_candidate_mask=positive_candidate_mask_b_to_a,
            )
"""


def backup(path: Path, stamp: str):
    if not path.exists():
        return None
    dst = path.with_name(path.name + f".bak_depth80_{stamp}")
    shutil.copy2(path, dst)
    return dst


def replace_once(text: str, old: str, new: str, label: str):
    if old not in text:
        raise RuntimeError(
            f"Could not find expected block for {label}; refusing to guess."
        )
    return text.replace(old, new, 1)


def patch_utils(stamp: str):
    if not UTILS.exists():
        raise FileNotFoundError(UTILS)

    text = UTILS.read_text(encoding="utf-8")

    if "def build_depth_limited_epipolar_candidate_mask(" in text:
        print("[utils] depth80 helper already present; skipping source patch.")
        return

    marker = "\ndef semantic_epipolar_rank_direction(\n"
    if marker not in text:
        raise RuntimeError(
            "Could not find semantic_epipolar_rank_direction insertion point."
        )

    backup_path = backup(UTILS, stamp)
    text = text.replace(marker, HELPER + marker, 1)

    text = replace_once(
        text, OLD_SEMANTIC_SIGNATURE, NEW_SEMANTIC_SIGNATURE, "semantic signature"
    )
    text = replace_once(
        text, OLD_POSITIVE, NEW_POSITIVE, "positive mask"
    )
    text = replace_once(
        text, OLD_CONFIG_READ, NEW_CONFIG_READ, "depth config read"
    )
    text = replace_once(
        text, OLD_TARGETS, NEW_TARGETS, "target grid selection"
    )
    text = replace_once(
        text, OLD_AFTER_QUERY_COORDS, NEW_AFTER_QUERY_COORDS, "finite segment masks"
    )
    text = replace_once(
        text, OLD_CALL_A, NEW_CALL_A, "A->B semantic call"
    )
    text = replace_once(
        text, OLD_CALL_B, NEW_CALL_B, "B->A semantic call"
    )

    UTILS.write_text(text, encoding="utf-8")
    print("[utils] patched:", UTILS)
    print("[utils] backup :", backup_path)


def patch_config(path: Path, is_debug: bool, stamp: str):
    if not path.exists():
        print("[config] skip missing:", path)
        return

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    tc = cfg["pipeline"]["training_config"]
    tc["view_consistency_depth_range_enabled"] = True
    tc["view_consistency_min_depth"] = 0.1
    tc["view_consistency_max_depth"] = 80.0

    if is_debug:
        tc["view_consistency_loss_warmup_steps"] = 0
        tc["view_consistency_max_sigma"] = 1.0
    else:
        tc["view_consistency_loss_warmup_steps"] = 1000
        tc["view_consistency_max_sigma"] = 0.5

    inference = cfg["pipeline"].get("inference_config", {})
    static_keys = inference.get(
        "autoregression_data_exception_for_take_sequence",
        [],
    )
    if "view_consistency_pair_mask" not in static_keys:
        static_keys.append("view_consistency_pair_mask")
    inference["autoregression_data_exception_for_take_sequence"] = static_keys

    backup_path = backup(path, stamp)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print("[config] updated:", path)
    print("[config] backup :", backup_path)
    print(
        "         range=",
        (tc["view_consistency_min_depth"], tc["view_consistency_max_depth"]),
        "warmup=",
        tc["view_consistency_loss_warmup_steps"],
        "max_sigma=",
        tc["view_consistency_max_sigma"],
    )


def verify():
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(UTILS)],
        check=True,
    )

    text = UTILS.read_text(encoding="utf-8")
    required = [
        "build_depth_limited_epipolar_candidate_mask",
        "positive_candidate_mask=positive_candidate_mask_a_to_b",
        "positive_candidate_mask=positive_candidate_mask_b_to_a",
        "view_consistency_depth_range_enabled",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(
            f"Verification failed, missing source tokens: {missing}"
        )

    for path, is_debug in CONFIGS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg["pipeline"]["training_config"]
        assert tc["view_consistency_depth_range_enabled"] is True
        assert float(tc["view_consistency_min_depth"]) == 0.1
        assert float(tc["view_consistency_max_depth"]) == 80.0

        if is_debug:
            assert int(tc["view_consistency_loss_warmup_steps"]) == 0
            assert float(tc["view_consistency_max_sigma"]) == 1.0
        else:
            assert int(tc["view_consistency_loss_warmup_steps"]) == 1000
            assert float(tc["view_consistency_max_sigma"]) == 0.5

    print("[verify] py_compile + config checks passed.")


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_utils(stamp)
    for path, is_debug in CONFIGS:
        patch_config(path, is_debug, stamp)
    verify()

    print()
    print("=== DEPTH-80 FORMAL LOSS PATCH DONE ===")
    print("Formal: depth=0.1..80m, query=lower-half, target=full-grid")
    print("Formal: warmup=1000, max_sigma=0.5")
    print("Debug smoke: warmup=0, max_sigma=1.0")
    print("No training was launched.")


if __name__ == "__main__":
    main()
