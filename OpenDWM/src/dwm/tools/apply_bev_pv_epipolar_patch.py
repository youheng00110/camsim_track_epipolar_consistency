#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/camsim/OpenDWM"
)

UTILS_SOURCE = r'''from typing import Optional

import einops
import torch


def scale_normalized_intrinsics_to_feature_grid(
    camera_intrinsics_norm: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Convert normalized intrinsics to the selected patch grid."""
    intrinsics = camera_intrinsics_norm.float().clone()
    intrinsics[..., 0, 0] *= float(feature_width)
    intrinsics[..., 0, 2] *= float(feature_width)
    intrinsics[..., 1, 1] *= float(feature_height)
    intrinsics[..., 1, 2] *= float(feature_height)
    return intrinsics


def build_fundamental_matrix_on_feature_grid(
    camera_intrinsics_a_norm: torch.Tensor,
    camera_intrinsics_b_norm: torch.Tensor,
    camera2referego_a: torch.Tensor,
    camera2referego_b: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Build F so that x_b^T F x_a = 0 on the patch grid."""
    intrinsics_a = scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_a_norm,
        feature_height,
        feature_width,
    )
    intrinsics_b = scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_b_norm,
        feature_height,
        feature_width,
    )
    camera_b_from_camera_a = (
        torch.linalg.inv(camera2referego_b.float())
        @ camera2referego_a.float()
    )
    rotation = camera_b_from_camera_a[:3, :3]
    translation = camera_b_from_camera_a[:3, 3]
    tx = torch.zeros(
        3,
        3,
        device=translation.device,
        dtype=torch.float32,
    )
    tx[0, 1] = -translation[2]
    tx[0, 2] = translation[1]
    tx[1, 0] = translation[2]
    tx[1, 2] = -translation[0]
    tx[2, 0] = -translation[1]
    tx[2, 1] = translation[0]
    fundamental = (
        torch.linalg.inv(intrinsics_b).transpose(0, 1)
        @ tx
        @ rotation
        @ torch.linalg.inv(intrinsics_a)
    )
    return fundamental / fundamental.norm().clamp(min=1e-08)


def build_patch_homogeneous_coordinates(
    flat_indices: torch.Tensor,
    feature_width: int,
) -> torch.Tensor:
    """Return patch-center coordinates [x, y, 1] for flattened indices."""
    y = torch.div(
        flat_indices,
        int(feature_width),
        rounding_mode="floor",
    ).float()
    x = flat_indices.remainder(int(feature_width)).float()
    ones = torch.ones_like(x)
    return torch.stack([x + 0.5, y + 0.5, ones], dim=-1)


def build_selected_box_patch_masks(
    box_images: torch.Tensor,
    selection: torch.Tensor,
    feature_height: int,
    feature_width: int,
    dilation_kernel: int,
) -> torch.Tensor:
    """Build foreground masks for [time_a, view_a, time_b, view_b] pairs."""
    if selection.ndim == 2:
        selection = selection.unsqueeze(1)
    if selection.ndim != 3 or selection.shape[-1] != 4:
        raise ValueError(
            "selection must have shape [B,K,4], got "
            f"{tuple(selection.shape)}"
        )

    batch_size, pair_count = selection.shape[:2]
    batch_ids = torch.arange(
        batch_size,
        device=box_images.device,
        dtype=torch.long,
    ).unsqueeze(1).expand(batch_size, pair_count)
    time_a_ids = selection[..., 0].long()
    view_a_ids = selection[..., 1].long()
    time_b_ids = selection[..., 2].long()
    view_b_ids = selection[..., 3].long()

    mask_a = box_images[
        batch_ids,
        time_a_ids,
        view_a_ids,
    ].amax(dim=2, keepdim=True)
    mask_b = box_images[
        batch_ids,
        time_b_ids,
        view_b_ids,
    ].amax(dim=2, keepdim=True)

    source_height = int(mask_a.shape[-2])
    source_width = int(mask_a.shape[-1])
    selected_masks = torch.cat(
        [
            mask_a.reshape(-1, 1, source_height, source_width),
            mask_b.reshape(-1, 1, source_height, source_width),
        ],
        dim=0,
    )

    dilation_kernel = max(int(dilation_kernel), 1)
    if dilation_kernel % 2 == 0:
        dilation_kernel += 1
    if dilation_kernel > 1:
        selected_masks = torch.nn.functional.max_pool2d(
            selected_masks,
            kernel_size=dilation_kernel,
            stride=1,
            padding=dilation_kernel // 2,
        )

    selected_masks = torch.nn.functional.adaptive_max_pool2d(
        selected_masks,
        output_size=(feature_height, feature_width),
    )
    selected_masks = selected_masks.gt(0.005)
    mask_a, mask_b = selected_masks.chunk(2, dim=0)
    mask_a = mask_a[:, 0].reshape(
        batch_size,
        pair_count,
        feature_height,
        feature_width,
    )
    mask_b = mask_b[:, 0].reshape(
        batch_size,
        pair_count,
        feature_height,
        feature_width,
    )
    return torch.stack([mask_a, mask_b], dim=2)


def sample_patch_indices(
    foreground_mask_flat: torch.Tensor,
    valid_mask_flat: torch.Tensor,
    max_foreground_patches: int,
    max_background_patches: int,
    generator: torch.Generator,
    device: torch.device,
):
    foreground_indices = (
        foreground_mask_flat & valid_mask_flat
    ).nonzero(as_tuple=False).flatten()
    background_indices = (
        valid_mask_flat & foreground_mask_flat.logical_not()
    ).nonzero(as_tuple=False).flatten()

    if foreground_indices.numel() > max_foreground_patches:
        perm = torch.randperm(
            int(foreground_indices.numel()),
            generator=generator,
            device="cpu",
        )[:max_foreground_patches].to(device)
        foreground_indices = foreground_indices.index_select(0, perm)
    if background_indices.numel() > max_background_patches:
        perm = torch.randperm(
            int(background_indices.numel()),
            generator=generator,
            device="cpu",
        )[:max_background_patches].to(device)
        background_indices = background_indices.index_select(0, perm)

    query_indices = torch.cat(
        [foreground_indices, background_indices],
        dim=0,
    )
    query_weights = torch.cat(
        [
            torch.full(
                (foreground_indices.numel(),),
                2.0,
                device=device,
                dtype=torch.float32,
            ),
            torch.ones(
                (background_indices.numel(),),
                device=device,
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )
    return query_indices, query_weights


def sample_stratified_times(
    time_ids,
    sample_time_count: int,
    generator: torch.Generator,
):
    if len(time_ids) <= sample_time_count:
        return list(time_ids)
    boundaries = torch.linspace(
        0,
        len(time_ids),
        steps=sample_time_count + 1,
        dtype=torch.float32,
    )
    picked = []
    for idx in range(sample_time_count):
        start = int(boundaries[idx].floor().item())
        end = int(boundaries[idx + 1].floor().item())
        if end <= start:
            end = min(start + 1, len(time_ids))
        if end <= start:
            end = len(time_ids)
        local = time_ids[start:end] or [
            time_ids[min(start, len(time_ids) - 1)]
        ]
        choice = int(
            torch.randint(
                0,
                len(local),
                (1,),
                generator=generator,
                device="cpu",
            ).item()
        )
        picked.append(int(local[choice]))
    return picked


def build_candidate_pairs(
    mask_row,
    masked_views: torch.Tensor,
    view_count: int,
):
    pairs_pref = []
    pairs_all = []
    has_masked_views = bool(masked_views.any().item())
    for a in range(view_count):
        for b in range(a + 1, view_count):
            if mask_row is not None and not bool(
                mask_row[a, b] or mask_row[b, a]
            ):
                continue
            pairs_all.append((a, b))
            if has_masked_views and bool(masked_views[a].item()) != bool(
                masked_views[b].item()
            ):
                pairs_pref.append(
                    (a, b) if bool(masked_views[a].item()) else (b, a)
                )
    return pairs_pref or list(pairs_all), pairs_all


def semantic_epipolar_rank_direction(
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
    """Epipolar-band semantic ranking without exact patch correspondences."""
    similarity = torch.matmul(
        query_features,
        target_features.transpose(0, 1),
    )
    target_lines = torch.matmul(
        fundamental_matrix,
        query_coordinates.transpose(0, 1),
    ).transpose(0, 1)
    numerator = torch.abs(
        torch.matmul(
            target_coordinates,
            target_lines.transpose(0, 1),
        ).transpose(0, 1)
    )
    denominator = torch.sqrt(
        target_lines[:, 0].square() + target_lines[:, 1].square()
    ).clamp(min=1e-06)

    epipolar_distance = numerator / denominator.unsqueeze(1)
    target_valid_mask = target_valid_mask.bool().unsqueeze(0).expand_as(
        epipolar_distance
    )
    positive_mask = target_valid_mask & epipolar_distance.le(
        float(epipolar_band_width)
    )
    if float(negative_band_scale) <= 1.0:
        negative_mask = target_valid_mask & positive_mask.logical_not()
    else:
        negative_mask = target_valid_mask & epipolar_distance.ge(
            float(epipolar_band_width) * float(negative_band_scale)
        )

    valid_query = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    temperature = max(float(temperature), 1e-06)
    safe_positive_mask = positive_mask.clone()
    safe_positive_mask[:, 0] = (
        safe_positive_mask[:, 0] | valid_query.logical_not()
    )
    positive_logits = similarity.masked_fill(
        safe_positive_mask.logical_not(),
        -torch.inf,
    )
    positive_attention = torch.softmax(
        positive_logits / temperature,
        dim=1,
    )
    positive_attention = torch.where(
        positive_mask,
        positive_attention,
        torch.zeros_like(positive_attention),
    )
    positive_score = (positive_attention * similarity).sum(dim=1)
    rank_delta = (
        similarity - positive_score.unsqueeze(1) + float(margin)
    ) / temperature

    pairwise_rank = torch.sigmoid(rank_delta) * negative_mask.float()
    negative_count = negative_mask.sum(dim=1).clamp(min=1)
    rank_loss = pairwise_rank.sum(dim=1) / negative_count
    valid_query_weight = valid_query.to(dtype=rank_loss.dtype)

    if query_weights is not None:
        query_weights = query_weights.to(
            rank_loss.device,
            dtype=rank_loss.dtype,
        ) * valid_query_weight
        return (rank_loss * query_weights).sum() / query_weights.sum().clamp(
            min=1e-06
        )
    return (rank_loss * valid_query_weight).sum() / valid_query_weight.sum().clamp(
        min=1.0
    )


def _choose_items(items, count: int, generator: torch.Generator):
    if len(items) == 0:
        return []
    if len(items) >= count:
        perm = torch.randperm(
            len(items),
            generator=generator,
            device="cpu",
        )[:count].tolist()
        return [items[index] for index in perm]
    return [
        items[
            int(
                torch.randint(
                    0,
                    len(items),
                    (1,),
                    generator=generator,
                    device="cpu",
                ).item()
            )
        ]
        for _ in range(count)
    ]


def sample_view_consistency_selection(
    batch: dict,
    training_config: dict,
    generator: torch.Generator,
    device: torch.device,
):
    """Sample unified [time_a, view_a, time_b, view_b] epipolar pairs.

    Cross-view pairs use the existing crossview_mask at the same time.
    Cross-frame pairs use the same camera slot at t and t + stride.
    """
    loss_weight = float(
        training_config.get("view_consistency_loss_weight", 0.0)
    )
    batch_size = int(batch["vae_images"].shape[0])
    frame_count = int(batch["vae_images"].shape[1])
    view_count = int(batch["vae_images"].shape[2])
    if loss_weight <= 0.0 or view_count <= 0 or frame_count <= 0:
        return None

    enable_crossview = bool(
        training_config.get("view_consistency_enable_crossview", True)
    )
    enable_crossframe = bool(
        training_config.get("view_consistency_enable_crossframe", True)
    )
    if not enable_crossview and not enable_crossframe:
        return None

    time_divisor = max(
        1,
        int(training_config.get("view_consistency_time_divisor", 16)),
    )
    pair_per_time = max(
        1,
        int(training_config.get("view_consistency_pairs_per_time", 2)),
    )
    crossframe_stride = max(
        1,
        int(training_config.get("view_consistency_crossframe_stride", 1)),
    )
    reference_count = max(
        0,
        int(training_config.get("reference_frame_count", 0)),
    )
    sample_time_count = max(1, frame_count // time_divisor)

    crossview_mask = batch.get("crossview_mask")
    if crossview_mask is not None:
        crossview_mask = crossview_mask.bool().cpu()
        if crossview_mask.ndim == 4 and crossview_mask.shape[1] == 1:
            crossview_mask = crossview_mask.squeeze(1)
        if crossview_mask.ndim == 2:
            crossview_mask = crossview_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        if crossview_mask.shape != (
            batch_size,
            view_count,
            view_count,
        ):
            raise ValueError(
                "crossview_mask must be [B,V,V], got "
                f"{tuple(crossview_mask.shape)}"
            )

    all_selections = []
    max_pair_count = 0
    for batch_index in range(batch_size):
        valid_times = list(range(min(reference_count, frame_count), frame_count))
        if len(valid_times) == 0:
            valid_times = [frame_count - 1]
        sampled_times = sample_stratified_times(
            valid_times,
            sample_time_count,
            generator,
        )

        mask_row = (
            None
            if crossview_mask is None
            else crossview_mask[batch_index]
        )
        _, crossview_pairs = build_candidate_pairs(
            mask_row,
            torch.zeros(view_count, dtype=torch.bool),
            view_count,
        )
        view_ids = list(range(view_count))
        sample_pairs = []

        for time_a in sampled_times:
            if enable_crossview:
                for view_a, view_b in _choose_items(
                    crossview_pairs,
                    pair_per_time,
                    generator,
                ):
                    sample_pairs.append(
                        [time_a, view_a, time_a, view_b]
                    )

            time_b = time_a + crossframe_stride
            if enable_crossframe and time_b < frame_count:
                for view_id in _choose_items(
                    view_ids,
                    pair_per_time,
                    generator,
                ):
                    sample_pairs.append(
                        [time_a, view_id, time_b, view_id]
                    )

        if len(sample_pairs) == 0:
            return None
        selection = torch.tensor(sample_pairs, dtype=torch.long)
        all_selections.append(selection)
        max_pair_count = max(max_pair_count, int(selection.shape[0]))

    padded_cpu = torch.full(
        (batch_size, max_pair_count, 4),
        -1,
        dtype=torch.long,
        device="cpu",
    )
    for batch_index, selection in enumerate(all_selections):
        padded_cpu[
            batch_index,
            : selection.shape[0],
        ] = selection
    return padded_cpu.to(device=device), padded_cpu


def compute_view_consistency_loss(
    batch: dict,
    projected_features: torch.Tensor,
    selection: torch.Tensor,
    selection_cpu: torch.Tensor,
    camera_intrinsics_norm: torch.Tensor,
    camera2referego: torch.Tensor,
    sigmas_cpu: torch.Tensor,
    training_config: dict,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if projected_features is None or selection is None:
        return camera_intrinsics_norm.sum() * 0.0

    feature_height = int(projected_features.shape[-2])
    feature_width = int(projected_features.shape[-1])
    box_patch_masks = None
    if "3dbox_images" in batch:
        box_images = batch["3dbox_images"]
        if box_images.device != device:
            box_images = box_images.to(device)
        box_patch_masks = build_selected_box_patch_masks(
            box_images=box_images,
            selection=selection,
            feature_height=feature_height,
            feature_width=feature_width,
            dilation_kernel=int(
                training_config.get(
                    "view_consistency_box_dilation_kernel",
                    31,
                )
            ),
        )

    max_sigma = float(
        training_config.get("view_consistency_max_sigma", 0.5)
    )
    max_foreground_patches = int(
        training_config.get(
            "view_consistency_max_foreground_patches",
            128,
        )
    )
    max_background_patches = int(
        training_config.get(
            "view_consistency_max_background_patches",
            128,
        )
    )
    min_patches = int(
        training_config.get("view_consistency_min_patches", 4)
    )
    epipolar_band_width = float(
        training_config.get(
            "view_consistency_epipolar_band_width",
            2.5,
        )
    )
    negative_band_scale = float(
        training_config.get(
            "view_consistency_negative_band_scale",
            1.0,
        )
    )
    margin = float(
        training_config.get("view_consistency_margin", 0.1)
    )
    temperature = float(
        training_config.get("view_consistency_temperature", 0.07)
    )
    lower_half_start_ratio = float(
        training_config.get(
            "view_consistency_lower_half_start_ratio",
            0.5,
        )
    )

    batch_size, pair_count = selection.shape[:2]
    pair_flat_features = einops.rearrange(
        projected_features,
        "b k p c h w -> b k p (h w) c",
    )
    lower_half_row = torch.arange(
        feature_height,
        device=device,
    ).unsqueeze(1).expand(feature_height, feature_width)
    lower_half_mask_flat = lower_half_row.ge(
        int(feature_height * lower_half_start_ratio)
    ).reshape(-1)
    target_indices = lower_half_mask_flat.nonzero(
        as_tuple=False
    ).flatten()
    if target_indices.numel() < min_patches:
        return projected_features.sum() * 0.0

    target_coordinates = build_patch_homogeneous_coordinates(
        target_indices,
        feature_width,
    )
    target_valid_mask = torch.ones(
        target_indices.shape[0],
        device=device,
        dtype=torch.bool,
    )
    pair_losses = []

    for batch_index in range(batch_size):
        sigma_value = float(sigmas_cpu[batch_index])
        if sigma_value > max_sigma:
            continue
        for pair_index in range(pair_count):
            time_a, view_a, time_b, view_b = selection_cpu[
                batch_index,
                pair_index,
            ].tolist()
            if min(time_a, view_a, time_b, view_b) < 0:
                continue

            fundamental_matrix = build_fundamental_matrix_on_feature_grid(
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
            )

            if box_patch_masks is not None:
                foreground_mask_a = box_patch_masks[
                    batch_index,
                    pair_index,
                    0,
                ].reshape(-1)
                foreground_mask_b = box_patch_masks[
                    batch_index,
                    pair_index,
                    1,
                ].reshape(-1)
            else:
                foreground_mask_a = torch.zeros_like(lower_half_mask_flat)
                foreground_mask_b = torch.zeros_like(lower_half_mask_flat)

            query_indices_a, query_weights_a = sample_patch_indices(
                foreground_mask_flat=foreground_mask_a,
                valid_mask_flat=lower_half_mask_flat,
                max_foreground_patches=max_foreground_patches,
                max_background_patches=max_background_patches,
                generator=generator,
                device=device,
            )
            query_indices_b, query_weights_b = sample_patch_indices(
                foreground_mask_flat=foreground_mask_b,
                valid_mask_flat=lower_half_mask_flat,
                max_foreground_patches=max_foreground_patches,
                max_background_patches=max_background_patches,
                generator=generator,
                device=device,
            )
            if (
                query_indices_a.numel() < min_patches
                or query_indices_b.numel() < min_patches
            ):
                continue

            features_a = pair_flat_features[
                batch_index,
                pair_index,
                0,
            ]
            features_b = pair_flat_features[
                batch_index,
                pair_index,
                1,
            ]
            query_features_a = torch.nn.functional.normalize(
                features_a.index_select(0, query_indices_a).float(),
                dim=-1,
            )
            query_features_b = torch.nn.functional.normalize(
                features_b.index_select(0, query_indices_b).float(),
                dim=-1,
            )
            target_features_a = torch.nn.functional.normalize(
                features_a.index_select(0, target_indices).float(),
                dim=-1,
            )
            target_features_b = torch.nn.functional.normalize(
                features_b.index_select(0, target_indices).float(),
                dim=-1,
            )
            query_coordinates_a = build_patch_homogeneous_coordinates(
                query_indices_a,
                feature_width,
            )
            query_coordinates_b = build_patch_homogeneous_coordinates(
                query_indices_b,
                feature_width,
            )

            loss_a_to_b = semantic_epipolar_rank_direction(
                query_features=query_features_a,
                target_features=target_features_b,
                query_coordinates=query_coordinates_a,
                target_coordinates=target_coordinates,
                target_valid_mask=target_valid_mask,
                fundamental_matrix=fundamental_matrix,
                epipolar_band_width=epipolar_band_width,
                negative_band_scale=negative_band_scale,
                margin=margin,
                temperature=temperature,
                query_weights=query_weights_a,
            )
            loss_b_to_a = semantic_epipolar_rank_direction(
                query_features=query_features_b,
                target_features=target_features_a,
                query_coordinates=query_coordinates_b,
                target_coordinates=target_coordinates,
                target_valid_mask=target_valid_mask,
                fundamental_matrix=fundamental_matrix.transpose(0, 1),
                epipolar_band_width=epipolar_band_width,
                negative_band_scale=negative_band_scale,
                margin=margin,
                temperature=temperature,
                query_weights=query_weights_b,
            )
            pair_losses.append(0.5 * (loss_a_to_b + loss_b_to_a))

    if len(pair_losses) == 0:
        return projected_features.sum() * 0.0
    return torch.stack(pair_losses).mean()
'''

PROJECTOR_SOURCE = r'''

class ViewConsistencyProjector(torch.nn.Module):
    """Lightweight patch projector used only by semantic consistency loss."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        hidden_channels: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                "ViewConsistencyProjector requires num_layers >= 2, "
                f"got {num_layers}."
            )

        layers = []
        layers.append(
            torch.nn.Conv2d(
                int(in_channels),
                int(hidden_channels),
                kernel_size=1,
            )
        )
        layers.append(torch.nn.SiLU())

        for _ in range(int(num_layers) - 2):
            layers.append(
                torch.nn.Conv2d(
                    int(hidden_channels),
                    int(hidden_channels),
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(torch.nn.SiLU())

        layers.append(
            torch.nn.Conv2d(
                int(hidden_channels),
                int(out_channels),
                kernel_size=1,
            )
        )
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.net(feature_map)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one marker, found {count}. "
            "The source differs from the reviewed version; aborting."
        )
    return text.replace(old, new, 1)


def patch_model(source: str) -> str:
    source = replace_once(
        source,
        "\nclass BEVConditionedSD3TransformerModel(diffusers.SD3Transformer2DModel):",
        PROJECTOR_SOURCE
        + "\n\nclass BEVConditionedSD3TransformerModel(diffusers.SD3Transformer2DModel):",
        "insert ViewConsistencyProjector",
    )

    source = replace_once(
        source,
        "        condition_image_adapter_config: Optional[dict] = None,\n"
        "        **kwargs,\n",
        "        condition_image_adapter_config: Optional[dict] = None,\n"
        "        view_consistency_config: Optional[dict] = None,\n"
        "        **kwargs,\n",
        "add view_consistency_config argument",
    )

    marker = (
        "        inner_dim = attention_head_dim * num_attention_heads\n\n"
        "        # PV image-condition branch."
    )
    replacement = '''        inner_dim = attention_head_dim * num_attention_heads

        consistency_config = dict(view_consistency_config or {})
        self.enable_view_consistency_features = bool(
            consistency_config.pop("enabled", False)
        )
        self.view_consistency_layer_id = int(
            consistency_config.pop("layer_id", 13)
        )
        consistency_projector_dim = int(
            consistency_config.pop("projector_dim", 128)
        )
        consistency_hidden_dim = int(
            consistency_config.pop("projector_hidden_dim", 256)
        )
        consistency_projector_layers = int(
            consistency_config.pop("projector_layers", 2)
        )
        if consistency_config:
            raise ValueError(
                "Unsupported view_consistency_config keys: "
                f"{sorted(consistency_config.keys())}"
            )

        self.view_consistency_projector = None
        if self.enable_view_consistency_features:
            if self.view_consistency_layer_id not in self.additional_layer_index:
                raise ValueError(
                    "The first epipolar version extracts features after the "
                    "complete BEV/condition/temporal/cross-view stage, so "
                    "view_consistency layer_id must be one of block_layers. "
                    f"Got layer_id={self.view_consistency_layer_id}, "
                    f"block_layers={self.block_layers}."
                )
            self.view_consistency_projector = ViewConsistencyProjector(
                in_channels=inner_dim,
                out_channels=consistency_projector_dim,
                hidden_channels=consistency_hidden_dim,
                num_layers=consistency_projector_layers,
            )

        # PV image-condition branch.'''
    source = replace_once(
        source,
        marker,
        replacement,
        "initialize view consistency module",
    )

    source = replace_once(
        source,
        "        condition_image_tensor: torch.Tensor = None,\n"
        "        return_dict: bool = False,\n",
        "        condition_image_tensor: torch.Tensor = None,\n"
        "        view_consistency_selection: torch.Tensor = None,\n"
        "        return_dict: bool = False,\n",
        "add view_consistency_selection argument",
    )

    source = replace_once(
        source,
        "        disable_crossview = torch.zeros_like(disable_temporal, dtype=torch.bool)\n\n"
        "        for layer_index, block in enumerate(self.transformer_blocks):\n",
        "        disable_crossview = torch.zeros_like(disable_temporal, dtype=torch.bool)\n"
        "        view_consistency_features = None\n\n"
        "        for layer_index, block in enumerate(self.transformer_blocks):\n",
        "initialize view_consistency_features",
    )

    extraction_marker = '''                hidden_states = self.forward_crossview_block_and_mix_result(
                    self.crossview_transformer_blocks[additional_index],
                    self.view_mixers[additional_index],
                    hidden_states,
                    camera_patch_embedding,
                    condition_keep,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    height,
                    crossview_attention_mask,
                    disable_crossview,
                )

        hidden_states = self.norm_out(hidden_states, temb)
'''
    extraction_replacement = '''                hidden_states = self.forward_crossview_block_and_mix_result(
                    self.crossview_transformer_blocks[additional_index],
                    self.view_mixers[additional_index],
                    hidden_states,
                    camera_patch_embedding,
                    condition_keep,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    height,
                    crossview_attention_mask,
                    disable_crossview,
                )

            if (
                self.enable_view_consistency_features
                and view_consistency_selection is not None
                and layer_index == self.view_consistency_layer_id
            ):
                selection = view_consistency_selection.to(
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                if selection.ndim == 2:
                    selection = selection.unsqueeze(1)
                if (
                    selection.ndim != 3
                    or selection.shape[0] != batch_size
                    or selection.shape[-1] != 4
                ):
                    raise ValueError(
                        "view_consistency_selection must be [B,K,4] with "
                        "[time_a, view_a, time_b, view_b], got "
                        f"{tuple(selection.shape)}"
                    )

                feature_grid = einops.rearrange(
                    hidden_states,
                    "(b t v) (h w) c -> b t v c h w",
                    b=batch_size,
                    t=sequence_length,
                    v=view_count,
                    h=height,
                    w=width,
                )
                pair_count = selection.shape[1]
                batch_ids = torch.arange(
                    batch_size,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(1).expand(batch_size, pair_count)
                time_a_ids = selection[..., 0].clamp(0, sequence_length - 1)
                view_a_ids = selection[..., 1].clamp(0, view_count - 1)
                time_b_ids = selection[..., 2].clamp(0, sequence_length - 1)
                view_b_ids = selection[..., 3].clamp(0, view_count - 1)

                feature_a = feature_grid[
                    batch_ids,
                    time_a_ids,
                    view_a_ids,
                ]
                feature_b = feature_grid[
                    batch_ids,
                    time_b_ids,
                    view_b_ids,
                ]
                feature_pair = torch.cat(
                    [
                        feature_a.reshape(-1, *feature_a.shape[-3:]),
                        feature_b.reshape(-1, *feature_b.shape[-3:]),
                    ],
                    dim=0,
                )
                projected_pair = self.view_consistency_projector(feature_pair)
                projected_a, projected_b = projected_pair.chunk(2, dim=0)
                projected_a = projected_a.reshape(
                    batch_size,
                    pair_count,
                    *projected_a.shape[1:],
                )
                projected_b = projected_b.reshape(
                    batch_size,
                    pair_count,
                    *projected_b.shape[1:],
                )
                view_consistency_features = torch.stack(
                    [projected_a, projected_b],
                    dim=2,
                )

        hidden_states = self.norm_out(hidden_states, temb)
'''
    source = replace_once(
        source,
        extraction_marker,
        extraction_replacement,
        "insert layer-13 feature extraction",
    )

    return_marker = '''        if return_dict:
            return {"noise_pred": output}
        return [output], encoder_hidden_states, pooled_projections
'''
    return_replacement = '''        if return_dict:
            result = {"noise_pred": output}
            if view_consistency_features is not None:
                result["view_consistency_features"] = view_consistency_features
            return result

        model_outputs = [output]
        if view_consistency_features is not None:
            model_outputs.append(view_consistency_features)
        return model_outputs, encoder_hidden_states, pooled_projections
'''
    source = replace_once(
        source,
        return_marker,
        return_replacement,
        "return view consistency features",
    )
    return source


def patch_pipeline(source: str) -> str:
    source = replace_once(
        source,
        "import dwm.utils.preview\n",
        "import dwm.utils.preview\nimport dwm.utils.view_consistency\n",
        "import view consistency utils",
    )

    geometry_method = r'''
    def prepare_view_consistency_geometry(
        self,
        batch: dict,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_intrinsics_norm = batch["camera_intrinsics"][:, :1].clone().float()
        image_size = batch["image_size"][:, :1].to(camera_intrinsics_norm)
        camera_intrinsics_norm[..., 0, 0] /= image_size[..., 0]
        camera_intrinsics_norm[..., 1, 1] /= image_size[..., 1]
        camera_intrinsics_norm[..., 0, 2] /= image_size[..., 0]
        camera_intrinsics_norm[..., 1, 2] /= image_size[..., 1]
        camera_intrinsics_norm = camera_intrinsics_norm.expand(
            -1,
            sequence_length,
            -1,
            -1,
            -1,
        ).contiguous()

        camera_to_ego = batch["camera_transforms"][:, :1].float().expand(
            -1,
            sequence_length,
            -1,
            -1,
            -1,
        ).contiguous()
        world_from_ego = batch["reference_ego_transforms"].double()
        if world_from_ego.shape[1] != sequence_length:
            raise ValueError(
                "reference_ego_transforms time length must match latent time: "
                f"{world_from_ego.shape[1]} vs {sequence_length}."
            )
        ego_to_initial = torch.linalg.solve(
            world_from_ego[:, :1],
            world_from_ego,
        ).float()
        camera2referego = (
            ego_to_initial[:, :, None] @ camera_to_ego
        )
        return (
            camera_intrinsics_norm.to(self.device),
            camera2referego.to(self.device),
        )

'''
    source = replace_once(
        source,
        "    def train_step(self, batch: dict, global_step: int):\n",
        geometry_method
        + "    def train_step(self, batch: dict, global_step: int):\n",
        "insert view consistency geometry preparation",
    )

    selection_marker = '''        model_conditions["disable_temporal"] = disable_temporal

        model_output, _, _ = self.model_wrapper(
            noisy_latents.to(self.model_dtype),
            timesteps,
            **model_conditions,
        )
'''
    selection_replacement = '''        model_conditions["disable_temporal"] = disable_temporal

        view_consistency_selection_result = (
            dwm.utils.view_consistency.sample_view_consistency_selection(
                batch=batch,
                training_config=self.training_config,
                generator=self.generator,
                device=self.device,
            )
        )
        if view_consistency_selection_result is None:
            view_consistency_selection = None
            view_consistency_selection_cpu = None
        else:
            (
                view_consistency_selection,
                view_consistency_selection_cpu,
            ) = view_consistency_selection_result
            model_conditions["view_consistency_selection"] = (
                view_consistency_selection
            )

        model_output, _, _ = self.model_wrapper(
            noisy_latents.to(self.model_dtype),
            timesteps,
            **model_conditions,
        )
        projected_features = (
            model_output[1]
            if view_consistency_selection is not None
            and len(model_output) > 1
            else None
        )
'''
    source = replace_once(
        source,
        selection_marker,
        selection_replacement,
        "sample epipolar pairs before model forward",
    )

    loss_marker = '''        denominator = (
            pixel_weight.sum() * predicted_latents.shape[3]
        ).clamp_min(1.0)
        loss = (squared_error * pixel_weight).sum() / denominator
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite BEV training loss at step {global_step + 1}."
            )

        self.loss_report_list.append({"loss": float(loss.detach())})
'''
    loss_replacement = '''        denominator = (
            pixel_weight.sum() * predicted_latents.shape[3]
        ).clamp_min(1.0)
        sd_loss = (squared_error * pixel_weight).sum() / denominator

        view_consistency_loss = sd_loss.new_zeros(())
        consistency_weight = float(
            self.training_config.get("view_consistency_loss_weight", 0.0)
        )
        if (
            consistency_weight > 0.0
            and view_consistency_selection is not None
        ):
            (
                camera_intrinsics_norm,
                camera2referego,
            ) = self.prepare_view_consistency_geometry(
                batch,
                sequence_length,
            )
            sigmas_cpu = sigmas.reshape(batch_size, -1)[:, 0].detach().cpu()
            view_consistency_loss = (
                dwm.utils.view_consistency.compute_view_consistency_loss(
                    batch=batch,
                    projected_features=projected_features,
                    selection=view_consistency_selection,
                    selection_cpu=view_consistency_selection_cpu,
                    camera_intrinsics_norm=camera_intrinsics_norm,
                    camera2referego=camera2referego,
                    sigmas_cpu=sigmas_cpu,
                    training_config=self.training_config,
                    generator=self.generator,
                    device=self.device,
                )
            )

        consistency_warmup_steps = int(
            self.training_config.get(
                "view_consistency_loss_warmup_steps",
                0,
            )
        )
        if consistency_warmup_steps > 0:
            warmup_ratio = min(
                float(global_step + 1) / float(consistency_warmup_steps),
                1.0,
            )
        else:
            warmup_ratio = 1.0
        effective_consistency_weight = consistency_weight * warmup_ratio
        loss = sd_loss + effective_consistency_weight * view_consistency_loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite BEV epipolar training loss at step "
                f"{global_step + 1}."
            )

        self.loss_report_list.append(
            {
                "loss": float(loss.detach()),
                "sd_loss": float(sd_loss.detach()),
                "view_consistency_loss": float(
                    view_consistency_loss.detach()
                ),
                "view_consistency_weight": float(
                    effective_consistency_weight
                ),
            }
        )
'''
    source = replace_once(
        source,
        loss_marker,
        loss_replacement,
        "add epipolar loss to BEV loss",
    )

    log_marker = '''    def log(self, global_step: int, log_steps: int):
        if not self.loss_report_list:
            return
        mean_loss = sum(
            item["loss"] for item in self.loss_report_list
        ) / len(self.loss_report_list)
        if self.should_save:
            print(
                f"Step {global_step} "
                f"({self.step_duration / log_steps:.1f} s/step), "
                f"loss: {mean_loss:.4f}",
                flush=True,
            )
            if self.summary is not None:
                self.summary.add_scalar(
                    "train/Loss",
                    mean_loss,
                    global_step,
                )
        self.loss_report_list.clear()
        self.step_duration = 0.0
'''
    log_replacement = '''    def log(self, global_step: int, log_steps: int):
        if not self.loss_report_list:
            return
        keys = self.loss_report_list[0].keys()
        mean_values = {
            key: sum(item[key] for item in self.loss_report_list)
            / len(self.loss_report_list)
            for key in keys
        }
        if self.should_save:
            loss_message = ", ".join(
                f"{key}: {value:.4f}"
                for key, value in mean_values.items()
            )
            print(
                f"Step {global_step} "
                f"({self.step_duration / log_steps:.1f} s/step), "
                f"{loss_message}",
                flush=True,
            )
            if self.summary is not None:
                for key, value in mean_values.items():
                    self.summary.add_scalar(
                        f"train/{key}",
                        value,
                        global_step,
                    )
        self.loss_report_list.clear()
        self.step_duration = 0.0
'''
    source = replace_once(
        source,
        log_marker,
        log_replacement,
        "extend loss logging",
    )
    return source


def patch_config(
    config: dict,
    loss_weight: float,
    warmup_steps: int,
) -> dict:
    pipeline = config["pipeline"]
    pipeline["_class_name"] = (
        "dwm.pipelines.lyh.bev_pv_epipolar.BEVPipeline"
    )
    pipeline["model"]["_class_name"] = (
        "dwm.models.lyh.bev_pv_plucker_epipolar."
        "BEVConditionedSD3TransformerModel"
    )
    pipeline["model"]["view_consistency_config"] = {
        "enabled": True,
        "layer_id": 13,
        "projector_dim": 128,
        "projector_hidden_dim": 256,
        "projector_layers": 2,
    }

    training = pipeline["training_config"]
    training.update(
        {
            "view_consistency_loss_weight": float(loss_weight),
            "view_consistency_loss_warmup_steps": int(warmup_steps),
            "view_consistency_time_divisor": 16,
            "view_consistency_pairs_per_time": 2,
            "view_consistency_enable_crossview": True,
            "view_consistency_enable_crossframe": True,
            "view_consistency_crossframe_stride": 1,
            "view_consistency_max_sigma": 0.5,
            "view_consistency_max_foreground_patches": 128,
            "view_consistency_max_background_patches": 128,
            "view_consistency_min_patches": 4,
            "view_consistency_epipolar_band_width": 2.5,
            "view_consistency_negative_band_scale": 1.0,
            "view_consistency_margin": 0.1,
            "view_consistency_temperature": 0.07,
            "view_consistency_lower_half_start_ratio": 0.5,
            "view_consistency_box_dilation_kernel": 31,
        }
    )

    output_path = config.get("output_path")
    if output_path:
        config["output_path"] = output_path.rstrip("/") + "_epipolar"
    return config


def write_new_file(path: Path, content: str, force: bool):
    if path.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {path}. "
            "Use --force only if you intentionally want to replace the "
            "generated experimental file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a BEV+PV SD3 epipolar-consistency experiment without "
            "modifying the original BEV-PV implementation."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--loss-weight",
        type=float,
        required=True,
        help=(
            "Coefficient lambda for L_total = L_sd + lambda * L_epipolar. "
            "The supplied Wan package does not provide an enabled final "
            "value, so this is intentionally required instead of guessed."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    if args.loss_weight <= 0.0:
        raise ValueError("--loss-weight must be > 0 for an enabled loss.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")

    root = args.root.resolve()
    base_pipeline = root / "src/dwm/pipelines/lyh/bev_pv.py"
    base_model = root / "src/dwm/models/lyh/bev_pv_plucker.py"
    base_config = root / "configs/lyh/PV_track_train.json"

    new_pipeline = root / "src/dwm/pipelines/lyh/bev_pv_epipolar.py"
    new_model = root / "src/dwm/models/lyh/bev_pv_plucker_epipolar.py"
    new_utils = root / "src/dwm/utils/view_consistency.py"
    new_config = root / "configs/lyh/PV_track_train_epipolar.json"

    for path in (base_pipeline, base_model, base_config):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

    print("=== git status before patch ===")
    try:
        subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=False,
        )
    except FileNotFoundError:
        print("git executable not found; continuing without status output.")

    model_source = base_model.read_text(encoding="utf-8")
    pipeline_source = base_pipeline.read_text(encoding="utf-8")
    config = json.loads(base_config.read_text(encoding="utf-8"))

    patched_model = patch_model(model_source)
    patched_pipeline = patch_pipeline(pipeline_source)
    patched_config = patch_config(
        config,
        args.loss_weight,
        args.warmup_steps,
    )

    write_new_file(new_utils, UTILS_SOURCE.rstrip() + "\n", args.force)
    write_new_file(new_model, patched_model, args.force)
    write_new_file(new_pipeline, patched_pipeline, args.force)
    write_new_file(
        new_config,
        json.dumps(patched_config, indent=4, ensure_ascii=False) + "\n",
        args.force,
    )

    print("\nCreated:")
    for path in (new_utils, new_model, new_pipeline, new_config):
        print(f"  {path.relative_to(root)}")

    print("\n=== syntax check ===")
    subprocess.run(
        [
            "python",
            "-m",
            "py_compile",
            str(new_utils),
            str(new_model),
            str(new_pipeline),
        ],
        check=True,
        cwd=root,
    )
    print("py_compile: OK")

    print("\n=== git diff --stat / status ===")
    subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=False,
    )
    print("\nNo training or evaluation was launched.")


if __name__ == "__main__":
    main()
