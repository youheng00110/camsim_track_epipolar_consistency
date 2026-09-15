from typing import Optional

import einops
import torch


def prepare_view_consistency_geometry(
    batch: dict,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use each camera's exposure pose, expressed in one common reference."""
    intrinsics = batch["camera_intrinsics"].to(device=device, dtype=torch.float32).clone()
    if intrinsics.ndim != 5 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera_intrinsics must be [B,T,V,3,3].")
    batch_size, time_count, view_count = intrinsics.shape[:3]
    if time_count != sequence_length:
        raise ValueError("camera_intrinsics time length must match latent time.")
    image_size = batch["image_size"].to(intrinsics)
    if image_size.shape != (batch_size, time_count, view_count, 2):
        raise ValueError("image_size must be [B,T,V,2] in width/height order.")
    intrinsics[..., 0, :] /= image_size[..., 0, None]
    intrinsics[..., 1, :] /= image_size[..., 1, None]

    camera_to_ego = batch["camera_transforms"].to(device=device, dtype=torch.float64)
    expected = (batch_size, time_count, view_count, 4, 4)
    if camera_to_ego.shape != expected:
        raise ValueError(f"camera_transforms must be {expected}.")
    world_from_ego = batch["ego_transforms"].to(device=device, dtype=torch.float64)
    if world_from_ego.shape == (batch_size, time_count, 4, 4):
        # A dataset may explicitly provide a synchronized per-frame pose.
        world_from_ego = world_from_ego[:, :, None].expand(expected)
    if world_from_ego.shape != expected:
        raise ValueError(f"ego_transforms must be {expected} or [B,T,4,4].")

    reference = batch.get("reference_ego_transforms")
    if reference is None:
        world_from_reference = world_from_ego[:, 0, 0]
    else:
        if reference.shape != (batch_size, time_count, 4, 4):
            raise ValueError("reference_ego_transforms must be [B,T,4,4].")
        world_from_reference = reference[:, 0].to(device=device, dtype=torch.float64)
    # Remove large world translations before casting relative geometry to FP32.
    ego_to_reference = torch.linalg.solve(
        world_from_reference[:, None, None], world_from_ego
    )
    camera_to_reference = ego_to_reference @ camera_to_ego
    return intrinsics, camera_to_reference.float()


def _feature_connected_zero(features: torch.Tensor) -> torch.Tensor:
    """A differentiable FP32 zero without a potentially overflowing FP16 sum."""
    return features.reshape(-1)[:1].float().sum() * 0.0


def scale_normalized_intrinsics_to_feature_grid(
    camera_intrinsics_norm: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Convert normalized intrinsics to the selected patch grid."""
    intrinsics = camera_intrinsics_norm.float().clone()
    intrinsics[..., 0, :] *= float(feature_width)
    intrinsics[..., 1, :] *= float(feature_height)
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
    positive_candidate_mask: Optional[torch.Tensor] = None,
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
    if float(negative_band_scale) <= 1.0:
        # Depth clipping may remove band pixels from positives, but does not
        # make them negatives. Such pixels are ignored in both sets.
        negative_mask = target_valid_mask & epipolar_distance.gt(
            float(epipolar_band_width)
        )
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

    Cross-view pairs use view_consistency_pair_mask at the same time.
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

    pair_mask = batch.get("view_consistency_pair_mask")
    if enable_crossview and pair_mask is None:
        raise KeyError(
            "view_consistency_pair_mask is required when "
            "view_consistency_enable_crossview=True. "
            "Do not reuse crossview_mask here: crossview_mask describes "
            "cross-view attention groups, while view_consistency_pair_mask "
            "describes valid pair-wise epipolar neighbors."
        )
    if pair_mask is not None:
        pair_mask = pair_mask.bool().cpu()
        if pair_mask.ndim == 4 and pair_mask.shape[1] == 1:
            pair_mask = pair_mask.squeeze(1)
        if pair_mask.ndim == 2:
            pair_mask = pair_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        if pair_mask.shape != (
            batch_size,
            view_count,
            view_count,
        ):
            raise ValueError(
                "view_consistency_pair_mask must be [B,V,V], got "
                f"{tuple(pair_mask.shape)}"
            )

    all_selections = []
    max_pair_count = 0

    configured_time_start = training_config.get(
        "view_consistency_time_start",
        None,
    )
    configured_time_end = training_config.get(
        "view_consistency_time_end",
        None,
    )

    for batch_index in range(batch_size):
        if configured_time_start is None and configured_time_end is None:
            # Backward-compatible behavior for configs without an explicit
            # view-consistency time window.
            valid_times = list(
                range(
                    min(reference_count, frame_count),
                    frame_count,
                )
            )
            if len(valid_times) == 0:
                valid_times = [frame_count - 1]
        else:
            # start is inclusive, end is exclusive.
            time_start = (
                reference_count
                if configured_time_start is None
                else int(configured_time_start)
            )
            time_end = (
                frame_count
                if configured_time_end is None
                else int(configured_time_end)
            )

            time_start = max(
                min(reference_count, frame_count),
                0,
                time_start,
            )
            time_end = min(
                frame_count,
                time_end,
            )

            # Cross-view and cross-frame share the same sampled source time.
            # If cross-frame is enabled, reserve t+stride so every sampled
            # time can produce both spatial and temporal pairs.
            if enable_crossframe:
                time_end = min(
                    time_end,
                    frame_count - crossframe_stride,
                )

            valid_times = list(range(time_start, time_end))
            if len(valid_times) == 0:
                return None

        sampled_times = sample_stratified_times(
            valid_times,
            sample_time_count,
            generator,
        )

        mask_row = (
            None
            if pair_mask is None
            else pair_mask[batch_index]
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
        return _feature_connected_zero(
            camera_intrinsics_norm if projected_features is None else projected_features
        )

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
    if bool(training_config.get("view_consistency_target_full_image", True)):
        # Target search extent is independent of query and depth restrictions.
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
        return _feature_connected_zero(projected_features)

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
                positive_candidate_mask=positive_candidate_mask_a_to_b,
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
                positive_candidate_mask=positive_candidate_mask_b_to_a,
            )
            pair_losses.append(0.5 * (loss_a_to_b + loss_b_to_a))

    if len(pair_losses) == 0:
        return _feature_connected_zero(projected_features)
    return torch.stack(pair_losses).mean()
