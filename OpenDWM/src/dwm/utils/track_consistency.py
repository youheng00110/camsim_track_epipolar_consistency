"""Track-ID supervised object-level consistency utilities."""

import torch

import dwm.utils.view_consistency


SPATIAL_PAIR = 0
TEMPORAL_PAIR = 1


def _feature_connected_zero(features: torch.Tensor) -> torch.Tensor:
    return features.reshape(-1)[:1].float().sum() * 0.0


def _vehicle_mask(
    classes: torch.Tensor,
    vehicle_class_ids: tuple[int, ...],
) -> torch.Tensor:
    result = torch.zeros_like(classes, dtype=torch.bool)
    for class_id in vehicle_class_ids:
        result |= classes == int(class_id)
    return result


def _valid_object_mask(
    track_ids: torch.Tensor,
    classes: torch.Tensor,
    visibility: torch.Tensor,
    vehicle_class_ids: tuple[int, ...],
) -> torch.Tensor:
    return (
        track_ids.gt(0)
        & visibility.bool()
        & _vehicle_mask(classes, vehicle_class_ids)
    )


def _pair_has_eligible_query(
    track_ids_a: torch.Tensor,
    classes_a: torch.Tensor,
    visibility_a: torch.Tensor,
    track_ids_b: torch.Tensor,
    classes_b: torch.Tensor,
    visibility_b: torch.Tensor,
    vehicle_class_ids: tuple[int, ...],
) -> bool:
    valid_a = _valid_object_mask(
        track_ids_a,
        classes_a,
        visibility_a,
        vehicle_class_ids,
    )
    valid_b = _valid_object_mask(
        track_ids_b,
        classes_b,
        visibility_b,
        vehicle_class_ids,
    )
    ids_a = set(track_ids_a[valid_a].tolist())
    ids_b = set(track_ids_b[valid_b].tolist())
    shared = ids_a.intersection(ids_b)
    return bool(shared) and len(ids_a) >= 2 and len(ids_b) >= 2


def _choose_pairs(
    pairs: list[list[int]],
    maximum: int,
    generator: torch.Generator,
) -> list[list[int]]:
    if maximum <= 0 or len(pairs) <= maximum:
        return pairs
    indices = torch.randperm(
        len(pairs),
        generator=generator,
        device="cpu",
    )[:maximum]
    return [pairs[int(index)] for index in indices]


def sample_track_consistency_selection(
    batch: dict,
    training_config: dict,
    generator: torch.Generator,
    device: torch.device,
):
    """Select eligible spatial and temporal pairs for Track-ID loss."""
    loss_weight = float(
        training_config.get("track_consistency_loss_weight", 0.0)
    )
    if loss_weight < 0.0:
        raise ValueError("track_consistency_loss_weight must be non-negative.")
    if loss_weight <= 0.0:
        return None

    required = (
        "bbox_token_track_ids",
        "bbox_token_classes",
        "bbox_token_masks",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            f"Track consistency requires batch keys: {missing}."
        )

    track_ids = batch["bbox_token_track_ids"].long().cpu()
    classes = batch["bbox_token_classes"].long().cpu()
    visibility = batch["bbox_token_masks"].bool().cpu()
    if track_ids.ndim != 3:
        raise ValueError(
            "bbox_token_track_ids must be [B,T,S], got "
            f"{tuple(track_ids.shape)}."
        )
    batch_size, frame_count, slot_count = track_ids.shape
    if classes.ndim != 4 or visibility.shape != classes.shape:
        raise ValueError(
            "bbox_token_classes and bbox_token_masks must be [B,T,V,S]."
        )
    if classes.shape[:2] != (batch_size, frame_count):
        raise ValueError("Track ID and bbox token time dimensions differ.")
    view_count = int(classes.shape[2])
    if classes.shape[3] != slot_count:
        raise ValueError("Track ID and bbox token slot dimensions differ.")

    vehicle_class_ids = tuple(
        int(value)
        for value in training_config[
            "track_consistency_vehicle_class_ids"
        ]
    )
    if not vehicle_class_ids:
        raise ValueError(
            "track_consistency_vehicle_class_ids must not be empty when "
            "Track consistency is enabled."
        )
    enable_spatial = bool(
        training_config.get("track_consistency_enable_spatial", True)
    )
    enable_temporal = bool(
        training_config.get("track_consistency_enable_temporal", True)
    )
    temporal_stride = int(
        training_config.get("track_consistency_temporal_stride", 1)
    )
    if temporal_stride <= 0:
        raise ValueError("track_consistency_temporal_stride must be positive.")
    max_spatial = int(
        training_config.get(
            "track_consistency_max_spatial_pairs_per_sample",
            8,
        )
    )
    max_temporal = int(
        training_config.get(
            "track_consistency_max_temporal_pairs_per_sample",
            8,
        )
    )
    if enable_spatial and max_spatial <= 0:
        raise ValueError(
            "track_consistency_max_spatial_pairs_per_sample must be positive."
        )
    if enable_temporal and max_temporal <= 0:
        raise ValueError(
            "track_consistency_max_temporal_pairs_per_sample must be positive."
        )

    pair_mask = batch.get("view_consistency_pair_mask")
    if enable_spatial:
        if pair_mask is None:
            raise KeyError(
                "Spatial Track consistency requires "
                "view_consistency_pair_mask; crossview_mask is not a valid "
                "replacement."
            )
        pair_mask = pair_mask.bool().cpu()
        if pair_mask.ndim == 4 and pair_mask.shape[1] == 1:
            pair_mask = pair_mask.squeeze(1)
        if pair_mask.ndim == 2:
            pair_mask = pair_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        if pair_mask.shape != (batch_size, view_count, view_count):
            raise ValueError(
                "view_consistency_pair_mask must be [B,V,V], got "
                f"{tuple(pair_mask.shape)}."
            )

    selections = []
    pair_types = []
    maximum_pair_count = 0
    for batch_index in range(batch_size):
        spatial_pairs = []
        temporal_pairs = []
        if enable_spatial:
            for time_id in range(frame_count):
                for view_a in range(view_count):
                    for view_b in range(view_a + 1, view_count):
                        if not bool(
                            pair_mask[batch_index, view_a, view_b]
                            or pair_mask[batch_index, view_b, view_a]
                        ):
                            continue
                        if _pair_has_eligible_query(
                            track_ids[batch_index, time_id],
                            classes[batch_index, time_id, view_a],
                            visibility[batch_index, time_id, view_a],
                            track_ids[batch_index, time_id],
                            classes[batch_index, time_id, view_b],
                            visibility[batch_index, time_id, view_b],
                            vehicle_class_ids,
                        ):
                            spatial_pairs.append(
                                [time_id, view_a, time_id, view_b]
                            )

        if enable_temporal:
            for time_a in range(frame_count - temporal_stride):
                time_b = time_a + temporal_stride
                for view_id in range(view_count):
                    if _pair_has_eligible_query(
                        track_ids[batch_index, time_a],
                        classes[batch_index, time_a, view_id],
                        visibility[batch_index, time_a, view_id],
                        track_ids[batch_index, time_b],
                        classes[batch_index, time_b, view_id],
                        visibility[batch_index, time_b, view_id],
                        vehicle_class_ids,
                    ):
                        temporal_pairs.append(
                            [time_a, view_id, time_b, view_id]
                        )

        spatial_pairs = _choose_pairs(
            spatial_pairs,
            max_spatial,
            generator,
        )
        temporal_pairs = _choose_pairs(
            temporal_pairs,
            max_temporal,
            generator,
        )
        sample_pairs = spatial_pairs + temporal_pairs
        sample_types = (
            [SPATIAL_PAIR] * len(spatial_pairs)
            + [TEMPORAL_PAIR] * len(temporal_pairs)
        )
        selection = torch.tensor(sample_pairs, dtype=torch.long).reshape(-1, 4)
        types = torch.tensor(sample_types, dtype=torch.long)
        selections.append(selection)
        pair_types.append(types)
        maximum_pair_count = max(maximum_pair_count, len(sample_pairs))

    if maximum_pair_count == 0:
        return None

    selection_cpu = torch.full(
        (batch_size, maximum_pair_count, 4),
        -1,
        dtype=torch.long,
    )
    pair_types_cpu = torch.full(
        (batch_size, maximum_pair_count),
        -1,
        dtype=torch.long,
    )
    for batch_index, selection in enumerate(selections):
        count = int(selection.shape[0])
        if count == 0:
            continue
        selection_cpu[batch_index, :count] = selection
        pair_types_cpu[batch_index, :count] = pair_types[batch_index]
    return (
        selection_cpu.to(device=device),
        selection_cpu,
        pair_types_cpu,
    )


def prepare_track_consistency_geometry(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized K and camera/box frames in a common reference."""
    frame_count = int(batch["bbox_token_track_ids"].shape[1])
    intrinsics, camera_to_common = (
        dwm.utils.view_consistency.prepare_view_consistency_geometry(
            batch,
            frame_count,
            device,
        )
    )
    reference = batch["reference_ego_transforms"].to(
        device=device,
        dtype=torch.float64,
    )
    if reference.ndim != 4 or reference.shape[-2:] != (4, 4):
        raise ValueError(
            "reference_ego_transforms must be [B,T,4,4]."
        )
    box_reference_to_common = torch.linalg.solve(
        reference[:, :1],
        reference,
    ).float()
    return intrinsics, camera_to_common, box_reference_to_common


def project_box_to_feature_mask(
    corners: torch.Tensor,
    camera_intrinsics_norm: torch.Tensor,
    camera_to_common: torch.Tensor,
    box_reference_to_common: torch.Tensor,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Rasterize the clipped 2D bounding rectangle of a projected 3D box."""
    mask = torch.zeros(
        feature_height,
        feature_width,
        device=corners.device,
        dtype=torch.bool,
    )
    corners_h = torch.cat(
        [corners.float(), torch.ones_like(corners[..., :1])],
        dim=-1,
    )
    common_points = (
        box_reference_to_common.float() @ corners_h.transpose(0, 1)
    )
    camera_points = torch.linalg.solve(
        camera_to_common.float(),
        common_points,
    )[:3].transpose(0, 1)
    positive_depth = camera_points[:, 2] > 1e-4
    if not bool(positive_depth.any()):
        return mask

    intrinsics = (
        dwm.utils.view_consistency.scale_normalized_intrinsics_to_feature_grid(
            camera_intrinsics_norm,
            feature_height,
            feature_width,
        )
    )
    projected = (
        intrinsics @ camera_points[positive_depth].transpose(0, 1)
    ).transpose(0, 1)
    xy = projected[:, :2] / projected[:, 2:].clamp_min(1e-4)
    x_min = max(int(torch.floor(xy[:, 0].min()).item()), 0)
    x_max = min(int(torch.ceil(xy[:, 0].max()).item()), feature_width)
    y_min = max(int(torch.floor(xy[:, 1].min()).item()), 0)
    y_max = min(int(torch.ceil(xy[:, 1].max()).item()), feature_height)
    if x_max <= x_min or y_max <= y_min:
        return mask
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def box_dimensions_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """Recover sorted physical side lengths from the standard 8-corner order."""
    dimensions = torch.stack(
        [
            torch.linalg.vector_norm(corners[4] - corners[0]),
            torch.linalg.vector_norm(corners[2] - corners[0]),
            torch.linalg.vector_norm(corners[1] - corners[0]),
        ]
    )
    return torch.sort(dimensions.clamp_min(1e-6)).values


def dimension_log_distance(
    dimensions_a: torch.Tensor,
    dimensions_b: torch.Tensor,
) -> torch.Tensor:
    return torch.abs(
        torch.log(dimensions_a.clamp_min(1e-6))
        - torch.log(dimensions_b.clamp_min(1e-6))
    ).mean()


def _collect_endpoint_objects(
    feature_map: torch.Tensor,
    track_ids: torch.Tensor,
    classes: torch.Tensor,
    visibility: torch.Tensor,
    corners: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_common: torch.Tensor,
    box_reference_to_common: torch.Tensor,
    vehicle_class_ids: tuple[int, ...],
    min_region_patches: int,
) -> tuple[dict, int]:
    valid = _valid_object_mask(
        track_ids,
        classes,
        visibility,
        vehicle_class_ids,
    )
    objects = {}
    skipped_small = 0
    for slot in valid.nonzero(as_tuple=False).flatten().tolist():
        region = project_box_to_feature_mask(
            corners[slot],
            intrinsics,
            camera_to_common,
            box_reference_to_common,
            int(feature_map.shape[-2]),
            int(feature_map.shape[-1]),
        )
        patch_count = int(region.sum().item())
        if patch_count < min_region_patches:
            skipped_small += 1
            continue
        pooled = feature_map[:, region].float().mean(dim=1)
        embedding = torch.nn.functional.normalize(
            pooled,
            dim=0,
        )
        track_id = int(track_ids[slot].item())
        if track_id in objects:
            raise ValueError(
                f"Duplicate visible Track ID {track_id} in one endpoint."
            )
        objects[track_id] = {
            "embedding": embedding,
            "class_id": int(classes[slot].item()),
            "dimensions": box_dimensions_from_corners(corners[slot]),
        }
    return objects, skipped_small


def directional_track_infonce(
    source_objects: dict,
    target_objects: dict,
    temperature: float,
    hard_negative_enabled: bool,
    hard_negative_size_threshold: float,
    hard_negative_weight: float,
) -> dict:
    """Compute one direction using every valid target different-ID object."""
    if temperature <= 0.0:
        raise ValueError("track_consistency_temperature must be positive.")
    if hard_negative_weight <= 0.0:
        raise ValueError(
            "track_consistency_hard_negative_weight must be positive."
        )

    losses = []
    positive_similarities = []
    negative_similarities = []
    hard_negative_similarities = []
    retrieval_correct = 0
    hard_negative_count = 0

    for track_id, source in source_objects.items():
        if track_id not in target_objects:
            continue
        negative_ids = [
            candidate
            for candidate in target_objects
            if candidate != track_id
        ]
        if not negative_ids:
            continue

        source_embedding = source["embedding"]
        positive_similarity = torch.dot(
            source_embedding,
            target_objects[track_id]["embedding"],
        )
        negative_values = []
        negative_weights = []
        for negative_id in negative_ids:
            negative = target_objects[negative_id]
            similarity = torch.dot(
                source_embedding,
                negative["embedding"],
            )
            is_hard = False
            if (
                hard_negative_enabled
                and source["class_id"] == negative["class_id"]
            ):
                size_distance = dimension_log_distance(
                    source["dimensions"],
                    negative["dimensions"],
                )
                is_hard = bool(
                    size_distance <= hard_negative_size_threshold
                )
            negative_values.append(similarity)
            negative_weights.append(
                hard_negative_weight if is_hard else 1.0
            )
            negative_similarities.append(similarity)
            if is_hard:
                hard_negative_count += 1
                hard_negative_similarities.append(similarity)

        negative_tensor = torch.stack(negative_values)
        weight_tensor = torch.tensor(
            negative_weights,
            device=negative_tensor.device,
            dtype=negative_tensor.dtype,
        )
        positive_logit = positive_similarity / temperature
        negative_logits = (
            negative_tensor / temperature + weight_tensor.log()
        )
        losses.append(
            torch.logsumexp(
                torch.cat([positive_logit.reshape(1), negative_logits]),
                dim=0,
            )
            - positive_logit
        )
        positive_similarities.append(positive_similarity)
        all_similarities = torch.cat(
            [positive_similarity.reshape(1), negative_tensor]
        )
        retrieval_correct += int(all_similarities.argmax().item() == 0)

    return {
        "losses": losses,
        "positive_similarities": positive_similarities,
        "negative_similarities": negative_similarities,
        "hard_negative_similarities": hard_negative_similarities,
        "valid_query_count": len(losses),
        "retrieval_correct": retrieval_correct,
        "hard_negative_count": hard_negative_count,
    }


def _mean_or_zero(
    values: list[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero


def compute_track_consistency_loss(
    batch: dict,
    projected_features: torch.Tensor,
    selection_cpu: torch.Tensor,
    pair_types_cpu: torch.Tensor,
    training_config: dict,
    device: torch.device,
) -> dict:
    """Compute bidirectional object-level Track-ID InfoNCE and diagnostics."""
    if projected_features is None:
        raise ValueError("Track selection was provided but features are missing.")
    if projected_features.ndim != 6 or projected_features.shape[2] != 2:
        raise ValueError(
            "track projected features must be [B,K,2,D,H,W], got "
            f"{tuple(projected_features.shape)}."
        )

    track_ids = batch["bbox_token_track_ids"].to(
        device=device,
        dtype=torch.long,
    )
    classes = batch["bbox_token_classes"].to(
        device=device,
        dtype=torch.long,
    )
    visibility = batch["bbox_token_masks"].to(
        device=device,
        dtype=torch.bool,
    )
    corners = batch["bbox_token_corners"].to(
        device=device,
        dtype=torch.float32,
    )
    intrinsics, camera_to_common, box_reference_to_common = (
        prepare_track_consistency_geometry(batch, device)
    )

    vehicle_class_ids = tuple(
        int(value)
        for value in training_config[
            "track_consistency_vehicle_class_ids"
        ]
    )
    min_region_patches = int(
        training_config.get("track_consistency_min_region_patches", 2)
    )
    if min_region_patches <= 0:
        raise ValueError(
            "track_consistency_min_region_patches must be positive."
        )
    temperature = float(
        training_config.get("track_consistency_temperature", 0.07)
    )
    hard_enabled = bool(
        training_config.get(
            "track_consistency_hard_negative_enabled",
            True,
        )
    )
    hard_threshold = float(
        training_config.get(
            "track_consistency_hard_negative_size_threshold",
            0.2,
        )
    )
    hard_weight = float(
        training_config.get(
            "track_consistency_hard_negative_weight",
            1.0,
        )
    )
    if hard_threshold < 0.0:
        raise ValueError(
            "track_consistency_hard_negative_size_threshold must be "
            "non-negative."
        )

    zero = _feature_connected_zero(projected_features)
    pair_losses = {SPATIAL_PAIR: [], TEMPORAL_PAIR: []}
    positive_similarities = []
    negative_similarities = []
    hard_negative_similarities = []
    valid_pair_count = 0
    valid_query_count = 0
    skipped_small_count = 0
    hard_negative_count = 0
    retrieval_correct = 0
    retrieval_total = 0
    spatial_retrieval_correct = 0
    spatial_retrieval_total = 0
    temporal_retrieval_correct = 0
    temporal_retrieval_total = 0

    batch_size, pair_count = selection_cpu.shape[:2]
    for batch_index in range(batch_size):
        for pair_index in range(pair_count):
            pair_type = int(pair_types_cpu[batch_index, pair_index].item())
            time_a, view_a, time_b, view_b = selection_cpu[
                batch_index,
                pair_index,
            ].tolist()
            if pair_type < 0 or min(time_a, view_a, time_b, view_b) < 0:
                continue

            objects_a, skipped_a = _collect_endpoint_objects(
                projected_features[batch_index, pair_index, 0],
                track_ids[batch_index, time_a],
                classes[batch_index, time_a, view_a],
                visibility[batch_index, time_a, view_a],
                corners[batch_index, time_a, view_a],
                intrinsics[batch_index, time_a, view_a],
                camera_to_common[batch_index, time_a, view_a],
                box_reference_to_common[batch_index, time_a],
                vehicle_class_ids,
                min_region_patches,
            )
            objects_b, skipped_b = _collect_endpoint_objects(
                projected_features[batch_index, pair_index, 1],
                track_ids[batch_index, time_b],
                classes[batch_index, time_b, view_b],
                visibility[batch_index, time_b, view_b],
                corners[batch_index, time_b, view_b],
                intrinsics[batch_index, time_b, view_b],
                camera_to_common[batch_index, time_b, view_b],
                box_reference_to_common[batch_index, time_b],
                vehicle_class_ids,
                min_region_patches,
            )
            skipped_small_count += skipped_a + skipped_b

            directions = (
                directional_track_infonce(
                    objects_a,
                    objects_b,
                    temperature,
                    hard_enabled,
                    hard_threshold,
                    hard_weight,
                ),
                directional_track_infonce(
                    objects_b,
                    objects_a,
                    temperature,
                    hard_enabled,
                    hard_threshold,
                    hard_weight,
                ),
            )
            directional_mean_losses = []
            pair_query_count = 0
            pair_retrieval_correct = 0
            for direction in directions:
                if direction["losses"]:
                    directional_mean_losses.append(
                        torch.stack(direction["losses"]).mean()
                    )
                positive_similarities.extend(
                    direction["positive_similarities"]
                )
                negative_similarities.extend(
                    direction["negative_similarities"]
                )
                hard_negative_similarities.extend(
                    direction["hard_negative_similarities"]
                )
                pair_query_count += direction["valid_query_count"]
                pair_retrieval_correct += direction["retrieval_correct"]
                hard_negative_count += direction["hard_negative_count"]

            if not directional_mean_losses:
                continue
            valid_pair_count += 1
            valid_query_count += pair_query_count
            retrieval_correct += pair_retrieval_correct
            retrieval_total += pair_query_count
            if pair_type == SPATIAL_PAIR:
                spatial_retrieval_correct += pair_retrieval_correct
                spatial_retrieval_total += pair_query_count
            elif pair_type == TEMPORAL_PAIR:
                temporal_retrieval_correct += pair_retrieval_correct
                temporal_retrieval_total += pair_query_count
            else:
                raise ValueError(f"Unknown Track pair type {pair_type}.")
            pair_losses[pair_type].append(
                torch.stack(directional_mean_losses).mean()
            )

    spatial_loss = _mean_or_zero(pair_losses[SPATIAL_PAIR], zero)
    temporal_loss = _mean_or_zero(pair_losses[TEMPORAL_PAIR], zero)
    spatial_weight = float(
        training_config.get("track_consistency_spatial_weight", 1.0)
    )
    temporal_weight = float(
        training_config.get("track_consistency_temporal_weight", 1.0)
    )
    if spatial_weight < 0.0 or temporal_weight < 0.0:
        raise ValueError("Track spatial/temporal weights must be non-negative.")
    track_loss = (
        spatial_weight * spatial_loss
        + temporal_weight * temporal_loss
    )
    positive_mean = _mean_or_zero(positive_similarities, zero)
    negative_mean = _mean_or_zero(negative_similarities, zero)
    hard_negative_mean = _mean_or_zero(
        hard_negative_similarities,
        zero,
    )
    return {
        "track_loss": track_loss,
        "track_spatial_loss": spatial_loss,
        "track_temporal_loss": temporal_loss,
        "track_valid_pair_count": float(valid_pair_count),
        "track_valid_query_count": float(valid_query_count),
        "track_skipped_small_region_count": float(skipped_small_count),
        "track_hard_negative_count": float(hard_negative_count),
        "track_positive_similarity": positive_mean,
        "track_negative_similarity": negative_mean,
        "track_similarity_gap": positive_mean - negative_mean,
        "track_retrieval_at_1": float(retrieval_correct)
        / max(retrieval_total, 1),
        "track_spatial_retrieval_at_1": float(spatial_retrieval_correct)
        / max(spatial_retrieval_total, 1),
        "track_temporal_retrieval_at_1": float(temporal_retrieval_correct)
        / max(temporal_retrieval_total, 1),
        "hard_negative_mean_similarity": hard_negative_mean,
    }
