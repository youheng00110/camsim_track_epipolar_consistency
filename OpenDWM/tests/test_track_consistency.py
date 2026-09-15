import torch

from dwm.models.lyh.bev_pv_plucker_epipolar import (
    BEVConditionedSD3TransformerModel,
    TrackConsistencyProjector,
    project_selected_feature_pairs,
)
from dwm.utils.track_consistency import (
    SPATIAL_PAIR,
    TEMPORAL_PAIR,
    compute_track_consistency_loss,
    directional_track_infonce,
    sample_track_consistency_selection,
)


def make_object(embedding, class_id=0, dimensions=(4.0, 2.0, 1.5)):
    return {
        "embedding": torch.nn.functional.normalize(
            torch.tensor(embedding, dtype=torch.float32),
            dim=0,
        ),
        "class_id": class_id,
        "dimensions": torch.tensor(dimensions, dtype=torch.float32),
    }


def make_box(center_x):
    template = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    return template + torch.tensor([center_x, 0.0, 5.0])


def directional_loss(source, target, hard_weight=1.0):
    result = directional_track_infonce(
        source,
        target,
        temperature=0.1,
        hard_negative_enabled=True,
        hard_negative_size_threshold=0.2,
        hard_negative_weight=hard_weight,
    )
    if not result["losses"]:
        return None, result
    return torch.stack(result["losses"]).mean(), result


def test_directional_infonce_semantics():
    source = {1: make_object([1.0, 0.0])}
    good_target = {
        1: make_object([1.0, 0.0]),
        2: make_object([0.0, 1.0]),
    }
    bad_target = {
        1: make_object([0.0, 1.0]),
        2: make_object([1.0, 0.0]),
    }
    good_loss, _ = directional_loss(source, good_target)
    bad_loss, _ = directional_loss(source, bad_target)
    assert good_loss < bad_loss

    no_negative_loss, no_negative = directional_loss(
        source,
        {1: make_object([1.0, 0.0])},
    )
    assert no_negative_loss is None
    assert no_negative["valid_query_count"] == 0

    no_positive_loss, no_positive = directional_loss(
        source,
        {2: make_object([0.0, 1.0])},
    )
    assert no_positive_loss is None
    assert no_positive["valid_query_count"] == 0

    base_loss, base = directional_loss(source, good_target, hard_weight=1.0)
    weighted_loss, weighted = directional_loss(
        source,
        good_target,
        hard_weight=2.0,
    )
    assert weighted_loss >= base_loss
    assert base["hard_negative_count"] == 1
    assert weighted["hard_negative_count"] == 1


def test_pair_sampler_matches_track_id_across_changed_slots():
    batch = {
        "bbox_token_track_ids": torch.tensor([[[1, 2, 0], [2, 1, 0]]]),
        "bbox_token_classes": torch.zeros(1, 2, 2, 3, dtype=torch.long),
        "bbox_token_masks": torch.tensor(
            [[[[1, 1, 0], [1, 1, 0]], [[1, 1, 0], [1, 1, 0]]]],
            dtype=torch.bool,
        ),
        "view_consistency_pair_mask": torch.tensor(
            [[[False, True], [True, False]]]
        ),
    }
    config = {
        "track_consistency_loss_weight": 1.0,
        "track_consistency_vehicle_class_ids": [0, 1, 2, 3, 4],
        "track_consistency_enable_spatial": True,
        "track_consistency_enable_temporal": True,
        "track_consistency_temporal_stride": 1,
        "track_consistency_max_spatial_pairs_per_sample": 8,
        "track_consistency_max_temporal_pairs_per_sample": 8,
    }
    selection, selection_cpu, pair_types = sample_track_consistency_selection(
        batch,
        config,
        torch.Generator().manual_seed(0),
        torch.device("cpu"),
    )
    assert torch.equal(selection, selection_cpu)
    selected = {
        tuple(pair.tolist()): int(pair_type)
        for pair, pair_type in zip(selection_cpu[0], pair_types[0])
        if int(pair_type) >= 0
    }
    assert selected[(0, 0, 1, 0)] == TEMPORAL_PAIR
    assert selected[(0, 0, 0, 1)] == SPATIAL_PAIR
    assert all(
        view_a != view_b
        for (_, view_a, _, view_b), pair_type in selected.items()
        if pair_type == SPATIAL_PAIR
    )


def test_selected_projector_connects_projector_and_backbone_gradients():
    hidden = torch.randn(4, 4, 8, requires_grad=True)
    selection = torch.tensor([[[0, 0, 1, 1]]])
    projector = TrackConsistencyProjector(
        in_channels=8,
        out_channels=4,
        hidden_channels=6,
        num_layers=2,
    )
    projected = project_selected_feature_pairs(
        hidden,
        selection,
        projector,
        batch_size=1,
        sequence_length=2,
        view_count=2,
        height=2,
        width=2,
        selection_name="test_selection",
    )
    assert projected.shape == (1, 1, 2, 4, 2, 2)
    source = {
        1: {
            "embedding": torch.nn.functional.normalize(
                projected[0, 0, 0, :, 0, 0], dim=0
            ),
            "class_id": 0,
            "dimensions": torch.tensor([1.5, 2.0, 4.0]),
        }
    }
    target = {
        1: {
            "embedding": torch.nn.functional.normalize(
                projected[0, 0, 1, :, 0, 0], dim=0
            ),
            "class_id": 0,
            "dimensions": torch.tensor([1.5, 2.0, 4.0]),
        },
        2: {
            "embedding": torch.nn.functional.normalize(
                projected[0, 0, 1, :, 1, 1], dim=0
            ),
            "class_id": 0,
            "dimensions": torch.tensor([1.5, 2.0, 4.0]),
        },
    }
    loss, _ = directional_loss(source, target, hard_weight=1.5)
    assert loss.requires_grad
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in projector.parameters()
    )


def test_model_builds_independent_track_projector_and_defaults_off():
    common_args = {
        "sample_size": 8,
        "patch_size": 2,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 14,
        "attention_head_dim": 4,
        "num_attention_heads": 2,
        "joint_attention_dim": 16,
        "caption_projection_dim": 8,
        "pooled_projection_dim": 8,
        "pos_embed_max_size": 8,
        "block_layers": [13],
        "bev_in_channels": 3,
        "bev_hidden_channels": 4,
        "bbox_config": {
            "hidden_dim": 8,
            "class_dim": 8,
            "temporal_heads": 2,
        },
    }
    old_model = BEVConditionedSD3TransformerModel(**common_args)
    assert not old_model.enable_track_consistency_features
    assert old_model.track_consistency_projector is None

    track_model = BEVConditionedSD3TransformerModel(
        **common_args,
        view_consistency_config={
            "enabled": True,
            "layer_id": 13,
            "projector_dim": 4,
            "projector_hidden_dim": 6,
            "projector_layers": 2,
        },
        track_consistency_config={
            "enabled": True,
            "layer_id": 13,
            "projector_dim": 4,
            "projector_hidden_dim": 6,
            "projector_layers": 2,
        },
    )
    assert track_model.enable_track_consistency_features
    assert (
        track_model.track_consistency_projector
        is not track_model.view_consistency_projector
    )
    track_parameter_ids = {
        id(parameter)
        for parameter in track_model.track_consistency_projector.parameters()
    }
    view_parameter_ids = {
        id(parameter)
        for parameter in track_model.view_consistency_projector.parameters()
    }
    assert track_parameter_ids.isdisjoint(view_parameter_ids)
    missing, unexpected = track_model.load_state_dict(
        old_model.state_dict(),
        strict=False,
    )
    assert any(key.startswith("track_consistency_projector.") for key in missing)
    assert not unexpected


def test_complete_track_loss_is_finite_and_connected():
    corners = torch.stack([make_box(-1.0), make_box(1.0)])
    corners = corners[None, None, None].expand(1, 2, 1, 2, 8, 3).clone()
    identity = torch.eye(4)
    batch = {
        "bbox_token_track_ids": torch.tensor([[[1, 2], [1, 2]]]),
        "bbox_token_classes": torch.zeros(1, 2, 1, 2, dtype=torch.long),
        "bbox_token_masks": torch.ones(1, 2, 1, 2, dtype=torch.bool),
        "bbox_token_corners": corners,
        "camera_intrinsics": torch.tensor(
            [[[[[16.0, 0.0, 16.0], [0.0, 9.0, 9.0], [0.0, 0.0, 1.0]]],
              [[[16.0, 0.0, 16.0], [0.0, 9.0, 9.0], [0.0, 0.0, 1.0]]]]]
        ),
        "image_size": torch.tensor([[[[32, 18]], [[32, 18]]]]),
        "camera_transforms": identity.reshape(1, 1, 1, 4, 4).expand(
            1, 2, 1, 4, 4
        ),
        "ego_transforms": identity.reshape(1, 1, 4, 4).expand(1, 2, 4, 4),
        "reference_ego_transforms": identity.reshape(1, 1, 4, 4).expand(
            1, 2, 4, 4
        ),
    }
    projected = torch.randn(1, 1, 2, 4, 18, 32, requires_grad=True)
    statistics = compute_track_consistency_loss(
        batch=batch,
        projected_features=projected,
        selection_cpu=torch.tensor([[[0, 0, 1, 0]]]),
        pair_types_cpu=torch.tensor([[TEMPORAL_PAIR]]),
        training_config={
            "track_consistency_vehicle_class_ids": [0, 1, 2, 3, 4],
            "track_consistency_min_region_patches": 2,
            "track_consistency_temperature": 0.07,
            "track_consistency_hard_negative_enabled": True,
            "track_consistency_hard_negative_size_threshold": 0.2,
            "track_consistency_hard_negative_weight": 1.5,
        },
        device=torch.device("cpu"),
    )
    loss = statistics["track_loss"]
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert statistics["track_valid_pair_count"] == 1
    assert statistics["track_valid_query_count"] == 4
    loss.backward()
    assert projected.grad is not None
    assert projected.grad.abs().sum() > 0


def test_tiny_model_forward_returns_only_selected_track_pairs():
    model = BEVConditionedSD3TransformerModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=14,
        attention_head_dim=4,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=8,
        pooled_projection_dim=8,
        pos_embed_max_size=8,
        block_layers=[13],
        bev_in_channels=3,
        bev_hidden_channels=4,
        bbox_config={
            "hidden_dim": 8,
            "class_dim": 8,
            "temporal_heads": 2,
        },
        track_consistency_config={
            "enabled": True,
            "layer_id": 13,
            "projector_dim": 4,
            "projector_hidden_dim": 6,
            "projector_layers": 2,
        },
        view_consistency_config={
            "enabled": True,
            "layer_id": 13,
            "projector_dim": 3,
            "projector_hidden_dim": 6,
            "projector_layers": 2,
        },
    ).eval()
    batch_size, time_count, view_count, slot_count = 1, 2, 2, 2
    identity = torch.eye(4)
    boxes = torch.stack([make_box(-1.0), make_box(1.0)])
    boxes = boxes.reshape(1, 1, 1, slot_count, 8, 3).expand(
        batch_size, time_count, view_count, slot_count, 8, 3
    )
    intrinsics = torch.tensor(
        [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 1, 3, 3).expand(1, 1, view_count, 3, 3)
    camera_to_ego = identity.reshape(1, 1, 1, 4, 4).expand(
        1, 1, view_count, 4, 4
    )
    with torch.no_grad():
        output, _, _ = model(
            sample=torch.randn(1, time_count, view_count, 4, 8, 8),
            timestep=torch.ones(1, time_count, view_count),
            encoder_hidden_states=torch.randn(
                1, time_count, view_count, 3, 16
            ),
            pooled_projections=torch.randn(1, time_count, view_count, 8),
            camera_intrinsics_norm=intrinsics,
            camera_to_ego=camera_to_ego,
            ego_to_initial=identity.reshape(1, 1, 4, 4).expand(
                1, time_count, 4, 4
            ),
            bbox_corners=boxes,
            bbox_classes=torch.zeros(
                1, time_count, view_count, slot_count, dtype=torch.long
            ),
            bbox_view_masks=torch.ones(
                1, time_count, view_count, slot_count, dtype=torch.bool
            ),
            bev_map=torch.randn(1, time_count, 3, 8, 8),
            crossview_attention_mask=torch.ones(
                1, view_count, view_count, dtype=torch.bool
            ),
            condition_keep=torch.ones(1, dtype=torch.bool),
            disable_temporal=torch.zeros(1, 1, 1, dtype=torch.bool),
            view_consistency_selection=torch.tensor([[[0, 0, 0, 1]]]),
            track_consistency_selection=torch.tensor([[[0, 0, 1, 1]]]),
        )
    assert len(output) == 3
    assert output[0].shape == (1, time_count, view_count, 4, 8, 8)
    assert output[1].shape == (1, 1, 2, 3, 4, 4)
    assert output[2].shape == (1, 1, 2, 4, 4, 4)
