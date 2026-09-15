import math

import torch

from dwm.utils.track_consistency import (
    TEMPORAL_PAIR,
    compute_track_consistency_loss,
    directional_track_infonce,
    sample_track_consistency_selection,
)
from test_track_consistency import make_box, make_object


def _directional(source, target, hard_weight=1.5):
    return directional_track_infonce(
        source,
        target,
        temperature=0.1,
        hard_negative_enabled=True,
        hard_negative_size_threshold=0.2,
        hard_negative_weight=hard_weight,
    )


def _geometry_batch(visible=True):
    identity = torch.eye(4)
    corners = torch.stack([make_box(-1.0), make_box(1.0)])
    corners = corners[None, None, None].expand(1, 2, 1, 2, 8, 3).clone()
    masks = torch.ones(1, 2, 1, 2, dtype=torch.bool)
    if not visible:
        masks[:, 1, :, 0] = False
    return {
        "bbox_token_track_ids": torch.tensor([[[1, 2], [1, 2]]]),
        "bbox_token_classes": torch.zeros(1, 2, 1, 2, dtype=torch.long),
        "bbox_token_masks": masks,
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


def _loss_config(min_patches=2):
    return {
        "track_consistency_vehicle_class_ids": [0, 1, 2, 3, 4],
        "track_consistency_min_region_patches": min_patches,
        "track_consistency_temperature": 0.07,
        "track_consistency_hard_negative_enabled": True,
        "track_consistency_hard_negative_size_threshold": 0.2,
        "track_consistency_hard_negative_weight": 1.5,
    }


def test_hard_weight_matches_denominator_formula():
    source = {1: make_object([1.0, 0.0])}
    target = {
        1: make_object([0.8, 0.6]),
        2: make_object([0.3, math.sqrt(1.0 - 0.3 ** 2)]),
        3: make_object(
            [0.3, math.sqrt(1.0 - 0.3 ** 2)],
            class_id=1,
        ),
    }
    weighted = _directional(source, target, hard_weight=1.5)
    unweighted = _directional(source, target, hard_weight=1.0)
    actual = weighted["losses"][0]
    positive_similarity = torch.dot(
        source[1]["embedding"], target[1]["embedding"]
    )
    negative_similarities = torch.stack(
        [
            torch.dot(source[1]["embedding"], target[2]["embedding"]),
            torch.dot(source[1]["embedding"], target[3]["embedding"]),
        ]
    )
    expected = torch.log(
        torch.exp(positive_similarity / 0.1)
        + 1.5 * torch.exp(negative_similarities[0] / 0.1)
        + torch.exp(negative_similarities[1] / 0.1)
    ) - positive_similarity / 0.1
    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual > unweighted["losses"][0]


def test_nontrivial_query_and_retrieval_rules():
    source = {1: make_object([1.0, 0.0])}
    good = _directional(
        source,
        {
            1: make_object([0.9, math.sqrt(1.0 - 0.9 ** 2)]),
            2: make_object([0.4, math.sqrt(1.0 - 0.4 ** 2)]),
            3: make_object([0.2, math.sqrt(1.0 - 0.2 ** 2)]),
        },
    )
    bad = _directional(
        source,
        {
            1: make_object([0.4, math.sqrt(1.0 - 0.4 ** 2)]),
            2: make_object([0.9, math.sqrt(1.0 - 0.9 ** 2)]),
        },
    )
    only_positive = _directional(source, {1: make_object([1.0, 0.0])})
    no_positive = _directional(source, {2: make_object([0.0, 1.0])})
    assert good["valid_query_count"] == 1
    assert good["retrieval_correct"] == 1
    assert bad["retrieval_correct"] == 0
    assert only_positive["valid_query_count"] == 0
    assert no_positive["valid_query_count"] == 0


def test_invisible_positive_is_not_selected():
    batch = _geometry_batch(visible=False)
    batch["view_consistency_pair_mask"] = torch.zeros(1, 1, 1, dtype=torch.bool)
    result = sample_track_consistency_selection(
        batch,
        {
            "track_consistency_loss_weight": 1.0,
            "track_consistency_vehicle_class_ids": [0, 1, 2, 3, 4],
            "track_consistency_enable_spatial": False,
            "track_consistency_enable_temporal": True,
            "track_consistency_temporal_stride": 1,
            "track_consistency_max_temporal_pairs_per_sample": 8,
            "track_consistency_max_spatial_pairs_per_sample": 8,
        },
        torch.Generator().manual_seed(0),
        torch.device("cpu"),
    )
    assert result is None


def test_roi_too_small_skips_queries():
    features = torch.randn(1, 1, 2, 4, 18, 32, requires_grad=True)
    statistics = compute_track_consistency_loss(
        batch=_geometry_batch(),
        projected_features=features,
        selection_cpu=torch.tensor([[[0, 0, 1, 0]]]),
        pair_types_cpu=torch.tensor([[TEMPORAL_PAIR]]),
        training_config=_loss_config(min_patches=1000),
        device=torch.device("cpu"),
    )
    assert statistics["track_valid_query_count"] == 0
    assert statistics["track_skipped_small_region_count"] == 4
    assert statistics["track_loss"].requires_grad


def test_cpu_optimization_improves_identity_gap():
    query = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
    positive = torch.tensor([1.0, 0.0])
    negative = torch.tensor([-1.0, 0.0])
    optimizer = torch.optim.SGD([query], lr=0.2)
    losses = []
    gaps = []
    for _ in range(30):
        optimizer.zero_grad()
        normalized_query = torch.nn.functional.normalize(query, dim=0)
        result = _directional(
            {
                1: {
                    "embedding": normalized_query,
                    "class_id": 0,
                    "dimensions": torch.tensor([1.5, 2.0, 4.0]),
                }
            },
            {
                1: make_object(positive.tolist()),
                2: make_object(negative.tolist()),
            },
            hard_weight=1.0,
        )
        loss = result["losses"][0]
        gap = torch.dot(normalized_query, positive) - torch.dot(
            normalized_query, negative
        )
        losses.append(float(loss.detach()))
        gaps.append(float(gap.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    assert gaps[-1] > gaps[0]


def test_track_disabled_and_total_loss_interface():
    disabled = sample_track_consistency_selection(
        {},
        {"track_consistency_loss_weight": 0.0},
        torch.Generator().manual_seed(0),
        torch.device("cpu"),
    )
    assert disabled is None
    sd_loss = torch.tensor(2.0)
    epipolar_loss = torch.tensor(3.0)
    track_loss = torch.tensor(5.0)
    total = sd_loss + 0.1 * epipolar_loss + 0.05 * track_loss
    assert torch.allclose(total, torch.tensor(2.55))
