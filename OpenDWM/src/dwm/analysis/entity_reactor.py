from __future__ import annotations

ENTITY_REACTOR_CAPTURE_VERSION = "v19.7.2-fullD-no-persistent-capture-20260819"
ENTITY_REACTOR_SELECTION_VERSION = "v19.6-camera-handoff-selection-20260817"

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class NoCrossCameraTransitionError(RuntimeError):
    """Raised when a clip contains no valid temporal camera hand-off track."""


def select_cross_camera_slots(
    observation_valid: torch.Tensor,
    crossview_mask: torch.Tensor,
    centers_lidar: torch.Tensor,
    track_valid: torch.Tensor,
    track_hash: torch.Tensor,
    max_tracks: int,
    minimum_cross_camera_events: int,
) -> tuple[torch.Tensor, dict]:
    """Keep tracks whose visibility really migrates from one camera to another over time.

    A camera pair A->B is a transition only when A starts earlier and also ends earlier
    than B.  The track must therefore have an A-only prefix and a B-only suffix.
    Same-time co-visibility alone does not qualify, and motion confined to one view does
    not qualify.
    """
    if observation_valid.ndim != 4:
        raise ValueError("observation_valid must be [B,T,V,S]")
    del centers_lidar, track_valid
    batch_size, _, view_count, slot_count = observation_valid.shape
    mask_matrix = crossview_mask
    if mask_matrix.ndim == 2:
        mask_matrix = mask_matrix.unsqueeze(0)
    if mask_matrix.shape[0] == 1 and batch_size > 1:
        mask_matrix = mask_matrix.expand(batch_size, -1, -1)

    selected_mask = torch.zeros((batch_size, slot_count), dtype=torch.bool)
    summaries = []
    minimum_transition_pairs = max(1, int(minimum_cross_camera_events))
    for batch_index in range(batch_size):
        eligible_slots = []
        transition_counts: dict[int, int] = {}
        transition_descriptions: dict[int, list[str]] = {}
        for slot_index in range(slot_count):
            transitions = []
            for first_view in range(view_count):
                first_times = torch.nonzero(
                    observation_valid[batch_index, :, first_view, slot_index],
                    as_tuple=False,
                ).flatten()
                if first_times.numel() == 0:
                    continue
                for second_view in range(first_view + 1, view_count):
                    allowed = bool(
                        mask_matrix[batch_index, first_view, second_view]
                        or mask_matrix[batch_index, second_view, first_view]
                    )
                    if not allowed:
                        continue
                    second_times = torch.nonzero(
                        observation_valid[batch_index, :, second_view, slot_index],
                        as_tuple=False,
                    ).flatten()
                    if second_times.numel() == 0:
                        continue

                    first_start = int(first_times.min().item())
                    first_end = int(first_times.max().item())
                    second_start = int(second_times.min().item())
                    second_end = int(second_times.max().item())
                    if first_start < second_start and first_end < second_end:
                        source_view, target_view = first_view, second_view
                        source_start, source_end = first_start, first_end
                        target_start, target_end = second_start, second_end
                    elif second_start < first_start and second_end < first_end:
                        source_view, target_view = second_view, first_view
                        source_start, source_end = second_start, second_end
                        target_start, target_end = first_start, first_end
                    else:
                        continue

                    has_source_only_prefix = source_start < target_start
                    has_target_only_suffix = source_end < target_end
                    if not has_source_only_prefix or not has_target_only_suffix:
                        continue
                    transitions.append(
                        (
                            int(source_view),
                            int(target_view),
                            int(source_start),
                            int(source_end),
                            int(target_start),
                            int(target_end),
                        )
                    )

            if len(transitions) < minimum_transition_pairs:
                continue
            eligible_slots.append(slot_index)
            transition_counts[slot_index] = len(transitions)
            transition_descriptions[slot_index] = [
                f"v{item[0]}->v{item[1]}:{item[2]}-{item[3]}|{item[4]}-{item[5]}"
                for item in transitions
            ]

        if not eligible_slots:
            raise NoCrossCameraTransitionError(
                "no tracked entity actually transitions from one camera view to another; "
                "same-view motion and same-time co-visibility are intentionally excluded"
            )

        eligible_slots.sort(
            key=lambda slot_index: (
                -int(transition_counts[slot_index]),
                int(track_hash[batch_index, slot_index].item()),
            )
        )
        selected_slots = list(eligible_slots)
        selected_mask[batch_index, selected_slots] = True

        selected_transition_count = sum(transition_counts[slot] for slot in selected_slots)
        selected_track_hashes = [
            int(track_hash[batch_index, slot].item())
            for slot in selected_slots
        ]
        selected_examples = {
            str(int(track_hash[batch_index, slot].item())): transition_descriptions[slot]
            for slot in selected_slots[:8]
        }
        summaries.append(
            {
                "selection_mode": "temporal_camera_handoff",
                "candidate_slot_count": int(slot_count),
                "eligible_transition_track_count": int(len(eligible_slots)),
                "selected_transition_track_count": int(len(selected_slots)),
                "selected_transition_pair_count": int(selected_transition_count),
                "selected_transition_track_hashes": selected_track_hashes,
                "minimum_transition_pairs": int(minimum_transition_pairs),
                "legacy_max_tracks_argument": int(max_tracks),
                "transition_examples": selected_examples,
            }
        )

    summary = summaries[0] if batch_size == 1 else {"per_batch": summaries}
    return selected_mask, summary


@dataclass
class ObservationGeometry:
    anchor_grid: torch.Tensor
    ring_grid: torch.Tensor
    anchor_visible: torch.Tensor
    observation_index: torch.Tensor
    geometry: torch.Tensor
    bbox_xyxy: torch.Tensor
    bbox_area_ratio: torch.Tensor
    track_hash: torch.Tensor
    class_id: torch.Tensor
    crossview_mask: torch.Tensor
    selected_track_hash: int


class EntityFeatureRecorder:
    """Samples one tracked 3D entity at canonical anchors from selected DiT stages."""

    def __init__(
        self,
        projection_dim: int = 64,
        projection_seed: int = 20260815,
        minimum_projected_area: float = 20.0,
        minimum_visible_anchors: int = 5,
        ring_expand_ratio: float = 0.20,
        max_cross_camera_tracks: int = 32,
        minimum_cross_camera_events: int = 1,
    ) -> None:
        # projection_dim/projection_seed remain accepted only so old launch scripts do not break.
        # Quantitative analysis always uses the original hidden dimension.
        del projection_dim, projection_seed
        self.minimum_projected_area = float(minimum_projected_area)
        self.minimum_visible_anchors = int(minimum_visible_anchors)
        self.ring_expand_ratio = float(ring_expand_ratio)
        self.max_cross_camera_tracks = int(max_cross_camera_tracks)
        self.minimum_cross_camera_events = int(minimum_cross_camera_events)
        if self.max_cross_camera_tracks <= 0 or self.minimum_cross_camera_events <= 0:
            raise ValueError("cross-camera track limits must be positive")
        self.geometry_plan: Optional[ObservationGeometry] = None
        self.token_height = 0
        self.token_width = 0
        self.batch_size = 0
        self.sequence_length = 0
        self.view_count = 0
        self.slot_count = 0
        self.stage_names: list[str] = []
        self.stage_features: list[torch.Tensor] = []
        self.metadata: dict = {}

    def configure(
        self,
        batch: dict,
        track_data: dict,
        token_height: int,
        token_width: int,
        metadata: Optional[dict] = None,
    ) -> None:
        required_batch = (
            "camera_intrinsics",
            "image_size",
            "lidar_to_camera",
            "crossview_mask",
        )
        missing_batch = [key for key in required_batch if key not in batch]
        if missing_batch:
            raise KeyError(f"batch is missing analysis keys {missing_batch}")

        camera_intrinsics = batch["camera_intrinsics"].detach().float().cpu()
        image_size = batch["image_size"].detach().float().cpu()
        lidar_to_camera = batch["lidar_to_camera"].detach().float().cpu()
        crossview_mask = batch["crossview_mask"].detach().bool().cpu()
        corners_lidar = track_data["corners_lidar"].detach().float().cpu()
        track_valid = track_data["valid"].detach().bool().cpu()
        class_id = track_data["class_id"].detach().long().cpu()
        track_hash = track_data["track_hash"].detach().long().cpu()

        if corners_lidar.ndim == 4:
            corners_lidar = corners_lidar.unsqueeze(0)
            track_valid = track_valid.unsqueeze(0)
            class_id = class_id.unsqueeze(0)
            track_hash = track_hash.unsqueeze(0)
        if crossview_mask.ndim == 2:
            crossview_mask = crossview_mask.unsqueeze(0)

        self.batch_size, self.sequence_length, self.view_count = camera_intrinsics.shape[:3]
        self.slot_count = int(corners_lidar.shape[2])
        self.token_height = int(token_height)
        self.token_width = int(token_width)

        centers = corners_lidar.mean(dim=-2, keepdim=True)
        anchors = torch.cat([corners_lidar, centers], dim=-2)
        homogeneous = torch.cat([anchors, torch.ones_like(anchors[..., :1])], dim=-1)
        camera_h = torch.einsum(
            "btvij,btskj->btvski",
            lidar_to_camera,
            homogeneous,
        )
        camera_xyz = camera_h[..., :3]
        projected = torch.einsum(
            "btvij,btvskj->btvski",
            camera_intrinsics,
            camera_xyz,
        )
        depth = camera_xyz[..., 2]
        safe_projected_depth = projected[..., 2].clamp_min(1e-6)
        pixel_u = projected[..., 0] / safe_projected_depth
        pixel_v = projected[..., 1] / safe_projected_depth

        width = image_size[..., 0].unsqueeze(-1).unsqueeze(-1)
        height = image_size[..., 1].unsqueeze(-1).unsqueeze(-1)
        positive_depth = depth > 0.10
        anchor_visible = (
            positive_depth
            & (pixel_u >= 0.0)
            & (pixel_u < width)
            & (pixel_v >= 0.0)
            & (pixel_v < height)
        )
        visible_corner_count = anchor_visible[..., :8].sum(dim=-1)

        positive_corners = positive_depth[..., :8]
        positive_u_min = torch.where(
            positive_corners,
            pixel_u[..., :8],
            torch.full_like(pixel_u[..., :8], float("inf")),
        ).amin(dim=-1)
        positive_v_min = torch.where(
            positive_corners,
            pixel_v[..., :8],
            torch.full_like(pixel_v[..., :8], float("inf")),
        ).amin(dim=-1)
        positive_u_max = torch.where(
            positive_corners,
            pixel_u[..., :8],
            torch.full_like(pixel_u[..., :8], float("-inf")),
        ).amax(dim=-1)
        positive_v_max = torch.where(
            positive_corners,
            pixel_v[..., :8],
            torch.full_like(pixel_v[..., :8], float("-inf")),
        ).amax(dim=-1)

        image_width = image_size[..., 0].unsqueeze(-1)
        image_height = image_size[..., 1].unsqueeze(-1)
        box_x0 = positive_u_min.clamp(min=0.0)
        box_y0 = positive_v_min.clamp(min=0.0)
        box_x1 = torch.minimum(positive_u_max, image_width - 1.0)
        box_y1 = torch.minimum(positive_v_max, image_height - 1.0)
        box_area = (box_x1 - box_x0).clamp_min(0.0) * (box_y1 - box_y0).clamp_min(0.0)
        image_area = (image_width * image_height).clamp_min(1.0)
        box_area_ratio = box_area / image_area
        observation_valid = (
            track_valid[:, :, None, :]
            & (visible_corner_count >= self.minimum_visible_anchors)
            & anchor_visible[..., -1]
            & torch.isfinite(box_area)
            & (box_area >= self.minimum_projected_area)
        )
        centers_lidar_for_selection = corners_lidar.mean(dim=-2)
        cross_camera_slot_mask, cross_camera_summary = select_cross_camera_slots(
            observation_valid,
            crossview_mask,
            centers_lidar_for_selection,
            track_valid,
            track_hash,
            max_tracks=self.max_cross_camera_tracks,
            minimum_cross_camera_events=self.minimum_cross_camera_events,
        )
        cross_camera_summary["selection_version"] = ENTITY_REACTOR_SELECTION_VERSION

        normalized_u = (pixel_u + 0.5) / width.clamp_min(1.0)
        normalized_v = (pixel_v + 0.5) / height.clamp_min(1.0)
        geometry = torch.stack(
            [normalized_u, normalized_v, torch.log(depth.clamp_min(1e-4))],
            dim=-1,
        )
        anchor_grid = torch.stack(
            [normalized_u * 2.0 - 1.0, normalized_v * 2.0 - 1.0],
            dim=-1,
        )

        box_width = (box_x1 - box_x0).clamp_min(1.0)
        box_height = (box_y1 - box_y0).clamp_min(1.0)
        ring_x0 = box_x0 - self.ring_expand_ratio * box_width
        ring_y0 = box_y0 - self.ring_expand_ratio * box_height
        ring_x1 = box_x1 + self.ring_expand_ratio * box_width
        ring_y1 = box_y1 + self.ring_expand_ratio * box_height
        ring_xm = 0.5 * (ring_x0 + ring_x1)
        ring_ym = 0.5 * (ring_y0 + ring_y1)
        ring_u = torch.stack(
            [ring_x0, ring_xm, ring_x1, ring_x1, ring_x1, ring_xm, ring_x0, ring_x0],
            dim=-1,
        )
        ring_v = torch.stack(
            [ring_y0, ring_y0, ring_y0, ring_ym, ring_y1, ring_y1, ring_y1, ring_ym],
            dim=-1,
        )
        ring_u = torch.maximum(ring_u, torch.zeros_like(ring_u))
        ring_v = torch.maximum(ring_v, torch.zeros_like(ring_v))
        ring_u = torch.minimum(ring_u, image_width.unsqueeze(-1) - 1.0)
        ring_v = torch.minimum(ring_v, image_height.unsqueeze(-1) - 1.0)
        ring_u = (ring_u + 0.5) / image_width.unsqueeze(-1).clamp_min(1.0)
        ring_v = (ring_v + 0.5) / image_height.unsqueeze(-1).clamp_min(1.0)
        ring_grid = torch.stack([ring_u * 2.0 - 1.0, ring_v * 2.0 - 1.0], dim=-1)

        observation_index = torch.nonzero(observation_valid, as_tuple=False)
        if observation_index.shape[0] == 0:
            raise RuntimeError("no tracked entity is sufficiently visible for analysis")
        b_index, t_index, v_index, s_index = observation_index.unbind(dim=1)
        observation_geometry = geometry[b_index, t_index, v_index, s_index]
        observation_anchor_visible = anchor_visible[b_index, t_index, v_index, s_index]
        observation_bbox = torch.stack(
            [
                box_x0[b_index, t_index, v_index, s_index],
                box_y0[b_index, t_index, v_index, s_index],
                box_x1[b_index, t_index, v_index, s_index],
                box_y1[b_index, t_index, v_index, s_index],
            ],
            dim=-1,
        )
        observation_area_ratio = box_area_ratio[b_index, t_index, v_index, s_index]
        observation_track_hash = track_hash[b_index, s_index]
        observation_class = class_id[b_index, s_index]
        selected_track_hash, selection_summary = self._select_representative_track(
            observation_index,
            observation_geometry,
            observation_area_ratio,
            observation_track_hash,
            crossview_mask,
        )

        self.geometry_plan = ObservationGeometry(
            anchor_grid=anchor_grid,
            ring_grid=ring_grid,
            anchor_visible=observation_anchor_visible,
            observation_index=observation_index,
            geometry=observation_geometry,
            bbox_xyxy=observation_bbox,
            bbox_area_ratio=observation_area_ratio,
            track_hash=observation_track_hash,
            class_id=observation_class,
            crossview_mask=crossview_mask,
            selected_track_hash=selected_track_hash,
        )
        self.stage_names.clear()
        self.stage_features.clear()
        self.metadata = dict(metadata or {})
        self.metadata.update(
            {
                "capture_version": ENTITY_REACTOR_CAPTURE_VERSION,
                "token_height": self.token_height,
                "token_width": self.token_width,
                "feature_dimension_policy": "full_hidden_dimension",
                "persistent_feature_capture": False,
                "hard_negative_policy": "same_class_only",
                "observation_count": int(observation_index.shape[0]),
                "selected_track_hash": int(selected_track_hash),
                "selected_track_summary": selection_summary,
                "cross_camera_sampling": cross_camera_summary,
                "canonical_anchor_count": 9,
            }
        )

    def _select_representative_track(
        self,
        observation_index: torch.Tensor,
        geometry: torch.Tensor,
        area_ratio: torch.Tensor,
        track_hash: torch.Tensor,
        crossview_mask: torch.Tensor,
    ) -> tuple[int, dict]:
        best_track = int(track_hash[0].item())
        best_score = float("-inf")
        best_summary: dict = {}
        unique_tracks = torch.unique(track_hash)
        mask_matrix = crossview_mask[0] if crossview_mask.ndim == 3 else crossview_mask
        for current_track_tensor in unique_tracks:
            current_track = int(current_track_tensor.item())
            rows = torch.nonzero(track_hash == current_track_tensor, as_tuple=False).flatten()
            current_index = observation_index[rows]
            current_geometry = geometry[rows, -1]
            current_area = area_ratio[rows]
            unique_times = torch.unique(current_index[:, 1])
            unique_views = torch.unique(current_index[:, 2])
            crossview_pair_count = 0
            for time_tensor in unique_times:
                time_rows = rows[observation_index[rows, 1] == time_tensor]
                time_views = observation_index[time_rows, 2].tolist()
                for source_position, source_view in enumerate(time_views):
                    for target_view in time_views[source_position + 1 :]:
                        if bool(mask_matrix[source_view, target_view] or mask_matrix[target_view, source_view]):
                            crossview_pair_count += 1
            horizontal_span = float(current_geometry[:, 0].max() - current_geometry[:, 0].min())
            vertical_span = float(current_geometry[:, 1].max() - current_geometry[:, 1].min())
            depth_span = float(current_geometry[:, 2].max() - current_geometry[:, 2].min())
            mean_area = float(current_area.mean())
            time_span = int(unique_times.max().item() - unique_times.min().item()) if unique_times.numel() else 0
            score = (
                2.0 * np.log1p(crossview_pair_count)
                + 0.80 * float(unique_views.numel())
                + 0.35 * float(unique_times.numel())
                + 0.10 * float(time_span)
                + 2.0 * horizontal_span
                + 1.0 * vertical_span
                + 0.8 * depth_span
                + 4.0 * np.sqrt(max(mean_area, 0.0))
            )
            if score <= best_score:
                continue
            best_score = score
            best_track = current_track
            best_summary = {
                "score": float(score),
                "observation_count": int(rows.numel()),
                "view_count": int(unique_views.numel()),
                "time_count": int(unique_times.numel()),
                "crossview_pair_count": int(crossview_pair_count),
                "time_span": int(time_span),
                "horizontal_span": horizontal_span,
                "vertical_span": vertical_span,
                "log_depth_span": depth_span,
                "mean_box_area_ratio": mean_area,
            }
        return best_track, best_summary

    def capture(self, stage_name: str, hidden_states: torch.Tensor) -> None:
        if self.geometry_plan is None or stage_name in self.stage_names:
            return
        hidden_states = self._normalize_hidden_shape(hidden_states)
        expected_btv = self.batch_size * self.sequence_length * self.view_count
        expected_tokens = self.token_height * self.token_width
        if hidden_states.shape[0] != expected_btv or hidden_states.shape[1] != expected_tokens:
            raise ValueError(
                f"stage {stage_name} shape {tuple(hidden_states.shape)} does not match "
                f"BTV={expected_btv}, tokens={expected_tokens}"
            )

        channel_count = int(hidden_states.shape[-1])
        normalized_hidden = F.layer_norm(hidden_states.float(), (channel_count,))
        normalized_hidden = normalized_hidden.transpose(1, 2).reshape(
            expected_btv,
            channel_count,
            self.token_height,
            self.token_width,
        )

        plan = self.geometry_plan
        anchor_grid = plan.anchor_grid.to(hidden_states.device, dtype=torch.float32)
        ring_grid = plan.ring_grid.to(hidden_states.device, dtype=torch.float32)
        anchor_count = int(anchor_grid.shape[-2])
        ring_count = int(ring_grid.shape[-2])
        anchor_grid_flat = anchor_grid.reshape(expected_btv, self.slot_count * anchor_count, 1, 2)
        ring_grid_flat = ring_grid.reshape(expected_btv, self.slot_count * ring_count, 1, 2)
        sampled_anchor = F.grid_sample(
            normalized_hidden,
            anchor_grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_ring = F.grid_sample(
            normalized_hidden,
            ring_grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_anchor = sampled_anchor[..., 0].transpose(1, 2).reshape(
            self.batch_size,
            self.sequence_length,
            self.view_count,
            self.slot_count,
            anchor_count,
            channel_count,
        )
        sampled_ring = sampled_ring[..., 0].transpose(1, 2).reshape(
            self.batch_size,
            self.sequence_length,
            self.view_count,
            self.slot_count,
            ring_count,
            channel_count,
        )
        local_background = sampled_ring.mean(dim=-2, keepdim=True)
        entity_constellation = sampled_anchor - local_background
        observation_index = plan.observation_index.to(hidden_states.device)
        b_index, t_index, v_index, s_index = observation_index.unbind(dim=1)
        entity_observations = entity_constellation[b_index, t_index, v_index, s_index]

        self.stage_names.append(stage_name)
        self.stage_features.append(entity_observations.detach().to("cpu", dtype=torch.float32))

    def build_probe_capture(self) -> dict:
        if self.geometry_plan is None:
            raise RuntimeError("configure the recorder before building probe events")
        plan = self.geometry_plan
        crossview_mask = plan.crossview_mask.numpy().astype(np.bool_, copy=False)
        if crossview_mask.ndim == 3:
            crossview_mask = crossview_mask[0]
        return {
            "path": Path("<eligibility-only>"),
            "geometry": plan.geometry.numpy().astype(np.float32, copy=False),
            "anchor_visible": plan.anchor_visible.numpy().astype(np.bool_, copy=False),
            "bbox_area_ratio": plan.bbox_area_ratio.numpy().astype(np.float32, copy=False),
            "observation_index": plan.observation_index.numpy().astype(np.int64, copy=False),
            "track_hash": plan.track_hash.numpy().astype(np.int64, copy=False),
            "class_id": plan.class_id.numpy().astype(np.int64, copy=False),
            "crossview_mask": crossview_mask,
            "metadata": dict(self.metadata),
        }

    def snapshot(self, logical_path: Optional[Path] = None) -> dict:
        if self.geometry_plan is None or not self.stage_features:
            raise RuntimeError("configure and run the model before taking a snapshot")
        plan = self.geometry_plan
        stage_names = list(self.stage_names)
        stage_features = list(self.stage_features)
        requested_stage_names = self.metadata.get("save_stage_names", None)
        if requested_stage_names is not None:
            requested = [str(name) for name in requested_stage_names]
            stage_index = {name: index for index, name in enumerate(stage_names)}
            missing = [name for name in requested if name not in stage_index]
            if missing:
                raise RuntimeError(
                    "requested intervention stages were not captured: " + ",".join(missing)
                )
            selected_indices = [stage_index[name] for name in requested]
            stage_names = [stage_names[index] for index in selected_indices]
            stage_features = [stage_features[index] for index in selected_indices]
        crossview_mask = plan.crossview_mask.numpy().astype(np.bool_, copy=False)
        if crossview_mask.ndim == 3:
            crossview_mask = crossview_mask[0]
        return {
            "path": Path(logical_path or "<memory-capture>"),
            "stage_names": stage_names,
            "features": [feature.numpy() for feature in stage_features],
            "geometry": plan.geometry.numpy().astype(np.float32, copy=False),
            "anchor_visible": plan.anchor_visible.numpy().astype(np.bool_, copy=False),
            "bbox_area_ratio": plan.bbox_area_ratio.numpy().astype(np.float32, copy=False),
            "observation_index": plan.observation_index.numpy().astype(np.int64, copy=False),
            "track_hash": plan.track_hash.numpy().astype(np.int64, copy=False),
            "class_id": plan.class_id.numpy().astype(np.int64, copy=False),
            "crossview_mask": crossview_mask,
            "metadata": dict(self.metadata),
        }

    def save(self, output_path: Path) -> Path:
        del output_path
        raise RuntimeError(
            "persistent per-sample feature captures are disabled; use snapshot() in memory"
        )

    def export_generated_entity_crops(
        self,
        images: torch.Tensor,
        output_path: Path,
        crop_size: tuple[int, int] = (144, 96),
    ) -> Optional[int]:
        if self.geometry_plan is None:
            raise RuntimeError("configure the recorder before exporting generated crops")
        plan = self.geometry_plan
        selected_track = int(plan.selected_track_hash)
        selected_rows = torch.nonzero(plan.track_hash == selected_track, as_tuple=False).flatten()
        if selected_rows.numel() == 0:
            return None

        image_cpu = images.detach().float().cpu().clamp(0.0, 1.0)
        index = plan.observation_index
        boxes = plan.bbox_xyxy
        crop_width, crop_height = int(crop_size[0]), int(crop_size[1])
        crops = []
        rows = []
        times = []
        views = []
        from PIL import Image

        for row_tensor in selected_rows:
            row = int(row_tensor.item())
            batch_index, time_index, view_index, _ = [int(value) for value in index[row].tolist()]
            x0, y0, x1, y1 = boxes[row].tolist()
            x0 = max(0, int(np.floor(x0)))
            y0 = max(0, int(np.floor(y0)))
            x1 = min(image_cpu.shape[-1], int(np.ceil(x1)) + 1)
            y1 = min(image_cpu.shape[-2], int(np.ceil(y1)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image_cpu[batch_index, time_index, view_index, :, y0:y1, x0:x1]
            crop_array = (crop.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            resized = Image.fromarray(crop_array).resize(
                (crop_width, crop_height),
                Image.Resampling.BICUBIC,
            )
            crops.append(np.asarray(resized, dtype=np.uint8))
            rows.append(row)
            times.append(time_index)
            views.append(view_index)

        if not crops:
            return None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            images=np.stack(crops, axis=0),
            rows=np.asarray(rows, dtype=np.int32),
            time=np.asarray(times, dtype=np.int16),
            view=np.asarray(views, dtype=np.int16),
            selected_track=np.asarray(selected_track, dtype=np.int64),
            source=np.asarray("predicted_x0"),
        )
        return selected_track

    def _normalize_hidden_shape(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            normalized = hidden_states
        elif hidden_states.ndim == 4:
            normalized = hidden_states.flatten(0, 1)
        else:
            raise ValueError(f"unsupported hidden state shape {tuple(hidden_states.shape)}")
        if normalized.shape[-1] <= 0 or normalized.shape[1] <= 0:
            raise ValueError(f"invalid hidden state shape {tuple(normalized.shape)}")
        return normalized



class _StemHook:
    def __init__(self, controller: "EntityReactorController") -> None:
        self.controller = controller

    def __call__(self, module, args, output):
        if not self.controller.active:
            return None
        self.controller._capture_stage("stem", output)
        self.controller.last_hidden = self.controller._normalize_runtime_hidden(output)
        self.controller.base_inputs.clear()
        self.controller.cond_inputs.clear()
        return None


class _BasePreHook:
    def __init__(self, controller: "EntityReactorController", layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args):
        if not self.controller.active:
            return None
        incoming = args[0]
        previous = self.controller.last_hidden
        if self.controller._layer_is_key(self.layer_index):
            input_state = previous if previous is not None else incoming
            self.controller._capture_stage(f"L{self.layer_index:02d}.in", input_state)
        if self.controller.model_kind == "pv":
            stage_name = f"L{self.layer_index:02d}.cond"
            incoming = self.controller._apply_gate(stage_name, previous, incoming)
            self.controller._capture_stage(stage_name, incoming)
            self.controller.last_hidden = self.controller._normalize_runtime_hidden(incoming)
        self.controller.base_inputs[self.layer_index] = incoming
        if incoming is args[0]:
            return None
        new_args = list(args)
        new_args[0] = incoming
        return tuple(new_args)


class _BasePostHook:
    def __init__(self, controller: "EntityReactorController", layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, output):
        if not self.controller.active:
            return None
        hidden = self.controller._extract_base_hidden(output)
        base_input = self.controller.base_inputs.get(self.layer_index, args[0])
        stage_name = f"L{self.layer_index:02d}.base"
        hidden = self.controller._apply_gate(stage_name, base_input, hidden)
        self.controller._capture_stage(stage_name, hidden)
        self.controller.last_hidden = self.controller._normalize_runtime_hidden(hidden)
        self.controller.base_inputs.pop(self.layer_index, None)
        return self.controller._replace_base_hidden(output, hidden)


class _BEVPreHook:
    def __init__(self, controller: "EntityReactorController", layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args):
        if not self.controller.active:
            return None
        incoming = args[0]
        stage_name = f"L{self.layer_index:02d}.bev"
        incoming = self.controller._apply_gate(stage_name, self.controller.last_hidden, incoming)
        self.controller._capture_stage(stage_name, incoming)
        self.controller.cond_inputs[self.layer_index] = incoming
        self.controller.last_hidden = self.controller._normalize_runtime_hidden(incoming)
        if incoming is args[0]:
            return None
        new_args = list(args)
        new_args[0] = incoming
        return tuple(new_args)


class _CondPostHook:
    def __init__(self, controller: "EntityReactorController", layer_index: int) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)

    def __call__(self, module, args, output):
        if not self.controller.active:
            return None
        base = self.controller.cond_inputs.get(self.layer_index, args[0])
        stage_name = f"L{self.layer_index:02d}.cond"
        processed = self.controller._apply_gate(stage_name, base, output)
        self.controller._capture_stage(stage_name, processed)
        self.controller.last_hidden = self.controller._normalize_runtime_hidden(processed)
        self.controller.cond_inputs.pop(self.layer_index, None)
        return processed


class _MixerHook:
    def __init__(
        self,
        controller: "EntityReactorController",
        layer_index: int,
        suffix: str,
    ) -> None:
        self.controller = controller
        self.layer_index = int(layer_index)
        self.suffix = str(suffix)

    def __call__(self, module, args, output):
        if not self.controller.active:
            return None
        base = self.controller._normalize_runtime_hidden(args[0])
        output_flat = self.controller._normalize_runtime_hidden(output)
        stage_name = f"L{self.layer_index:02d}.{self.suffix}"
        processed_flat = self.controller._apply_gate(stage_name, base, output_flat)
        self.controller._capture_stage(stage_name, processed_flat)
        self.controller.last_hidden = processed_flat
        if output.ndim == 4:
            return processed_flat.reshape_as(output)
        if output.ndim == 3:
            return processed_flat
        raise RuntimeError(f"unsupported mixer output shape at {stage_name}")




class _CameraEmbeddingCaptureHook:
    def __init__(self, controller: "EntityReactorController", encoding_kind: str) -> None:
        self.controller = controller
        self.encoding_kind = str(encoding_kind)

    def __call__(self, module, args, output):
        del module, args
        if not self.controller.active or not isinstance(output, torch.Tensor):
            return None
        camera_embedding = output
        if self.encoding_kind == "implicit":
            if camera_embedding.ndim != 2:
                raise RuntimeError(
                    f"implicit camera embedding must be [BTV,D], got {tuple(camera_embedding.shape)}"
                )
            camera_embedding = camera_embedding.unsqueeze(1)
        elif self.encoding_kind in ("explicit", "petr"):
            if camera_embedding.ndim != 4:
                raise RuntimeError(
                    f"{self.encoding_kind} camera embedding must be [BTV,H,W,D], "
                    f"got {tuple(camera_embedding.shape)}"
                )
            camera_embedding = camera_embedding.flatten(1, 2)
        self.controller.current_camera_embedding = camera_embedding
        return None


class _PVTemporalCameraProxy:
    def __init__(self, controller: "EntityReactorController", original) -> None:
        self.controller = controller
        self.original = original

    def __call__(self, *args, **kwargs):
        if len(args) < 4:
            return self.original(*args, **kwargs)
        temporal_block = args[0]
        layer_index = self.controller.temporal_layer_by_block.get(id(temporal_block), None)
        if layer_index is None:
            return self.original(*args, **kwargs)
        alpha = self.controller._gate_alpha(f"L{int(layer_index):02d}.cam")
        camera_embedding = self.controller.current_camera_embedding
        if alpha == 1.0 or camera_embedding is None:
            return self.original(*args, **kwargs)
        sequence_embedding = args[3]
        if not isinstance(sequence_embedding, torch.Tensor):
            return self.original(*args, **kwargs)
        camera_embedding = camera_embedding.to(
            device=sequence_embedding.device,
            dtype=sequence_embedding.dtype,
        )
        if camera_embedding.shape != sequence_embedding.shape:
            try:
                camera_embedding = camera_embedding.expand_as(sequence_embedding)
            except RuntimeError:
                return self.original(*args, **kwargs)
        modified = list(args)
        modified[3] = sequence_embedding - (1.0 - alpha) * camera_embedding
        return self.original(*modified, **kwargs)


class _PVCrossviewCameraProxy:
    def __init__(self, controller: "EntityReactorController", original) -> None:
        self.controller = controller
        self.original = original

    def __call__(self, *args, **kwargs):
        if len(args) < 4:
            return self.original(*args, **kwargs)
        crossview_block = args[0]
        layer_index = self.controller.crossview_layer_by_block.get(id(crossview_block), None)
        if layer_index is None:
            return self.original(*args, **kwargs)
        alpha = self.controller._gate_alpha(f"L{int(layer_index):02d}.cam")
        if alpha == 1.0 or not isinstance(args[3], torch.Tensor):
            return self.original(*args, **kwargs)
        modified = list(args)
        modified[3] = args[3] * alpha
        return self.original(*modified, **kwargs)


class _PVTVAsCrossviewCameraProxy:
    """Treat the TV branch as the Cross-view branch for Entity Reactor analysis.

    The TV branch receives tv_emb = time_emb + camera_emb.  Camera intervention
    must remove only the camera term and keep the time term intact.
    """

    def __init__(self, controller: "EntityReactorController", original) -> None:
        self.controller = controller
        self.original = original

    def __call__(self, *args, **kwargs):
        if len(args) < 4:
            return self.original(*args, **kwargs)
        tv_block = args[0]
        layer_index = self.controller.crossview_layer_by_block.get(id(tv_block), None)
        if layer_index is None:
            return self.original(*args, **kwargs)
        alpha = self.controller._gate_alpha(f"L{int(layer_index):02d}.cam")
        camera_embedding = self.controller.current_camera_embedding
        if alpha == 1.0 or camera_embedding is None:
            return self.original(*args, **kwargs)
        tv_embedding = args[3]
        if not isinstance(tv_embedding, torch.Tensor):
            return self.original(*args, **kwargs)
        camera_embedding = camera_embedding.to(
            device=tv_embedding.device,
            dtype=tv_embedding.dtype,
        )
        if camera_embedding.shape != tv_embedding.shape:
            try:
                camera_embedding = camera_embedding.expand_as(tv_embedding)
            except RuntimeError:
                return self.original(*args, **kwargs)
        modified = list(args)
        modified[3] = tv_embedding - (1.0 - alpha) * camera_embedding
        return self.original(*modified, **kwargs)


class _BEVCrossviewCameraProxy:
    def __init__(self, controller: "EntityReactorController", original) -> None:
        self.controller = controller
        self.original = original

    def __call__(self, *args, **kwargs):
        if len(args) < 4:
            return self.original(*args, **kwargs)
        crossview_block = args[0]
        layer_index = self.controller.crossview_layer_by_block.get(id(crossview_block), None)
        if layer_index is None:
            return self.original(*args, **kwargs)
        alpha = self.controller._gate_alpha(f"L{int(layer_index):02d}.cam")
        if alpha == 1.0 or not isinstance(args[3], torch.Tensor):
            return self.original(*args, **kwargs)
        modified = list(args)
        modified[3] = args[3] * alpha
        return self.original(*modified, **kwargs)

class _FinalPreHook:
    def __init__(self, controller: "EntityReactorController") -> None:
        self.controller = controller

    def __call__(self, module, args):
        if not self.controller.active:
            return None
        hidden = args[0]
        self.controller._capture_stage("final", hidden)
        self.controller.last_hidden = self.controller._normalize_runtime_hidden(hidden)
        self.controller.base_inputs.clear()
        self.controller.cond_inputs.clear()
        return None


class EntityReactorController:
    """Architecture-aware hook controller with unchanged pipeline entry point."""

    def __init__(
        self,
        capture_patterns: Optional[list[str]] = None,
        gate_spec: Optional[dict[str, float]] = None,
        projection_dim: int = 64,
        projection_seed: int = 20260815,
        max_cross_camera_tracks: int = 32,
    ) -> None:
        self.capture_patterns = list(capture_patterns or [])
        self.gate_spec = dict(gate_spec or {})
        self.recorder = EntityFeatureRecorder(
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            max_cross_camera_tracks=max_cross_camera_tracks,
        )
        self.pipeline = None
        self.model = None
        self.model_kind = ""
        self.handles: list = []
        self.active = False
        self.last_hidden: Optional[torch.Tensor] = None
        self.base_inputs: dict[int, torch.Tensor] = {}
        self.cond_inputs: dict[int, torch.Tensor] = {}
        self.output_path: Optional[Path] = None
        self.latest_capture: Optional[dict] = None
        self.key_layers: tuple[int, ...] = ()
        self.num_layers = 0
        self.temporal_layers: tuple[int, ...] = ()
        self.crossview_layers: tuple[int, ...] = ()
        self.condition_layers: tuple[int, ...] = ()
        self.camera_layers: tuple[int, ...] = ()
        self.current_camera_embedding: Optional[torch.Tensor] = None
        self.temporal_layer_by_block: dict[int, int] = {}
        self.crossview_layer_by_block: dict[int, int] = {}
        self.tv_as_crossview = False
        self._patched_methods: dict[str, tuple[bool, object]] = {}

    def attach(self, pipeline) -> None:
        if self.handles:
            return
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model_kind = self._detect_model_kind(self.model)
        self.num_layers = len(self.model.transformer_blocks)
        self.temporal_layers = tuple(int(value) for value in (getattr(self.model, "temporal_block_layers", None) or []))
        self.tv_as_crossview = bool(
            getattr(self.model, "enable_tv", False)
            and hasattr(self.model, "tv_block_layers")
            and hasattr(self.model, "tv_mixers")
            and hasattr(self.model, "tv_transformer_blocks")
        )
        if self.tv_as_crossview:
            self.crossview_layers = tuple(
                int(value) for value in (getattr(self.model, "tv_block_layers", None) or [])
            )
        else:
            self.crossview_layers = tuple(
                int(value) for value in (getattr(self.model, "crossview_block_layers", None) or [])
            )
        if self.model_kind == "bev":
            self.condition_layers = tuple(int(value) for value in self.model.block_layers)
            self.temporal_layers = self.condition_layers
            self.crossview_layers = self.condition_layers
            self.camera_layers = self.crossview_layers
        else:
            self.condition_layers = tuple(sorted(set(self.temporal_layers) | set(self.crossview_layers)))
            camera_layer_set = set(self.crossview_layers)
            camera_enters_temporal = bool(
                self.temporal_layers
                and not getattr(self.model, "disable_view_emb_on_temporal_module", True)
            )
            if camera_enters_temporal:
                camera_layer_set.update(self.temporal_layers)
            self.camera_layers = tuple(sorted(camera_layer_set))
        structural_layers = sorted(
            set(self.camera_layers)
            | set(self.condition_layers)
            | set(self.temporal_layers)
            | set(self.crossview_layers)
        )
        if not structural_layers:
            requested_count = min(6, self.num_layers)
            structural_layers = sorted(
                set(
                    int(round(value))
                    for value in np.linspace(0, self.num_layers - 1, requested_count)
                )
            )
        self.key_layers = tuple(structural_layers)

        self.handles.append(self.model.pos_embed.register_forward_hook(_StemHook(self)))
        for layer_index, block in enumerate(self.model.transformer_blocks):
            self.handles.append(block.register_forward_pre_hook(_BasePreHook(self, layer_index)))
            self.handles.append(block.register_forward_hook(_BasePostHook(self, layer_index)))

        if self.model_kind == "bev":
            for additional_index, layer_index in enumerate(self.condition_layers):
                cond_block = self.model.cond_cross_blocks[additional_index]
                self.handles.append(cond_block.register_forward_pre_hook(_BEVPreHook(self, layer_index)))
                self.handles.append(cond_block.register_forward_hook(_CondPostHook(self, layer_index)))
                self.handles.append(
                    self.model.time_mixers[additional_index].register_forward_hook(
                        _MixerHook(self, layer_index, "temp")
                    )
                )
                self.handles.append(
                    self.model.view_mixers[additional_index].register_forward_hook(
                        _MixerHook(self, layer_index, "view")
                    )
                )
        else:
            for additional_index, layer_index in enumerate(self.temporal_layers):
                self.handles.append(
                    self.model.time_mixers[additional_index].register_forward_hook(
                        _MixerHook(self, layer_index, "temp")
                    )
                )
            crossview_mixers = (
                self.model.tv_mixers
                if self.tv_as_crossview
                else self.model.view_mixers
            )
            for additional_index, layer_index in enumerate(self.crossview_layers):
                self.handles.append(
                    crossview_mixers[additional_index].register_forward_hook(
                        _MixerHook(self, layer_index, "view")
                    )
                )

        self.temporal_layer_by_block = {}
        if hasattr(self.model, "temporal_transformer_blocks"):
            for additional_index, layer_index in enumerate(self.temporal_layers):
                if additional_index < len(self.model.temporal_transformer_blocks):
                    self.temporal_layer_by_block[id(self.model.temporal_transformer_blocks[additional_index])] = int(layer_index)

        self.crossview_layer_by_block = {}
        crossview_blocks = None
        if self.tv_as_crossview:
            crossview_blocks = self.model.tv_transformer_blocks
        elif hasattr(self.model, "crossview_transformer_blocks"):
            crossview_blocks = self.model.crossview_transformer_blocks
        if crossview_blocks is not None:
            for additional_index, layer_index in enumerate(self.crossview_layers):
                if additional_index < len(crossview_blocks):
                    self.crossview_layer_by_block[id(crossview_blocks[additional_index])] = int(layer_index)

        if self.tv_as_crossview and hasattr(self.model, "forward_tv_full_block_and_mix_result"):
            original_crossview = self.model.forward_tv_full_block_and_mix_result
            self._patch_model_method(
                "forward_tv_full_block_and_mix_result",
                _PVTVAsCrossviewCameraProxy(self, original_crossview),
            )
        elif hasattr(self.model, "forward_crossview_block_and_mix_result"):
            original_crossview = self.model.forward_crossview_block_and_mix_result
            if self.model_kind == "bev":
                self._patch_model_method(
                    "forward_crossview_block_and_mix_result",
                    _BEVCrossviewCameraProxy(self, original_crossview),
                )
            else:
                self._patch_model_method(
                    "forward_crossview_block_and_mix_result",
                    _PVCrossviewCameraProxy(self, original_crossview),
                )

        if self.model_kind == "pv" and hasattr(self.model, "forward_temporal_block_and_mix_result"):
            original_temporal = self.model.forward_temporal_block_and_mix_result
            self._patch_model_method(
                "forward_temporal_block_and_mix_result",
                _PVTemporalCameraProxy(self, original_temporal),
            )
            perspective_type = str(getattr(self.model, "perspective_modeling_type", "")).lower()
            camera_module = None
            if perspective_type == "implicit" and hasattr(self.model, "view_embedding"):
                camera_module = self.model.view_embedding
            elif perspective_type == "explicit" and hasattr(self.model, "rayencoder"):
                camera_module = self.model.rayencoder
            elif perspective_type == "petr" and hasattr(self.model, "petr_encoder"):
                camera_module = self.model.petr_encoder
            if camera_module is not None:
                self.handles.append(
                    camera_module.register_forward_hook(
                        _CameraEmbeddingCaptureHook(self, perspective_type)
                    )
                )

        self.handles.append(self.model.norm_out.register_forward_pre_hook(_FinalPreHook(self)))
        pipeline._entity_reactor_controller = self

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._restore_model_methods()
        if self.pipeline is not None and hasattr(self.pipeline, "_entity_reactor_controller"):
            delattr(self.pipeline, "_entity_reactor_controller")
        self.pipeline = None
        self.model = None
        self.active = False
        self.last_hidden = None
        self.current_camera_embedding = None

    def configure_capture(
        self,
        batch: dict,
        track_data: dict,
        token_height: int,
        token_width: int,
        output_path: Path,
        metadata: Optional[dict] = None,
    ) -> None:
        complete_metadata = dict(metadata or {})
        complete_metadata.update(self.architecture_metadata())
        complete_metadata["gate_spec"] = self.gate_spec
        self.recorder.configure(
            batch,
            track_data,
            token_height=token_height,
            token_width=token_width,
            metadata=complete_metadata,
        )
        self.output_path = Path(output_path)

    def architecture_metadata(self) -> dict:
        stage_modules = {
            "camera": list(self.camera_layers),
            "condition": list(self.condition_layers),
            "temporal": list(self.temporal_layers),
            "crossview": list(self.crossview_layers),
        }
        return {
            "model_kind": self.model_kind,
            "num_layers": int(self.num_layers),
            "key_layers": list(self.key_layers),
            "stage_modules": stage_modules,
            "capture_policy": "relation-mechanism-key-layers-only",
            "camera_gate_scope": "camera embedding at each consumer layer; cross-view computation remains enabled",
        }

    def begin_forward(self) -> None:
        self.active = True
        self.last_hidden = None
        self.current_camera_embedding = None
        self.latest_capture = None
        self.base_inputs.clear()
        self.cond_inputs.clear()
        self.recorder.stage_names.clear()
        self.recorder.stage_features.clear()

    def end_forward(self, save: bool = True) -> Optional[Path]:
        self.active = False
        output = None
        if save:
            self.latest_capture = self.recorder.snapshot(self.output_path)
            output = self.output_path
        self.last_hidden = None
        self.current_camera_embedding = None
        self.base_inputs.clear()
        self.cond_inputs.clear()
        return output

    def _layer_is_key(self, layer_index: int) -> bool:
        if int(layer_index) in self.key_layers:
            return True
        for pattern in self.capture_patterns:
            if fnmatch.fnmatch(f"L{int(layer_index):02d}.*", pattern):
                return True
        return False

    def _capture_stage(self, stage_name: str, hidden_states: torch.Tensor) -> None:
        if not self.active:
            return
        if self.capture_patterns:
            matched = any(fnmatch.fnmatch(stage_name, pattern) for pattern in self.capture_patterns)
            if not matched:
                return
        elif stage_name not in ("stem", "final"):
            layer_text = stage_name.split(".", 1)[0]
            layer_index = int(layer_text[1:]) if layer_text.startswith("L") else -1
            if layer_index not in self.key_layers:
                return
        normalized = self._normalize_runtime_hidden(hidden_states)
        self.recorder.capture(stage_name, normalized)

    def _patch_model_method(self, name: str, replacement) -> None:
        if self.model is None:
            raise RuntimeError("cannot patch model method before attach")
        if name in self._patched_methods:
            return
        had_instance_value = name in self.model.__dict__
        old_instance_value = self.model.__dict__.get(name, None)
        self._patched_methods[name] = (had_instance_value, old_instance_value)
        setattr(self.model, name, replacement)

    def _restore_model_methods(self) -> None:
        if self.model is None:
            self._patched_methods.clear()
            return
        for name, state in self._patched_methods.items():
            had_instance_value, old_instance_value = state
            if had_instance_value:
                setattr(self.model, name, old_instance_value)
            elif name in self.model.__dict__:
                delattr(self.model, name)
        self._patched_methods.clear()

    def _gate_alpha(self, stage_name: str) -> float:
        alpha = 1.0
        for pattern, value in self.gate_spec.items():
            if fnmatch.fnmatch(stage_name, pattern):
                alpha = float(value)
        return alpha

    def _apply_gate(
        self,
        stage_name: str,
        module_input: Optional[torch.Tensor],
        module_output: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._gate_alpha(stage_name)
        if alpha == 1.0 or module_input is None:
            return module_output
        if module_input.shape != module_output.shape:
            return module_output
        gated = module_input + alpha * (module_output - module_input)
        if gated.dtype != module_output.dtype or gated.device != module_output.device:
            gated = gated.to(device=module_output.device, dtype=module_output.dtype)
        return gated

    def _normalize_runtime_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            normalized = hidden_states
        elif hidden_states.ndim == 4:
            normalized = hidden_states.flatten(0, 1)
        else:
            raise ValueError(f"unsupported runtime hidden shape {tuple(hidden_states.shape)}")
        if normalized.ndim != 3:
            raise RuntimeError("runtime hidden normalization did not produce [BTV,N,D]")
        return normalized

    def _extract_base_hidden(self, output) -> torch.Tensor:
        if isinstance(output, tuple) and len(output) >= 2:
            hidden = output[1]
        elif isinstance(output, list) and len(output) >= 2:
            hidden = output[1]
        else:
            raise TypeError(f"unexpected base block output type {type(output)}")
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("base block hidden output is not a tensor")
        return hidden

    def _replace_base_hidden(self, output, hidden: torch.Tensor):
        if isinstance(output, tuple):
            values = list(output)
            values[1] = hidden
            replaced = tuple(values)
        elif isinstance(output, list):
            replaced = list(output)
            replaced[1] = hidden
        else:
            raise TypeError(f"unexpected base block output type {type(output)}")
        return replaced

    def _detect_model_kind(self, model) -> str:
        is_bev = hasattr(model, "cond_cross_blocks") and hasattr(model, "bev_control")
        if is_bev and not hasattr(model, "block_layers"):
            raise RuntimeError("BEV model is missing block_layers")
        if not hasattr(model, "transformer_blocks") or not hasattr(model, "pos_embed"):
            raise TypeError("Entity Reactor requires an SD3-style transformer model")
        return "bev" if is_bev else "pv"


def entity_reactor_forward(
    pipeline,
    model_input: torch.Tensor,
    timesteps: torch.Tensor,
    model_conditions: dict,
):
    """Unchanged Pipe contract around ``pipeline.model_wrapper``."""
    controller = getattr(pipeline, "_entity_reactor_controller", None)
    if controller is not None:
        controller.begin_forward()
    completed = False
    try:
        result = pipeline.model_wrapper(
            model_input,
            timesteps,
            **model_conditions,
        )
        completed = True
        return result
    finally:
        if controller is not None:
            controller.end_forward(save=completed)
