"""AV2 BEV adapter built on the current track_pv Argoverse reader."""

import bisect
import numpy as np
import pyarrow.feather
import torch

import dwm.datasets.common
from dwm.datasets.bevs.common import (
    argoverse_class_id,
    assign_stable_slot,
    assert_unique_tracks,
    require_track_id,
)
from dwm.datasets.track_pv.argoverse import MotionDataset as TrackPVMotionDataset


class MotionDataset(TrackPVMotionDataset):
    """Track-PV AV2 reader plus BEV-specific outputs."""

    def __init__(self, *args, layout_token_settings=None, max_boxes=64, **kwargs):
        layout = {} if layout_token_settings is None else dict(layout_token_settings)
        self.layout_max_boxes = int(layout.get("max_boxes", max_boxes))
        self.max_annotation_time_error_ns = int(
            layout.get("max_annotation_time_error_ns", 150_000_000)
        )

        hdmap_bev_settings = kwargs.get("hdmap_bev_settings")
        if hdmap_bev_settings is not None:
            hdmap_bev_settings = dict(hdmap_bev_settings)
            # Keep static RGB map here. DatasetAdapter appends 10 bbox channels.
            hdmap_bev_settings["append_dynamic_object_mask"] = False
            kwargs["hdmap_bev_settings"] = hdmap_bev_settings

        # Disable track_pv's per-frame bbox packing; rebuild stable slots below.
        super().__init__(*args, layout_token_settings=None, **kwargs)

    def get_bev_map_frame_reference(self, frame_items):
        for item in frame_items:
            if str(item.get("sensor", "")).lower() == "lidar":
                return item
        return super().get_bev_map_frame_reference(frame_items)

    def _read_feather(self, item, filename):
        path = f"{item['split']}/{item['scene_id']}/{filename}"
        with self.fs.open(path) as f:
            return pyarrow.feather.read_table(f).to_pydict()

    def _nearest_annotation_range_safe(self, annotations, timestamp):
        timestamps = annotations.get("timestamp_ns", [])
        if len(timestamps) == 0:
            return None, None, None

        pos = bisect.bisect_left(timestamps, timestamp)
        if pos == 0:
            nearest = timestamps[0]
        elif pos >= len(timestamps):
            nearest = timestamps[-1]
        else:
            prev_ts = timestamps[pos - 1]
            next_ts = timestamps[pos]
            nearest = (
                prev_ts
                if abs(timestamp - prev_ts) <= abs(next_ts - timestamp)
                else next_ts
            )

        nearest = int(nearest)
        timestamp = int(timestamp)
        if abs(nearest - timestamp) > self.max_annotation_time_error_ns:
            # >150ms (default): empty bbox condition for this frame.
            return None, None, None

        start = bisect.bisect_left(timestamps, nearest)
        stop = bisect.bisect_right(timestamps, nearest)
        return nearest, start, stop

    def _annotation_corners(
        self, annotations, row, target_timestamp, annotation_timestamp, poses
    ):
        world_from_current = self.get_transform(
            poses, "timestamp_ns", target_timestamp
        ).astype(np.float32)
        world_from_annotation = self.get_transform(
            poses, "timestamp_ns", annotation_timestamp
        ).astype(np.float32)
        current_from_annotation = np.linalg.solve(
            world_from_current, world_from_annotation
        ).astype(np.float32)

        scale = np.diag(
            [annotations[k][row] for k in self.shape_keys] + [1.0]
        ).astype(np.float32)
        annotation_from_box = dwm.datasets.common.get_transform(
            [annotations[k][row] for k in self.rotation_keys],
            [annotations[k][row] for k in self.translation_keys],
        ).astype(np.float32)
        template = np.asarray(
            self.default_3dbox_corner_template, dtype=np.float32
        ).T
        points = current_from_annotation @ annotation_from_box @ scale @ template
        return torch.tensor(points[:3].T, dtype=torch.float32)

    def _stable_layout_tokens(
        self, item, annotations, poses, extrinsics, intrinsics
    ):
        if "track_uuid" not in annotations:
            raise KeyError("Argoverse annotations.feather lacks official track_uuid.")

        slot_by_track = {}
        class_by_track = {}
        frame_records = []

        for frame_items in item["segment"]:
            ref = self.get_bev_map_frame_reference(frame_items)
            target_ts = int(ref["timestamp"])
            anno_ts, start, stop = self._nearest_annotation_range_safe(
                annotations, target_ts
            )

            records = []
            frame_track_ids = []
            if start is not None:
                for row in range(start, stop):
                    track_id = require_track_id(
                        annotations["track_uuid"][row], "Argoverse"
                    )
                    class_id = argoverse_class_id(annotations["category"][row])
                    slot = assign_stable_slot(
                        track_id,
                        class_id,
                        slot_by_track,
                        class_by_track,
                        self.layout_max_boxes,
                    )
                    if slot is None:
                        continue
                    records.append((slot, class_id, row, target_ts, anno_ts))
                    frame_track_ids.append(track_id)

            assert_unique_tracks(frame_track_ids, "Argoverse")
            frame_records.append(records)

        camera_frames = [
            [
                sample
                for sample in frame_items
                if str(sample.get("sensor", "")).startswith("cameras/")
            ]
            for frame_items in item["segment"]
        ]
        time_count = len(camera_frames)
        if time_count == 0:
            raise ValueError("Argoverse BEV adapter received an empty segment.")
        view_count = len(camera_frames[0])
        if view_count == 0:
            raise ValueError("Argoverse BEV adapter requires camera views.")
        if any(len(frame) != view_count for frame in camera_frames):
            raise ValueError("Argoverse camera view count changes inside sequence.")

        corners = torch.zeros(
            time_count, view_count, self.layout_max_boxes, 8, 3,
            dtype=torch.float32,
        )
        classes = torch.zeros(
            time_count, view_count, self.layout_max_boxes, dtype=torch.long
        )
        masks = torch.zeros(
            time_count, view_count, self.layout_max_boxes, dtype=torch.float32
        )

        for t, records in enumerate(frame_records):
            for slot, class_id, row, target_ts, anno_ts in records:
                box = self._annotation_corners(
                    annotations, row, target_ts, anno_ts, poses
                )
                box_np = box.numpy()
                for v, sample in enumerate(camera_frames[t]):
                    corners[t, v, slot] = box
                    classes[t, v, slot] = class_id
                    if self._layout_box_visible_in_av2_camera(
                        box_np,
                        target_ts,
                        extrinsics,
                        intrinsics,
                        poses,
                        sample,
                    ):
                        masks[t, v, slot] = 1.0

        return corners, classes, masks

    def __getitem__(self, index):
        # images/camera geometry/3dbox/static BEV map all come from track_pv.
        result = super().__getitem__(index)
        item = self.items[index]

        annotations = self._read_feather(item, "annotations.feather")
        poses = self._read_feather(item, "city_SE3_egovehicle.feather")
        extrinsics = self._read_feather(
            item, "calibration/egovehicle_SE3_sensor.feather"
        )
        intrinsics = self._read_feather(
            item, "calibration/intrinsics.feather"
        )

        result["reference_ego_transforms"] = torch.stack(
            [
                self.get_transform(
                    poses,
                    "timestamp_ns",
                    int(self.get_bev_map_frame_reference(frame)["timestamp"]),
                    "pt",
                )
                for frame in item["segment"]
            ],
            dim=0,
        )

        corners, classes, masks = self._stable_layout_tokens(
            item, annotations, poses, extrinsics, intrinsics
        )
        result["bbox_token_corners"] = corners
        result["bbox_token_classes"] = classes
        result["bbox_token_masks"] = masks

        if self.hdmap_bev_settings is not None and "hdmap_bev_images" not in result:
            raise KeyError(
                "Expected hdmap_bev_images from track_pv AV2 reader."
            )
        return result
