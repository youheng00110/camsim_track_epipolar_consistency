import bisect

import numpy as np
import pyarrow.feather
import torch

import dwm.datasets.common
from dwm.datasets.argoverse import MotionDataset as OpenDWMMotionDataset
from dwm.datasets.bevs.common import (
    argoverse_class_id,
    assign_stable_slot,
    assert_unique_tracks,
    keep_training_keys,
    require_track_id,
)


class MotionDataset(OpenDWMMotionDataset):
    """Argoverse 2 input restricted to the fields used by the current pipe."""

    def __init__(
        self,
        fs,
        sequence_length,
        fps_stride_tuples,
        sensor_channels,
        hdmap_bev_settings,
        stub_key_data_dict,
        balanced_json_path,
        index_json_path,
        dataset_root,
        max_boxes: int = 64,
        split: str = "train",
    ):
        camera_channels = [
            channel for channel in sensor_channels
            if str(channel).startswith("cameras/")
        ]
        if not camera_channels:
            raise ValueError("Argoverse requires at least one camera channel.")
        if not sensor_channels or sensor_channels[0] != "lidar":
            raise ValueError("Argoverse requires lidar as the frame reference.")
        self.layout_max_boxes = int(max_boxes)
        hdmap_bev_settings = dict(hdmap_bev_settings)
        hdmap_bev_settings["append_dynamic_object_mask"] = False
        super().__init__(
            fs=fs,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            sensor_channels=sensor_channels,
            hide_lidar=True,
            enable_synchronization_check=True,
            enable_camera_transforms=True,
            enable_ego_transforms=True,
            _3dbox_image_settings={},
            hdmap_image_settings=None,
            _3dbox_bev_settings=None,
            hdmap_bev_settings=hdmap_bev_settings,
            image_description_settings=None,
            stub_key_data_dict=stub_key_data_dict,
            balanced_json_path=balanced_json_path,
            index_json_path=index_json_path,
            dataset_root=dataset_root,
            scene_dirs=None,
            layout_token_settings=None,
            split=split,
        )

    def _read_feather(self, item, filename):
        path = f"{item['split']}/{item['scene_id']}/{filename}"
        with self.fs.open(path) as file_obj:
            return pyarrow.feather.read_table(file_obj).to_pydict()

    def _nearest_annotation_range(self, annotations, timestamp):
        annotation_timestamps = annotations["timestamp_ns"]
        position = bisect.bisect_left(annotation_timestamps, timestamp)
        if position == 0:
            nearest = annotation_timestamps[0]
        elif position >= len(annotation_timestamps):
            nearest = annotation_timestamps[-1]
        else:
            previous_time = annotation_timestamps[position - 1]
            next_time = annotation_timestamps[position]
            if abs(timestamp - previous_time) <= abs(next_time - timestamp):
                nearest = previous_time
            else:
                nearest = next_time
        if abs(int(nearest) - int(timestamp)) > 100_000_000:
            raise RuntimeError(
                "Argoverse annotation timestamp mismatch exceeds 100000000 ns: "
                f"camera={int(timestamp)}, annotation={int(nearest)}"
            )
        start = bisect.bisect_left(annotation_timestamps, nearest)
        stop = bisect.bisect_right(annotation_timestamps, nearest)
        return int(nearest), start, stop

    def _annotation_corners(self, annotations, row, target_timestamp, anno_timestamp, poses):
        world_from_current_ego = self.get_transform(
            poses,
            "timestamp_ns",
            target_timestamp,
        ).astype(np.float32)
        world_from_annotation_ego = self.get_transform(
            poses,
            "timestamp_ns",
            anno_timestamp,
        ).astype(np.float32)
        current_ego_from_annotation_ego = np.linalg.solve(
            world_from_current_ego,
            world_from_annotation_ego,
        ).astype(np.float32)
        scale = np.diag(
            [annotations[key][row] for key in self.shape_keys] + [1.0]
        ).astype(np.float32)
        annotation_ego_from_box = dwm.datasets.common.get_transform(
            [annotations[key][row] for key in self.rotation_keys],
            [annotations[key][row] for key in self.translation_keys],
        ).astype(np.float32)
        template = np.asarray(
            self.default_3dbox_corner_template,
            dtype=np.float32,
        ).T
        points = (
            current_ego_from_annotation_ego
            @ annotation_ego_from_box
            @ scale
            @ template
        )
        return torch.tensor(points[:3].T, dtype=torch.float32)

    def _box_visible(
        self,
        corners,
        reference_timestamp,
        extrinsics,
        intrinsics,
        poses,
        sample_data,
    ):
        sensor_name = sample_data["sensor"][8:]
        intrinsic_index = bisect.bisect_left(
            intrinsics["sensor_name"],
            sensor_name,
        )
        if (
            intrinsic_index >= len(intrinsics["sensor_name"])
            or intrinsics["sensor_name"][intrinsic_index] != sensor_name
        ):
            return False
        image_width, image_height = [
            int(intrinsics[key][intrinsic_index])
            for key in self.intrinsic_size_keys
        ]
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [
                intrinsics[key][intrinsic_index]
                for key in self.intrinsic_focal_keys
            ],
            [
                intrinsics[key][intrinsic_index]
                for key in self.intrinsic_center_keys
            ],
        )
        ego_from_camera = self.get_transform(
            extrinsics,
            "sensor_name",
            sensor_name,
        ).astype(np.float32)
        world_from_reference = self.get_transform(
            poses,
            "timestamp_ns",
            reference_timestamp,
        ).astype(np.float32)
        world_from_camera_ego = self.get_transform(
            poses,
            "timestamp_ns",
            sample_data["timestamp"],
        ).astype(np.float32)
        camera_from_reference = np.linalg.solve(
            world_from_camera_ego @ ego_from_camera,
            world_from_reference,
        ).astype(np.float32)
        corners_np = corners.detach().cpu().numpy()
        corners_h = np.concatenate(
            [corners_np, np.ones((8, 1), dtype=np.float32)],
            axis=1,
        )
        projected = (intrinsic @ camera_from_reference @ corners_h.T).T
        valid = projected[:, 2] > 1e-5
        if not np.any(valid):
            return False
        projected = projected[valid]
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        return not (
            u.max() < 0.0
            or u.min() >= image_width
            or v.max() < 0.0
            or v.min() >= image_height
        )

    def _stable_layout_tokens(
        self,
        item,
        annotations,
        poses,
        extrinsics,
        intrinsics,
    ):
        if "track_uuid" not in annotations:
            raise KeyError(
                "Argoverse annotations.feather lacks official track_uuid."
            )
        slot_by_track = {}
        class_by_track = {}
        frame_records = []

        for frame_items in item["segment"]:
            ref_sample = frame_items[0]
            target_timestamp = int(ref_sample["timestamp"])
            anno_timestamp, start, stop = self._nearest_annotation_range(
                annotations,
                target_timestamp,
            )
            records = []
            frame_track_ids = []
            if start is not None:
                for row in range(start, stop):
                    track_id = require_track_id(
                        annotations["track_uuid"][row],
                        "Argoverse",
                    )
                    class_id = argoverse_class_id(
                        annotations["category"][row]
                    )
                    slot = assign_stable_slot(
                        track_id,
                        class_id,
                        slot_by_track,
                        class_by_track,
                        self.layout_max_boxes,
                    )
                    if slot is None:
                        continue
                    records.append(
                        (slot, class_id, row, target_timestamp, anno_timestamp)
                    )
                    frame_track_ids.append(track_id)
            assert_unique_tracks(frame_track_ids, "Argoverse")
            frame_records.append(records)

        camera_frames = [
            [
                sample_data for sample_data in frame_items
                if sample_data["sensor"].startswith("cameras/")
            ]
            for frame_items in item["segment"]
        ]
        time_count = len(item["segment"])
        view_count = len(camera_frames[0])
        corners = torch.zeros(
            time_count,
            view_count,
            self.layout_max_boxes,
            8,
            3,
            dtype=torch.float32,
        )
        classes = torch.zeros(
            time_count,
            view_count,
            self.layout_max_boxes,
            dtype=torch.long,
        )
        masks = torch.zeros(
            time_count,
            view_count,
            self.layout_max_boxes,
            dtype=torch.float32,
        )

        for time_index, records in enumerate(frame_records):
            for slot, class_id, row, target_timestamp, anno_timestamp in records:
                box_corners = self._annotation_corners(
                    annotations,
                    row,
                    target_timestamp,
                    anno_timestamp,
                    poses,
                )
                for view_index, sample_data in enumerate(camera_frames[time_index]):
                    corners[time_index, view_index, slot] = box_corners
                    classes[time_index, view_index, slot] = class_id
                    visible = self._box_visible(
                        box_corners,
                        target_timestamp,
                        extrinsics,
                        intrinsics,
                        poses,
                        sample_data,
                    )
                    if visible:
                        masks[time_index, view_index, slot] = 1.0

        return corners, classes, masks

    def __getitem__(self, index):
        result = super().__getitem__(index)
        item = self.items[index]
        annotations = self._read_feather(item, "annotations.feather")
        poses = self._read_feather(item, "city_SE3_egovehicle.feather")
        extrinsics = self._read_feather(
            item,
            "calibration/egovehicle_SE3_sensor.feather",
        )
        intrinsics = self._read_feather(
            item,
            "calibration/intrinsics.feather",
        )
        result["reference_ego_transforms"] = torch.stack(
            [
                self.get_transform(
                    poses,
                    "timestamp_ns",
                    int(frame_items[0]["timestamp"]),
                    "pt",
                )
                for frame_items in item["segment"]
            ]
        )
        corners, classes, masks = self._stable_layout_tokens(
            item,
            annotations,
            poses,
            extrinsics,
            intrinsics,
        )
        result["bbox_token_corners"] = corners
        result["bbox_token_classes"] = classes
        result["bbox_token_masks"] = masks
        return keep_training_keys(result, self.stub_key_data_dict)
