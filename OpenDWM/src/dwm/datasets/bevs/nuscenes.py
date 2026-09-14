import numpy as np
import torch

import dwm.datasets.common
from dwm.datasets.nuscenes import MotionDataset as OpenDWMMotionDataset
from dwm.datasets.bevs.common import (
    assign_stable_slot,
    assert_unique_tracks,
    keep_training_keys,
    nuscenes_class_id,
    require_track_id,
)


class MotionDataset(OpenDWMMotionDataset):
    """nuScenes input restricted to the fields used by the current training pipe."""

    def __init__(
        self,
        fs,
        dataset_name,
        sequence_length,
        fps_stride_tuples,
        split,
        sensor_channels,
        hdmap_bev_settings,
        stub_key_data_dict,
        max_boxes: int = 64,
    ):
        camera_channels = [
            channel for channel in sensor_channels
            if str(channel).startswith("CAM")
        ]
        if not camera_channels:
            raise ValueError("nuScenes requires at least one camera channel.")
        if not sensor_channels or sensor_channels[0] != "LIDAR_TOP":
            raise ValueError("nuScenes requires LIDAR_TOP as the first frame reference.")
        self.layout_max_boxes = int(max_boxes)
        hdmap_bev_settings = dict(hdmap_bev_settings)
        hdmap_bev_settings["append_dynamic_object_mask"] = False
        super().__init__(
            fs=fs,
            dataset_name=dataset_name,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            split=split,
            sensor_channels=sensor_channels,
            keyframe_only=True,
            enable_synchronization_check=False,
            enable_scene_description=False,
            enable_camera_transforms=True,
            enable_ego_transforms=True,
            enable_sample_data=False,
            _3dbox_image_settings={},
            hdmap_image_settings=None,
            image_segmentation_settings=None,
            foreground_region_image_settings=None,
            _3dbox_bev_settings=None,
            hdmap_bev_settings=hdmap_bev_settings,
            image_description_settings=None,
            stub_key_data_dict=stub_key_data_dict,
            balanced_json_path=None,
            layout_token_settings=None,
        )

    def _frame_annotations(self, frame_items):
        # Match the original OpenDWM layout-token coordinate frame.
        # With the current config, frame_items[0] is LIDAR_TOP, while the
        # original dataset selects the first camera sample as the BEV/box
        # reference. nuScenes sensors are asynchronous, so using LIDAR_TOP
        # changes the ego pose and shifts every box corner.
        ref_sample_data = self.get_bev_map_frame_reference(frame_items)
        sample = self.query(
            self.tables,
            self.indices,
            "sample",
            ref_sample_data["sample_token"],
        )
        annotations = self.query_range(
            self.tables,
            self.indices,
            "sample_annotation",
            sample["token"],
            column_name="sample_token",
        )
        return ref_sample_data, annotations

    def _annotation_class_id(self, annotation):
        instance = self.query(
            self.tables,
            self.indices,
            "instance",
            annotation["instance_token"],
        )
        category = self.query(
            self.tables,
            self.indices,
            "category",
            instance["category_token"],
        )
        return nuscenes_class_id(category["name"])

    def _annotation_corners(self, annotation, ref_sample_data):
        ego_to_global = dwm.datasets.common.get_transform(
            ref_sample_data["rotation"],
            ref_sample_data["translation"],
            "pt",
        )
        global_to_ego = torch.linalg.inv(ego_to_global)
        box_to_global = dwm.datasets.common.get_transform(
            annotation["rotation"],
            annotation["translation"],
            "pt",
        )
        size = torch.tensor(
            [annotation["size"][1], annotation["size"][0], annotation["size"][2]],
            dtype=torch.float32,
        )
        template = torch.tensor(
            self.default_3dbox_corner_template,
            dtype=torch.float32,
        )
        template[:, :3] *= size.view(1, 3)
        return (global_to_ego @ box_to_global @ template.t()).t()[:, :3]

    def _box_visible(self, corners, ref_sample_data, camera_sample_data):
        calibrated_sensor = self.query(
            self.tables,
            self.indices,
            "calibrated_sensor",
            camera_sample_data["calibrated_sensor_token"],
        )
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(
            calibrated_sensor["camera_intrinsic"],
            dtype=np.float32,
        )
        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"],
            calibrated_sensor["translation"],
        ).astype(np.float32)
        world_from_ref = dwm.datasets.common.get_transform(
            ref_sample_data["rotation"],
            ref_sample_data["translation"],
        ).astype(np.float32)
        world_from_camera_ego = dwm.datasets.common.get_transform(
            camera_sample_data["rotation"],
            camera_sample_data["translation"],
        ).astype(np.float32)
        camera_from_ref = np.linalg.solve(
            world_from_camera_ego @ ego_from_camera,
            world_from_ref,
        ).astype(np.float32)
        corners_np = corners.detach().cpu().numpy()
        corners_h = np.concatenate(
            [corners_np, np.ones((8, 1), dtype=np.float32)],
            axis=1,
        )
        projected = (intrinsic @ camera_from_ref @ corners_h.T).T
        valid = projected[:, 2] > 1e-5
        if not np.any(valid):
            return False
        projected = projected[valid]
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        width = int(camera_sample_data["width"])
        height = int(camera_sample_data["height"])
        return not (
            u.max() < 0.0
            or u.min() >= width
            or v.max() < 0.0
            or v.min() >= height
        )

    def _stable_layout_tokens(self, segment):
        slot_by_track = {}
        class_by_track = {}
        frame_records = []

        for frame_items in segment:
            ref_sample_data, annotations = self._frame_annotations(frame_items)
            current_records = []
            current_track_ids = []
            for annotation in annotations:
                if "instance_token" not in annotation:
                    raise KeyError(
                        "nuScenes sample_annotation lacks official instance_token."
                    )
                track_id = require_track_id(
                    annotation["instance_token"],
                    "nuScenes",
                )
                class_id = self._annotation_class_id(annotation)
                slot = assign_stable_slot(
                    track_id,
                    class_id,
                    slot_by_track,
                    class_by_track,
                    self.layout_max_boxes,
                )
                if slot is None:
                    continue
                current_records.append((slot, class_id, annotation))
                current_track_ids.append(track_id)
            assert_unique_tracks(current_track_ids, "nuScenes")
            frame_records.append((ref_sample_data, current_records))

        camera_frames = [
            [
                sample_data for sample_data in frame_items
                if self.check_sensor(
                    self.tables,
                    self.indices,
                    sample_data,
                    modality="camera",
                )
            ]
            for frame_items in segment
        ]
        time_count = len(segment)
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

        for time_index, (ref_sample_data, records) in enumerate(frame_records):
            for slot, class_id, annotation in records:
                box_corners = self._annotation_corners(
                    annotation,
                    ref_sample_data,
                )
                for view_index, camera_sample_data in enumerate(camera_frames[time_index]):
                    corners[time_index, view_index, slot] = box_corners
                    classes[time_index, view_index, slot] = class_id
                    if self._box_visible(
                        box_corners,
                        ref_sample_data,
                        camera_sample_data,
                    ):
                        masks[time_index, view_index, slot] = 1.0

        return corners, classes, masks

    def __getitem__(self, index):
        result = super().__getitem__(index)
        item = self.items[index]
        segment = [
            [
                self.query(self.tables, self.indices, "sample_data", token)
                for token in frame_tokens
            ]
            for frame_tokens in item["segment"]
        ]
        reference_ego_transforms = []
        for frame_items in segment:
            reference_sample_data = self.get_bev_map_frame_reference(frame_items)
            reference_ego_transforms.append(
                dwm.datasets.common.get_transform(
                    reference_sample_data["rotation"],
                    reference_sample_data["translation"],
                    "pt",
                )
            )
        result["reference_ego_transforms"] = torch.stack(
            reference_ego_transforms
        )

        corners, classes, masks = self._stable_layout_tokens(segment)
        result["bbox_token_corners"] = corners
        result["bbox_token_classes"] = classes
        result["bbox_token_masks"] = masks
        return keep_training_keys(result, self.stub_key_data_dict)
