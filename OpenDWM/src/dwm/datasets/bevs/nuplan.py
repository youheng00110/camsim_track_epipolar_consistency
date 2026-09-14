import numpy as np
import torch

import dwm.datasets.common
import dwm.datasets.nuplan as nuplan_base
from dwm.datasets.bevs.common import (
    assign_stable_slot,
    assert_unique_tracks,
    keep_training_keys,
    nuplan_class_id,
    require_track_id,
)


class MotionDataset(nuplan_base.MotionDataset):
    """NuPlan input restricted to the fields used by the current pipe."""

    def __init__(
        self,
        pkl_path,
        sensor_root,
        cache_root,
        dataset_root,
        map_root,
        balanced_json_path,
        sequence_length,
        fps_stride_tuples,
        sensor_channels,
        hdmap_bev_settings,
        stub_key_data_dict,
        max_boxes: int = 64,
    ):
        self.layout_max_boxes = int(max_boxes)
        hdmap_bev_settings = dict(hdmap_bev_settings)
        hdmap_bev_settings["append_dynamic_object_mask"] = False
        super().__init__(
            pkl_path=pkl_path,
            sensor_root=sensor_root,
            cache_root=cache_root,
            dataset_root=dataset_root,
            map_root=map_root,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            sensor_channels=sensor_channels,
            enable_synchronization_check=True,
            enable_sample_data=False,
            enable_camera_transforms=True,
            enable_ego_transforms=True,
            _3dbox_image_settings={},
            hdmap_image_settings=None,
            projected_pc_settings=None,
            balanced_json_path=balanced_json_path,
            stub_key_data_dict=stub_key_data_dict,
            image_description_settings=None,
            layout_token_settings=None,
            hdmap_bev_settings=hdmap_bev_settings,
        )

    def _stable_layout_tokens(self, seq, cam_infos, img_sizes, cam_intrinsics):
        slot_by_track = {}
        class_by_track = {}
        frame_records = []

        for info in seq:
            if "track_token" not in info:
                raise KeyError(
                    "NuPlan info lacks official track_token from nuplan_info.py."
                )
            boxes = info.get("gt_boxes", [])
            names = info.get("gt_names", [])
            track_tokens = info["track_token"]
            if len(track_tokens) != len(boxes):
                raise ValueError(
                    "NuPlan track_token and gt_boxes lengths do not match."
                )
            records = []
            frame_track_ids = []
            for box_index, box in enumerate(boxes):
                track_id = require_track_id(
                    track_tokens[box_index],
                    "NuPlan",
                )
                class_name = names[box_index] if box_index < len(names) else ""
                class_id = nuplan_class_id(class_name)
                slot = assign_stable_slot(
                    track_id,
                    class_id,
                    slot_by_track,
                    class_by_track,
                    self.layout_max_boxes,
                )
                if slot is None:
                    continue
                records.append((slot, class_id, box_index))
                frame_track_ids.append(track_id)
            assert_unique_tracks(frame_track_ids, "NuPlan")
            frame_records.append(records)

        time_count = len(seq)
        view_count = len(self.sensor_channels)
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
            info = seq[time_index]
            for slot, class_id, box_index in records:
                box = np.asarray(info["gt_boxes"][box_index][:7], dtype=np.float32)
                box_corners_np = self._mdtoken_box_corners_lidar_xyz(box)
                box_corners = torch.from_numpy(box_corners_np).float()
                for view_index in range(view_count):
                    corners[time_index, view_index, slot] = box_corners
                    classes[time_index, view_index, slot] = class_id
                    visible = self._layout_box_visible_in_nuplan_camera(
                        box_corners_np,
                        cam_infos[time_index][view_index],
                        img_sizes[time_index][view_index],
                        cam_intrinsics[time_index][view_index],
                    )
                    if visible:
                        masks[time_index, view_index, slot] = 1.0

        return corners, classes, masks

    def __getitem__(self, index):
        item = self.items[index]
        scene = item["scene"]
        seq = [self.scenes[scene][frame_index] for frame_index in item["indices"]]
        result = {
            "fps": torch.tensor(float(item["fps"]), dtype=torch.float32),
        }

        images = []
        cam_intrinsics = []
        image_sizes = []
        cam_infos = []
        for info in seq:
            paths = self._get_sensor_paths(info)
            row_images = []
            row_intrinsics = []
            row_sizes = []
            row_infos = []
            for camera_name, image_path in zip(self.sensor_channels, paths):
                image = nuplan_base._try_open_png(image_path)
                if image is None:
                    raise RuntimeError(
                        f"Failed to read NuPlan camera image: {image_path}"
                    )
                camera_info = self._get_cam_info(info, camera_name)
                if camera_info is None:
                    raise KeyError(
                        f"Missing NuPlan camera calibration: {camera_name}"
                    )
                row_infos.append(camera_info)
                intrinsic = np.asarray(
                    camera_info["camera_intrinsics"],
                    dtype=np.float32,
                ).reshape(3, 3)
                row_images.append(image)
                row_intrinsics.append(intrinsic)
                row_sizes.append(image.size)
            images.append(row_images)
            cam_intrinsics.append(row_intrinsics)
            image_sizes.append(row_sizes)
            cam_infos.append(row_infos)

        result["images"] = images
        result["camera_intrinsics"] = torch.tensor(
            np.asarray(cam_intrinsics),
            dtype=torch.float32,
        )
        result["image_size"] = torch.tensor(
            np.asarray(
                [
                    [[width, height] for width, height in frame_sizes]
                    for frame_sizes in image_sizes
                ]
            ),
            dtype=torch.long,
        )

        box_images = []
        for time_index, info in enumerate(seq):
            row_images = []
            for view_index, camera_name in enumerate(self.sensor_channels):
                width, height = image_sizes[time_index][view_index]
                camera_info = cam_infos[time_index][view_index]
                if camera_info is None:
                    raise KeyError(
                        f"Missing NuPlan camera calibration: {camera_name}"
                    )
                intrinsic = cam_intrinsics[time_index][view_index]
                lidar_to_image = nuplan_base._lidar2image_from_caminfo(
                    camera_info,
                    intrinsic,
                )
                row_images.append(
                    self._get_3dbox_image_from_gtline(
                        info,
                        (width, height),
                        lidar_to_image,
                    )
                )
            box_images.append(row_images)
        result["3dbox_images"] = box_images

        result["hdmap_bev_images"] = [
            self._get_hdmap_bev_image(info)
            for info in seq
        ]

        corners, classes, masks = self._stable_layout_tokens(
            seq,
            cam_infos,
            image_sizes,
            cam_intrinsics,
        )
        result["bbox_token_corners"] = corners
        result["bbox_token_classes"] = classes
        result["bbox_token_masks"] = masks

        camera_transforms = []
        for row_infos in cam_infos:
            row_transforms = []
            for camera_info in row_infos:
                if camera_info is None:
                    raise KeyError("Missing NuPlan camera calibration.")
                row_transforms.append(
                    nuplan_base._se3_from_qt(
                        camera_info["sensor2ego_rotation"],
                        camera_info["sensor2ego_translation"],
                    )
                )
            camera_transforms.append(row_transforms)
        result["camera_transforms"] = torch.tensor(
            np.asarray(camera_transforms),
            dtype=torch.float32,
        )

        ego_transforms = []
        for info in seq:
            world_from_ego = np.asarray(
                info[self.ego2global_key],
                dtype=np.float32,
            )
            ego_transforms.append(
                [world_from_ego] * len(self.sensor_channels)
            )
        result["ego_transforms"] = torch.tensor(
            np.asarray(ego_transforms),
            dtype=torch.float32,
        )
        result["reference_ego_transforms"] = torch.tensor(
            np.asarray(
                [
                    np.asarray(info[self.ego2global_key], dtype=np.float32)
                    for info in seq
                ]
            ),
            dtype=torch.float32,
        )

        dwm.datasets.common.add_stub_key_data(
            self.stub_key_data_dict,
            result,
        )
        return keep_training_keys(result, self.stub_key_data_dict)
