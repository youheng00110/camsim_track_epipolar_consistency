import io
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
import waymo_open_dataset.dataset_pb2 as waymo_pb

import dwm.datasets.common
from dwm.datasets.waymo import MotionDataset as OpenDWMMotionDataset
from dwm.datasets.bevs.common import (
    assign_stable_slot,
    assert_unique_tracks,
    keep_training_keys,
    require_track_id,
    waymo_class_id,
)


class MotionDataset(OpenDWMMotionDataset):
    """Waymo input restricted to the fields used by the current pipe."""

    DRIVABLE_LANE_WIDTH_METERS = 4.0

    def __init__(
        self,
        fs,
        info_dict_path,
        sequence_length,
        fps_stride_tuples,
        sensor_channels,
        hdmap_bev_settings,
        stub_key_data_dict,
        balanced_json_path,
        dataset_root,
        max_boxes: int = 64,
        split: str = "training",
    ):
        camera_channels = [
            channel for channel in sensor_channels
            if str(channel).startswith("CAM")
        ]
        if not camera_channels:
            raise ValueError("Waymo requires at least one camera channel.")
        self.layout_max_boxes = int(max_boxes)
        hdmap_bev_settings = dict(hdmap_bev_settings)
        hdmap_bev_settings["append_dynamic_object_mask"] = False
        super().__init__(
            fs=fs,
            info_dict_path=info_dict_path,
            sequence_length=sequence_length,
            fps_stride_tuples=fps_stride_tuples,
            sensor_channels=camera_channels,
            enable_camera_transforms=True,
            enable_ego_transforms=True,
            _3dbox_image_settings={},
            hdmap_image_settings=None,
            _3dbox_bev_settings=None,
            hdmap_bev_settings=hdmap_bev_settings,
            image_description_settings=None,
            stub_key_data_dict=stub_key_data_dict,
            balanced_json_path=balanced_json_path,
            dataset_root=dataset_root,
            layout_token_settings=None,
            split=split,
        )

    @staticmethod
    def _feature_points(feature, feature_type):
        feature_data = getattr(feature, feature_type)
        field_name = "polygon" if feature_type in {"crosswalk", "driveway"} else "polyline"
        points = getattr(feature_data, field_name)
        return [[point.x, point.y, point.z, 1.0] for point in points]

    def _static_bev_image(self, map_features, frame):
        settings = self.hdmap_bev_settings
        bev_size = settings.get("bev_size", [256, 256])
        bev_from_ego = np.asarray(
            settings.get(
                "bev_from_ego_transform",
                self.default_bev_from_ego_transform,
            ),
            dtype=np.float32,
        )
        world_from_ego = np.asarray(
            frame.pose.transform,
            dtype=np.float32,
        ).reshape(4, 4)
        bev_from_world = bev_from_ego @ np.linalg.inv(world_from_ego)
        image = Image.new("RGB", tuple(bev_size))
        draw = ImageDraw.Draw(image)
        pixel_per_meter = max(
            abs(float(bev_from_ego[0, 0])),
            abs(float(bev_from_ego[1, 1])),
        )
        drivable_width = max(
            1,
            int(round(self.DRIVABLE_LANE_WIDTH_METERS * pixel_per_meter)),
        )
        lane_width = max(1, int(settings.get("pen_width", 2)))

        for feature in map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type not in {"lane", "driveway"}:
                continue
            if feature_type == "lane" and int(feature.lane.type) == 3:
                continue
            points = self._feature_points(feature, feature_type)
            if not points:
                continue
            points_np = np.asarray(points, dtype=np.float32).T
            projected = bev_from_world @ points_np
            xy = [
                (float(projected[0, index]), float(projected[1, index]))
                for index in range(projected.shape[1])
            ]
            if feature_type == "lane" and len(xy) >= 2:
                draw.line(xy, fill=(0, 0, 255), width=drivable_width)
            elif feature_type == "driveway" and len(xy) >= 3:
                draw.polygon(xy, fill=(0, 0, 255))

        for feature in map_features:
            if feature.WhichOneof("feature_data") != "road_line":
                continue
            points = self._feature_points(feature, "road_line")
            if len(points) < 2:
                continue
            points_np = np.asarray(points, dtype=np.float32).T
            projected = bev_from_world @ points_np
            xy = [
                (float(projected[0, index]), float(projected[1, index]))
                for index in range(projected.shape[1])
            ]
            draw.line(xy, fill=(0, 255, 0), width=lane_width)

        for feature in map_features:
            if feature.WhichOneof("feature_data") != "crosswalk":
                continue
            points = self._feature_points(feature, "crosswalk")
            if len(points) < 3:
                continue
            points_np = np.asarray(points, dtype=np.float32).T
            projected = bev_from_world @ points_np
            xy = [
                (float(projected[0, index]), float(projected[1, index]))
                for index in range(projected.shape[1])
            ]
            draw.polygon(xy, fill=(255, 0, 0))

        return image

    def _label_corners(self, label):
        template = torch.tensor(
            self.default_3dbox_corner_template,
            dtype=torch.float32,
        )[:, :3]
        box = label.box
        scale = torch.tensor(
            [box.length, box.width, box.height],
            dtype=torch.float32,
        )
        yaw = float(box.heading)
        rotation = torch.tensor(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        center = torch.tensor(
            [box.center_x, box.center_y, box.center_z],
            dtype=torch.float32,
        )
        return (template * scale.view(1, 3)) @ rotation.t() + center.view(1, 3)

    def _box_visible(self, corners, camera_calibration):
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4],
        )
        correction = np.asarray(
            self.extrinsic_correction,
            dtype=np.float32,
        )
        ego_from_camera = np.asarray(
            camera_calibration.extrinsic.transform,
            dtype=np.float32,
        ).reshape(4, 4)
        image_from_ego = intrinsic @ correction @ np.linalg.inv(ego_from_camera)
        corners_np = corners.detach().cpu().numpy()
        corners_h = np.concatenate(
            [corners_np, np.ones((8, 1), dtype=np.float32)],
            axis=1,
        )
        projected = (image_from_ego @ corners_h.T).T
        valid = projected[:, 2] > 1e-5
        if not np.any(valid):
            return False
        projected = projected[valid]
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        width = int(camera_calibration.width)
        height = int(camera_calibration.height)
        return not (
            u.max() < 0.0
            or u.min() >= width
            or v.max() < 0.0
            or v.min() >= height
        )

    def _stable_layout_tokens(self, frames, camera_channels):
        slot_by_track = {}
        class_by_track = {}
        frame_records = []

        for frame in frames:
            records = []
            frame_track_ids = []
            for label in frame.laser_labels:
                track_id = require_track_id(label.id, "Waymo")
                class_id = waymo_class_id(label.type)
                slot = assign_stable_slot(
                    track_id,
                    class_id,
                    slot_by_track,
                    class_by_track,
                    self.layout_max_boxes,
                )
                if slot is None:
                    continue
                records.append((slot, class_id, track_id, label))
                frame_track_ids.append(track_id)
            assert_unique_tracks(frame_track_ids, "Waymo")
            frame_records.append(records)

        time_count = len(frames)
        view_count = len(camera_channels)
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
        track_ids = torch.zeros(
            time_count,
            self.layout_max_boxes,
            dtype=torch.long,
        )
        local_id_by_track = {
            track_id: local_id
            for local_id, track_id in enumerate(slot_by_track, start=1)
        }

        for time_index, records in enumerate(frame_records):
            frame = frames[time_index]
            calibrations = [
                self.find_by_name(
                    frame.context.camera_calibrations,
                    self.sensor_name_id_dict[channel],
                )
                for channel in camera_channels
            ]
            for slot, class_id, track_id, label in records:
                track_ids[time_index, slot] = local_id_by_track[track_id]
                box_corners = self._label_corners(label)
                for view_index, calibration in enumerate(calibrations):
                    corners[time_index, view_index, slot] = box_corners
                    classes[time_index, view_index, slot] = class_id
                    if self._box_visible(
                        box_corners,
                        calibration,
                    ):
                        masks[time_index, view_index, slot] = 1.0

        return corners, classes, masks, track_ids

    def __getitem__(self, index):
        item = self.items[index]
        scene_id = item["scene"]
        camera_channels = list(self.sensor_channels)
        segment = [
            self.sample_info_lookup[f"{scene_id};{timestamp}"]
            for timestamp in item["segment"]
        ]
        all_frames = self.sample_info_dict[scene_id]
        scene_filename = f"segment-{scene_id}_with_camera_labels.tfrecord"
        split = item.get("split", self.split)
        if self.dataset_root:
            scene_path = os.path.join(
                "individual_files",
                split,
                scene_filename,
            )
        else:
            scene_path = os.path.join(
                "individual_files",
                "training",
                scene_filename,
            )
        if not self.fs.exists(scene_path):
            raise FileNotFoundError(f"Waymo record not found at {scene_path}")

        frames = [waymo_pb.Frame() for _ in segment]
        scene_frame = waymo_pb.Frame()
        with self.fs.open(scene_path, "rb") as file_obj:
            _, first_length, first_offset = all_frames[0]
            file_obj.seek(first_offset)
            scene_frame.ParseFromString(file_obj.read(first_length))
            for frame_index, frame_info in enumerate(segment):
                _, length, offset = frame_info
                file_obj.seek(offset)
                frames[frame_index].ParseFromString(file_obj.read(length))

        result = {
            "fps": torch.tensor(float(item["fps"]), dtype=torch.float32),
        }
        images = []
        intrinsics = []
        image_sizes = []
        camera_transforms = []
        ego_transforms = []
        box_images = []

        correction_inverse = torch.linalg.inv(
            torch.tensor(self.extrinsic_correction, dtype=torch.float32)
        )
        for frame in frames:
            row_images = []
            row_intrinsics = []
            row_sizes = []
            row_camera_transforms = []
            row_box_images = []
            for camera_channel in camera_channels:
                camera_id = self.sensor_name_id_dict[camera_channel]
                image_data = self.find_by_name(frame.images, camera_id)
                calibration = self.find_by_name(
                    frame.context.camera_calibrations,
                    camera_id,
                )
                if image_data is None:
                    raise KeyError(f"Missing Waymo camera image: {camera_channel}")
                if calibration is None:
                    raise KeyError(
                        f"Missing Waymo camera calibration: {camera_channel}"
                    )
                with io.BytesIO(image_data.image) as buffer:
                    image = Image.open(buffer)
                    image.load()
                row_images.append(image)
                row_intrinsics.append(
                    dwm.datasets.common.make_intrinsic_matrix(
                        calibration.intrinsic[0:2],
                        calibration.intrinsic[2:4],
                        "pt",
                    )
                )
                row_sizes.append(
                    torch.tensor(
                        [calibration.width, calibration.height],
                        dtype=torch.long,
                    )
                )
                ego_from_camera = torch.tensor(
                    calibration.extrinsic.transform,
                    dtype=torch.float32,
                ).reshape(4, 4)
                row_camera_transforms.append(
                    ego_from_camera @ correction_inverse
                )
                row_box_images.append(
                    self.get_3dbox_image(
                        frame.laser_labels,
                        calibration,
                        self._3dbox_image_settings,
                    )
                )
            images.append(row_images)
            intrinsics.append(torch.stack(row_intrinsics))
            image_sizes.append(torch.stack(row_sizes))
            camera_transforms.append(torch.stack(row_camera_transforms))
            box_images.append(row_box_images)
            world_from_ego = torch.tensor(
                frame.pose.transform,
                dtype=torch.float32,
            ).reshape(4, 4)
            ego_transforms.append(
                torch.stack([world_from_ego] * len(camera_channels))
            )

        result["images"] = images
        result["camera_intrinsics"] = torch.stack(intrinsics)
        result["image_size"] = torch.stack(image_sizes)
        result["camera_transforms"] = torch.stack(camera_transforms)
        result["ego_transforms"] = torch.stack(ego_transforms)
        result["reference_ego_transforms"] = torch.stack(
            [frame_poses[0] for frame_poses in ego_transforms]
        )
        result["3dbox_images"] = box_images
        result["hdmap_bev_images"] = [
            self._static_bev_image(scene_frame.map_features, frame)
            for frame in frames
        ]

        corners, classes, masks, track_ids = self._stable_layout_tokens(
            frames,
            camera_channels,
        )
        result["bbox_token_corners"] = corners
        result["bbox_token_classes"] = classes
        result["bbox_token_masks"] = masks
        result["bbox_token_track_ids"] = track_ids

        if self.image_description_settings is not None:
            image_captions = []
            for frame_info in segment:
                row_captions = []
                for camera_channel in camera_channels:
                    camera_id = self.sensor_name_id_dict[camera_channel]
                    row_captions.append(
                        self.get_image_description(
                            self.image_descriptions,
                            self.time_list_dict,
                            scene_id,
                            frame_info[0],
                            camera_id,
                        )
                    )
                image_captions.append(
                    dwm.datasets.common.align_image_description_crossview(
                        row_captions,
                        self.image_description_settings,
                    )
                )
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        caption,
                        self.image_description_settings,
                        self.image_desc_rs,
                    )
                    for caption in frame_captions
                ]
                for frame_captions in image_captions
            ]

        dwm.datasets.common.add_stub_key_data(
            self.stub_key_data_dict,
            result,
        )
        return keep_training_keys(result, self.stub_key_data_dict)
