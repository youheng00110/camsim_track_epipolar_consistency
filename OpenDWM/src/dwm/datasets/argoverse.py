import bisect
import dwm.common
import dwm.datasets.common
import dwm.fs.czip
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import pyarrow.feather
import re
import torch
import os
import glob
from tqdm import tqdm


class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the Argoverse 2 Sensor dataset.

    Args:
        fs (dwm.fs.czip.CombinedZipFileSystem): The file system for the dataset
            content files.
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        sensor_channels (list): The string list of required views in
            "cameras/ring_front_center", "cameras/ring_front_left",
            "cameras/ring_front_right", "cameras/ring_rear_left",
            "cameras/ring_rear_right", "cameras/ring_side_left",
            "cameras/ring_side_right", "lidar", following the Argoverse sensor
            name.
        enable_synchronization_check (bool): When this feature is enabled, if
            the timestamp error of a certain frame in the read video clip
            exceeds half of the frame interval, this video clip will be
            discarded. True as default.
        enable_camera_transforms (bool): If set to True, the data item will
            include the "camera_transforms", "camera_intrinsics", "image_size"
            if camera modality exists, and include "lidar_transforms" if LiDAR
            modality exists. For a detailed definition of transforms, please
            refer to the dataset README.
        enable_ego_transforms (bool): If set to True, the data item will
            include the "ego_transforms". For a detailed definition of
            transforms, please refer to the dataset README.
        _3dbox_image_settings (dict or None): If set, the data item will
            include the "3dbox_images".
        hdmap_image_settings (dict or None): If set, the data item will include
            the "hdmap_images".
        _3dbox_bev_settings (dict or None): If set, the data item will include
            the "3dbox_bev_images".
        hdmap_bev_settings (dict or None): If set, the data item will include
            the "hdmap_bev_images".
        image_description_settings (dict or None): If set, the data item will
            include the "image_description". The "path" in the setting is for
            the content JSON file. The "time_list_dict_path" in the setting is
            for the file to seek the nearest labelled time points. Please refer
            to dwm.datasets.common.make_image_description_string() for details.
        stub_key_data_dict (dict or None): The dict of stub key and data, to
            align with other datasets with keys and data missing in this
            dataset. Please refer to dwm.datasets.common.add_stub_key_data()
            for details.
    """

    point_keys = ["x", "y", "z"]
    shape_keys = ["length_m", "width_m", "height_m"]
    rotation_keys = ["qw", "qx", "qy", "qz"]
    translation_keys = ["tx_m", "ty_m", "tz_m"]
    intrinsic_focal_keys = ["fx_px", "fy_px"]
    intrinsic_center_keys = ["cx_px", "cy_px"]
    intrinsic_size_keys = ["width_px", "height_px"]

    default_3dbox_color_table = {
        "BICYCLIST": (255, 0, 0),
        "MOTORCYCLIST": (255, 0, 0),
        "PEDESTRIAN": (255, 0, 0),
        "BICYCLE": (128, 255, 0),
        "MOTORCYCLE": (0, 255, 128),
        "BOX_TRUCK": (255, 255, 0),
        "BUS": (128, 0, 255),
        "LARGE_VEHICLE": (0, 0, 255),
        "REGULAR_VEHICLE": (0, 0, 255),
        "SCHOOL_BUS": (128, 0, 255),
        "TRUCK": (255, 255, 0),
        "TRUCK_CAB": (255, 255, 0),
        "VEHICULAR_TRAILER": (255, 255, 255)
    }
    default_hdmap_color_table = {
        "drivable_areas": (0, 0, 255),
        "lane_segments": (0, 255, 0),
        "pedestrian_crossings": (255, 0, 0)
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        (6, 3), (6, 5)
    ]
    default_bev_from_ego_transform = [
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1]
    ]
    default_bev_3dbox_corner_template = [
        [-0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1],
        [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1]
    ]
    default_bev_3dbox_edge_indices = [(0, 2), (2, 3), (3, 1), (1, 0)]

    @staticmethod
    def enumerate_segments(
        channel_sample_data_list: list, sequence_length: int, fps, stride,
        enable_synchronization_check: bool
    ):
        # stride > 0:
        #   * FPS == 0: offset between segment beginings are by index.
        #   * FPS > 0: offset between segment beginings are by second.

        # ✅ 先过滤空 channel
        csdl = [sdl for sdl in channel_sample_data_list if len(sdl) > 0]

        if len(csdl) == 0:
            return

        # ✅ 用第一个非空的作为 reference
        reference_channel = csdl[0]

        channel_timestamp_list = [
            [i["timestamp"] for i in sdl] for sdl in csdl
        ]

        if fps == 0:
            for t in range(0, len(reference_channel), max(1, stride)):
            # frames are extracted by the index
                # find the indices of the first frame of channels matching the
                # given timestamp
                ct0 = [
                    dwm.datasets.common.find_nearest(
                        tl, csdl[0][t]["timestamp"])
                    for tl in channel_timestamp_list
                ]

                if all([
                    t0 + sequence_length <= len(sdl)
                    for t0, sdl in zip(ct0, csdl)
                ]):
                    # TODO: the lidar frequency is different from the camera
                    # frequency, fix it for temporal training
                    yield [
                        [sdl[t0 + i] for t0, sdl in zip(ct0, csdl)]
                        for i in range(sequence_length)
                    ]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(sdl, sequence_duration, stride):
                s = sdl[-1]["timestamp"] / 1000000000 - sequence_duration
                t = sdl[0]["timestamp"] / 1000000000
                while t <= s:
                    yield t
                    t += stride

            for t in enumerate_begin_time(
                reference_channel, sequence_length / fps, stride
            ):
                # find the indices of the first frame of channels matching the
                # given timestamp
                ct0 = [t * 1000000000 for _ in csdl]
                channel_expected_times = [
                    [t0 + i / fps * 1000000000 for i in range(sequence_length)]
                    for t0 in ct0
                ]
                channel_candidates = [
                    [
                        sdl[dwm.datasets.common.find_nearest(timestamps, i)]
                        for i in expected_times
                    ]
                    for sdl, timestamps, expected_times in zip(
                        csdl, channel_timestamp_list, channel_expected_times)
                ]
                max_time_error = max([
                    abs(i0["timestamp"] - i1)
                    for candidates, expected_times in zip(
                        channel_candidates, channel_expected_times)
                    for i0, i1 in zip(candidates, expected_times)
                ])
                if (
                    not enable_synchronization_check or
                    max_time_error <= 500000000 / fps
                ):
                    yield [
                        [candidates[i] for candidates in channel_candidates]
                        for i in range(sequence_length)
                    ]

    @staticmethod
    def feather_query(
        feather_dict: dict, key_column: str, queried_key,
        queried_columns: list
    ):
        keys = feather_dict[key_column]
        index = bisect.bisect_left(keys, queried_key)
        if index < 0 or index >= len(keys) or keys[index] != queried_key:
            raise Exception("The key {} is not found".format(queried_key))

        return [feather_dict[k][index] for k in queried_columns]

    @staticmethod
    def get_transform(
        pose_dict: dict, key_column: str, queried_key, output_type: str = "np"
    ):
        keys = pose_dict[key_column]
        index = bisect.bisect_left(keys, queried_key)
        if index < 0 or index >= len(keys) or keys[index] != queried_key:
            raise Exception("The key {} is not found".format(queried_key))

        return dwm.datasets.common.get_transform(
            [pose_dict[k][index] for k in MotionDataset.rotation_keys],
            [pose_dict[k][index] for k in MotionDataset.translation_keys],
            output_type)

    @staticmethod
    def get_3dbox_image(
        annotations: dict, ref_timestamp: int, extrinsics: dict,
        intrinsics: dict, poses: dict, sample_data: dict,
        _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 10)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the referenced ego space to the image space
        sensor_name = sample_data["sensor"][8:]

        i = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
        image_size = tuple(
            [intrinsics[j][i] for j in MotionDataset.intrinsic_size_keys])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [intrinsics[j][i] for j in MotionDataset.intrinsic_focal_keys],
            [intrinsics[j][i] for j in MotionDataset.intrinsic_center_keys])

        ego_from_camera = MotionDataset.get_transform(
            extrinsics, "sensor_name", sensor_name)
        world_from_ref = MotionDataset.get_transform(
            poses, "timestamp_ns", ref_timestamp)
        world_from_ego = MotionDataset.get_transform(
            poses, "timestamp_ns", sample_data["timestamp"])
        camera_from_ref = np.linalg.solve(
            world_from_ego @ ego_from_camera, world_from_ref)
        image_from_ref = intrinsic @ camera_from_ref

        # draw annotations to the image
        def list_annotation():
            a0 = bisect.bisect_left(annotations["timestamp_ns"], ref_timestamp)
            a1 = bisect.bisect_right(
                annotations["timestamp_ns"], ref_timestamp)
            for i in range(a0, a1):
                yield i

        def get_world_transform(i):
            scale = np.diag(
                [annotations[j][i] for j in MotionDataset.shape_keys] + [1])
            ref_from_annotation = dwm.datasets.common.get_transform(
                [annotations[j][i] for j in MotionDataset.rotation_keys],
                [annotations[j][i] for j in MotionDataset.translation_keys])
            return ref_from_annotation @ scale

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_ref, list_annotation, get_world_transform,
            lambda i: annotations["category"][i], pen_width, color_table,
            corner_templates, edge_indices)

        return image

    @staticmethod
    def get_hdmap_image(
        map: dict, extrinsics: dict, intrinsics: dict, poses: dict,
        sample_data: dict, hdmap_image_settings: dict
    ):
        # options
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 10)
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the world (map) space to the image space
        sensor_name = sample_data["sensor"][8:]

        i = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
        image_size = tuple(
            [intrinsics[j][i] for j in MotionDataset.intrinsic_size_keys])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [intrinsics[j][i] for j in MotionDataset.intrinsic_focal_keys],
            [intrinsics[j][i] for j in MotionDataset.intrinsic_center_keys])

        ego_from_camera = MotionDataset.get_transform(
            extrinsics, "sensor_name", sensor_name)
        world_from_ego = MotionDataset.get_transform(
            poses, "timestamp_ns", sample_data["timestamp"])
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera)
        image_from_world = intrinsic @ camera_from_world

        # draw map elements to the image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        if "lane_segments" in color_table and "lane_segments" in map:
            pen_color = tuple(color_table["lane_segments"])
            for i in map["lane_segments"].values():
                if i["is_intersection"]:
                    continue

                line_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["left_lane_boundary"]
                ]).transpose()
                p = image_from_world @ line_nodes
                for j in range(1, p.shape[1]):
                    xy = dwm.datasets.common.project_line(
                        p[:, j - 1], p[:, j], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

                line_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["right_lane_boundary"]
                ]).transpose()
                p = image_from_world @ line_nodes
                for j in range(1, p.shape[1]):
                    xy = dwm.datasets.common.project_line(
                        p[:, j - 1], p[:, j], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        if "drivable_areas" in color_table and "drivable_areas" in map:
            pen_color = tuple(color_table["drivable_areas"])
            for i in map["drivable_areas"].values():
                polygon_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["area_boundary"]
                ]).transpose()
                p = image_from_world @ polygon_nodes
                m = p.shape[1]
                for j in range(m):
                    xy = dwm.datasets.common.project_line(
                        p[:, j], p[:, (j + 1) % m], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        if "pedestrian_crossings" in color_table and \
                "pedestrian_crossings" in map:
            pen_color = tuple(color_table["pedestrian_crossings"])
            for i in map["pedestrian_crossings"].values():
                e1, e2 = i["edge1"], i["edge2"]
                polygon_nodes = np.array([
                    [e1[0][j] for j in MotionDataset.point_keys] + [1],
                    [e1[1][j] for j in MotionDataset.point_keys] + [1],
                    [e2[1][j] for j in MotionDataset.point_keys] + [1],
                    [e2[0][j] for j in MotionDataset.point_keys] + [1]
                ]).transpose()
                p = image_from_world @ polygon_nodes
                m = p.shape[1]
                for j in range(m):
                    xy = dwm.datasets.common.project_line(
                        p[:, j], p[:, (j + 1) % m], far_z=max_distance)
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(
        annotations: dict, sample_data: dict, _3dbox_bev_settings: dict
    ):
        # options
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_bev_settings.get(
            "corner_templates",
            MotionDataset.default_bev_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get(
            "edge_indices", MotionDataset.default_bev_3dbox_edge_indices)

        # get the transform from the referenced ego space to the BEV space
        bev_from_ego = np.array(bev_from_ego_transform)

        # draw annotations to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        timestamp = sample_data["timestamp"]
        a0 = bisect.bisect_left(annotations["timestamp_ns"], timestamp)
        a1 = bisect.bisect_right(annotations["timestamp_ns"], timestamp)
        for i in range(a0, a1):
            category = annotations["category"][i]
            if category in color_table:
                pen_color = tuple(color_table[category])
                scale = np.diag(
                    [annotations[j][i] for j in MotionDataset.shape_keys] + [1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    [annotations[j][i] for j in MotionDataset.rotation_keys],
                    [annotations[j][i] for j in MotionDataset.translation_keys])
                p = bev_from_ego @ ego_from_annotation @ scale @ \
                    corner_templates_np
                if fill_box:
                    draw.polygon(
                        [(p[0, a], p[1, a]) for a, _ in edge_indices],
                        fill=pen_color, width=pen_width)
                else:
                    for a, b in edge_indices:
                        draw.line(
                            (p[0, a], p[1, a], p[0, b], p[1, b]),
                            fill=pen_color, width=pen_width)

        return image


    @staticmethod
    def _mdtoken_dynamic_class_id(category_name):
        name = str(category_name).lower()

        if "truck" in name:
            return 1
        if "bus" in name:
            return 3
        if "trailer" in name:
            return 4
        if "barrier" in name:
            return 5
        if "motorcycle" in name:
            return 6
        if "bicycle" in name or "cyclist" in name or "bike" in name:
            return 7
        if "pedestrian" in name or "ped" in name:
            return 8
        if "cone" in name:
            return 9
        if "vehicle" in name or "car" in name:
            return 0
        if "regular_vehicle" in name or "large_vehicle" in name:
            return 0

        return 0

    @staticmethod
    def _mdtoken_yaw_from_quaternion(q):
        q = np.asarray(q, dtype=np.float32).reshape(-1)

        if q.shape[0] != 4:
            return 0.0

        qw, qx, qy, qz = [float(v) for v in q]

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _mdtoken_box7_from_dict(obj):
        if not isinstance(obj, dict):
            return None, None

        name = (
            obj.get("category_name", None)
            or obj.get("category", None)
            or obj.get("label", None)
            or obj.get("name", None)
            or obj.get("type", None)
            or "car"
        )

        if all(k in obj for k in ["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m"]):
            x = float(obj["tx_m"])
            y = float(obj["ty_m"])
            z = float(obj["tz_m"])
            length = float(obj["length_m"])
            width = float(obj["width_m"])
            height = float(obj["height_m"])

            if all(k in obj for k in ["qw", "qx", "qy", "qz"]):
                yaw = MotionDataset._mdtoken_yaw_from_quaternion(
                    [obj["qw"], obj["qx"], obj["qy"], obj["qz"]]
                )
            else:
                yaw = float(obj.get("yaw", obj.get("heading", 0.0)))

            return np.asarray(
                [x, y, z, length, width, height, yaw],
                dtype=np.float32,
            ), name

        if "translation" in obj and ("size" in obj or "dimensions" in obj):
            t = np.asarray(obj["translation"], dtype=np.float32).reshape(-1)
            s = np.asarray(
                obj.get("size", obj.get("dimensions")),
                dtype=np.float32,
            ).reshape(-1)

            if t.shape[0] < 3 or s.shape[0] < 3:
                return None, name

            if "rotation" in obj:
                yaw = MotionDataset._mdtoken_yaw_from_quaternion(obj["rotation"])
            else:
                yaw = float(obj.get("yaw", obj.get("heading", 0.0)))

            return np.asarray(
                [t[0], t[1], t[2], s[0], s[1], s[2], yaw],
                dtype=np.float32,
            ), name

        if "center" in obj and ("size" in obj or "dimensions" in obj):
            c = np.asarray(obj["center"], dtype=np.float32).reshape(-1)
            s = np.asarray(
                obj.get("size", obj.get("dimensions")),
                dtype=np.float32,
            ).reshape(-1)

            if c.shape[0] < 3 or s.shape[0] < 3:
                return None, name

            yaw = float(obj.get("yaw", obj.get("heading", 0.0)))

            return np.asarray(
                [c[0], c[1], c[2], s[0], s[1], s[2], yaw],
                dtype=np.float32,
            ), name

        return None, name

    @staticmethod
    def _mdtoken_box_corners_lidar_xyz(box7):
        box7 = np.asarray(box7[:7], dtype=np.float32)

        if not np.isfinite(box7).all():
            return None

        x, y, z, length, width, height, yaw = [float(v) for v in box7]

        local = np.array(
            [
                [-0.5 * length, -0.5 * width, -0.5 * height],
                [-0.5 * length, -0.5 * width,  0.5 * height],
                [-0.5 * length,  0.5 * width, -0.5 * height],
                [-0.5 * length,  0.5 * width,  0.5 * height],
                [ 0.5 * length, -0.5 * width, -0.5 * height],
                [ 0.5 * length, -0.5 * width,  0.5 * height],
                [ 0.5 * length,  0.5 * width, -0.5 * height],
                [ 0.5 * length,  0.5 * width,  0.5 * height],
            ],
            dtype=np.float32,
        )

        c = np.cos(yaw)
        s = np.sin(yaw)

        rot = np.array(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        center = np.asarray([x, y, z], dtype=np.float32)
        return local @ rot.T + center[None, :]

    @staticmethod
    def _mdtoken_order_bev_polygon(pts):
        pts = np.asarray(pts, dtype=np.float32)
        center = pts.mean(axis=0)

        angles = np.arctan2(
            pts[:, 1] - center[1],
            pts[:, 0] - center[0],
        )

        order = np.argsort(angles)

        return [
            (float(pts[i, 0]), float(pts[i, 1]))
            for i in order
        ]

    @staticmethod
    def _mdtoken_extract_dynamic_objects(sample_data, hdmap_bev_settings):
        if sample_data is None or not isinstance(sample_data, dict):
            return []

        box_key = hdmap_bev_settings.get("dynamic_boxes_key", None)
        name_key = hdmap_bev_settings.get("dynamic_names_key", None)

        box_candidates = []
        name_candidates = []

        if box_key is not None and box_key in sample_data:
            box_candidates.append(sample_data[box_key])

        for k in ["gt_boxes", "boxes", "bbox", "bboxes", "cuboids"]:
            if k in sample_data:
                box_candidates.append(sample_data[k])

        if name_key is not None and name_key in sample_data:
            name_candidates.append(sample_data[name_key])

        for k in ["gt_names", "names", "labels", "categories", "classes"]:
            if k in sample_data:
                name_candidates.append(sample_data[k])

        if len(box_candidates) > 0:
            boxes = box_candidates[0]
            names = name_candidates[0] if len(name_candidates) > 0 else []

            if isinstance(names, np.ndarray):
                names = names.tolist()

            objects = []
            for i, box in enumerate(boxes):
                name = names[i] if isinstance(names, (list, tuple)) and i < len(names) else "car"

                if isinstance(box, dict):
                    box7, dict_name = MotionDataset._mdtoken_box7_from_dict(box)
                    if dict_name is not None:
                        name = dict_name
                else:
                    box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
                    if box_arr.shape[0] < 7:
                        continue
                    box7 = box_arr[:7]

                if box7 is None:
                    continue

                objects.append((box7, name))

            return objects

        for k in ["annotations", "annos", "objects", "tracks", "track_labels"]:
            if k not in sample_data:
                continue

            records = sample_data[k]
            objects = []

            if isinstance(records, dict):
                records = list(records.values())

            for obj in records:
                box7, name = MotionDataset._mdtoken_box7_from_dict(obj)
                if box7 is not None:
                    objects.append((box7, name))

            if len(objects) > 0:
                return objects

        return []

    @staticmethod
    def _get_dynamic_object_bev_masks_from_sample_data(
        sample_data,
        hdmap_bev_settings,
        bev_h,
        bev_w,
        bev_from_ego,
        num_obj_classes=10,
    ):
        obj_masks = np.zeros(
            (bev_h, bev_w, num_obj_classes),
            dtype=np.uint8,
        )

        objects = MotionDataset._mdtoken_extract_dynamic_objects(
            sample_data,
            hdmap_bev_settings,
        )

        if len(objects) == 0:
            return obj_masks

        for box7, name in objects:
            corners = MotionDataset._mdtoken_box_corners_lidar_xyz(box7)
            if corners is None:
                continue

            cls_id = int(MotionDataset._mdtoken_dynamic_class_id(name))

            if cls_id < 0 or cls_id >= num_obj_classes:
                continue

            bottom_idx = np.argsort(corners[:, 2])[:4]
            bottom = corners[bottom_idx].astype(np.float32)

            pts_h = np.concatenate(
                [
                    bottom,
                    np.ones((bottom.shape[0], 1), dtype=np.float32),
                ],
                axis=1,
            )

            uv = (bev_from_ego @ pts_h.T).T[:, :2].astype(np.float32)

            if uv[:, 0].max() < 0:
                continue
            if uv[:, 0].min() >= bev_w:
                continue
            if uv[:, 1].max() < 0:
                continue
            if uv[:, 1].min() >= bev_h:
                continue

            poly = MotionDataset._mdtoken_order_bev_polygon(uv)

            channel_img = Image.fromarray(obj_masks[:, :, cls_id], mode="L")
            draw = ImageDraw.Draw(channel_img)
            draw.polygon(poly, fill=255)
            obj_masks[:, :, cls_id] = np.asarray(channel_img, dtype=np.uint8)

        return obj_masks


    @staticmethod
    def get_hdmap_bev_image(
        map: dict, poses: dict, sample_data: dict,
        hdmap_bev_settings: dict
    ):
        # options
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the world (map) space to the BEV space
        world_from_ego = MotionDataset.get_transform(
            poses, "timestamp_ns", sample_data["timestamp"])
        ego_from_world = np.linalg.inv(world_from_ego)
        bev_from_ego = np.array(bev_from_ego_transform)
        bev_from_world = bev_from_ego @ ego_from_world

        # draw map elements to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        if "drivable_areas" in color_table and "drivable_areas" in map:
            pen_color = tuple(color_table["drivable_areas"])
            for i in map["drivable_areas"].values():
                polygon_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["area_boundary"]
                ]).transpose()
                p = bev_from_world @ polygon_nodes
                draw.polygon(
                    [(p[0, j], p[1, j]) for j in range(p.shape[1])],
                    fill=pen_color, width=pen_width)

        if "pedestrian_crossings" in color_table and \
                "pedestrian_crossings" in map:
            pen_color = tuple(color_table["pedestrian_crossings"])
            for i in map["pedestrian_crossings"].values():
                e1, e2 = i["edge1"], i["edge2"]
                polygon_nodes = np.array([
                    [e1[0][j] for j in MotionDataset.point_keys] + [1],
                    [e1[1][j] for j in MotionDataset.point_keys] + [1],
                    [e2[1][j] for j in MotionDataset.point_keys] + [1],
                    [e2[0][j] for j in MotionDataset.point_keys] + [1]
                ]).transpose()
                p = bev_from_world @ polygon_nodes
                draw.polygon(
                    [(p[0, j], p[1, j]) for j in range(p.shape[1])],
                    fill=pen_color, width=pen_width)

        if "lane_segments" in color_table and "lane_segments" in map:
            pen_color = tuple(color_table["lane_segments"])
            for i in map["lane_segments"].values():
                if i["is_intersection"]:
                    continue

                line_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["left_lane_boundary"]
                ]).transpose()
                p = bev_from_world @ line_nodes
                for j in range(1, p.shape[1]):
                    draw.line(
                        (p[0, j - 1], p[1, j - 1], p[0, j], p[1, j]),
                        fill=pen_color, width=pen_width)

                line_nodes = np.array([
                    [j[k] for k in MotionDataset.point_keys] + [1]
                    for j in i["right_lane_boundary"]
                ]).transpose()
                p = bev_from_world @ line_nodes
                for j in range(1, p.shape[1]):
                    draw.line(
                        (p[0, j - 1], p[1, j - 1], p[0, j], p[1, j]),
                        fill=pen_color, width=pen_width)

        if bool(hdmap_bev_settings.get("append_dynamic_object_mask", False)):
            num_obj_classes = int(hdmap_bev_settings.get("num_object_classes", 10))

            obj_masks = MotionDataset._get_dynamic_object_bev_masks_from_sample_data(
                sample_data,
                hdmap_bev_settings,
                int(bev_size[1]),
                int(bev_size[0]),
                bev_from_ego,
                num_obj_classes=num_obj_classes,
            )

            image_np = np.asarray(image, dtype=np.uint8)
            image_np = np.concatenate(
                [image_np, obj_masks],
                axis=-1,
            )

            return image_np

        return image

    @staticmethod
    def get_image_description(
        image_descriptions: dict, time_list_dict: dict, split: str,
        scene_id: str, sample_data: dict
    ):
        scene_camera = "sensor/{}/{}|{}".format(
            split, scene_id, sample_data["sensor"])
        time_list = time_list_dict[scene_camera]
        nearest_time = dwm.datasets.common.find_nearest(
            time_list, sample_data["timestamp"], return_item=True)
        return image_descriptions["{}|{}".format(scene_camera, nearest_time)]
    @staticmethod
    def _layout_box_visible_in_av2_camera(
        corners_np,
        ref_timestamp,
        extrinsics,
        intrinsics,
        poses,
        sample_data,
    ):
        if sample_data is None:
            return False

        if not str(sample_data.get("sensor", "")).startswith("cameras"):
            return False

        corners_np = np.asarray(corners_np, dtype=np.float32)

        if corners_np.shape != (8, 3):
            return False

        if not np.isfinite(corners_np).all():
            return False

        sensor_name = sample_data["sensor"][8:]

        try:
            idx = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
            if idx < 0 or idx >= len(intrinsics["sensor_name"]):
                return False
            if intrinsics["sensor_name"][idx] != sensor_name:
                return False

            image_w, image_h = [
                intrinsics[key][idx]
                for key in MotionDataset.intrinsic_size_keys
            ]

            image_w = int(image_w)
            image_h = int(image_h)

            if image_w <= 0 or image_h <= 0:
                return False

            intrinsic = np.eye(4, dtype=np.float32)
            intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
                [
                    intrinsics[key][idx]
                    for key in MotionDataset.intrinsic_focal_keys
                ],
                [
                    intrinsics[key][idx]
                    for key in MotionDataset.intrinsic_center_keys
                ],
            )

            ego_from_camera = MotionDataset.get_transform(
                extrinsics,
                "sensor_name",
                sensor_name,
            ).astype(np.float32)

            world_from_ref = MotionDataset.get_transform(
                poses,
                "timestamp_ns",
                ref_timestamp,
            ).astype(np.float32)

            world_from_camera_ego = MotionDataset.get_transform(
                poses,
                "timestamp_ns",
                sample_data["timestamp"],
            ).astype(np.float32)

            camera_from_ref = np.linalg.solve(
                world_from_camera_ego @ ego_from_camera,
                world_from_ref,
            ).astype(np.float32)

            image_from_ref = intrinsic @ camera_from_ref

        except Exception:
            return False

        corners_h = np.concatenate(
            [
                corners_np,
                np.ones((corners_np.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )

        proj = (image_from_ref @ corners_h.T).T
        depth = proj[:, 2]
        valid_depth = depth > 1e-5

        if not np.any(valid_depth):
            return False

        proj = proj[valid_depth]
        depth = depth[valid_depth]

        u = proj[:, 0] / depth
        v = proj[:, 1] / depth

        if u.max() < 0.0:
            return False
        if u.min() >= float(image_w):
            return False
        if v.max() < 0.0:
            return False
        if v.min() >= float(image_h):
            return False

        return True

    def get_layout_token_boxes(
        self,
        annotations,
        sample_data,
        poses,
        frame_items=None,
        extrinsics=None,
        intrinsics=None,
    ):
        max_boxes = self.layout_token_settings.get("max_boxes", 64)
        max_allowed_error = self.layout_token_settings.get(
            "max_annotation_time_error_ns",
            100 * 1000000,
        )

        if frame_items is None:
            frame_items = []

        camera_items = [
            i for i in frame_items
            if str(i.get("sensor", "")).startswith("cameras")
        ]

        view_count = max(1, len(camera_items))

        corners = torch.zeros(view_count, max_boxes, 8, 3, dtype=torch.float32)
        classes = torch.zeros(view_count, max_boxes, dtype=torch.long)
        masks = torch.zeros(view_count, max_boxes, dtype=torch.float32)

        corner_template = np.array(
            self.default_3dbox_corner_template,
            dtype=np.float32,
        ).T

        target_ts = int(sample_data["timestamp"])
        annotation_timestamps = annotations["timestamp_ns"]

        idx = bisect.bisect_left(annotation_timestamps, target_ts)

        if idx == 0:
            nearest_anno_ts = annotation_timestamps[0]
        elif idx >= len(annotation_timestamps):
            nearest_anno_ts = annotation_timestamps[-1]
        else:
            prev_ts = annotation_timestamps[idx - 1]
            next_ts = annotation_timestamps[idx]
            nearest_anno_ts = (
                prev_ts
                if abs(target_ts - prev_ts) <= abs(next_ts - target_ts)
                else next_ts
            )

        if abs(int(nearest_anno_ts) - target_ts) > max_allowed_error:
            return corners, classes, masks

        a0 = bisect.bisect_left(annotation_timestamps, nearest_anno_ts)
        a1 = bisect.bisect_right(annotation_timestamps, nearest_anno_ts)

        world_from_current_ego = MotionDataset.get_transform(
            poses,
            "timestamp_ns",
            target_ts,
        ).astype(np.float32)

        world_from_annotation_ego = MotionDataset.get_transform(
            poses,
            "timestamp_ns",
            nearest_anno_ts,
        ).astype(np.float32)

        current_ego_from_annotation_ego = np.linalg.solve(
            world_from_current_ego,
            world_from_annotation_ego,
        ).astype(np.float32)

        kept = 0

        for i in range(a0, a1):
            if kept >= max_boxes:
                break

            category = annotations["category"][i]

            scale = np.diag(
                [annotations[j][i] for j in self.shape_keys] + [1.0]
            ).astype(np.float32)

            annotation_ego_from_box = dwm.datasets.common.get_transform(
                [annotations[j][i] for j in self.rotation_keys],
                [annotations[j][i] for j in self.translation_keys],
            ).astype(np.float32)

            current_ego_from_box = (
                current_ego_from_annotation_ego
                @ annotation_ego_from_box
            )

            p = current_ego_from_box @ scale @ corner_template
            ego_corners = torch.tensor(p[:3].T, dtype=torch.float32)
            ego_corners_np = ego_corners.numpy()

            class_id = int(self.get_layout_token_class_id(category))

            for view_idx in range(view_count):
                corners[view_idx, kept] = ego_corners
                classes[view_idx, kept] = class_id

                if len(camera_items) == 0:
                    masks[view_idx, kept] = 1.0
                    continue

                if extrinsics is None or intrinsics is None:
                    visible = False
                else:
                    visible = MotionDataset._layout_box_visible_in_av2_camera(
                        ego_corners_np,
                        target_ts,
                        extrinsics,
                        intrinsics,
                        poses,
                        camera_items[view_idx],
                    )

                if visible:
                    masks[view_idx, kept] = 1.0

            kept += 1

        return corners, classes, masks
    def get_layout_token_class_id(self, category_name):
        if category_name in ["REGULAR_VEHICLE"]:
            return 0
        if category_name in ["TRUCK", "BOX_TRUCK", "TRUCK_CAB"]:
            return 1
        if category_name in ["LARGE_VEHICLE"]:
            return 1
        if category_name in ["BUS", "SCHOOL_BUS"]:
            return 3
        if category_name in ["VEHICULAR_TRAILER"]:
            return 4
        if category_name in ["MOTORCYCLE"]:
            return 6
        if category_name in ["BICYCLE"]:
            return 7
        if category_name in ["PEDESTRIAN"]:
            return 8
        if category_name in ["BICYCLIST", "MOTORCYCLIST"]:
            return 8

        return 0
    def get_bev_map_frame_reference(self, frame_items):
        reference_channel = None
        if self.layout_token_settings is not None:
            reference_channel = self.layout_token_settings.get(
                "bev_map_reference_channel",
                None,
            )

        if reference_channel is not None:
            for item in frame_items:
                channel = item.get("channel", item.get("sensor", None))
                if channel == reference_channel:
                    return item

        for item in frame_items:
            channel = str(item.get("channel", item.get("sensor", ""))).lower()
            if "cam" in channel or "camera" in channel:
                return item

        if len(frame_items) == 0:
            raise ValueError(
                "Empty frame_items when selecting BEV map reference."
            )

        return frame_items[0]
    def __init__(
        self, fs: dwm.fs.czip.CombinedZipFileSystem, sequence_length: int,
        fps_stride_tuples: list,
        sensor_channels: list = ["cameras/ring_front_left"],
        hide_lidar: bool = False, enable_synchronization_check: bool = True,
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False, _3dbox_image_settings=None,
        hdmap_image_settings=None, _3dbox_bev_settings=None,
        hdmap_bev_settings=None, image_description_settings=None,
        stub_key_data_dict=None,
        balanced_json_path=None,
        index_json_path=None,
        dataset_root=None,
        scene_dirs=None,
        layout_token_settings: dict = None,
        split: str = "train" 
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.hide_lidar = hide_lidar
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.index_json_path = index_json_path
        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict
        self.split = split
        self.dataset_root = dataset_root
        self.layout_token_settings = layout_token_settings
        #if index_json_path is None:
         #   raise RuntimeError("index_json_path required")

        #with open(index_json_path, "r") as f:
        #    index_entries = json.load(f)

        #print("[Dataset] Index entries:", len(index_entries))
        ###从json读取###3
        filename_dict = {}
        scene_channel_sample_data = {}
        scene_map_dict = {}
        scene_split_dict = {}
         # 重新定义正则，以适配你 JSON 里的路径格式 (train/scene_id/sensors/cameras/...)
        # 注意：正则需要匹配你 JSON 中 "files" 列表里的字符串格式
        pattern = re.compile(
            "^(?P<split>\\w+)/(?P<scene_id>[^/]+)/sensors/cameras/"
            "(?P<sensor_channel>{})/"
            "(?P<timestamp>\\d+).+$".format("|".join([s.split('/')[-1] for s in sensor_channels])))
        
        map_pattern = re.compile(
            "^(?P<split>\\w+)/(?P<scene_id>[^/]+)/map/"
            "log_map_archive_.+.json$")

        # 获取所有 JSON 文件
        json_files = glob.glob(os.path.join(index_json_path, "*.json"))
        
        

        for json_path in json_files:
            with open(json_path, 'r') as f:
                scene_info = json.load(f)

            scene_id = scene_info["scene_name"]
            files = scene_info.get("files", [])

            for rel_file_path in files:

                # ✅ 关键：只保留当前 split
                if not rel_file_path.startswith(self.split):
                    continue

                # --- 匹配传感器 ---
                match = re.match(pattern, rel_file_path)
                if match is not None:
                    split = match.group("split")
                    sensor_channel = match.group("sensor_channel")

                    full_sensor_name = f"cameras/{sensor_channel}"
                    timestamp = match.group("timestamp")

                    # ✅ 统一 split（避免混乱）
                    scene_split_dict[scene_id] = self.split

                    if scene_id not in scene_channel_sample_data:
                        scene_channel_sample_data[scene_id] = [[] for _ in sensor_channels]

                    full_disk_path = os.path.join(self.dataset_root, rel_file_path)

                    filename_dict["{}/{}/{}".format(
                        scene_id, full_sensor_name, timestamp
                    )] = full_disk_path

                    sample_data = {
                        "timestamp": int(timestamp),
                        "sensor": full_sensor_name
                    }

                    for i, s_name in enumerate(sensor_channels):
                        if s_name == full_sensor_name:
                            scene_channel_sample_data[scene_id][i].append(sample_data)

                # --- 匹配地图 ---
                map_match = re.match(map_pattern, rel_file_path)
                if map_match is not None:
                    scene_map_dict[scene_id] = {
                        "split": self.split,  # ✅ 统一
                        "filename": os.path.join(self.dataset_root, rel_file_path)
                    }

        self.filename_dict = dwm.common.SerializedReadonlyDict(filename_dict)
        self.scene_map_dict = dwm.common.SerializedReadonlyDict(scene_map_dict)
       
        
        # ===============================
        # 读取 motion intervals
        # ===============================

        use_balance = balanced_json_path is not None

        if use_balance:
            with open(balanced_json_path, "r") as f:
                raw_entries = json.load(f)

            

            self.motion_intervals = []

            for e in raw_entries:
                self.motion_intervals.append({
                    "scene": e["scene_name"],
                    "start_ts": e["start_timestamp"],
                    "end_ts": e["end_timestamp"],
                    "angle": e["angle"],
                    "dist": e["dist"]
                })

            self.motion_intervals_by_scene = {}

            for interval in self.motion_intervals:
                scene = interval["scene"]

                if scene not in self.motion_intervals_by_scene:
                    self.motion_intervals_by_scene[scene] = []

                self.motion_intervals_by_scene[scene].append(interval)

        else:
            #print("[avrgoDataset] No balanced_json provided → skip filtering")
            self.motion_intervals_by_scene = {}

        # ===============================
        # DEBUG
        # ===============================

        #print("\n========== avrgoSENSOR DATA DEBUG ==========")

        #print(f"[avrgoDataset] Total scenes: {len(scene_channel_sample_data)}")

        example_scene = list(scene_channel_sample_data.keys())[0]
        #print(f"[avrgoDataset] Example scene: {example_scene}")

        #

        #print("================avrgo=======================\n")

        
            #print("\n==========avrgo BALANCED JSON DEBUG ==========")
            #print(f"[Dataset] Motion intervals loaded: {len(raw_entries)}")
            #print("[Dataset] Interval scenes:", len(self.motion_intervals_by_scene))
            #print("==================avrgo=======================\n")

                ####### 构建 items ########


        items = []

        total_windows = 0
        matched_windows_before_downsample = 0
        matched_windows = 0

        DEFAULT_OVERLAP_RATIO = 0.39

        KEEP_RATIO = 1
        DOWNSAMPLE_SEED = 1234
        rng = random.Random(DOWNSAMPLE_SEED)

        for scene_id, csd in scene_channel_sample_data.items():

            # 👉 只有在使用 balance 时才过滤 scene
            if use_balance and scene_id not in self.motion_intervals_by_scene:
                continue

            # --- 必须排序 ---
            for i in range(len(csd)):
                csd[i].sort(key=lambda x: x["timestamp"])
            # ----------------

            scene_intervals = self.motion_intervals_by_scene.get(scene_id, [])

            for fps_stride_cfg in self.fps_stride_tuples:
                if len(fps_stride_cfg) == 2:
                    fps, stride = fps_stride_cfg
                    overlap_ratio = DEFAULT_OVERLAP_RATIO
                elif len(fps_stride_cfg) == 3:
                    fps, stride, overlap_ratio = fps_stride_cfg
                else:
                    raise ValueError(
                        "Each item in fps_stride_tuples must be "
                        "(fps, stride) or (fps, stride, overlap_ratio), "
                        f"but got: {fps_stride_cfg}"
                    )

                for segment in MotionDataset.enumerate_segments(
                    csd,
                    self.sequence_length,
                    fps,
                    stride,
                    enable_synchronization_check
                ):

                    total_windows += 1

                    ts_start = segment[0][0]["timestamp"]

                    if fps > 0:
                        window_duration = int(self.sequence_length / fps * 1e9)
                        ts_end = ts_start + window_duration
                    else:
                        ts_end = segment[-1][0]["timestamp"]
                        window_duration = ts_end - ts_start

                    matched_interval = None

                    if use_balance:
                        tolerance_ns = 50_000_000  # 50ms

                        for interval in scene_intervals:
                            overlap_start = max(ts_start - tolerance_ns, interval["start_ts"])
                            overlap_end = min(ts_end + tolerance_ns, interval["end_ts"])

                            overlap = overlap_end - overlap_start

                            if overlap <= 0:
                                continue

                            if overlap >= overlap_ratio * window_duration:
                                matched_interval = interval
                                break

                        if matched_interval is None:
                            continue

                    matched_windows_before_downsample += 1

                    if rng.random() >= KEEP_RATIO:
                        continue

                    matched_windows += 1

                    items.append({
                        "segment": segment,
                        "fps": fps,
                        "scene_id": scene_id,
                        "split": scene_split_dict[scene_id],
                        "angle": matched_interval["angle"] if use_balance else 0.0,
                        "dist": matched_interval["dist"] if use_balance else 0.0,
                    })

        # ===============================
        # 最终统计
        # ========================3=======

       # print(f"[avrgoDataset] Total windows: {total_windows}")

        

        

        self.items = dwm.common.SerializedReadonlyList(items)

        if image_description_settings is not None:
            with open(
                image_description_settings["path"], "r", encoding="utf-8"
            ) as f:
                self.image_descriptions = json.load(f)

            self.image_desc_rs = np.random.RandomState(
                image_description_settings["seed"]
                if "seed" in image_description_settings else None)

            with open(
                image_description_settings["time_list_dict_path"], "r",
                encoding="utf-8"
            ) as f:
                self.time_list_dict = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        #if index == 0:
        #    print("\n==========avrgo GETITEM DEBUG ==========")
        #    print("Scene:", item["scene_id"])
        #    print("Segment length:", len(item["segment"]))
        #    print("FPS:", item["fps"])
        #    print("Angle:", item["angle"])
        #   print("Dist:", item["dist"])
        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32),
            "angle": torch.tensor(item["angle"], dtype=torch.float32),
            "dist": torch.tensor(item["dist"], dtype=torch.float32),
            #"pts": torch.tensor([
                #[
                    #(j["timestamp"] - item["segment"][0][0]["timestamp"])
                    #/ 1000000
                   # for j in i
                  #  if (
                 #       j["sensor"].startswith("cameras") or
                #        (j["sensor"] == "lidar" and not self.hide_lidar)
               #     )
              #  ]
             #   for i in item["segment"]
            #], dtype=torch.float32)#
        }

        images, lidar_points = [], []
        for i in item["segment"]:
            images_i, lidar_points_i = [], []
            for j in i:
                path = self.filename_dict[
                    "{}/{}/{}".format(
                        item["scene_id"], j["sensor"], j["timestamp"])
                ]

                if j["sensor"].startswith("cameras"):
                    with open(path, "rb") as f:
                        image = Image.open(f)
                        image.load()

                    images_i.append(image)

                elif j["sensor"] == "lidar" and not self.hide_lidar:
                    with open(path, "rb") as f:
                        points = pyarrow.feather.read_feather(f)

                    np_points = points.to_numpy()[:, :3]
                    lidar_points_i.append(
                        torch.tensor(np_points, dtype=torch.float32))

            if len(images_i) > 0:
                images.append(images_i)

            if len(lidar_points_i) > 0:
                lidar_points.append(lidar_points_i[0])

        if len(images) > 0:
            result["images"] = images  # [sequence_length, view_count]

        if len(lidar_points) > 0 and not self.hide_lidar:
            result["lidar_points"] = lidar_points  # [sequence_length]

        extrinsics, intrinsics = None, None

        if self.enable_camera_transforms:
            if "images" in result:

                # ===== extrinsics =====
                extrinsic_path = os.path.join(
                    item["split"],
                    item["scene_id"],
                    "calibration",
                    "egovehicle_SE3_sensor.feather"
                )

                with self.fs.open(extrinsic_path) as f:
                    extrinsics = pyarrow.feather.read_table(f).to_pydict()

                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        MotionDataset.get_transform(
                            extrinsics, "sensor_name", j["sensor"][8:], "pt"
                        )
                        for j in i if j["sensor"].startswith("cameras")
                    ])
                    for i in item["segment"]
                ])

                # ===== intrinsics =====
                intrinsic_path = os.path.join(
                    item["split"],
                    item["scene_id"],
                    "calibration",
                    "intrinsics.feather"
                )

                with self.fs.open(intrinsic_path) as f:
                    intrinsics = pyarrow.feather.read_table(f).to_pydict()

                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        dwm.datasets.common.make_intrinsic_matrix(
                            MotionDataset.feather_query(
                                intrinsics, "sensor_name", j["sensor"][8:],
                                MotionDataset.intrinsic_focal_keys
                            ),
                            MotionDataset.feather_query(
                                intrinsics, "sensor_name", j["sensor"][8:],
                                MotionDataset.intrinsic_center_keys
                            ),
                            "pt"
                        )
                        for j in i if j["sensor"].startswith("cameras")
                    ])
                    for i in item["segment"]
                ])
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            MotionDataset.feather_query(
                                intrinsics, "sensor_name", j["sensor"][8:],
                                MotionDataset.intrinsic_size_keys),
                            dtype=torch.long)
                        for j in i
                        if j["sensor"].startswith("cameras")
                    ])
                    for i in item["segment"]
                ])

            if "lidar_points" in result and not self.hide_lidar:
                # the LiDAR points are already in the ego space
                result["lidar_transforms"] = torch.stack([
                    torch.stack([
                        torch.eye(4)
                        for j in i
                        if j["sensor"] == "lidar"
                    ])
                    for i in item["segment"]
                ])

        poses = None
        if self.enable_ego_transforms:
            pose_path = "{}/{}/city_SE3_egovehicle.feather"\
                .format(item["split"], item["scene_id"])
            with self.fs.open(pose_path) as f:
                poses = pyarrow.feather.read_table(f).to_pydict()

            result["ego_transforms"] = torch.stack([
                torch.stack([
                    MotionDataset.get_transform(
                        poses, "timestamp_ns", j["timestamp"], "pt")
                    for j in i
                    if (
                        j["sensor"].startswith("cameras") #or
                        #(j["sensor"] == "lidar" and not self.hide_lidar)
                    )
                ])
                for i in item["segment"]
            ])

        annotations = None
        if self._3dbox_image_settings is not None:
            if poses is None:
                pose_path = "{}/{}/city_SE3_egovehicle.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(pose_path) as f:
                    poses = pyarrow.feather.read_table(f).to_pydict()

            if extrinsics is None:
                extrinsic_path = \
                    "{}/{}/calibration/egovehicle_SE3_sensor.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(extrinsic_path) as f:
                    extrinsics = pyarrow.feather.read_table(f).to_pydict()

            if intrinsics is None:
                intrinsic_path = "{}/{}/calibration/intrinsics.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(intrinsic_path) as f:
                    intrinsics = pyarrow.feather.read_table(f).to_pydict()

            if self.sensor_channels[0] != "lidar":
                raise Exception(
                    "\"lidar\" should be the first item of the sensor channel "
                    "for 3D box image condition.")

            annotation_path = "{}/{}/annotations.feather"\
                .format(item["split"], item["scene_id"])
            with self.fs.open(annotation_path) as f:
                annotations = pyarrow.feather.read_table(f).to_pydict()

                # ========== 核心修复：标注时间戳匹配 ==========
            result["3dbox_images"] = []
            annotation_timestamps = annotations["timestamp_ns"]
            max_allowed_error = 100 * 1000000  # 最大允许100ms时间误差（对应LiDAR 10Hz间隔）

            for time_step in item["segment"]:
                # 拿到当前时间步的参考时间戳（同时间步所有相机共享一个参考时间）
                current_cam_ts = time_step[0]["timestamp"]
                
                # 匹配最近的标注时间戳
                idx = bisect.bisect_left(annotation_timestamps, current_cam_ts)
                if idx == 0:
                    nearest_anno_ts = annotation_timestamps[0]
                elif idx == len(annotation_timestamps):
                    nearest_anno_ts = annotation_timestamps[-1]
                else:
                    # 选前后更近的时间戳
                    d_prev = current_cam_ts - annotation_timestamps[idx-1]
                    d_next = annotation_timestamps[idx] - current_cam_ts
                    nearest_anno_ts = annotation_timestamps[idx-1] if d_prev < d_next else annotation_timestamps[idx]
                
                # 时间误差过大，生成空图避免报错
                if abs(nearest_anno_ts - current_cam_ts) > max_allowed_error:
                    empty_cam_images = []
                    for cam_data in time_step:
                        if cam_data["sensor"].startswith("cameras"):
                            # 读取相机对应图像尺寸
                            sensor_name = cam_data["sensor"][8:]
                            cam_idx = bisect.bisect_left(intrinsics["sensor_name"], sensor_name)
                            img_w, img_h = [intrinsics[k][cam_idx] for k in self.intrinsic_size_keys]
                            empty_cam_images.append(Image.new("RGB", (img_w, img_h)))
                    result["3dbox_images"].append(empty_cam_images)
                    continue
                
                # 匹配到有效标注，生成3D框图像
                cam_box_images = [
                    self.get_3dbox_image(
                        annotations, nearest_anno_ts, extrinsics, intrinsics,
                        poses, cam_data, self._3dbox_image_settings
                    )
                    for cam_data in time_step
                    if cam_data["sensor"].startswith("cameras")
                ]
                result["3dbox_images"].append(cam_box_images)

        map = None
        if self.hdmap_image_settings is not None:
            if poses is None:
                pose_path = "{}/{}/city_SE3_egovehicle.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(pose_path) as f:
                    poses = pyarrow.feather.read_table(f).to_pydict()

            if extrinsics is None:
                extrinsic_path = \
                    "{}/{}/calibration/egovehicle_SE3_sensor.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(extrinsic_path) as f:
                    extrinsics = pyarrow.feather.read_table(f).to_pydict()

            if intrinsics is None:
                intrinsic_path = "{}/{}/calibration/intrinsics.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(intrinsic_path) as f:
                    intrinsics = pyarrow.feather.read_table(f).to_pydict()

            abs_path = self.scene_map_dict[item["scene_id"]]["filename"]

            root = self.dataset_root.rstrip("/") + "/"

            if abs_path.startswith(root):
                map_path = abs_path[len(root):]
            else:
                raise ValueError(f"path not under fs root:\n{abs_path}\n{root}")
            
            
            # ✅ 补上这一步！！！！
            with self.fs.open(map_path, "r", encoding="utf-8") as f:
                map = json.load(f)
            result["hdmap_images"] = [
                [
                    MotionDataset.get_hdmap_image(
                        map, extrinsics, intrinsics, poses, j,
                        self.hdmap_image_settings)
                    for j in i
                    if j["sensor"].startswith("cameras")
                ]
                for i in item["segment"]
            ]

        if self._3dbox_bev_settings is not None:
            if annotations is None:
                annotation_path = "{}/{}/annotations.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(annotation_path) as f:
                    annotations = pyarrow.feather.read_table(f).to_pydict()

            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    annotations, j, self._3dbox_bev_settings)
                for i in item["segment"]
                for j in i
                if j["sensor"] == "lidar"
            ]

        if self.hdmap_bev_settings is not None:
            if poses is None:
                pose_path = "{}/{}/city_SE3_egovehicle.feather"\
                    .format(item["split"], item["scene_id"])
                with self.fs.open(pose_path) as f:
                    poses = pyarrow.feather.read_table(f).to_pydict()

            if map is None:
                abs_path = self.scene_map_dict[item["scene_id"]]["filename"]
                root = self.dataset_root.rstrip("/") + "/"

                if abs_path.startswith(root):
                    map_path = abs_path[len(root):]
                else:
                    raise ValueError(f"path not under fs root:\n{abs_path}\n{root}")

                with self.fs.open(map_path, "r", encoding="utf-8") as f:
                    map = json.load(f)

            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(
                    map,
                    poses,
                    self.get_bev_map_frame_reference(i),
                    self.hdmap_bev_settings,
                )
                for i in item["segment"]
            ]

        if self.image_description_settings is not None:
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.image_descriptions, self.time_list_dict,
                        item["split"], item["scene_id"], j)
                    for j in i
                    if j["sensor"].startswith("cameras")
                ], self.image_description_settings)
                for i in item["segment"]
            ]
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        j, self.image_description_settings, self.image_desc_rs)
                    for j in i
                ]
                for i in image_captions
            ]
        if self.layout_token_settings is not None:
            if annotations is None:
                annotation_path = "{}/{}/annotations.feather" \
                    .format(item["split"], item["scene_id"])
                with self.fs.open(annotation_path) as f:
                    annotations = pyarrow.feather.read_table(f).to_pydict()

            if poses is None:
                pose_path = "{}/{}/city_SE3_egovehicle.feather" \
                    .format(item["split"], item["scene_id"])
                with self.fs.open(pose_path) as f:
                    poses = pyarrow.feather.read_table(f).to_pydict()

            if extrinsics is None:
                extrinsic_path = \
                    "{}/{}/calibration/egovehicle_SE3_sensor.feather" \
                    .format(item["split"], item["scene_id"])
                with self.fs.open(extrinsic_path) as f:
                    extrinsics = pyarrow.feather.read_table(f).to_pydict()

            if intrinsics is None:
                intrinsic_path = "{}/{}/calibration/intrinsics.feather" \
                    .format(item["split"], item["scene_id"])
                with self.fs.open(intrinsic_path) as f:
                    intrinsics = pyarrow.feather.read_table(f).to_pydict()

            bbox_token_corners_list = []
            bbox_token_classes_list = []
            bbox_token_masks_list = []

            for time_step in item["segment"]:
                ref_sample_data = self.get_bev_map_frame_reference(time_step)

                bbox_corners, bbox_classes, bbox_masks = \
                    self.get_layout_token_boxes(
                        annotations,
                        ref_sample_data,
                        poses,
                        frame_items=time_step,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                    )

                bbox_token_corners_list.append(bbox_corners)
                bbox_token_classes_list.append(bbox_classes)
                bbox_token_masks_list.append(bbox_masks)

            result["bbox_token_corners"] = torch.stack(
                bbox_token_corners_list,
                dim=0,
            )
            result["bbox_token_classes"] = torch.stack(
                bbox_token_classes_list,
                dim=0,
            )
            result["bbox_token_masks"] = torch.stack(
                bbox_token_masks_list,
                dim=0,
            )
        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        return result
