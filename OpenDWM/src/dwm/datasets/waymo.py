import dwm.common
import dwm.datasets.common
import dwm.datasets.waymo_common as wc
import fsspec
import io
import random
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import transforms3d
import os
import waymo_open_dataset.dataset_pb2 as waymo_pb
import zlib


class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the Waymo Perception dataset.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the data records.
        info_dict_path (str): The path to the info dict file, which contains
            the offset of the data at each timestamp in the record relative to
            the beginning of the file, is used for fast seek during random
            access.
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        sensor_channels (list): The string list of required views in
            "LIDAR_TOP", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_SIDE_LEFT", "CAM_SIDE_RIGHT", following the Waymo sensor name.
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

    sensor_name_id_dict = {
        "CAM_FRONT": 1,
        "CAM_FRONT_LEFT": 2,
        "CAM_FRONT_RIGHT": 3,
        "CAM_SIDE_LEFT": 4,
        "CAM_SIDE_RIGHT": 5,
        "LIDAR_TOP": 1,
        "LIDAR_FRONT": 2,
        "LIDAR_SIDE_LEFT": 3,
        "LIDAR_SIDE_RIGHT": 4,
        "LIDAR_REAR": 5
    }
    box_type_dict = {
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "SIGN": 3,
        "CYCLIST": 4
    }
    map_element_type_dict = {
        "road_line": "polyline",
        "lane": "polyline",
        "road_edge": "polyline",
        "crosswalk": "polygon",
        "driveway": "polygon",
        "speed_bump": "polygon"
    }

    extrinsic_correction = [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    default_3dbox_color_table = {
        "PEDESTRIAN": (255, 0, 0),
        "CYCLIST": (0, 255, 0),
        "VEHICLE": (0, 0, 255)
    }
    default_hdmap_color_table = {
        "crosswalk": (255, 0, 0),
        "road_edge": (0, 0, 255),
        "road_line": (0, 255, 0)
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
    def find_by_name(list_to_search, queried_name):
            for item in list_to_search:
                # 这里的 int() 强制转换能解决枚举对比问题
                if int(item.name) == int(queried_name):
                    return item
            return None

    @staticmethod
    def enumerate_segments(
        sample_list: list, sequence_length: int, fps, stride
    ):
        # enumerate segments for each scene
        timestamps = [i[0] for i in sample_list]
        if fps == 0:
            # frames are extracted by the index.
            stop = len(timestamps) - sequence_length + 1
            for t in range(0, stop, max(1, stride)):
                yield timestamps[t:t+sequence_length]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(timestamps, sequence_duration, stride):
                s = timestamps[-1] / 1000000 - sequence_duration
                t = timestamps[0] / 1000000
                while t <= s:
                    yield t
                    t += stride

            for t in enumerate_begin_time(
                timestamps, sequence_length / fps, stride
            ):
                candidates = [
                    dwm.datasets.common.find_nearest(
                        timestamps, (t + i / fps) * 1000000, return_item=True)
                    for i in range(sequence_length)
                ]
                yield candidates

    @staticmethod
    def get_images_and_lidar_points(
        sensor_channels: list, frame: waymo_pb.Frame
    ):
        images = []
        lidar_points = []
        for i in sensor_channels:
            if i.startswith("LIDAR"):
                laser = MotionDataset.find_by_name(
                    frame.lasers, MotionDataset.sensor_name_id_dict[i])
                range_image = waymo_pb.MatrixFloat()
                range_image.ParseFromString(
                    zlib.decompress(
                        laser.ri_return1.range_image_compressed))
                range_image = np.array(range_image.data, np.float32)\
                    .reshape(range_image.shape.dims)

                laser_calibration = wc.laser_calibration_to_dict(
                    MotionDataset.find_by_name(
                        frame.context.laser_calibrations,
                        MotionDataset.sensor_name_id_dict[i]))

                if i == "LIDAR_TOP":
                    range_image_top_pose = waymo_pb.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        zlib.decompress(
                            laser.ri_return1.range_image_pose_compressed))
                    range_image_top_pose = np\
                        .array(range_image_top_pose.data, np.float32)\
                        .reshape(range_image_top_pose.shape.dims)
                    frame_pose = np.array(frame.pose.transform, np.float32)\
                        .reshape(4, 4)
                else:
                    range_image_top_pose = None
                    frame_pose = None

                lidar_points.append(
                    torch.tensor(
                        wc.convert_range_image_to_cartesian(
                            range_image, laser_calibration,
                            range_image_top_pose, frame_pose),
                        dtype=torch.float32))

            elif i.startswith("CAM"):
                frame_image = MotionDataset.find_by_name(
                    frame.images, MotionDataset.sensor_name_id_dict[i])
                with io.BytesIO(frame_image.image) as f:
                    image = Image.open(f)
                    image.load()

                images.append(image)

        return images, lidar_points

    @staticmethod
    def get_3dbox_image(
        laser_labels, camera_calibration, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 8)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the ego space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform).reshape(4, 4)
        image_from_ego = intrinsic @ ec @ np.linalg.inv(ego_from_camera)

        # draw annotations to the image
        def list_annotation():
            for i in laser_labels:
                yield i

        def get_world_transform(i):
            scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
            ego_from_annotation = dwm.datasets.common.get_transform(
                transforms3d.euler.euler2quat(
                    0, 0, i.box.heading).tolist(),
                [i.box.center_x, i.box.center_y, i.box.center_z])
            return ego_from_annotation @ scale

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_ego, list_annotation, get_world_transform,
            lambda i: i.type, pen_width, native_color_table, corner_templates,
            edge_indices)

        return image

    @staticmethod
    def draw_polygon_to_image(
        polygon: list, draw: ImageDraw, transform: np.array,
        max_distance: float, pen_color: tuple, pen_width: int
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        m = len(polygon)
        for i in range(m):
            xy = dwm.datasets.common.project_line(
                p[:, i], p[:, (i + 1) % m], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def draw_line_to_image(
        line: list, draw: ImageDraw, transform: np.array, max_distance: float,
        pen_color: tuple, pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            xy = dwm.datasets.common.project_line(
                p[:, i - 1], p[:, i], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_image(
        map_features, camera_calibration, pose, hdmap_image_settings: dict
    ):
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0
        pen_width = hdmap_image_settings["pen_width"] \
            if "pen_width" in hdmap_image_settings else 8
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0

        # get the transform from the world space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction, np.float32)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform, np.float32).reshape(4, 4)
        world_from_ego = np.array(pose.transform, np.float32).reshape(4, 4)
        image_from_world = intrinsic @ ec @ \
            np.linalg.inv(world_from_ego @ ego_from_camera)

        # draw annotations to the image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []
            for item in items:
                coors_3d.append([item.x, item.y, item.z])
            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(laser_labels, _3dbox_bev_settings: dict):
        # options
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

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
        for i in laser_labels:
            category = i.type
            if category in native_color_table:
                pen_color = tuple(native_color_table[category])
                scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    transforms3d.euler.euler2quat(
                        0, 0, i.box.heading).tolist(),
                    [i.box.center_x, i.box.center_y, i.box.center_z])
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
    def draw_polygon_bev_to_image(
        polygon: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int, solid: bool = True
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], 0, 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        draw.polygon(
            [(p[0, i], p[1, i]) for i in range(p.shape[1])],
            fill=pen_color if solid else None,
            outline=None if solid else pen_color, width=pen_width)

    @staticmethod
    def draw_line_bev_to_image(
        line: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], 0, 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            draw.line(
                (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i]),
                fill=pen_color, width=pen_width)

    @staticmethod

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
    def get_hdmap_bev_image(map_features, frame, hdmap_bev_settings: dict):
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform,
        )
        color_table = hdmap_bev_settings.get(
            "color_table",
            MotionDataset.default_hdmap_color_table,
        )

        world_from_ego = np.array(
            frame.pose.transform,
            np.float32,
        ).reshape(4, 4)

        ego_from_world = np.linalg.inv(world_from_ego).astype(np.float32)

        map_pose_offset = np.array(
            [
                frame.map_pose_offset.x,
                frame.map_pose_offset.y,
                frame.map_pose_offset.z,
            ],
            dtype=np.float32,
        )

        raw_ego_from_map_ego = np.eye(4, dtype=np.float32)
        raw_ego_from_map_ego[:3, 3] = -map_pose_offset

        bev_from_ego = np.array(bev_from_ego_transform, np.float32)

        bev_from_world = bev_from_ego @ raw_ego_from_map_ego @ ego_from_world

        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}

        for feat in map_features:
            type_ = feat.WhichOneof("feature_data")
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []

            for item in items:
                coors_3d.append([item.x, item.y, item.z])

            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_bev_to_image(
                        i,
                        draw,
                        bev_from_world,
                        c,
                        pen_width,
                    )

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_bev_to_image(
                        i,
                        draw,
                        bev_from_world,
                        c,
                        pen_width,
                    )
        if os.environ.get("DWM_DEBUG_WAYMO_MAP_OFFSET", "0") == "1":
            print(
                "[WAYMO_MAP_OFFSET]",
                "ts=", int(frame.timestamp_micros),
                "offset=", map_pose_offset.tolist(),
                "pose_t=", world_from_ego[:3, 3].tolist(),
                flush=True,
            )
        if False and bool(hdmap_bev_settings.get("append_dynamic_object_mask", False)):
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
        image_descriptions: dict, time_list_dict: dict, scene_key: str,
        timestamp: int, camera_id: int
    ):
        nearest_time = dwm.datasets.common.find_nearest(
            time_list_dict[scene_key], timestamp, return_item=True)
        key = "{}|{}|{}".format(scene_key, nearest_time, camera_id)
        return image_descriptions[key]
    @staticmethod
    def _layout_box_visible_in_waymo_camera(corners_np, camera_calibration):
        if camera_calibration is None:
            return False

        corners_np = np.asarray(corners_np, dtype=np.float32)

        if corners_np.shape != (8, 3):
            return False

        if not np.isfinite(corners_np).all():
            return False

        image_w = int(camera_calibration.width)
        image_h = int(camera_calibration.height)

        if image_w <= 0 or image_h <= 0:
            return False

        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4],
        )

        ec = np.asarray(MotionDataset.extrinsic_correction, dtype=np.float32)
        ego_from_camera = np.asarray(
            camera_calibration.extrinsic.transform,
            dtype=np.float32,
        ).reshape(4, 4)

        try:
            image_from_ego = intrinsic @ ec @ np.linalg.inv(ego_from_camera)
        except np.linalg.LinAlgError:
            return False

        corners_h = np.concatenate(
            [
                corners_np,
                np.ones((corners_np.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )

        proj = (image_from_ego @ corners_h.T).T
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

    def get_layout_token_boxes(self, frame, camera_calibrations=None):
        max_boxes = self.layout_token_settings.get("max_boxes", 64)

        if camera_calibrations is None:
            camera_calibrations = []

        camera_calibrations = list(camera_calibrations)
        view_count = max(1, len(camera_calibrations))

        corners = torch.zeros(view_count, max_boxes, 8, 3, dtype=torch.float32)
        classes = torch.zeros(view_count, max_boxes, dtype=torch.long)
        masks = torch.zeros(view_count, max_boxes, dtype=torch.float32)

        corner_template = torch.tensor(
            self.default_3dbox_corner_template,
            dtype=torch.float32,
        )

        kept = 0

        for label in frame.laser_labels:
            if kept >= max_boxes:
                break

            if int(label.type) <= 0:
                continue

            box = label.box

            scale = torch.tensor(
                [box.length, box.width, box.height],
                dtype=torch.float32,
            )

            yaw = float(box.heading)
            cos_yaw = torch.tensor(np.cos(yaw), dtype=torch.float32)
            sin_yaw = torch.tensor(np.sin(yaw), dtype=torch.float32)

            rot = torch.tensor(
                [
                    [cos_yaw, -sin_yaw, 0.0],
                    [sin_yaw,  cos_yaw, 0.0],
                    [0.0,      0.0,     1.0],
                ],
                dtype=torch.float32,
            )

            local_corners = corner_template[:, :3] * scale.view(1, 3)
            ego_corners = local_corners @ rot.t()
            ego_corners = ego_corners + torch.tensor(
                [box.center_x, box.center_y, box.center_z],
                dtype=torch.float32,
            ).view(1, 3)

            class_id = int(self.get_layout_token_class_id(label.type))
            ego_corners_np = ego_corners.numpy()

            for view_idx in range(view_count):
                corners[view_idx, kept] = ego_corners
                classes[view_idx, kept] = class_id

                if len(camera_calibrations) == 0:
                    masks[view_idx, kept] = 1.0
                    continue

                visible = self._layout_box_visible_in_waymo_camera(
                    ego_corners_np,
                    camera_calibrations[view_idx],
                )

                if visible:
                    masks[view_idx, kept] = 1.0

            kept += 1

        return corners, classes, masks
    def get_layout_token_class_id(self, label_type):
        if int(label_type) == 1:
            return 0      # vehicle
        if int(label_type) == 2:
            return 8      # pedestrian
        if int(label_type) == 3:
            return 9      # sign / cone-like, if used
        if int(label_type) == 4:
            return 7      # cyclist

        return 0
    ###增加筛选平衡后的json########
    def __init__(
        self,
        fs,
        info_dict_path,
        sequence_length,
        fps_stride_tuples,
        sensor_channels=["CAM_FRONT"],
        enable_camera_transforms=False,
        enable_ego_transforms=False,
        _3dbox_image_settings=None,
        hdmap_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
        balanced_json_path=None,
        dataset_root=None,
        layout_token_settings: dict = None,
        split: str = "train" 
    ):

        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.dataset_root = dataset_root
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict
        self.split =split
        self.layout_token_settings = layout_token_settings
        self.items = []

        # ===============================
        # 1️⃣ 读取 info_dict
        # ===============================

        with open(info_dict_path, "r") as f:
            raw_sample_info_dict = json.load(f)

        self.sample_info_dict = {
            scene_id: sorted(sample_list, key=lambda x: x[0])
            for scene_id, sample_list in raw_sample_info_dict.items()
        }

        self.sample_info_lookup = dwm.common.SerializedReadonlyDict({
            f"{scene_id};{sample_info[0]}": sample_info
            for scene_id, sample_list in self.sample_info_dict.items()
            for sample_info in sample_list
        })

        # ===============================
        # 2️⃣ balanced_json（可选）
        # ===============================

        use_balance = balanced_json_path is not None

        if use_balance:
            with open(balanced_json_path, 'r') as f:
                raw_entries = json.load(f)

            #print(f"[waymoDataset] Motion intervals loaded: {len(raw_entries)}")

            self.motion_intervals = []

            for e in raw_entries:
                self.motion_intervals.append({
                    "scene": e["seq_id"],
                    "start_idx": e["start_idx"],
                    "end_idx": e["end_idx"],
                    "start_ts": e["start_timestamp"],
                    "end_ts": e["end_timestamp"],
                    "angle": e["angle"],
                    "dist": e["dist"]
                })

            # scene -> intervals
            self.motion_intervals_by_scene = {}

            for interval in self.motion_intervals:
                scene = interval["scene"]
                if scene not in self.motion_intervals_by_scene:
                    self.motion_intervals_by_scene[scene] = []
                self.motion_intervals_by_scene[scene].append(interval)

            #print("[waymoDataset] Interval scenes:", len(self.motion_intervals_by_scene))

        else:
            #print("[waymoDataset] No balanced_json → no filtering")
            self.motion_intervals_by_scene = {}
        # ===============================
        # 3️⃣ enumerate windows
        # ===============================

        total_windows = 0
        matched_windows_before_downsample = 0
        matched_windows = 0

        DEFAULT_OVERLAP_RATIO = 0.8

        KEEP_RATIO = 1
        DOWNSAMPLE_SEED = 1234
        rng = random.Random(DOWNSAMPLE_SEED)

        for scene_id, sample_list in self.sample_info_dict.items():

            if use_balance and scene_id not in self.motion_intervals_by_scene:
                continue

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

                if not (0.0 <= overlap_ratio <= 1.0):
                    raise ValueError(
                        f"overlap_ratio must be in [0, 1], got: {overlap_ratio}"
                    )

                # 这里保持原版语义：
                # fps > 0 时，stride 就是“相邻 clip 起点间隔的秒数”
                # 不要再乘 fps
                for segment in MotionDataset.enumerate_segments(
                    sample_list,
                    self.sequence_length,
                    fps,
                    stride
                ):
                    if len(segment) == 0:
                        continue

                    total_windows += 1

                    window_ts_start = segment[0]
                    window_ts_end = segment[-1]
                    window_duration = window_ts_end - window_ts_start

                    if window_duration <= 0:
                        continue

                    matched_interval = None

                    if use_balance:
                        for interval in scene_intervals:
                            overlap_start = max(window_ts_start, interval["start_ts"])
                            overlap_end = min(window_ts_end, interval["end_ts"])

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

                    self.items.append({
                        "scene": scene_id,
                        "segment": segment,
                        "fps": fps,
                        "split": split,
                        "angle": matched_interval["angle"] if use_balance else 0.0,
                        "dist": matched_interval["dist"] if use_balance else 0.0,
                    })
        # ===============================
        # 4️⃣ stats
        # ===============================

        #print("waymo[Dataset] Window enumeration finished")
        #print("waymoTotal windows:", total_windows)

        
        #if len(self.items) > 0:
         #   print("[waymoDataset DEBUG] Example item:")
          #  print(self.items[0])

        # 👉 最终封装（只保留这个）
        self.items = dwm.common.SerializedReadonlyList(self.items)
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
        # 取出当前样本对应的元信息
        item = self.items[index]
        scene_id = item["scene"]

        # 只保留相机通道；后面 images / hdmap / 3dbox 这些都是按相机视角组织
        camera_only_channels = [
            j for j in self.sensor_channels
            if j.startswith("CAM")
        ]
        # 当前样本里相机视角数
        view_count = len(camera_only_channels)

        # 防御性检查：确保 __init__ 中已经初始化了 sample_info_dict
        if not hasattr(self, "sample_info_dict"):
            raise AttributeError(
                "sample_info_dict not initialized. Check __init__ logic."
            )

        all_frames = self.sample_info_dict[scene_id]

        segment = [
            self.sample_info_lookup[f"{scene_id};{ts}"]
            for ts in item["segment"]
        ]

        if len(segment) == 0:
            raise ValueError(
                f"No frames found for scene={scene_id}, segment={item['segment']}"
            )

        # 防御性检查：当前窗口不能为空
        if len(segment) == 0:
            raise ValueError(
                f"No frames found for {scene_id} at "
                f"{item['start_idx']}:{item['end_idx']}"
            )

        # 初始化返回结果
        result = {
            # 当前 clip 的 fps
            "fps": torch.tensor(item["fps"]).float(),
            # pts: 每一帧相对首帧的时间偏移
            # 这里沿 view 维复制，保证形状和多视角输入对齐
            #"pts": torch.tensor(
             #   [
              #      [(i[0] - segment[0][0]) / 1000] * view_count
               #     for i in segment
                #],
                #dtype=torch.float32
            #),
            # balanced_json 里附带的运动属性
            "angle": torch.tensor(item["angle"]).float(),
            "dist": torch.tensor(item["dist"]).float(),
        }

        # 构造当前 scene 对应的 tfrecord 文件名
        scene_filename = f"segment-{scene_id}_with_camera_labels.tfrecord"
        # 优先从 item 中拿 split；否则退回 self.split；再否则默认 training
        split = item.get("split", self.split if hasattr(self, "split") else "training")

        # 这里保持你当前的路径逻辑不变：
        # 如果给了 dataset_root，则按 individual_files/split/filename 拼
        # 否则退回默认 training 路径
        if self.dataset_root:
            scene_path = os.path.join(
                "individual_files",
                split,
                scene_filename
            )
        else:
            scene_path = os.path.join(
                "individual_files",
                "training",
                scene_filename
            )

        # 文件不存在就直接报错，避免后面静默失败
        if not self.fs.exists(scene_path):
            raise FileNotFoundError(f"Waymo record not found at: {scene_path}")

        # frames: 当前 segment 内逐帧解析出来的 Frame
        frames = [waymo_pb.Frame() for _ in segment]
        # scene_frame: 专门用来承载整段 scene 的 map_features
        # 这里仍然沿用你现在的使用方式：从该 scene 第一帧读取 map 信息
        scene_frame = waymo_pb.Frame()

        # 打开 tfrecord，按 offset 进行随机读取
        with self.fs.open(scene_path, "rb") as f:
            # 如果需要画 hdmap / hdmap_bev，则先单独读取 scene 的第一帧
            # 目的是拿到 scene_frame.map_features
            if (
                self.hdmap_image_settings is not None or
                self.hdmap_bev_settings is not None
            ):
                _, first_length, first_offset = all_frames[0]
                f.seek(first_offset)
                scene_frame.ParseFromString(f.read(first_length))

            # 逐帧读取当前窗口里的 frame 数据
            for i_id, frame_info in enumerate(segment):
                _, length, offset = frame_info
                f.seek(offset)
                frames[i_id].ParseFromString(f.read(length))

        # 下面这三组分别缓存：
        # images      : 原始 PIL 图像，按 [T][V] 组织
        # intrinsics  : 相机内参，按 [T, V, 3, 3] 组织
        # extrinsics  : 相机外参，按 [T, V, 4, 4] 组织
        images = []
        intrinsics = []
        extrinsics = []

        # 遍历当前窗口中的每一帧
        for f_data in frames:
            frame_images = []
            frame_intr = []
            frame_extr = []

            # 遍历当前帧中的每个相机
            for cam_name in camera_only_channels:
                cam_id = MotionDataset.sensor_name_id_dict[cam_name]

                # 先取图像数据
                img_data = MotionDataset.find_by_name(f_data.images, cam_id)
                if img_data is not None:
                    # 正常读到图像时，解析成 PIL.Image
                    with io.BytesIO(img_data.image) as f:
                        img = Image.open(f)
                        img.load()
                        frame_images.append(img)
                else:
                    # 容错：如果某个相机图缺失，则补一张黑图占位
                    # 这样可以保证多视角维度不乱
                    frame_images.append(Image.new("RGB", (448, 256), (0, 0, 0)))

                # 再取该相机的标定信息
                calib = MotionDataset.find_by_name(
                    f_data.context.camera_calibrations,
                    cam_id
                )
                if calib is not None:
                    # 外参：直接从 protobuf 中读出 4x4
                    ext = np.array(
                        calib.extrinsic.transform,
                        dtype=np.float32
                    ).reshape(4, 4)

                    # 内参：只构造基础 3x3 pinhole 矩阵
                    ins = np.eye(3, dtype=np.float32)
                    ins[0, 0] = calib.intrinsic[0]
                    ins[1, 1] = calib.intrinsic[1]
                    ins[0, 2] = calib.intrinsic[2]
                    ins[1, 2] = calib.intrinsic[3]

                    frame_intr.append(torch.from_numpy(ins).float())
                    frame_extr.append(torch.from_numpy(ext).float())
                else:
                    # 容错：标定缺失时给单位阵
                    frame_intr.append(torch.eye(3))
                    frame_extr.append(torch.eye(4))

            # 当前帧处理完成后，写入总列表
            images.append(frame_images)
            intrinsics.append(torch.stack(frame_intr))
            extrinsics.append(torch.stack(frame_extr))

        # 将原始图像与相机参数写回 result
        result["images"] = images
        result["camera_intrinsics"] = torch.stack(intrinsics)

        # ---------------------------
        # 相机几何相关输出
        # ---------------------------
        if self.enable_camera_transforms:
            if "images" in result:
                # 为每一帧、每一个相机取出 calibration
                camera_calibrations = [
                    [
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j]
                        )
                        for j in self.sensor_channels
                        if j.startswith("CAM")
                    ]
                    for i in frames
                ]

                # extrinsic_correction 的逆，用于和原版保持一致的相机坐标定义
                ec_inv = torch.linalg.inv(
                    torch.tensor(
                        MotionDataset.extrinsic_correction,
                        dtype=torch.float32
                    )
                )

                # camera_transforms: [T, V, 4, 4]
                # 将数据集原始外参与 correction 结合，得到统一约定下的相机变换
                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            j.extrinsic.transform,
                            dtype=torch.float32
                        ).reshape(4, 4) @ ec_inv
                        for j in i
                    ])
                    for i in camera_calibrations
                ])

                # camera_intrinsics: [T, V, 3, 3]
                # 这里用仓库原有的 make_intrinsic_matrix("pt") 形式覆盖前面手动拼的 K
                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        dwm.datasets.common.make_intrinsic_matrix(
                            j.intrinsic[0:2],
                            j.intrinsic[2:4],
                            "pt"
                        )
                        for j in i
                    ])
                    for i in camera_calibrations
                ])

                # image_size: [T, V, 2]
                # 这里约定为 [width, height]
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor([j.width, j.height], dtype=torch.long)
                        for j in i
                    ])
                    for i in camera_calibrations
                ])

            # 如果 result 中存在 lidar_points，则额外构造 lidar_transforms
            # 你当前这版里大多情况下不会进来，因为前面没实际填充 lidar_points
            #if "lidar_points" in result:
            #    result["lidar_transforms"] = torch.stack([
            #        torch.stack([
            #            torch.eye(4)
            #            for j in self.sensor_channels
            #            if j.startswith("LIDAR")
            #        ])
            #        for _ in frames
            #    ])

        # ---------------------------
        # ego pose 相关输出
        # ---------------------------
        if self.enable_ego_transforms:
            # ego_transforms: [T, V, 4, 4]
            # 同一帧下对每个 sensor 复制一份 ego pose，保持维度兼容
            camera_channels = [
                s for s in self.sensor_channels
                if s.startswith("CAM") or s.startswith("cameras")
            ]

            result["ego_transforms"] = torch.stack([
                torch.stack([
                    torch.tensor(
                        i.pose.transform,
                        dtype=torch.float32
                    ).reshape(4, 4)
                    for _ in camera_channels
                ])
                for i in frames
            ])

        # ---------------------------
        # 3D box 图像
        # ---------------------------
        if self._3dbox_image_settings is not None:
            # 对每一帧的每个相机，利用激光标注 + 当前相机标定投影出 3D box
            result["3dbox_images"] = [
                [
                    MotionDataset.get_3dbox_image(
                        i.laser_labels,
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j]
                        ),
                        self._3dbox_image_settings
                    )
                    for j in self.sensor_channels
                    if j.startswith("CAM")
                ]
                for i in frames
            ]

        # ---------------------------
        # HD map 图像
        # ---------------------------
        if self.hdmap_image_settings is not None:
            # 注意这里用的是 scene_frame.map_features
            # 所以 scene_frame 必须在前面先正确读取，否则这里会全黑
            result["hdmap_images"] = [
                [
                    MotionDataset.get_hdmap_image(
                        scene_frame.map_features,
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j]
                        ),
                        i.pose,
                        self.hdmap_image_settings
                    )
                    for j in self.sensor_channels
                    if j.startswith("CAM")
                ]
                for i in frames
            ]

        # ---------------------------
        # 3D box 的 BEV 图
        # ---------------------------
        if self._3dbox_bev_settings is not None:
            # 这里仍然沿用你当前写法：
            # 对每个 frame、每个 lidar 通道生成一张 bev 图
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    i.laser_labels,
                    self._3dbox_bev_settings
                )
                for i in frames
                for j in self.sensor_channels
                if j.startswith("LIDAR")
            ]

        # ---------------------------
        # HD map 的 BEV 图
        # ---------------------------
        if self.hdmap_bev_settings is not None:
            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(
                    scene_frame.map_features,
                    i,
                    self.hdmap_bev_settings,
                )
                for i in frames
            ]

        # ---------------------------
        # 图像描述文本
        # ---------------------------
        if self.image_description_settings is not None:
            # 先按 timestamp 对齐到最近的标注时间点，再做 cross-view 聚合
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.image_descriptions,
                        self.time_list_dict,
                        item["scene"],
                        i[0],
                        MotionDataset.sensor_name_id_dict[j]
                    )
                    for j in self.sensor_channels
                    if "LIDAR" not in j
                ], self.image_description_settings)
                for i in segment
            ]

            # 再将结构化 caption 转成最终字符串
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        j,
                        self.image_description_settings,
                        self.image_desc_rs
                    )
                    for j in i
                ]
                for i in image_captions
            ]

        # 给缺失字段补 stub，和其他数据集对齐
        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        if self.layout_token_settings is not None:
            bbox_token_corners_list = []
            bbox_token_classes_list = []
            bbox_token_masks_list = []

            for frame in frames:
                camera_calibrations = [
                    MotionDataset.find_by_name(
                        frame.context.camera_calibrations,
                        MotionDataset.sensor_name_id_dict[cam_name],
                    )
                    for cam_name in camera_only_channels
                ]

                bbox_corners, bbox_classes, bbox_masks = \
                    self.get_layout_token_boxes(
                        frame,
                        camera_calibrations=camera_calibrations,
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

        if "angle" in item:
            result["angle"] = torch.tensor(item["angle"]).float()

        if "dist" in item:
            result["dist"] = torch.tensor(item["dist"]).float()

        # 返回最终样本
        return result