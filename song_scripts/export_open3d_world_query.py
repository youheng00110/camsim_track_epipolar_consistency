#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click nuScenes exporter for a paper-style shared-world visualization.

Run
    python export_nuscenes_shared_world.py

What it does
    1. Selects one nuScenes validation clip.
    2. Accumulates raw LIDAR_TOP sweeps in world coordinates.
    3. Projects each LiDAR sweep into all six RGB cameras and samples real RGB.
    4. Removes points inside dynamic 3D boxes before accumulation.
    5. Voxel-fuses the colored point cloud.
    6. Exports the ego trajectory and N multi-view image planes at real poses.
    7. Writes a local Open3D-friendly bundle and a zip archive.

No projected_pc_settings and no external cfg are used.
"""

import json
import shutil
import sys
from pathlib import Path

import fsspec
import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
OPEN_DWM_SRC = THIS_DIR.parent / "OpenDWM" / "src"
if OPEN_DWM_SRC.exists() and str(OPEN_DWM_SRC) not in sys.path:
    sys.path.insert(0, str(OPEN_DWM_SRC))

import dwm.datasets.common
from dwm.datasets.nuscenes import MotionDataset


# =============================================================================
# Only edit this block when you want another clip / output style.
# =============================================================================

NUSCENES_DATASET_DIR = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/dataset/nus_local/interp_12Hz_trainval"
)
NUSCENES_ROOT = NUSCENES_DATASET_DIR.parent
DATASET_NAME = NUSCENES_DATASET_DIR.name
SPLIT = "val"

# The current nuscenes.py has an unbound DEFAULT_OVERLAP_RATIO bug for
# 2-element fps_stride_tuples in the non-balanced path, so keep 3 elements.
SEQUENCE_LENGTH = 30
FPS_STRIDE = (2, 6, 0.9)
SAMPLE_INDEX = 9

SENSOR_CHANNELS = [
    "LIDAR_TOP",
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

# RGB cameras used to colorize raw LiDAR.
COLOR_CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

# Cameras shown as image planes in the final 3D scene.
DISPLAY_VIEWS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
]

NUM_DISPLAY_TIMES = 5
IMAGE_MAX_WIDTH = 0

# Colored-point-cloud accumulation.
REMOVE_DYNAMIC_POINTS = True
DYNAMIC_BOX_XY_PAD = 0.50
CROP_MARGIN = 60.0
Z_BELOW = 5.0
Z_ABOVE = 14.0
VOXEL_SIZE = 0.04
MAX_OUTPUT_POINTS = 1500000

# 3D image-plane layout.
IMAGE_PLANE_DISTANCE = 4.0
IMAGE_PLANE_SCALE = 0.85
VISUAL_LIFT = 4.0

OUTPUT_DIR = THIS_DIR.parent / "song_output" / "debug" / "world_query_bundle"
MAKE_ZIP = True
RANDOM_SEED = 0


def resolve_nuscenes_root():
    table_file = NUSCENES_DATASET_DIR / "sample.json"
    if not table_file.exists():
        raise FileNotFoundError(
            f"nuScenes table not found: {table_file}"
        )
    return NUSCENES_ROOT.resolve()


def build_dataset(root):
    fs = DirFileSystem(
        path=str(root),
        fs=fsspec.filesystem("file"),
    )
    dataset = MotionDataset(
        fs=fs,
        dataset_name=DATASET_NAME,
        sequence_length=SEQUENCE_LENGTH,
        fps_stride_tuples=[list(FPS_STRIDE)],
        split=SPLIT,
        sensor_channels=SENSOR_CHANNELS,
        keyframe_only=True,
        enable_synchronization_check=False,
        enable_scene_description=False,
        enable_camera_transforms=False,
        enable_ego_transforms=False,
        enable_sample_data=False,
        _3dbox_image_settings=None,
        hdmap_image_settings=None,
        image_segmentation_settings=None,
        foreground_region_image_settings=None,
        _3dbox_bev_settings=None,
        hdmap_bev_settings=None,
        image_description_settings=None,
        stub_key_data_dict=None,
        balanced_json_path=None,
        layout_token_settings=None,
        enable_3dbox_records=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("nuScenes dataset contains zero generated clips.")
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(dataset):
        raise IndexError(
            f"SAMPLE_INDEX={SAMPLE_INDEX} but dataset has {len(dataset)} clips."
        )
    return dataset, fs


def get_segment_metadata(dataset):
    item = dataset.items[SAMPLE_INDEX]
    ref_idx = SENSOR_CHANNELS.index("CAM_FRONT")
    ego_poses = []
    sample_tokens = []

    for row_tokens in item["segment"]:
        ref_sd = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "sample_data",
            row_tokens[ref_idx],
        )
        world_from_ego = dwm.datasets.common.get_transform(
            ref_sd["rotation"],
            ref_sd["translation"],
        ).astype(np.float64)
        ego_poses.append(world_from_ego)
        sample_tokens.append(ref_sd["sample_token"])

    ego_poses = np.stack(ego_poses, axis=0)
    xyz = ego_poses[:, :3, 3]
    step_distance = np.linalg.norm(
        np.diff(xyz[:, :2], axis=0),
        axis=1,
    )
    cumulative = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.cumsum(step_distance),
        ]
    )
    return item, ego_poses, sample_tokens, cumulative


def select_display_time_ids(cumulative):
    frame_count = len(cumulative)
    count = max(1, min(NUM_DISPLAY_TIMES, frame_count))
    total_distance = float(cumulative[-1])

    if total_distance > 1e-3:
        targets = np.linspace(0.0, total_distance, count)
        ids = np.asarray(
            [
                int(np.argmin(np.abs(cumulative - target)))
                for target in targets
            ],
            dtype=np.int64,
        )
    else:
        ids = np.rint(
            np.linspace(0, frame_count - 1, count)
        ).astype(np.int64)

    ids = np.unique(ids)
    if len(ids) < count:
        fallback = np.rint(
            np.linspace(0, frame_count - 1, count)
        ).astype(np.int64)
        ids = np.unique(np.concatenate([ids, fallback]))
    return ids[:count].tolist()


def remove_dynamic_points(dataset, sample_token, xyz_world):
    if not REMOVE_DYNAMIC_POINTS or len(xyz_world) == 0:
        return np.ones(len(xyz_world), dtype=bool)

    annotations = MotionDataset.query_range(
        dataset.tables,
        dataset.indices,
        "sample_annotation",
        sample_token,
        column_name="sample_token",
    )
    keep = np.ones(len(xyz_world), dtype=bool)

    for annotation in annotations:
        instance = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "instance",
            annotation["instance_token"],
        )
        category = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "category",
            instance["category_token"],
        )
        category_name = str(category["name"])

        # Match Point-as-Skeleton's background-building rule:
        # movable_object.* and static_object.* are not treated as moving actors.
        if category_name.startswith(("movable_object.", "static_object.")):
            continue

        center = np.asarray(
            annotation["translation"],
            dtype=np.float64,
        )
        rotation = dwm.datasets.common.get_transform(
            annotation["rotation"],
            [0.0, 0.0, 0.0],
        )[:3, :3].astype(np.float64)

        # nuScenes size = [width, length, height].
        width = float(annotation["size"][0]) + 2.0 * DYNAMIC_BOX_XY_PAD
        length = float(annotation["size"][1]) + 2.0 * DYNAMIC_BOX_XY_PAD
        height = float(annotation["size"][2])

        active_ids = np.flatnonzero(keep)
        if len(active_ids) == 0:
            break

        rel_world = xyz_world[active_ids] - center[None, :]
        local = rel_world @ rotation
        inside = (
            (np.abs(local[:, 0]) <= 0.5 * length)
            & (np.abs(local[:, 1]) <= 0.5 * width)
            & (np.abs(local[:, 2]) <= 0.5 * height)
        )
        if np.any(inside):
            keep[active_ids[inside]] = False

    return keep


def colorize_lidar_with_cameras(
    dataset,
    fs,
    row_tokens,
    xyz_world,
):
    row_sample_data = [
        MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "sample_data",
            token,
        )
        for token in row_tokens
    ]
    by_channel = dict(
        zip(SENSOR_CHANNELS, row_sample_data)
    )

    best_score = np.full(
        len(xyz_world),
        np.inf,
        dtype=np.float32,
    )
    colors = np.zeros(
        (len(xyz_world), 3),
        dtype=np.uint8,
    )

    xyz1 = np.concatenate(
        [
            xyz_world.astype(np.float64),
            np.ones(
                (len(xyz_world), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    for camera_name in COLOR_CAMERAS:
        camera_sd = by_channel[camera_name]
        calibration = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "calibrated_sensor",
            camera_sd["calibrated_sensor_token"],
        )

        ego_from_camera = dwm.datasets.common.get_transform(
            calibration["rotation"],
            calibration["translation"],
        ).astype(np.float64)
        world_from_camera_ego = dwm.datasets.common.get_transform(
            camera_sd["rotation"],
            camera_sd["translation"],
        ).astype(np.float64)
        world_from_camera = (
            world_from_camera_ego @ ego_from_camera
        )
        camera_from_world = np.linalg.inv(
            world_from_camera
        )

        xyz_camera = (
            xyz1 @ camera_from_world.T
        )[:, :3]
        depth = xyz_camera[:, 2]
        positive = depth > 1e-4
        if not np.any(positive):
            continue

        intrinsic = np.asarray(
            calibration["camera_intrinsic"],
            dtype=np.float64,
        )
        projection = (
            xyz_camera @ intrinsic.T
        )
        u = projection[:, 0] / np.maximum(
            projection[:, 2],
            1e-8,
        )
        v = projection[:, 1] / np.maximum(
            projection[:, 2],
            1e-8,
        )

        with fs.open(
            camera_sd["filename"],
            "rb",
        ) as stream:
            image = Image.open(stream).convert("RGB")
            image.load()
        image_np = np.asarray(image)
        height, width = image_np.shape[:2]

        visible = (
            positive
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
        )
        visible_ids = np.flatnonzero(visible)
        if len(visible_ids) == 0:
            continue

        uu = np.rint(
            u[visible_ids]
        ).astype(np.int32)
        vv = np.rint(
            v[visible_ids]
        ).astype(np.int32)

        # If a point is visible in multiple cameras, prefer the projection
        # nearer the image center to reduce seam / edge distortion.
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        score = (
            ((u[visible_ids] - cx) / max(0.5 * width, 1.0)) ** 2
            + ((v[visible_ids] - cy) / max(0.5 * height, 1.0)) ** 2
        ).astype(np.float32)

        better = score < best_score[visible_ids]
        update_ids = visible_ids[better]
        if len(update_ids) == 0:
            continue

        update_u = uu[better]
        update_v = vv[better]
        colors[update_ids] = image_np[
            update_v,
            update_u,
            :3,
        ]
        best_score[update_ids] = score[better]

    colored = np.isfinite(best_score)
    return colors, colored


def accumulate_colored_point_cloud(
    dataset,
    fs,
    item,
    full_ego_world,
):
    trajectory_xyz = full_ego_world[:, :3, 3]
    x_min = float(
        trajectory_xyz[:, 0].min() - CROP_MARGIN
    )
    x_max = float(
        trajectory_xyz[:, 0].max() + CROP_MARGIN
    )
    y_min = float(
        trajectory_xyz[:, 1].min() - CROP_MARGIN
    )
    y_max = float(
        trajectory_xyz[:, 1].max() + CROP_MARGIN
    )
    ego_z = float(
        np.median(trajectory_xyz[:, 2])
    )
    z_min = ego_z - Z_BELOW
    z_max = ego_z + Z_ABOVE

    lidar_idx = SENSOR_CHANNELS.index(
        "LIDAR_TOP"
    )
    xyz_chunks = []
    rgb_chunks = []

    for frame_index, row_tokens in enumerate(
        item["segment"]
    ):
        lidar_sd = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "sample_data",
            row_tokens[lidar_idx],
        )
        calibration = MotionDataset.query(
            dataset.tables,
            dataset.indices,
            "calibrated_sensor",
            lidar_sd["calibrated_sensor_token"],
        )

        ego_from_lidar = dwm.datasets.common.get_transform(
            calibration["rotation"],
            calibration["translation"],
        ).astype(np.float64)
        world_from_ego = dwm.datasets.common.get_transform(
            lidar_sd["rotation"],
            lidar_sd["translation"],
        ).astype(np.float64)
        world_from_lidar = (
            world_from_ego @ ego_from_lidar
        )

        raw = np.frombuffer(
            fs.cat_file(lidar_sd["filename"]),
            dtype=np.float32,
        )
        if raw.size % 5 != 0:
            raise RuntimeError(
                f"Unexpected LIDAR_TOP packet: "
                f"{lidar_sd['filename']}"
            )
        raw = raw.reshape(-1, 5)
        xyz_lidar = raw[:, :3].astype(
            np.float64,
            copy=False,
        )
        xyz_lidar1 = np.concatenate(
            [
                xyz_lidar,
                np.ones(
                    (len(xyz_lidar), 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )
        xyz_world = (
            xyz_lidar1 @ world_from_lidar.T
        )[:, :3]

        crop = (
            (xyz_world[:, 0] >= x_min)
            & (xyz_world[:, 0] <= x_max)
            & (xyz_world[:, 1] >= y_min)
            & (xyz_world[:, 1] <= y_max)
            & (xyz_world[:, 2] >= z_min)
            & (xyz_world[:, 2] <= z_max)
        )
        xyz_world = xyz_world[crop]
        if len(xyz_world) == 0:
            print(
                f"[{frame_index + 1:02d}/{len(item['segment']):02d}] "
                "0 points after crop"
            )
            continue

        static_mask = remove_dynamic_points(
            dataset,
            lidar_sd["sample_token"],
            xyz_world,
        )
        xyz_world = xyz_world[static_mask]
        if len(xyz_world) == 0:
            print(
                f"[{frame_index + 1:02d}/{len(item['segment']):02d}] "
                "0 static points"
            )
            continue

        rgb, colored_mask = colorize_lidar_with_cameras(
            dataset,
            fs,
            row_tokens,
            xyz_world,
        )
        xyz_world = xyz_world[colored_mask]
        rgb = rgb[colored_mask]

        if len(xyz_world) > 0:
            xyz_chunks.append(
                xyz_world.astype(np.float32)
            )
            rgb_chunks.append(rgb)

        print(
            f"[{frame_index + 1:02d}/{len(item['segment']):02d}] "
            f"colored static points {len(xyz_world):,}"
        )

    if not xyz_chunks:
        raise RuntimeError(
            "No colored LiDAR points were accumulated."
        )

    xyz = np.concatenate(
        xyz_chunks,
        axis=0,
    )
    rgb = np.concatenate(
        rgb_chunks,
        axis=0,
    )
    return xyz, rgb


def voxel_fuse_colored_points(xyz, rgb):
    if VOXEL_SIZE <= 0.0:
        return xyz, rgb

    keys = np.floor(
        xyz / float(VOXEL_SIZE)
    ).astype(np.int64)
    _, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True,
    )
    count = np.bincount(
        inverse
    ).astype(np.float64)

    fused_xyz = np.stack(
        [
            np.bincount(
                inverse,
                weights=xyz[:, axis],
            )
            for axis in range(3)
        ],
        axis=1,
    )
    fused_xyz /= count[:, None]

    fused_rgb = np.stack(
        [
            np.bincount(
                inverse,
                weights=rgb[:, axis].astype(np.float64),
            )
            for axis in range(3)
        ],
        axis=1,
    )
    fused_rgb /= count[:, None]
    fused_rgb = np.clip(
        np.rint(fused_rgb),
        0,
        255,
    ).astype(np.uint8)
    fused_xyz = fused_xyz.astype(np.float32)

    if (
        MAX_OUTPUT_POINTS > 0
        and len(fused_xyz) > MAX_OUTPUT_POINTS
    ):
        rng = np.random.default_rng(
            RANDOM_SEED
        )
        ids = rng.choice(
            len(fused_xyz),
            size=MAX_OUTPUT_POINTS,
            replace=False,
        )
        ids.sort()
        fused_xyz = fused_xyz[ids]
        fused_rgb = fused_rgb[ids]

    return fused_xyz, fused_rgb


def make_local_transform(selected_ego_world):
    origin = selected_ego_world[
        :, :3, 3
    ].mean(axis=0)
    local_from_world = np.eye(
        4,
        dtype=np.float64,
    )
    local_from_world[:3, 3] = -origin
    return local_from_world, origin


def write_binary_ply_xyzrgb(path, xyz, rgb):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["red"] = rgb[:, 0]
    data["green"] = rgb[:, 1]
    data["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as stream:
        stream.write(
            header.encode("ascii")
        )
        data.tofile(stream)


def write_line_obj(path, vertices, edges):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as stream:
        for point in vertices:
            stream.write(
                f"v {point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f}\n"
            )
        for start, end in edges:
            stream.write(
                f"l {int(start) + 1} "
                f"{int(end) + 1}\n"
            )


def write_textured_plane_obj(
    obj_path,
    vertices,
    image_rel_path,
):
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    mtl_path = obj_path.with_suffix(
        ".mtl"
    )
    material_name = "camera_image"

    with open(
        mtl_path,
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(
            f"newmtl {material_name}\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\n"
            "illum 1\n"
            f"map_Kd {image_rel_path}\n"
        )

    with open(
        obj_path,
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(
            f"mtllib {mtl_path.name}\n"
        )
        for point in vertices:
            stream.write(
                f"v {point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f}\n"
            )
        stream.write(
            "vt 0.0 1.0\n"
            "vt 1.0 1.0\n"
            "vt 1.0 0.0\n"
            "vt 0.0 0.0\n"
            f"usemtl {material_name}\n"
            "f 1/1 2/2 3/3\n"
            "f 1/1 3/3 4/4\n"
        )


def export_camera_images_and_planes(
    dataset,
    fs,
    item,
    display_ids,
    local_from_world,
    output_dir,
):
    image_dir = output_dir / "images"
    plane_dir = output_dir / "planes"
    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    plane_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera_records = []
    frustum_vertices = []
    frustum_edges = []
    stem_vertices = []
    stem_edges = []

    for ordinal, frame_index in enumerate(
        display_ids
    ):
        row_tokens = item["segment"][
            frame_index
        ]
        row_sample_data = [
            MotionDataset.query(
                dataset.tables,
                dataset.indices,
                "sample_data",
                token,
            )
            for token in row_tokens
        ]
        by_channel = dict(
            zip(
                SENSOR_CHANNELS,
                row_sample_data,
            )
        )

        for channel in DISPLAY_VIEWS:
            camera_sd = by_channel[channel]
            calibration = MotionDataset.query(
                dataset.tables,
                dataset.indices,
                "calibrated_sensor",
                camera_sd["calibrated_sensor_token"],
            )
            ego_from_camera = dwm.datasets.common.get_transform(
                calibration["rotation"],
                calibration["translation"],
            ).astype(np.float64)
            world_from_ego = dwm.datasets.common.get_transform(
                camera_sd["rotation"],
                camera_sd["translation"],
            ).astype(np.float64)
            world_from_camera = (
                world_from_ego
                @ ego_from_camera
            )
            local_from_camera = (
                local_from_world
                @ world_from_camera
            )

            with fs.open(
                camera_sd["filename"],
                "rb",
            ) as stream:
                image = Image.open(stream).convert(
                    "RGB"
                )
                image.load()

            original_width = image.width
            original_height = image.height
            if (
                IMAGE_MAX_WIDTH > 0
                and image.width > IMAGE_MAX_WIDTH
            ):
                scale = (
                    IMAGE_MAX_WIDTH
                    / float(image.width)
                )
                new_height = max(
                    1,
                    int(
                        round(
                            image.height
                            * scale
                        )
                    ),
                )
                image = image.resize(
                    (
                        IMAGE_MAX_WIDTH,
                        new_height,
                    ),
                    Image.Resampling.LANCZOS,
                )

            image_name = (
                f"t{ordinal:02d}_{channel}.png"
            )
            image_path = (
                image_dir / image_name
            )
            image.save(image_path)

            intrinsic = np.asarray(
                calibration["camera_intrinsic"],
                dtype=np.float64,
            )
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])

            pixels = np.asarray(
                [
                    [0.0, 0.0],
                    [original_width, 0.0],
                    [
                        original_width,
                        original_height,
                    ],
                    [0.0, original_height],
                ],
                dtype=np.float64,
            )
            corners_camera = np.zeros(
                (4, 3),
                dtype=np.float64,
            )
            corners_camera[:, 0] = (
                (pixels[:, 0] - cx)
                / max(fx, 1e-8)
                * IMAGE_PLANE_DISTANCE
            )
            corners_camera[:, 1] = (
                (pixels[:, 1] - cy)
                / max(fy, 1e-8)
                * IMAGE_PLANE_DISTANCE
            )
            corners_camera[:, 2] = (
                IMAGE_PLANE_DISTANCE
            )
            center = corners_camera.mean(
                axis=0,
                keepdims=True,
            )
            corners_camera = (
                center
                + (
                    corners_camera - center
                )
                * IMAGE_PLANE_SCALE
            )

            corners1 = np.concatenate(
                [
                    corners_camera,
                    np.ones(
                        (4, 1),
                        dtype=np.float64,
                    ),
                ],
                axis=1,
            )
            plane_vertices = (
                corners1
                @ local_from_camera.T
            )[:, :3]
            plane_vertices[:, 2] += (
                VISUAL_LIFT
            )

            real_camera_center = (
                local_from_camera[:3, 3]
                .copy()
            )
            visual_camera_center = (
                real_camera_center.copy()
            )
            visual_camera_center[2] += (
                VISUAL_LIFT
            )

            plane_name = (
                f"t{ordinal:02d}_{channel}.obj"
            )
            write_textured_plane_obj(
                plane_dir / plane_name,
                plane_vertices,
                f"../images/{image_name}",
            )

            base_vertex = len(
                frustum_vertices
            )
            frustum_vertices.append(
                visual_camera_center
            )
            frustum_vertices.extend(
                plane_vertices.tolist()
            )
            for corner_index in range(4):
                frustum_edges.append(
                    (
                        base_vertex,
                        base_vertex
                        + 1
                        + corner_index,
                    )
                )
            for corner_index in range(4):
                frustum_edges.append(
                    (
                        base_vertex
                        + 1
                        + corner_index,
                        base_vertex
                        + 1
                        + (
                            (corner_index + 1)
                            % 4
                        ),
                    )
                )

            stem_base = len(
                stem_vertices
            )
            stem_vertices.append(
                real_camera_center
            )
            stem_vertices.append(
                visual_camera_center
            )
            stem_edges.append(
                (
                    stem_base,
                    stem_base + 1,
                )
            )

            camera_records.append(
                {
                    "ordinal": int(
                        ordinal
                    ),
                    "time_index": int(
                        frame_index
                    ),
                    "channel": channel,
                    "sample_token": camera_sd[
                        "sample_token"
                    ],
                    "image_file": (
                        f"images/{image_name}"
                    ),
                    "plane_file": (
                        f"planes/{plane_name}"
                    ),
                    "local_from_camera": (
                        local_from_camera.tolist()
                    ),
                    "plane_vertices": (
                        plane_vertices.tolist()
                    ),
                    "visual_camera_center": (
                        visual_camera_center.tolist()
                    ),
                    "camera_center_visual": (
                        visual_camera_center.tolist()
                    ),
                }
            )

    write_line_obj(
        output_dir / "camera_frustums.obj",
        np.asarray(
            frustum_vertices,
            dtype=np.float32,
        ),
        frustum_edges,
    )
    write_line_obj(
        output_dir / "camera_stems.obj",
        np.asarray(
            stem_vertices,
            dtype=np.float32,
        ),
        stem_edges,
    )
    return camera_records


def main():
    root = resolve_nuscenes_root()
    print(f"nuScenes root  {root}")
    print(f"dataset       {DATASET_NAME}")
    print(f"split         {SPLIT}")

    dataset, fs = build_dataset(root)
    item, full_ego_world, sample_tokens, cumulative = (
        get_segment_metadata(dataset)
    )
    display_ids = select_display_time_ids(
        cumulative
    )

    print(
        f"dataset clips  {len(dataset):,}"
    )
    print(
        f"selected clip  {SAMPLE_INDEX}"
    )
    print(
        f"display frames  {display_ids}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    for child_name in [
        "images",
        "planes",
    ]:
        child = OUTPUT_DIR / child_name
        if child.exists():
            shutil.rmtree(child)

    raw_xyz_world, raw_rgb = (
        accumulate_colored_point_cloud(
            dataset,
            fs,
            item,
            full_ego_world,
        )
    )
    print(
        f"raw colored points  "
        f"{len(raw_xyz_world):,}"
    )

    fused_xyz_world, fused_rgb = (
        voxel_fuse_colored_points(
            raw_xyz_world,
            raw_rgb,
        )
    )
    print(
        f"voxel fused points  "
        f"{len(fused_xyz_world):,}"
    )

    selected_ego_world = full_ego_world[
        np.asarray(
            display_ids,
            dtype=np.int64,
        )
    ]
    local_from_world, origin_world = (
        make_local_transform(
            selected_ego_world
        )
    )

    fused_xyz1 = np.concatenate(
        [
            fused_xyz_world.astype(
                np.float64
            ),
            np.ones(
                (
                    len(fused_xyz_world),
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )
    fused_xyz_local = (
        fused_xyz1 @ local_from_world.T
    )[:, :3].astype(np.float32)

    write_binary_ply_xyzrgb(
        OUTPUT_DIR
        / "scene_colored.ply",
        fused_xyz_local,
        fused_rgb,
    )
    np.savez_compressed(
        OUTPUT_DIR
        / "scene_colored.npz",
        xyz=fused_xyz_local,
        rgb=fused_rgb,
    )

    full_ego_local = np.einsum(
        "ij,tjk->tik",
        local_from_world,
        full_ego_world,
    )
    full_trajectory = full_ego_local[
        :, :3, 3
    ].astype(np.float32)
    trajectory_edges = [
        (index, index + 1)
        for index in range(
            len(full_trajectory) - 1
        )
    ]
    write_line_obj(
        OUTPUT_DIR
        / "trajectory.obj",
        full_trajectory,
        trajectory_edges,
    )

    selected_ids_array = np.asarray(
        display_ids,
        dtype=np.int64,
    )
    selected_trajectory = (
        full_trajectory[
            selected_ids_array
        ]
    )
    selected_ego_local = (
        full_ego_local[
            selected_ids_array
        ]
    )
    np.savez_compressed(
        OUTPUT_DIR
        / "bundle_arrays.npz",
        full_trajectory=full_trajectory,
        selected_trajectory=(
            selected_trajectory
        ),
        selected_time_ids=np.asarray(
            display_ids,
            dtype=np.int64,
        ),
    )

    camera_records = (
        export_camera_images_and_planes(
            dataset,
            fs,
            item,
            display_ids,
            local_from_world,
            OUTPUT_DIR,
        )
    )

    first_sample = MotionDataset.query(
        dataset.tables,
        dataset.indices,
        "sample",
        sample_tokens[0],
    )
    scene = MotionDataset.query(
        dataset.tables,
        dataset.indices,
        "scene",
        first_sample["scene_token"],
    )
    log = MotionDataset.query(
        dataset.tables,
        dataset.indices,
        "log",
        scene["log_token"],
    )

    manifest = {
        "format_version": 2,
        "point_cloud_source": (
            "raw LIDAR_TOP + RGB camera "
            "projection"
        ),
        "dataset_name": DATASET_NAME,
        "split": SPLIT,
        "sample_index": SAMPLE_INDEX,
        "scene_name": scene["name"],
        "location": log["location"],
        "sensor_channels": SENSOR_CHANNELS,
        "color_cameras": COLOR_CAMERAS,
        "display_views": DISPLAY_VIEWS,
        "display_time_ids": display_ids,
        "origin_world": (
            origin_world.tolist()
        ),
        "voxel_size": VOXEL_SIZE,
        "remove_dynamic_points": (
            REMOVE_DYNAMIC_POINTS
        ),
        "visual_lift": VISUAL_LIFT,
        "image_plane_distance": (
            IMAGE_PLANE_DISTANCE
        ),
        "image_plane_scale": (
            IMAGE_PLANE_SCALE
        ),
        "scene_file": (
            "scene_colored.ply"
        ),
        "scene_npz_file": (
            "scene_colored.npz"
        ),
        "trajectory_file": (
            "trajectory.obj"
        ),
        "frustums_file": (
            "camera_frustums.obj"
        ),
        "stems_file": (
            "camera_stems.obj"
        ),
        "arrays_file": (
            "bundle_arrays.npz"
        ),
        "selected_ego_local": (
            selected_ego_local.tolist()
        ),
        "cameras": camera_records,
    }
    with open(
        OUTPUT_DIR / "manifest.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            manifest,
            stream,
            indent=2,
        )

    archive_path = None
    if MAKE_ZIP:
        archive_path = shutil.make_archive(
            str(OUTPUT_DIR),
            "zip",
            root_dir=str(
                OUTPUT_DIR.parent
            ),
            base_dir=OUTPUT_DIR.name,
        )

    print()
    print("Done")
    print(
        f"bundle  {OUTPUT_DIR}"
    )
    print(
        f"PLY     "
        f"{OUTPUT_DIR / 'scene_colored.ply'}"
    )
    print(
        f"planes  "
        f"{OUTPUT_DIR / 'planes'}"
    )
    print(
        f"images  "
        f"{OUTPUT_DIR / 'images'}"
    )
    if archive_path is not None:
        print(
            f"zip     {archive_path}"
        )


if __name__ == "__main__":
    main()