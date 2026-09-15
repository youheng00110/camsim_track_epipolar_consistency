import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


DEFAULT_BOX_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (6, 3), (6, 5),
)


class ResizeInstanceFlow:
    def __init__(self, size):
        if len(size) != 2:
            raise ValueError("ResizeInstanceFlow size must be [H, W].")
        self.size = [int(size[0]), int(size[1])]

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = TF.resize(
                image,
                self.size,
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
            return TF.to_tensor(image)

        if not torch.is_tensor(image):
            image = torch.from_numpy(np.asarray(image))
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                "Instance-flow image must be HWC/CHW RGB, got {}.".format(
                    tuple(image.shape)
                )
            )
        image = image.float()
        if image.detach().amax() > 1.0:
            image = image / 255.0
        return torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=tuple(self.size),
            mode="nearest",
        )[0]


def encode_instance_flow_color(offset_xyz, settings):
    scale = np.asarray(
        settings.get("offset_scale", [5.0, 5.0, 2.0]),
        dtype=np.float32,
    )
    if scale.shape != (3,):
        raise ValueError(
            "instance_flow_image_settings['offset_scale'] must contain 3 values."
        )
    scale = np.maximum(np.abs(scale), 1e-6)
    normalized = np.asarray(offset_xyz, dtype=np.float32) / scale
    encoding = str(settings.get("encoding", "tanh")).lower()
    if encoding == "tanh":
        normalized = np.tanh(normalized)
    elif encoding == "linear":
        normalized = np.clip(normalized, -1.0, 1.0)
    else:
        raise ValueError(
            "instance-flow encoding must be 'tanh' or 'linear', got {!r}.".format(
                encoding
            )
        )
    rgb = np.rint((normalized + 1.0) * 127.5)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return tuple(int(value) for value in rgb.tolist())


def transform_point(transform, point_xyz):
    point_xyz = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    point_h = np.concatenate([point_xyz, np.ones((1,), dtype=np.float32)])
    return (np.asarray(transform, dtype=np.float32) @ point_h)[:3]


def project_box_hull(
    corners_frame,
    camera_from_frame,
    intrinsic,
    image_height,
    image_width,
    near_plane,
    edge_indices=DEFAULT_BOX_EDGES,
):
    corners_frame = np.asarray(corners_frame, dtype=np.float32).reshape(8, 3)
    corners_h = np.concatenate(
        [corners_frame, np.ones((8, 1), dtype=np.float32)],
        axis=1,
    )
    camera_corners = (
        np.asarray(camera_from_frame, dtype=np.float32) @ corners_h.T
    )[:3]

    clipped_points = []
    for corner_id in range(camera_corners.shape[1]):
        point = camera_corners[:, corner_id]
        if float(point[2]) >= near_plane:
            clipped_points.append(point)

    for corner_a, corner_b in edge_indices:
        point_a = camera_corners[:, corner_a]
        point_b = camera_corners[:, corner_b]
        depth_a = float(point_a[2])
        depth_b = float(point_b[2])
        crosses = (
            depth_a < near_plane <= depth_b
            or depth_b < near_plane <= depth_a
        )
        if not crosses:
            continue
        denominator = depth_b - depth_a
        if abs(denominator) < 1e-8:
            continue
        ratio = (near_plane - depth_a) / denominator
        clipped_points.append(point_a + ratio * (point_b - point_a))

    if len(clipped_points) < 3:
        return None, None

    camera_points = np.stack(clipped_points, axis=1).astype(np.float32)
    projected = np.asarray(intrinsic, dtype=np.float32) @ camera_points
    uv = (projected[:2] / np.maximum(projected[2:3], 1e-6)).T
    if not np.isfinite(uv).all():
        return None, None

    margin_x = float(image_width) * 4.0
    margin_y = float(image_height) * 4.0
    uv[:, 0] = np.clip(uv[:, 0], -margin_x, margin_x + image_width)
    uv[:, 1] = np.clip(uv[:, 1], -margin_y, margin_y + image_height)
    hull = cv2.convexHull(uv.reshape(-1, 1, 2).astype(np.float32))
    if hull is None or int(hull.shape[0]) < 3:
        return None, None

    hull = np.rint(hull.reshape(-1, 2)).astype(np.int32)
    outside = (
        int(hull[:, 0].max()) < 0
        or int(hull[:, 0].min()) >= image_width
        or int(hull[:, 1].max()) < 0
        or int(hull[:, 1].min()) >= image_height
    )
    if outside:
        return None, None

    center_frame = corners_frame.mean(axis=0)
    center_camera = transform_point(camera_from_frame, center_frame)
    center_depth = float(center_camera[2])
    if center_depth < near_plane:
        center_depth = float(np.median(camera_points[2]))
    return hull, center_depth


def scaled_camera_spec(camera_spec, settings):
    intrinsic = np.asarray(camera_spec["intrinsic"], dtype=np.float32).copy()
    source_height = max(int(camera_spec["image_height"]), 1)
    source_width = max(int(camera_spec["image_width"]), 1)
    render_size = settings.get("render_size")
    if render_size is None:
        return intrinsic, source_height, source_width
    if len(render_size) != 2:
        raise ValueError("instance_flow_image_settings['render_size'] must be [H, W].")
    image_height = int(render_size[0])
    image_width = int(render_size[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("instance-flow render size must be positive.")
    intrinsic[0] *= float(image_width) / float(source_width)
    intrinsic[1] *= float(image_height) / float(source_height)
    return intrinsic, image_height, image_width


def render_instance_flow_images(frame_records, camera_specs, settings):
    if len(frame_records) != len(camera_specs):
        raise ValueError(
            "frame_records and camera_specs must have the same time length."
        )

    near_plane = float(settings.get("near_plane", 0.1))
    last_visible_position = {}
    output_rows = []

    for time_index, records in enumerate(frame_records):
        annotation_by_track = {}
        for record in records:
            track_id = str(record["track_id"])
            if not track_id:
                continue
            if track_id in annotation_by_track:
                raise ValueError(
                    "Duplicate track ID {!r} at time {}.".format(
                        track_id,
                        time_index,
                    )
                )
            position_reference = np.asarray(
                record["position_reference"],
                dtype=np.float32,
            ).reshape(3)
            previous_position = last_visible_position.get(track_id)
            has_visible_history = previous_position is not None
            flow_color = (
                encode_instance_flow_color(
                    position_reference - previous_position,
                    settings,
                )
                if has_visible_history
                else (0, 0, 0)
            )
            annotation_by_track[track_id] = {
                "track_id": track_id,
                "position_reference": position_reference,
                "corners_frame": np.asarray(
                    record["corners_frame"],
                    dtype=np.float32,
                ).reshape(8, 3),
                "has_visible_history": has_visible_history,
                "flow_color": flow_color,
            }

        visible_tracks = set()
        rendered_views = []
        for camera_spec in camera_specs[time_index]:
            intrinsic, image_height, image_width = scaled_camera_spec(
                camera_spec,
                settings,
            )
            flow_image = np.zeros(
                (image_height, image_width, 3),
                dtype=np.uint8,
            )
            projection_records = []
            for annotation in annotation_by_track.values():
                hull, depth = project_box_hull(
                    annotation["corners_frame"],
                    camera_spec["camera_from_frame"],
                    intrinsic,
                    image_height,
                    image_width,
                    near_plane,
                )
                if hull is None:
                    continue
                projection_records.append(
                    {
                        "track_id": annotation["track_id"],
                        "hull": hull,
                        "depth": depth,
                    }
                )

            projection_records.sort(
                key=lambda record: float(record["depth"]),
                reverse=True,
            )
            for projection_record in projection_records:
                track_id = projection_record["track_id"]
                visible_tracks.add(track_id)
                annotation = annotation_by_track[track_id]
                if not annotation["has_visible_history"]:
                    continue
                cv2.fillConvexPoly(
                    flow_image,
                    projection_record["hull"],
                    color=annotation["flow_color"],
                    lineType=cv2.LINE_8,
                )
            rendered_views.append(Image.fromarray(flow_image, mode="RGB"))

        for track_id in visible_tracks:
            last_visible_position[track_id] = annotation_by_track[track_id][
                "position_reference"
            ]
        output_rows.append(rendered_views)

    return output_rows
