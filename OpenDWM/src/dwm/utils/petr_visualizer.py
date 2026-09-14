from typing import Sequence
import os

import torch
import numpy as np
from PIL import Image


def make_depth_bins(
    depth_num: int,
    depth_start: float,
    depth_max: float,
    lid: bool,
    device: torch.device,
) -> torch.Tensor:
    if depth_num <= 0:
        raise ValueError("depth_num should be positive.")

    if depth_max <= depth_start:
        raise ValueError("depth_max should be larger than depth_start.")

    index = torch.arange(
        start=0,
        end=depth_num,
        step=1,
        device=device,
        dtype=torch.float32,
    )

    if lid:
        index_1 = index + 1.0
        bin_size = (depth_max - depth_start) / (
            depth_num * (1.0 + depth_num)
        )
        depth_bins = depth_start + bin_size * index * index_1
    else:
        bin_size = (depth_max - depth_start) / depth_num
        depth_bins = depth_start + bin_size * index

    return depth_bins


def make_token_intrinsics(
    camera_intrinsics_norm: torch.Tensor,
    token_height: int,
    token_width: int,
) -> torch.Tensor:
    if camera_intrinsics_norm.ndim != 4:
        raise ValueError(
            "camera_intrinsics_norm should be [T, V, 3, 3] or [B,T,V,3,3] "
            "after selecting batch/time. This function expects [V,3,3]."
        )

    raise RuntimeError(
        "make_token_intrinsics received wrong shape. "
        "Please call make_token_intrinsics_from_view_tensor instead."
    )


def make_token_intrinsics_from_view_tensor(
    camera_intrinsics_norm: torch.Tensor,
    token_height: int,
    token_width: int,
) -> torch.Tensor:
    if camera_intrinsics_norm.ndim != 3:
        raise ValueError(
            f"camera_intrinsics_norm should be [V,3,3], "
            f"got {tuple(camera_intrinsics_norm.shape)}."
        )

    if camera_intrinsics_norm.shape[-2:] != (3, 3):
        raise ValueError(
            f"camera_intrinsics_norm should be [V,3,3], "
            f"got {tuple(camera_intrinsics_norm.shape)}."
        )

    k_token = camera_intrinsics_norm.clone().float()
    k_token[..., 0, 0] = k_token[..., 0, 0] * float(token_width)
    k_token[..., 1, 1] = k_token[..., 1, 1] * float(token_height)
    k_token[..., 0, 2] = k_token[..., 0, 2] * float(token_width)
    k_token[..., 1, 2] = k_token[..., 1, 2] * float(token_height)

    return k_token


def unproject_pixels_to_refego(
    pixels: torch.Tensor,
    depth_bins: torch.Tensor,
    camera_intrinsics_token: torch.Tensor,
    camera2referego: torch.Tensor,
) -> torch.Tensor:
    if pixels.ndim != 2 or pixels.shape[-1] != 3:
        raise ValueError(f"pixels should be [P,3], got {tuple(pixels.shape)}.")

    if depth_bins.ndim != 1:
        raise ValueError(
            f"depth_bins should be [D], got {tuple(depth_bins.shape)}."
        )

    if camera_intrinsics_token.ndim != 3:
        raise ValueError(
            f"camera_intrinsics_token should be [V,3,3], "
            f"got {tuple(camera_intrinsics_token.shape)}."
        )

    if camera2referego.ndim != 3:
        raise ValueError(
            f"camera2referego should be [V,4,4], "
            f"got {tuple(camera2referego.shape)}."
        )

    image_points = pixels[:, None, :] * depth_bins[None, :, None]
    k_inv = torch.linalg.inv(camera_intrinsics_token.float())

    points_cam = torch.einsum("vij,pdj->vpdi", k_inv, image_points.float())
    points_ref = torch.einsum(
        "vij,vpdj->vpdi",
        camera2referego[:, :3, :3].float(),
        points_cam,
    )
    points_ref = points_ref + camera2referego[:, None, None, :3, 3].float()

    return points_ref


def write_ply_segments(
    output_path: str,
    segments: list,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vertex_count = len(segments) * 2
    edge_count = len(segments)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {edge_count}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p0, p1, color in segments:
            r, g, b = color
            f.write(
                "{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                    float(p0[0]), float(p0[1]), float(p0[2]), r, g, b
                )
            )
            f.write(
                "{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                    float(p1[0]), float(p1[1]), float(p1[2]), r, g, b
                )
            )

        for edge_id, (_, _, color) in enumerate(segments):
            r, g, b = color
            v0 = edge_id * 2
            v1 = edge_id * 2 + 1
            f.write(f"{v0} {v1} {r} {g} {b}\n")


def save_valid_ratio_heatmap(
    valid_ratio: torch.Tensor,
    output_path: str,
    scale: int = 16,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ratio = valid_ratio.detach().float().cpu().clamp(0.0, 1.0).numpy()
    h, w = ratio.shape

    red = np.zeros_like(ratio)
    green = np.zeros_like(ratio)
    blue = np.zeros_like(ratio)

    low_mask = ratio < 0.5
    high_mask = np.logical_not(low_mask)

    green[low_mask] = ratio[low_mask] * 2.0
    blue[low_mask] = 1.0 - ratio[low_mask] * 2.0

    red[high_mask] = (ratio[high_mask] - 0.5) * 2.0
    green[high_mask] = (1.0 - ratio[high_mask]) * 2.0

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    image.save(output_path)

    stat_path = output_path.replace(".png", ".txt")
    with open(stat_path, "w") as f:
        f.write(f"shape={tuple(valid_ratio.shape)}\n")
        f.write(f"min={float(valid_ratio.min())}\n")
        f.write(f"mean={float(valid_ratio.mean())}\n")
        f.write(f"max={float(valid_ratio.max())}\n")


def build_frustum_and_ray_segments(
    camera_intrinsics_token: torch.Tensor,
    camera2referego: torch.Tensor,
    token_height: int,
    token_width: int,
    frustum_depth: float,
    ray_depth: float,
    ray_grid_size: int,
) -> list:
    view_colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]

    device = camera_intrinsics_token.device
    view_count = camera_intrinsics_token.shape[0]
    segments = []

    corner_pixels = torch.tensor(
        [
            [0.5, 0.5, 1.0],
            [float(token_width) - 0.5, 0.5, 1.0],
            [float(token_width) - 0.5, float(token_height) - 0.5, 1.0],
            [0.5, float(token_height) - 0.5, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    frustum_depth_bins = torch.tensor(
        [frustum_depth],
        device=device,
        dtype=torch.float32,
    )

    frustum_points = unproject_pixels_to_refego(
        corner_pixels,
        frustum_depth_bins,
        camera_intrinsics_token,
        camera2referego,
    )[:, :, 0, :]

    xs = torch.linspace(
        0.5,
        float(token_width) - 0.5,
        steps=ray_grid_size,
        device=device,
        dtype=torch.float32,
    )
    ys = torch.linspace(
        0.5,
        float(token_height) - 0.5,
        steps=ray_grid_size,
        device=device,
        dtype=torch.float32,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ray_pixels = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.ones_like(grid_x).reshape(-1)],
        dim=-1,
    )

    ray_depth_bins = torch.tensor(
        [ray_depth],
        device=device,
        dtype=torch.float32,
    )

    ray_points = unproject_pixels_to_refego(
        ray_pixels,
        ray_depth_bins,
        camera_intrinsics_token,
        camera2referego,
    )[:, :, 0, :]

    axis_len = min(float(ray_depth) * 0.08, 3.0)
    axis_cam = torch.tensor(
        [
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        device=device,
        dtype=torch.float32,
    )

    for view_id in range(view_count):
        color = view_colors[view_id % len(view_colors)]
        center = camera2referego[view_id, :3, 3].float()

        corners = frustum_points[view_id]
        for corner_id in range(4):
            segments.append((center, corners[corner_id], color))

        segments.append((corners[0], corners[1], color))
        segments.append((corners[1], corners[2], color))
        segments.append((corners[2], corners[3], color))
        segments.append((corners[3], corners[0], color))

        rays = ray_points[view_id]
        for point_id in range(rays.shape[0]):
            segments.append((center, rays[point_id], color))

        axis_ref = torch.einsum(
            "ij,kj->ki",
            camera2referego[view_id, :3, :3].float(),
            axis_cam,
        )
        axis_ref = axis_ref + center[None, :]

        segments.append((center, axis_ref[0], (255, 0, 0)))
        segments.append((center, axis_ref[1], (0, 255, 0)))
        segments.append((center, axis_ref[2], (0, 0, 255)))

    return segments


def visualize_petr_geometry(
    model,
    batch: dict,
    camera_intrinsics_norm: torch.Tensor,
    camera2referego: torch.Tensor,
    latents_shape: Sequence[int],
    common_config: dict,
    tag: str = "petr_geometry",
) -> None:
    output_root = common_config.get(
        "petr_visualize_output_root",
        "./debug_petr_vis",
    )

    batch_index = int(common_config.get("petr_visualize_batch_index", 0))
    time_index = int(common_config.get("petr_visualize_time_index", 0))
    max_views = int(common_config.get("petr_visualize_max_views", 8))
    ray_depth = float(common_config.get("petr_visualize_ray_depth", 30.0))
    frustum_depth = float(common_config.get("petr_visualize_frustum_depth", 20.0))
    ray_grid_size = int(common_config.get("petr_visualize_ray_grid_size", 3))

    patch_size = int(getattr(model.config, "patch_size", 2))
    latent_height = int(latents_shape[-2])
    latent_width = int(latents_shape[-1])
    token_height = latent_height // patch_size
    token_width = latent_width // patch_size

    if token_height <= 0 or token_width <= 0:
        raise ValueError(
            f"Invalid token grid: token_height={token_height}, "
            f"token_width={token_width}, latents_shape={latents_shape}."
        )

    petr_encoder = getattr(model, "petr_encoder", None)

    if petr_encoder is not None:
        depth_num = int(getattr(petr_encoder, "depth_num", 64))
        depth_start = float(getattr(petr_encoder, "depth_start", 1.0))
        depth_max = float(getattr(petr_encoder, "depth_max", 80.0))
        lid = bool(getattr(petr_encoder, "LID", False))
        position_min = petr_encoder.position_min.detach().cpu().float()
        position_max = petr_encoder.position_max.detach().cpu().float()
    else:
        depth_num = int(common_config.get("petr_visualize_depth_num", 64))
        depth_start = float(common_config.get("petr_visualize_depth_start", 1.0))
        depth_max = float(common_config.get("petr_visualize_depth_max", 80.0))
        lid = bool(common_config.get("petr_visualize_lid", False))
        position_range = common_config.get(
            "petr_visualize_position_range",
            [-80.0, -80.0, -5.0, 80.0, 80.0, 5.0],
        )
        position_min = torch.tensor(position_range[:3], dtype=torch.float32)
        position_max = torch.tensor(position_range[3:], dtype=torch.float32)

    camera_intrinsics_norm = camera_intrinsics_norm.detach().cpu().float()
    camera2referego = camera2referego.detach().cpu().float()

    batch_index = min(batch_index, camera_intrinsics_norm.shape[0] - 1)
    time_index = min(time_index, camera_intrinsics_norm.shape[1] - 1)
    view_count = min(max_views, camera_intrinsics_norm.shape[2])

    k_norm_view = camera_intrinsics_norm[
        batch_index,
        time_index,
        :view_count,
    ]
    cam2ref_view = camera2referego[
        batch_index,
        time_index,
        :view_count,
    ]

    k_token_view = make_token_intrinsics_from_view_tensor(
        k_norm_view,
        token_height,
        token_width,
    )

    depth_bins = make_depth_bins(
        depth_num=depth_num,
        depth_start=depth_start,
        depth_max=depth_max,
        lid=lid,
        device=torch.device("cpu"),
    )

    segments = build_frustum_and_ray_segments(
        camera_intrinsics_token=k_token_view,
        camera2referego=cam2ref_view,
        token_height=token_height,
        token_width=token_width,
        frustum_depth=frustum_depth,
        ray_depth=ray_depth,
        ray_grid_size=ray_grid_size,
    )

    save_dir = os.path.join(
        output_root,
        tag,
        f"b{batch_index}_t{time_index}",
    )
    os.makedirs(save_dir, exist_ok=True)

    ply_path = os.path.join(save_dir, "frustum_rays.ply")
    write_ply_segments(ply_path, segments)

    ys, xs = torch.meshgrid(
        torch.arange(token_height, dtype=torch.float32) + 0.5,
        torch.arange(token_width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    all_pixels = torch.stack(
        [
            xs.reshape(-1),
            ys.reshape(-1),
            torch.ones_like(xs).reshape(-1),
        ],
        dim=-1,
    )

    points_ref = unproject_pixels_to_refego(
        pixels=all_pixels,
        depth_bins=depth_bins,
        camera_intrinsics_token=k_token_view,
        camera2referego=cam2ref_view,
    )

    position_span = torch.clamp(position_max - position_min, min=1e-5)
    norm_coords = (
        points_ref - position_min.view(1, 1, 1, 3)
    ) / position_span.view(1, 1, 1, 3)

    valid_mask = (norm_coords > 0.0).all(dim=-1) & (norm_coords < 1.0).all(dim=-1)
    valid_ratio = valid_mask.float().mean(dim=-1)
    valid_ratio = valid_ratio.reshape(view_count, token_height, token_width)

    for view_id in range(view_count):
        heatmap_path = os.path.join(
            save_dir,
            f"valid_ratio_view{view_id:02d}.png",
        )
        save_valid_ratio_heatmap(valid_ratio[view_id], heatmap_path)

    meta_path = os.path.join(save_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"tag={tag}\n")
        f.write(f"batch_index={batch_index}\n")
        f.write(f"time_index={time_index}\n")
        f.write(f"view_count={view_count}\n")
        f.write(f"latent_shape={tuple(latents_shape)}\n")
        f.write(f"token_height={token_height}\n")
        f.write(f"token_width={token_width}\n")
        f.write(f"depth_num={depth_num}\n")
        f.write(f"depth_start={depth_start}\n")
        f.write(f"depth_max={depth_max}\n")
        f.write(f"frustum_depth={frustum_depth}\n")
        f.write(f"ray_depth={ray_depth}\n")
        f.write(f"position_min={position_min.tolist()}\n")
        f.write(f"position_max={position_max.tolist()}\n")

        if "camera_names" in batch:
            f.write(f"camera_names={batch['camera_names']}\n")

        for view_id in range(view_count):
            f.write("\n")
            f.write(f"[view {view_id}]\n")
            f.write(f"K_token=\n{k_token_view[view_id].numpy()}\n")
            f.write(
                "camera_translation={}\n".format(
                    cam2ref_view[view_id, :3, 3].numpy().tolist()
                )
            )
            f.write(
                "valid_ratio min/mean/max = {:.6f} {:.6f} {:.6f}\n".format(
                    float(valid_ratio[view_id].min()),
                    float(valid_ratio[view_id].mean()),
                    float(valid_ratio[view_id].max()),
                )
            )

    print(
        "[PETR_VIS] saved frustum/ray and valid_ratio heatmaps to {}".format(
            save_dir
        ),
        flush=True,
    )