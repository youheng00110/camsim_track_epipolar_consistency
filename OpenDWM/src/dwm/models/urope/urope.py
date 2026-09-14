# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for OpenDWM from:
# https://github.com/Applied-Intuition-Open-Source/URoPE
# The geometry path is written explicitly for OpenDWM's camera-major tokens and
# supports OpenCV camera coordinates without the original OpenGL axis flip.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert rigid 4x4 transforms."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(
            "SE(3) transforms must end with shape (4, 4), got {}.".format(
                tuple(transforms.shape)
            )
        )

    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inv,
        transforms[..., :3, 3],
    )
    output[..., 3, 3] = 1.0
    return output


def _invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 pinhole intrinsics. Skew is assumed to be zero."""
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            "Intrinsics must end with shape (3, 3), got {}.".format(
                tuple(intrinsics.shape)
            )
        )

    output = torch.zeros_like(intrinsics)
    output[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    output[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    output[..., 0, 2] = (
        -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    )
    output[..., 1, 2] = (
        -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    )
    output[..., 2, 2] = 1.0
    return output


def _convert_viewmats(
    viewmats: torch.Tensor,
    camera_convention: str,
) -> torch.Tensor:
    """Return world-to-camera matrices in OpenCV coordinates."""
    if camera_convention == "opencv":
        return viewmats

    if camera_convention != "opengl":
        raise ValueError(
            "camera_convention must be 'opencv' or 'opengl', got {!r}.".format(
                camera_convention
            )
        )

    axis_transform = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=viewmats.device,
        dtype=viewmats.dtype,
    )
    return axis_transform[None, None] @ viewmats


def _rope_coefficients(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feature_dim: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build split-order RoPE coefficients."""
    if positions.ndim != 3:
        raise ValueError(
            "positions must be [batch, heads, tokens], got {}.".format(
                tuple(positions.shape)
            )
        )
    if feature_dim % 2 != 0:
        raise ValueError(
            "Each RoPE feature block must be even, got {}.".format(feature_dim)
        )

    frequency_count = feature_dim // 2
    frequencies = freq_scale * (
        freq_base
        ** (
            -torch.arange(
                frequency_count,
                device=positions.device,
                dtype=torch.float32,
            )
            / frequency_count
        )
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    return (
        torch.cos(angles).to(dtype=output_dtype),
        torch.sin(angles).to(dtype=output_dtype),
    )


def _apply_rope(
    features: torch.Tensor,
    coefficients: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply split-order RoPE to one feature block."""
    cosine, sine = coefficients
    if cosine.shape[2] != features.shape[2]:
        if features.shape[2] % cosine.shape[2] != 0:
            raise ValueError(
                "RoPE token count {} cannot expand to {}.".format(
                    cosine.shape[2], features.shape[2]
                )
            )
        repeat_count = features.shape[2] // cosine.shape[2]
        cosine = cosine.repeat(1, 1, repeat_count, 1)
        sine = sine.repeat(1, 1, repeat_count, 1)

    if cosine.shape[-1] != features.shape[-1] // 2:
        raise ValueError(
            "Coefficient dimension {} does not match feature dimension {}.".format(
                cosine.shape[-1], features.shape[-1]
            )
        )

    first_half = features[..., : features.shape[-1] // 2]
    second_half = features[..., features.shape[-1] // 2 :]

    if inverse:
        rotated_first = cosine * first_half - sine * second_half
        rotated_second = sine * first_half + cosine * second_half
    else:
        rotated_first = cosine * first_half + sine * second_half
        rotated_second = -sine * first_half + cosine * second_half

    return torch.cat([rotated_first, rotated_second], dim=-1)


def _apply_xy_rope(
    features: torch.Tensor,
    x_coefficients: Tuple[torch.Tensor, torch.Tensor],
    y_coefficients: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Use half of each head for x and half for y."""
    if features.shape[-1] % 4 != 0:
        raise ValueError(
            "URoPE head_dim must be divisible by 4, got {}.".format(
                features.shape[-1]
            )
        )

    block_dim = features.shape[-1] // 2
    x_features = features[..., :block_dim]
    y_features = features[..., block_dim:]
    x_output = _apply_rope(x_features, x_coefficients, inverse=inverse)
    y_output = _apply_rope(y_features, y_coefficients, inverse=inverse)
    return torch.cat([x_output, y_output], dim=-1)


def _prepare_local_geometry_coefficients(
    query_viewmats: torch.Tensor,
    query_intrinsics: torch.Tensor,
    source_viewmats: torch.Tensor,
    source_intrinsics: torch.Tensor,
    patch_width: int,
    patch_height: int,
    depth_count: int,
    group_size: int,
    min_depth: float,
    max_depth: float,
    freq_base: float,
    freq_scale: float,
    head_dim: int,
    camera_convention: str,
    output_dtype: torch.dtype,
    clamp_min: float,
    clamp_max: float,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """Build URoPE coefficients only for selected source cameras."""
    query_viewmats = _convert_viewmats(
        query_viewmats[:, None].float(),
        camera_convention,
    )[:, 0]
    source_viewmats = _convert_viewmats(
        source_viewmats.float(),
        camera_convention,
    )
    query_intrinsics = query_intrinsics.float()
    source_intrinsics = source_intrinsics.float()

    batch_size, source_count = source_viewmats.shape[:2]
    patch_count = patch_width * patch_height
    source_camera_to_world = _invert_se3(source_viewmats)
    source_intrinsics_inv = _invert_intrinsics(
        source_intrinsics
    )

    grid_x, grid_y = torch.meshgrid(
        torch.arange(
            patch_width,
            device=source_viewmats.device,
            dtype=torch.float32,
        ),
        torch.arange(
            patch_height,
            device=source_viewmats.device,
            dtype=torch.float32,
        ),
        indexing="xy",
    )
    image_points = torch.stack(
        [
            grid_x + 0.5,
            grid_y + 0.5,
            torch.ones_like(grid_x),
        ],
        dim=-1,
    ).reshape(
        1,
        1,
        patch_count,
        3,
    )
    image_points = image_points.expand(
        batch_size,
        source_count,
        -1,
        -1,
    )

    ray_directions_camera = torch.einsum(
        "bsij,bspj->bspi",
        source_intrinsics_inv,
        image_points,
    )
    ray_origins_world = source_camera_to_world[..., :3, 3]
    ray_directions_world = torch.einsum(
        "bsij,bspj->bspi",
        source_camera_to_world[..., :3, :3],
        ray_directions_camera,
    )

    world_to_query_rotation = query_viewmats[..., :3, :3]
    world_to_query_translation = query_viewmats[..., :3, 3]
    ray_origins_query = torch.einsum(
        "bij,bsj->bsi",
        world_to_query_rotation,
        ray_origins_world,
    )
    ray_origins_query = (
        ray_origins_query
        + world_to_query_translation[:, None, :]
    )
    ray_directions_query = torch.einsum(
        "bij,bspj->bspi",
        world_to_query_rotation,
        ray_directions_world,
    )

    depth_indices = torch.arange(
        depth_count,
        device=source_viewmats.device,
        dtype=torch.float32,
    )
    depth_bin_size = (max_depth - min_depth) / depth_count
    depths = min_depth + depth_bin_size * depth_indices

    points_query = ray_origins_query[:, None, :, None, :]
    points_query = points_query + (
        ray_directions_query[:, None]
        * depths[None, :, None, None, None]
    )
    projected_points = torch.einsum(
        "bij,bdspj->bdspi",
        query_intrinsics,
        points_query,
    )

    projected_z = projected_points[..., 2].abs() + 1e-5
    key_x = projected_points[..., 0] / projected_z
    key_y = projected_points[..., 1] / projected_z
    key_x = key_x.clamp(
        min=clamp_min,
        max=clamp_max,
    )
    key_y = key_y.clamp(
        min=clamp_min,
        max=clamp_max,
    )

    key_x = key_x.reshape(
        batch_size,
        depth_count,
        source_count * patch_count,
    ).repeat_interleave(
        group_size,
        dim=1,
    )
    key_y = key_y.reshape(
        batch_size,
        depth_count,
        source_count * patch_count,
    ).repeat_interleave(
        group_size,
        dim=1,
    )

    active_head_count = depth_count * group_size
    query_x = (grid_x + 0.5).reshape(
        1,
        1,
        patch_count,
    ).expand(
        batch_size,
        active_head_count,
        patch_count,
    )
    query_y = (grid_y + 0.5).reshape(
        1,
        1,
        patch_count,
    ).expand(
        batch_size,
        active_head_count,
        patch_count,
    )

    rope_block_dim = head_dim // 2
    query_x_coefficients = _rope_coefficients(
        query_x,
        freq_base,
        freq_scale,
        rope_block_dim,
        output_dtype,
    )
    query_y_coefficients = _rope_coefficients(
        query_y,
        freq_base,
        freq_scale,
        rope_block_dim,
        output_dtype,
    )
    key_x_coefficients = _rope_coefficients(
        key_x,
        freq_base,
        freq_scale,
        rope_block_dim,
        output_dtype,
    )
    key_y_coefficients = _rope_coefficients(
        key_y,
        freq_base,
        freq_scale,
        rope_block_dim,
        output_dtype,
    )
    return (
        query_x_coefficients,
        query_y_coefficients,
        key_x_coefficients,
        key_y_coefficients,
    )


class URoPEDotProductAttention(torch.nn.Module):
    """Local cross-view URoPE around PyTorch SDPA."""

    def __init__(
        self,
        head_num: int,
        head_dim: int,
        min_depth: float = 2.0,
        max_depth: float = 20.0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        group_size: int = 4,
        leaveout_head: int = 0,
        camera_convention: str = "opencv",
        clamp_min: float = -64.0,
        clamp_max: float = 96.0,
    ):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                "URoPE requires head_dim divisible by 4, got {}.".format(
                    head_dim
                )
            )
        if leaveout_head < 0 or leaveout_head >= head_num:
            raise ValueError(
                "leaveout_head must be in [0, head_num), got {}.".format(
                    leaveout_head
                )
            )

        active_head_count = head_num - leaveout_head
        if active_head_count % group_size != 0:
            raise ValueError(
                "head_num - leaveout_head must be divisible by group_size. "
                "Got {} - {} and group_size {}.".format(
                    head_num,
                    leaveout_head,
                    group_size,
                )
            )
        if not 0.0 < min_depth < max_depth:
            raise ValueError(
                "Depth range must satisfy 0 < min_depth < max_depth."
            )

        self.head_num = head_num
        self.head_dim = head_dim
        self.leaveout_head = leaveout_head
        self.group_size = group_size
        self.depth_count = active_head_count // group_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.camera_convention = camera_convention
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_viewmats: torch.Tensor,
        query_intrinsics: torch.Tensor,
        source_viewmats: torch.Tensor,
        source_intrinsics: torch.Tensor,
        patch_width: int,
        patch_height: int,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if key.shape != value.shape:
            raise ValueError(
                "URoPE key and value must have the same shape."
            )
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(
                "URoPE expects Q/K/V with shape [B,H,L,D]."
            )

        batch_size, head_count, query_length, head_dim = (
            query.shape
        )
        key_batch_size, key_head_count, key_length, key_head_dim = (
            key.shape
        )
        patch_count = patch_width * patch_height
        source_count = source_viewmats.shape[1]

        if (
            key_batch_size != batch_size
            or key_head_count != head_count
            or key_head_dim != head_dim
        ):
            raise ValueError(
                "URoPE Q and K batch/head dimensions do not match."
            )
        if head_count != self.head_num or head_dim != self.head_dim:
            raise ValueError(
                "QKV heads are ({}, {}), configured as ({}, {}).".format(
                    head_count,
                    head_dim,
                    self.head_num,
                    self.head_dim,
                )
            )
        if query_length != patch_count:
            raise ValueError(
                "Local URoPE expects query length H*W = {}, got {}.".format(
                    patch_count,
                    query_length,
                )
            )
        if key_length != source_count * patch_count:
            raise ValueError(
                "Local URoPE expects key length K*H*W = {}, got {}.".format(
                    source_count * patch_count,
                    key_length,
                )
            )
        if query_viewmats.shape != (batch_size, 4, 4):
            raise ValueError(
                "query_viewmats has invalid shape {}.".format(
                    tuple(query_viewmats.shape)
                )
            )
        if query_intrinsics.shape != (batch_size, 3, 3):
            raise ValueError(
                "query_intrinsics has invalid shape {}.".format(
                    tuple(query_intrinsics.shape)
                )
            )
        if source_viewmats.shape != (
            batch_size,
            source_count,
            4,
            4,
        ):
            raise ValueError(
                "source_viewmats has invalid shape {}.".format(
                    tuple(source_viewmats.shape)
                )
            )
        if source_intrinsics.shape != (
            batch_size,
            source_count,
            3,
            3,
        ):
            raise ValueError(
                "source_intrinsics has invalid shape {}.".format(
                    tuple(source_intrinsics.shape)
                )
            )

        (
            query_x_coefficients,
            query_y_coefficients,
            key_x_coefficients,
            key_y_coefficients,
        ) = _prepare_local_geometry_coefficients(
            query_viewmats=query_viewmats,
            query_intrinsics=query_intrinsics,
            source_viewmats=source_viewmats,
            source_intrinsics=source_intrinsics,
            patch_width=patch_width,
            patch_height=patch_height,
            depth_count=self.depth_count,
            group_size=self.group_size,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            head_dim=self.head_dim,
            camera_convention=self.camera_convention,
            output_dtype=query.dtype,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
        )

        active_head_count = head_count - self.leaveout_head
        query_active = _apply_xy_rope(
            query[:, :active_head_count],
            query_x_coefficients,
            query_y_coefficients,
        )
        key_active = _apply_xy_rope(
            key[:, :active_head_count],
            key_x_coefficients,
            key_y_coefficients,
        )
        value_active = _apply_xy_rope(
            value[:, :active_head_count],
            key_x_coefficients,
            key_y_coefficients,
        )

        if self.leaveout_head > 0:
            query = torch.cat(
                [
                    query_active,
                    query[:, active_head_count:],
                ],
                dim=1,
            )
            key = torch.cat(
                [
                    key_active,
                    key[:, active_head_count:],
                ],
                dim=1,
            )
            value = torch.cat(
                [
                    value_active,
                    value[:, active_head_count:],
                ],
                dim=1,
            )
        else:
            query = query_active
            key = key_active
            value = value_active

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device=query.device
            )
            if attention_mask.shape[-2:] != (
                query_length,
                key_length,
            ):
                raise ValueError(
                    "Local URoPE attention mask ends with {}, expected "
                    "({}, {}).".format(
                        tuple(attention_mask.shape[-2:]),
                        query_length,
                        key_length,
                    )
                )

        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        output_active = _apply_xy_rope(
            output[:, :active_head_count],
            query_x_coefficients,
            query_y_coefficients,
            inverse=True,
        )
        if self.leaveout_head > 0:
            output = torch.cat(
                [
                    output_active,
                    output[:, active_head_count:],
                ],
                dim=1,
            )
        else:
            output = output_active

        return output

