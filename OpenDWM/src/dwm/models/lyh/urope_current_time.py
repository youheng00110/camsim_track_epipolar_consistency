"""
Selective URoPE for a 9-slot TV context.

Only one contiguous K/V token interval receives camera-geometry URoPE.
All other K/V tokens remain vanilla. Q remains in its original basis.

For selected K/V, apply the relative transform R_q^{-1} R_k. This lets
selected URoPE keys and unselected vanilla temporal keys coexist in a single
scaled-dot-product attention with one global softmax.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from dwm.models.urope.urope import (
    URoPEDotProductAttention,
    _apply_xy_rope,
    _prepare_local_geometry_coefficients,
)


class CurrentTimeURoPEDotProductAttention(URoPEDotProductAttention):
    """URoPE only on a selected current-time K/V interval."""

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
        geometry_key_start: int,
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

        batch_size, head_count, query_length, head_dim = query.shape
        (
            key_batch_size,
            key_head_count,
            key_length,
            key_head_dim,
        ) = key.shape

        patch_count = patch_width * patch_height
        source_count = source_viewmats.shape[1]
        geometry_key_length = source_count * patch_count
        geometry_key_end = geometry_key_start + geometry_key_length

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
                "Current-time URoPE expects query length H*W = {}, "
                "got {}.".format(patch_count, query_length)
            )
        if geometry_key_start < 0 or geometry_key_end > key_length:
            raise ValueError(
                "URoPE geometry interval [{}, {}) is outside K/V "
                "length {}.".format(
                    geometry_key_start,
                    geometry_key_end,
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
        key_active = key[:, :active_head_count]
        value_active = value[:, :active_head_count]

        selected_key = key_active[
            :,
            :,
            geometry_key_start:geometry_key_end,
        ]
        selected_value = value_active[
            :,
            :,
            geometry_key_start:geometry_key_end,
        ]

        # Relative URoPE on selected current-time K/V only:
        # R_q^{-1} R_k.
        selected_key = _apply_xy_rope(
            selected_key,
            key_x_coefficients,
            key_y_coefficients,
        )
        selected_key = _apply_xy_rope(
            selected_key,
            query_x_coefficients,
            query_y_coefficients,
            inverse=True,
        )

        selected_value = _apply_xy_rope(
            selected_value,
            key_x_coefficients,
            key_y_coefficients,
        )
        selected_value = _apply_xy_rope(
            selected_value,
            query_x_coefficients,
            query_y_coefficients,
            inverse=True,
        )

        key_active = torch.cat(
            [
                key_active[:, :, :geometry_key_start],
                selected_key,
                key_active[:, :, geometry_key_end:],
            ],
            dim=2,
        )
        value_active = torch.cat(
            [
                value_active[:, :, :geometry_key_start],
                selected_value,
                value_active[:, :, geometry_key_end:],
            ],
            dim=2,
        )

        if self.leaveout_head > 0:
            key = torch.cat(
                [key_active, key[:, active_head_count:]],
                dim=1,
            )
            value = torch.cat(
                [value_active, value[:, active_head_count:]],
                dim=1,
            )
        else:
            key = key_active
            value = value_active

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=query.device)
            if attention_mask.shape[-2:] != (
                query_length,
                key_length,
            ):
                raise ValueError(
                    "Current-time URoPE attention mask ends with {}, "
                    "expected ({}, {}).".format(
                        tuple(attention_mask.shape[-2:]),
                        query_length,
                        key_length,
                    )
                )

        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
