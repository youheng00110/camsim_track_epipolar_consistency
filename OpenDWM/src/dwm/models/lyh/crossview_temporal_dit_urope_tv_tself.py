from typing import Optional

import diffusers
import diffusers.models.attention_processor
import torch

from dwm.models.crossview_temporal_dit_PLUCKER_TVROW import (
    DiTCrossviewTemporalConditionModel as TVModelBase,
    build_tv_view_index_from_crossview_mask,
)
from dwm.models.lyh.urope_current_time import (
    CurrentTimeURoPEDotProductAttention,
)


class VTURoPETVAttentionBlock(torch.nn.Module):
    """
    Joint TV attention with URoPE applied inside the same attention operation.

    Query:
        current (t, v), all HW tokens -> [N, HW, C]

    Context:
        [t-1, t, t+1] x [left, self, right] x HW
        -> [N, 9*HW, C]

    Temporal position:
        supplied by the original TV time embedding before Q/K/V projection.

    Camera geometry:
        TV still attends all 9 time-view slots. URoPE geometry is computed
        only for CURRENT time t x [left, self, right]. The t-1/t+1 K/V
        slots remain vanilla and do not reuse current-time geometry.

    No camera SlotID embedding is added.
    """

    def __init__(
        self,
        inner_dim: int,
        context_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        urope_config: Optional[dict] = None,
        qk_norm=None,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.context_dim = context_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_attention_dim = num_attention_heads * attention_head_dim
        self.dropout = float(dropout)

        self.norm_q = torch.nn.LayerNorm(inner_dim)
        self.norm_context = torch.nn.LayerNorm(context_dim)

        # Keep the original TV parameter names/shapes for checkpoint reuse.
        self.q_proj = torch.nn.Linear(
            inner_dim,
            self.inner_attention_dim,
            bias=False,
        )
        self.k_proj = torch.nn.Linear(
            context_dim,
            self.inner_attention_dim,
            bias=False,
        )
        self.v_proj = torch.nn.Linear(
            context_dim,
            self.inner_attention_dim,
            bias=False,
        )
        self.out_proj = torch.nn.Linear(
            self.inner_attention_dim,
            inner_dim,
            bias=False,
        )

        self.qk_norm_q = None
        self.qk_norm_k = None
        if qk_norm is not None:
            qk_norm_helper = diffusers.models.attention_processor.Attention(
                query_dim=self.inner_attention_dim,
                cross_attention_dim=self.inner_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                qk_norm=qk_norm,
                bias=False,
            )
            self.qk_norm_q = qk_norm_helper.norm_q
            self.qk_norm_k = qk_norm_helper.norm_k

        self.urope_attention = CurrentTimeURoPEDotProductAttention(
            head_num=num_attention_heads,
            head_dim=attention_head_dim,
            **(urope_config or {}),
        )

        self.norm_ff = torch.nn.LayerNorm(inner_dim)
        self.ff_in = torch.nn.Linear(
            inner_dim,
            inner_dim * ff_mult,
        )
        self.ff_act = torch.nn.GELU(approximate="tanh")
        self.ff_out = torch.nn.Linear(
            inner_dim * ff_mult,
            inner_dim,
        )

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        context_hidden_states: torch.Tensor,
        *,
        query_viewmats: torch.Tensor,
        query_intrinsics: torch.Tensor,
        source_viewmats: torch.Tensor,
        source_intrinsics: torch.Tensor,
        patch_width: int,
        patch_height: int,
        geometry_key_start: int,
        context_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = query_hidden_states

        query_hidden_states = self.norm_q(query_hidden_states)
        context_hidden_states = self.norm_context(context_hidden_states)

        q = self.q_proj(query_hidden_states)
        k = self.k_proj(context_hidden_states)
        v = self.v_proj(context_hidden_states)

        batch_size, query_length, _ = q.shape
        context_length = k.shape[1]

        q = q.view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.attention_head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            context_length,
            self.num_attention_heads,
            self.attention_head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            context_length,
            self.num_attention_heads,
            self.attention_head_dim,
        ).transpose(1, 2)

        if self.qk_norm_q is not None:
            q = self.qk_norm_q(q)
        if self.qk_norm_k is not None:
            k = self.qk_norm_k(k)

        attention_mask = None
        if context_attention_mask is not None:
            if context_attention_mask.shape != (
                batch_size,
                context_length,
            ):
                raise ValueError(
                    "context_attention_mask should be [N,K], "
                    f"but got {tuple(context_attention_mask.shape)} "
                    f"for N={batch_size}, K={context_length}."
                )
            attention_mask = context_attention_mask.to(
                device=q.device,
                dtype=torch.bool,
            )[:, None, None, :].expand(
                batch_size,
                1,
                query_length,
                context_length,
            )

        attention_output = self.urope_attention(
            q,
            k,
            v,
            query_viewmats=query_viewmats,
            query_intrinsics=query_intrinsics,
            source_viewmats=source_viewmats,
            source_intrinsics=source_intrinsics,
            patch_width=patch_width,
            patch_height=patch_height,
            geometry_key_start=geometry_key_start,
            attention_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous().view(
            batch_size,
            query_length,
            self.inner_attention_dim,
        )

        hidden_states = residual + self.out_proj(attention_output)
        hidden_states = hidden_states + self.ff_out(
            self.ff_act(
                self.ff_in(
                    self.norm_ff(hidden_states)
                )
            )
        )
        return hidden_states


class DiTCrossviewTemporalConditionModel(TVModelBase):
    """
    OpenDWM DiT with URoPE directly inside TV attention.

    This is NOT:
        cross-view URoPE attention -> TV attention

    It is one joint attention:
        TV selects the local temporal-view context;
        URoPE provides camera geometry inside that same attention;
        the original TV time embedding provides temporal position.

    Independent cross-view attention is disabled, so Camera SlotID /
    view_pos_embeds are not used.
    """

    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        projection_class_embeddings_input_dim: int = None,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = False,
        enable_temporal: bool = False,
        enable_tv: bool = True,
        urope_config: Optional[dict] = None,
        crossview_attention_type: str = "full",
        temporal_attention_type: str = None,
        tv_attention_type: str = "full",
        merge_factor: float = 2,
        merge_strategy: str = "learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        tv_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        tv_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "urope",
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mask_module=None,
        tv_time_radius: int = 1,
        tv_view_radius: int = 1,
        tv_height_chunk_size: int = 0,
        tv_full_batch_chunk_size: int = 1,
        **kwargs,
    ):
        # Accepted for old config compatibility, but intentionally unused.
        del enable_crossview
        del crossview_gradient_checkpointing

        if perspective_modeling_type != "urope":
            raise ValueError(
                "URoPE-inside-TV requires "
                "perspective_modeling_type='urope'."
            )
        if not enable_tv:
            raise ValueError(
                "URoPE-inside-TV requires enable_tv=True."
            )
        if tv_attention_type != "full":
            raise ValueError(
                "URoPE-inside-TV currently requires "
                "tv_attention_type='full'."
            )

        self.urope_config = {
            "min_depth": 2.0,
            "max_depth": 20.0,
            "freq_base": 100.0,
            "freq_scale": 1.0,
            "group_size": 4,
            "leaveout_head": 0,
            "camera_convention": "opencv",
            **(urope_config or {}),
        }

        # No independent cross-view branch. TV still receives and uses
        # crossview_attention_mask to choose left/self/right source views.
        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            projection_class_embeddings_input_dim=(
                projection_class_embeddings_input_dim
            ),
            condition_image_adapter_config=condition_image_adapter_config,
            enable_crossview=False,
            enable_temporal=enable_temporal,
            enable_tv=True,
            crossview_attention_type=crossview_attention_type,
            temporal_attention_type=temporal_attention_type,
            tv_attention_type=tv_attention_type,
            merge_factor=merge_factor,
            merge_strategy=merge_strategy,
            crossview_block_layers=crossview_block_layers,
            temporal_block_layers=temporal_block_layers,
            tv_block_layers=tv_block_layers,
            crossview_gradient_checkpointing=False,
            temporal_gradient_checkpointing=(
                temporal_gradient_checkpointing
            ),
            tv_gradient_checkpointing=tv_gradient_checkpointing,
            mixer_type=mixer_type,
            perspective_modeling_type="urope",
            disable_view_emb_on_temporal_module=(
                disable_view_emb_on_temporal_module
            ),
            qk_norm_on_additional_modules=(
                qk_norm_on_additional_modules
            ),
            mask_module=mask_module,
            tv_time_radius=tv_time_radius,
            tv_view_radius=tv_view_radius,
            tv_height_chunk_size=tv_height_chunk_size,
            tv_full_batch_chunk_size=tv_full_batch_chunk_size,
            **kwargs,
        )

        self.register_to_config(
            enable_crossview=False,
            enable_tv=True,
            perspective_modeling_type="urope",
        )

        inner_dim = attention_head_dim * num_attention_heads
        self.tv_transformer_blocks = torch.nn.ModuleList([
            VTURoPETVAttentionBlock(
                inner_dim=inner_dim,
                context_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                urope_config=self.urope_config,
                qk_norm=qk_norm_on_additional_modules,
            )
            for _ in range(len(self.tv_block_layers))
        ])

    def forward(
        self,
        *args,
        camera_intrinsics_norm=None,
        camera2referego=None,
        **kwargs,
    ):
        if camera_intrinsics_norm is None or camera2referego is None:
            raise ValueError(
                "URoPE-inside-TV requires camera_intrinsics_norm "
                "and camera2referego."
            )

        # Keep geometry for checkpoint recomputation during backward.
        self._urope_tv_camera_intrinsics_norm = camera_intrinsics_norm
        self._urope_tv_camera2referego = camera2referego

        return super().forward(
            *args,
            camera_intrinsics_norm=camera_intrinsics_norm,
            camera2referego=camera2referego,
            **kwargs,
        )

    def forward_tv_full_block_and_mix_result(
        self,
        tv_block: torch.nn.Module,
        mixer,
        hidden_states: torch.Tensor,
        tv_emb: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        width: int,
        height: int,
        disable_tv: torch.BoolTensor,
        crossview_attention_mask: torch.Tensor,
        crossview_attention_index: torch.Tensor,
    ):
        del crossview_attention_index

        if self.tv_attention_type != "full":
            raise ValueError(
                "forward_tv_full_block_and_mix_result requires "
                f"tv_attention_type='full', got "
                f"{self.tv_attention_type!r}."
            )

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states should be [B*T*V, HW, C], "
                f"but got {tuple(hidden_states.shape)}."
            )

        camera_intrinsics_norm = getattr(
            self,
            "_urope_tv_camera_intrinsics_norm",
            None,
        )
        camera2referego = getattr(
            self,
            "_urope_tv_camera2referego",
            None,
        )
        if camera_intrinsics_norm is None or camera2referego is None:
            raise ValueError(
                "URoPE TV geometry cache is empty. "
                "Call model.forward with camera_intrinsics_norm and "
                "camera2referego."
            )

        device = hidden_states.device
        dtype = hidden_states.dtype
        channel = hidden_states.shape[-1]
        token_count = height * width
        expected_flat_batch = (
            batch_size * sequence_length * view_count
        )

        if hidden_states.shape[0] != expected_flat_batch:
            raise ValueError(
                "Flattened batch mismatch: "
                f"hidden_states.shape[0]={hidden_states.shape[0]}, "
                f"expected={expected_flat_batch}."
            )
        if hidden_states.shape[1] != token_count:
            raise ValueError(
                "Spatial token mismatch: "
                f"hidden_states.shape[1]={hidden_states.shape[1]}, "
                f"height*width={token_count}."
            )

        # With perspective_modeling_type='urope', view_cam_emb in TVModelBase
        # stays zero, so tv_emb contains temporal position only.
        tv_hidden_states = hidden_states + tv_emb.to(
            device=device,
            dtype=dtype,
        )
        tv_hidden_states = tv_hidden_states.reshape(
            batch_size,
            sequence_length,
            view_count,
            token_count,
            channel,
        )

        intrinsics = camera_intrinsics_norm.to(
            device=device,
            dtype=torch.float32,
        ).clone().reshape(
            batch_size,
            sequence_length,
            view_count,
            3,
            3,
        )
        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width
        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height
        intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width
        intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height

        camera2referego = camera2referego.to(
            device=device,
            dtype=torch.float32,
        ).reshape(
            batch_size,
            sequence_length,
            view_count,
            4,
            4,
        )
        viewmats = torch.linalg.inv(camera2referego)

        time_base = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.long,
        )
        time_offsets = torch.tensor(
            [-1, 0, 1],
            device=device,
            dtype=torch.long,
        )
        raw_time_index = (
            time_base[:, None] + time_offsets[None, :]
        )

        # Preserve the current TV semantics: boundary time slots clamp and
        # still participate.
        time_index = raw_time_index.clamp(
            0,
            sequence_length - 1,
        )
        time_valid_mask = torch.ones_like(
            time_index,
            dtype=torch.bool,
        )

        # Preserve current TV semantics: missing left/right uses self.
        view_index, view_valid_mask = (
            build_tv_view_index_from_crossview_mask(
                crossview_attention_mask,
                batch_size,
                view_count,
                device,
            )
        )

        flat_batch_index = torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        )[:, None, None].expand(
            batch_size,
            sequence_length,
            view_count,
        ).reshape(-1)

        flat_time_index = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.long,
        )[None, :, None].expand(
            batch_size,
            sequence_length,
            view_count,
        ).reshape(-1)

        flat_view_index = torch.arange(
            view_count,
            device=device,
            dtype=torch.long,
        )[None, None, :].expand(
            batch_size,
            sequence_length,
            view_count,
        ).reshape(-1)

        target_count = expected_flat_batch
        chunk_size = min(
            self.tv_full_batch_chunk_size,
            target_count,
        )
        output_chunks = []

        for chunk_start in range(
            0,
            target_count,
            chunk_size,
        ):
            chunk_end = min(
                chunk_start + chunk_size,
                target_count,
            )

            target_b = flat_batch_index[
                chunk_start:chunk_end
            ]
            target_t = flat_time_index[
                chunk_start:chunk_end
            ]
            target_v = flat_view_index[
                chunk_start:chunk_end
            ]
            current_chunk_size = target_b.shape[0]

            query_hidden_states = tv_hidden_states[
                target_b,
                target_t,
                target_v,
            ]

            source_time_index = time_index[target_t]
            source_time_valid = time_valid_mask[target_t]
            source_view_index = view_index[
                target_b,
                target_v,
            ]
            source_view_valid = view_valid_mask[
                target_b,
                target_v,
            ]

            local_hidden_states = tv_hidden_states[
                target_b[:, None, None],
                source_time_index[:, :, None],
                source_view_index[:, None, :],
            ]

            expected_local_shape = (
                current_chunk_size,
                3,
                3,
                token_count,
                channel,
            )
            if tuple(local_hidden_states.shape) != (
                expected_local_shape
            ):
                raise RuntimeError(
                    "TV-full gather shape mismatch: "
                    f"got {tuple(local_hidden_states.shape)}, "
                    f"expected {expected_local_shape}."
                )

            context_hidden_states = (
                local_hidden_states.reshape(
                    current_chunk_size,
                    9 * token_count,
                    channel,
                )
            )

            context_slot_valid = (
                source_time_valid[:, :, None]
                & source_view_valid[:, None, :]
            )
            context_attention_mask = (
                context_slot_valid[
                    :,
                    :,
                    :,
                    None,
                ].expand(
                    current_chunk_size,
                    3,
                    3,
                    token_count,
                ).reshape(
                    current_chunk_size,
                    9 * token_count,
                )
            )

            # Target camera geometry.
            query_viewmats = viewmats[
                target_b,
                target_t,
                target_v,
            ]
            query_intrinsics = intrinsics[
                target_b,
                target_t,
                target_v,
            ]

            # Current-time-only URoPE geometry.
            # TV content remains 3 time x 3 view = 9 slots.
            # Geometry exists only at target_t for [left,self,right].
            # There is NO geometry reuse onto t-1/t+1.
            source_viewmats = viewmats[
                target_b[:, None],
                target_t[:, None],
                source_view_index,
            ]
            source_intrinsics = intrinsics[
                target_b[:, None],
                target_t[:, None],
                source_view_index,
            ]

            expected_source_viewmats_shape = (
                current_chunk_size, 3, 4, 4,
            )
            expected_source_intrinsics_shape = (
                current_chunk_size, 3, 3, 3,
            )
            if tuple(source_viewmats.shape) != expected_source_viewmats_shape:
                raise RuntimeError(
                    "Current-time URoPE source_viewmats mismatch: "
                    f"got {tuple(source_viewmats.shape)}, "
                    f"expected {expected_source_viewmats_shape}."
                )
            if tuple(source_intrinsics.shape) != expected_source_intrinsics_shape:
                raise RuntimeError(
                    "Current-time URoPE source_intrinsics mismatch: "
                    f"got {tuple(source_intrinsics.shape)}, "
                    f"expected {expected_source_intrinsics_shape}."
                )

            tv_chunk = tv_block(
                query_hidden_states,
                context_hidden_states,
                query_viewmats=query_viewmats,
                query_intrinsics=query_intrinsics,
                source_viewmats=source_viewmats,
                source_intrinsics=source_intrinsics,
                patch_width=width,
                patch_height=height,
                # Flatten order is time-major:
                # [t-1:L/S/R][t:L/S/R][t+1:L/S/R].
                geometry_key_start=3 * token_count,
                context_attention_mask=(
                    context_attention_mask
                ),
            )

            if tv_chunk.shape != query_hidden_states.shape:
                raise RuntimeError(
                    "URoPE-TV output shape mismatch: "
                    f"output={tuple(tv_chunk.shape)}, "
                    f"query={tuple(query_hidden_states.shape)}."
                )

            output_chunks.append(tv_chunk)

        tv_hidden_states = torch.cat(
            output_chunks,
            dim=0,
        )
        if tv_hidden_states.shape != hidden_states.shape:
            raise RuntimeError(
                "URoPE-TV final shape mismatch: "
                f"tv={tuple(tv_hidden_states.shape)}, "
                f"hidden={tuple(hidden_states.shape)}."
            )

        if mixer is None:
            return tv_hidden_states

        return mixer(
            hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                token_count,
                channel,
            ),
            tv_hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                token_count,
                channel,
            ),
            image_only_indicator=disable_tv,
        ).flatten(0, 1)
