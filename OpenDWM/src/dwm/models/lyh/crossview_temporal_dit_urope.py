from typing import Optional

import diffusers
import einops
import torch

from dwm.models.crossview_temporal_dit import (
    DiTCrossviewTemporalConditionModel as OpenDWMModelBase,
)
from dwm.models.urope.block import VTURoPEAttentionBlock


class DiTCrossviewTemporalConditionModel(OpenDWMModelBase):
    """OpenDWM DiT with full cross-view URoPE self-attention."""

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
        urope_config: Optional[dict] = None,
        crossview_attention_type: str = "full",
        temporal_attention_type: str = None,
        merge_factor: float = 2,
        merge_strategy: str = "learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "urope",
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mask_module=None,
        **kwargs,
    ):
        if perspective_modeling_type != "urope":
            raise ValueError(
                "This model requires perspective_modeling_type='urope'."
            )
        if crossview_attention_type != "full":
            raise ValueError(
                "The first URoPE reproduction supports full attention only."
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

        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            projection_class_embeddings_input_dim=(
                projection_class_embeddings_input_dim
            ),
            condition_image_adapter_config=condition_image_adapter_config,
            enable_crossview=enable_crossview,
            enable_temporal=enable_temporal,
            crossview_attention_type=crossview_attention_type,
            temporal_attention_type=temporal_attention_type,
            merge_factor=merge_factor,
            merge_strategy=merge_strategy,
            crossview_block_layers=crossview_block_layers,
            temporal_block_layers=temporal_block_layers,
            crossview_gradient_checkpointing=(
                crossview_gradient_checkpointing
            ),
            temporal_gradient_checkpointing=temporal_gradient_checkpointing,
            mixer_type=mixer_type,
            perspective_modeling_type=perspective_modeling_type,
            disable_view_emb_on_temporal_module=(
                disable_view_emb_on_temporal_module
            ),
            qk_norm_on_additional_modules=qk_norm_on_additional_modules,
            mask_module=mask_module,
            **kwargs,
        )

        if enable_crossview:
            inner_dim = attention_head_dim * num_attention_heads
            self.crossview_transformer_blocks = torch.nn.ModuleList([
                VTURoPEAttentionBlock(
                    dim=inner_dim,
                    time_mix_inner_dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    qk_norm=qk_norm_on_additional_modules,
                    urope_config=self.urope_config,
                )
                for _ in range(len(crossview_block_layers))
            ])

    def forward(
        self,
        *args,
        camera_intrinsics_norm=None,
        camera2referego=None,
        **kwargs,
    ):
        self._urope_camera_intrinsics_norm = camera_intrinsics_norm
        self._urope_camera2referego = camera2referego

        return super().forward(
            *args,
            camera_intrinsics_norm=camera_intrinsics_norm,
            camera2referego=camera2referego,
            **kwargs,
        )

    def forward_crossview_block_and_mix_result(
        self,
        crossview_block,
        mixer,
        hidden_states,
        view_emb,
        batch_size,
        sequence_length,
        view_count,
        width,
        height,
        disable_crossview,
        crossview_attention_mask,
        crossview_attention_index,
        camera_intrinsics_norm=None,
        camera2referego=None,
    ):
        del view_emb

        if camera_intrinsics_norm is None:
            camera_intrinsics_norm = getattr(
                self,
                "_urope_camera_intrinsics_norm",
                None,
            )

        if camera2referego is None:
            camera2referego = getattr(
                self,
                "_urope_camera2referego",
                None,
            )

        if camera_intrinsics_norm is None or camera2referego is None:
            raise ValueError(
                "URoPE requires camera_intrinsics_norm and camera2referego."
            )

        bt_count = batch_size * sequence_length
        patch_count = height * width
        hidden_states_by_view = hidden_states.reshape(
            bt_count,
            view_count,
            patch_count,
            hidden_states.shape[-1],
        )

        if crossview_attention_mask is not None:
            camera_mask = crossview_attention_mask.to(
                device=hidden_states.device
            ).bool()
            if camera_mask.ndim == 2:
                camera_mask = camera_mask.unsqueeze(0)
            elif camera_mask.ndim == 4 and camera_mask.shape[1] == 1:
                camera_mask = camera_mask[:, 0]

            if camera_mask.ndim != 3:
                raise ValueError(
                    "crossview_attention_mask must be [V,V], [B,V,V], "
                    "or [B*T,V,V], got {}.".format(
                        tuple(camera_mask.shape)
                    )
                )
            if camera_mask.shape[-2:] != (view_count, view_count):
                raise ValueError(
                    "crossview_attention_mask ends with {}, expected "
                    "({}, {}).".format(
                        tuple(camera_mask.shape[-2:]),
                        view_count,
                        view_count,
                    )
                )

            mask_batch_size = camera_mask.shape[0]
            if mask_batch_size == 1 and batch_size > 1:
                camera_mask = camera_mask.expand(
                    batch_size,
                    -1,
                    -1,
                )
                mask_batch_size = batch_size

            if mask_batch_size == batch_size:
                camera_mask = camera_mask.repeat_interleave(
                    sequence_length,
                    dim=0,
                )
            elif mask_batch_size != bt_count:
                raise ValueError(
                    "Camera mask batch must be B or B*T, got {} "
                    "for B={} and T={}.".format(
                        mask_batch_size,
                        batch_size,
                        sequence_length,
                    )
                )

            diagonal = torch.arange(
                view_count,
                device=hidden_states.device,
            )
            camera_mask = camera_mask.clone()
            camera_mask[:, diagonal, diagonal] = True

            local_view_count = int(
                camera_mask.sum(dim=-1).max().item()
            )
            all_view_indices = torch.arange(
                view_count,
                device=hidden_states.device,
            ).reshape(1, 1, view_count)
            all_view_indices = all_view_indices.expand(
                bt_count,
                view_count,
                view_count,
            )
            invalid_index = torch.full_like(
                all_view_indices,
                view_count,
            )
            source_view_index = torch.where(
                camera_mask,
                all_view_indices,
                invalid_index,
            )
            source_view_index = source_view_index.sort(
                dim=-1
            ).values[..., :local_view_count]
            source_view_valid = source_view_index < view_count

            query_view_index = torch.arange(
                view_count,
                device=hidden_states.device,
            ).reshape(1, view_count, 1)
            query_view_index = query_view_index.expand(
                bt_count,
                view_count,
                local_view_count,
            )
            source_view_index = torch.where(
                source_view_valid,
                source_view_index,
                query_view_index,
            )
        else:
            if crossview_attention_index is None:
                raise ValueError(
                    "Local URoPE requires crossview_attention_mask or "
                    "crossview_attention_index."
                )

            source_view_index = crossview_attention_index.to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            if source_view_index.ndim == 2:
                if source_view_index.shape[-1] % view_count != 0:
                    raise ValueError(
                        "crossview_attention_index width must be divisible "
                        "by view_count, got {} for V={}.".format(
                            source_view_index.shape[-1],
                            view_count,
                        )
                    )
                source_view_index = source_view_index.reshape(
                    source_view_index.shape[0],
                    view_count,
                    -1,
                )
            elif source_view_index.ndim != 3:
                raise ValueError(
                    "crossview_attention_index must be [B,V*K] or "
                    "[B,V,K], got {}.".format(
                        tuple(source_view_index.shape)
                    )
                )

            index_batch_size = source_view_index.shape[0]
            if index_batch_size == 1 and batch_size > 1:
                source_view_index = source_view_index.expand(
                    batch_size,
                    -1,
                    -1,
                )
                index_batch_size = batch_size

            if index_batch_size == batch_size:
                source_view_index = source_view_index.repeat_interleave(
                    sequence_length,
                    dim=0,
                )
            elif index_batch_size != bt_count:
                raise ValueError(
                    "Cross-view index batch must be B or B*T, got {} "
                    "for B={} and T={}.".format(
                        index_batch_size,
                        batch_size,
                        sequence_length,
                    )
                )

            if source_view_index.shape[1] != view_count:
                raise ValueError(
                    "crossview_attention_index has {} query views, "
                    "expected {}.".format(
                        source_view_index.shape[1],
                        view_count,
                    )
                )
            if (
                source_view_index.min().item() < 0
                or source_view_index.max().item() >= view_count
            ):
                raise ValueError(
                    "crossview_attention_index contains an invalid "
                    "camera index."
                )
            local_view_count = source_view_index.shape[-1]
            source_view_valid = torch.ones_like(
                source_view_index,
                dtype=torch.bool,
            )

        bt_index = torch.arange(
            bt_count,
            device=hidden_states.device,
        ).reshape(bt_count, 1, 1)

        context_hidden_states = hidden_states_by_view[
            bt_index,
            source_view_index,
        ]
        context_hidden_states = context_hidden_states.reshape(
            bt_count * view_count,
            local_view_count * patch_count,
            hidden_states.shape[-1],
        )
        query_hidden_states = hidden_states_by_view.reshape(
            bt_count * view_count,
            patch_count,
            hidden_states.shape[-1],
        )

        intrinsics = camera_intrinsics_norm.clone().float()
        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width
        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height
        intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width
        intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height
        intrinsics = intrinsics.reshape(
            bt_count,
            view_count,
            3,
            3,
        ).to(device=hidden_states.device)

        viewmats = torch.linalg.inv(camera2referego.float())
        viewmats = viewmats.reshape(
            bt_count,
            view_count,
            4,
            4,
        ).to(device=hidden_states.device)

        query_intrinsics = intrinsics.reshape(
            bt_count * view_count,
            3,
            3,
        )
        query_viewmats = viewmats.reshape(
            bt_count * view_count,
            4,
            4,
        )
        source_intrinsics = intrinsics[
            bt_index,
            source_view_index,
        ].reshape(
            bt_count * view_count,
            local_view_count,
            3,
            3,
        )
        source_viewmats = viewmats[
            bt_index,
            source_view_index,
        ].reshape(
            bt_count * view_count,
            local_view_count,
            4,
            4,
        )

        source_token_valid = source_view_valid.unsqueeze(-1).expand(
            bt_count,
            view_count,
            local_view_count,
            patch_count,
        ).reshape(
            bt_count * view_count,
            local_view_count * patch_count,
        )
        local_attention_mask = source_token_valid[
            :,
            None,
            None,
            :,
        ].expand(
            -1,
            1,
            patch_count,
            -1,
        )

        crossview_hidden_states = crossview_block(
            query_hidden_states,
            context_hidden_states=context_hidden_states,
            query_viewmats=query_viewmats,
            query_intrinsics=query_intrinsics,
            source_viewmats=source_viewmats,
            source_intrinsics=source_intrinsics,
            patch_height=height,
            patch_width=width,
            self_attention_mask=local_attention_mask,
        )

        if mixer is None:
            return crossview_hidden_states

        original_states = hidden_states.view(
            batch_size,
            sequence_length * view_count,
            *hidden_states.shape[1:],
        )
        crossview_states = crossview_hidden_states.view(
            batch_size,
            sequence_length * view_count,
            *crossview_hidden_states.shape[1:],
        )
        return mixer(
            original_states,
            crossview_states,
            image_only_indicator=disable_crossview,
        ).flatten(0, 1)
