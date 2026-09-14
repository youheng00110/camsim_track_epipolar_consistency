from typing import Optional

import diffusers
import torch

from dwm.models.crossview_temporal_dit_PLUCKER_TVROW import (
    DiTCrossviewTemporalConditionModel as TVModelBase,
)
from dwm.models.crossview_temporal_dit_urope import (
    DiTCrossviewTemporalConditionModel as LocalURoPEModel,
)
from dwm.models.urope.block import VTURoPEAttentionBlock


class DiTCrossviewTemporalConditionModel(TVModelBase):
    """OpenDWM DiT with local cross-view URoPE followed by TV attention."""

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
        enable_tv: bool = False,
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
        if perspective_modeling_type != "urope":
            raise ValueError(
                "URoPE+TV requires perspective_modeling_type='urope'."
            )
        if crossview_attention_type != "full":
            raise ValueError(
                "URoPE+TV requires crossview_attention_type='full'."
            )
        if enable_tv and tv_attention_type != "full":
            raise ValueError(
                "URoPE+TV currently requires tv_attention_type='full'."
            )

        resolved_urope_config = {
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
            enable_tv=enable_tv,
            crossview_attention_type=crossview_attention_type,
            temporal_attention_type=temporal_attention_type,
            tv_attention_type=tv_attention_type,
            merge_factor=merge_factor,
            merge_strategy=merge_strategy,
            crossview_block_layers=crossview_block_layers,
            temporal_block_layers=temporal_block_layers,
            tv_block_layers=tv_block_layers,
            crossview_gradient_checkpointing=(
                crossview_gradient_checkpointing
            ),
            temporal_gradient_checkpointing=temporal_gradient_checkpointing,
            tv_gradient_checkpointing=tv_gradient_checkpointing,
            mixer_type=mixer_type,
            perspective_modeling_type=perspective_modeling_type,
            disable_view_emb_on_temporal_module=(
                disable_view_emb_on_temporal_module
            ),
            qk_norm_on_additional_modules=qk_norm_on_additional_modules,
            mask_module=mask_module,
            tv_time_radius=tv_time_radius,
            tv_view_radius=tv_view_radius,
            tv_height_chunk_size=tv_height_chunk_size,
            tv_full_batch_chunk_size=tv_full_batch_chunk_size,
            **kwargs,
        )

        self.urope_config = resolved_urope_config
        self.run_crossview_with_tv = True

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
                for _ in range(len(self.crossview_block_layers))
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
        return LocalURoPEModel.forward_crossview_block_and_mix_result(
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
            camera_intrinsics_norm,
            camera2referego,
        )
