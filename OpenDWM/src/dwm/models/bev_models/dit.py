import math
from typing import Optional, Union

import diffusers
import einops
import torch

from dwm.models.crossview_temporal import AlphaBlender, VTSelfAttentionBlock
from .condition import (
    ConditionCrossAttention,
    EgoTrajectoryConditionEncoder,
    TemporalBBoxConditionEncoder,
    TemporalBEVResidualAdapter,
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_octaves: int, start_octave: int = 0):
        super().__init__()
        self.num_octaves = int(num_octaves)
        self.start_octave = int(start_octave)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, coordinate_dim = coordinates.shape
        octaves = torch.arange(
            self.start_octave,
            self.start_octave + self.num_octaves,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        multipliers = (2.0 ** octaves) * math.pi
        scaled = coordinates.unsqueeze(-1) * multipliers
        sine = torch.sin(scaled).reshape(
            batch_size,
            point_count,
            coordinate_dim * self.num_octaves,
        )
        cosine = torch.cos(scaled).reshape(
            batch_size,
            point_count,
            coordinate_dim * self.num_octaves,
        )
        return torch.cat([sine, cosine], dim=-1)


class PluckerEncoder(torch.nn.Module):
    def __init__(
        self,
        dir_octaves: int = 4,
        moment_octaves: int = 8,
        out_dim: int = 1536,
    ):
        super().__init__()
        self.dir_encoding = PositionalEncoding(dir_octaves)
        self.moment_encoding = PositionalEncoding(moment_octaves)
        encoded_dim = 3 * dir_octaves * 2 + 3 * moment_octaves * 2
        self.proj = torch.nn.Linear(encoded_dim, out_dim, bias=False)

    def forward(
        self,
        ray_directions: torch.Tensor,
        ray_moments: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, height, width = ray_directions.shape[:3]
        ray_directions = ray_directions.flatten(1, 2)
        ray_moments = ray_moments.flatten(1, 2)
        direction_features = self.dir_encoding(ray_directions)
        moment_features = self.moment_encoding(ray_moments)
        features = torch.cat([direction_features, moment_features], dim=-1)
        features = features.view(batch_size, height, width, -1)
        return self.proj(features)

def get_rays(
    camera_intrinsics: torch.Tensor,
    camera_to_ego: torch.Tensor,
    target_size: Union[int, tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = camera_to_ego.device
    output_dtype = camera_to_ego.dtype

    camera_intrinsics = camera_intrinsics.float()
    camera_to_ego = camera_to_ego.float()

    if isinstance(target_size, int):
        height = target_size
        width = target_size
    else:
        height, width = target_size

    pixel_x, pixel_y = torch.meshgrid(
        torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        ) + 0.5,
        torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        ) + 0.5,
        indexing="xy",
    )

    homogeneous_pixels = torch.stack(
        [
            pixel_x.reshape(-1),
            pixel_y.reshape(-1),
            torch.ones(
                height * width,
                device=device,
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )

    camera_directions = torch.linalg.solve(
        camera_intrinsics,
        homogeneous_pixels.unsqueeze(0),
    )

    ray_directions = (
        camera_to_ego[:, :3, :3]
        @ camera_directions
    )
    ray_directions = torch.nn.functional.normalize(
        ray_directions,
        dim=1,
    )
    ray_directions = ray_directions.transpose(
        1,
        2,
    ).reshape(
        -1,
        height,
        width,
        3,
    )

    ray_origins = camera_to_ego[:, :3, 3]

    return (
        ray_origins.to(output_dtype),
        ray_directions.to(output_dtype),
    )

class BEVConditionedSD3TransformerModel(diffusers.SD3Transformer2DModel):
    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        block_layers=(1, 5, 9, 13, 17, 21),
        merge_factor: float = 2.0,
        bbox_config: Optional[dict] = None,
        bev_in_channels: int = 3,
        bev_hidden_channels: int = 256,
        trajectory_translation_scale: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs,
        )
        self.block_layers = tuple(int(layer) for layer in block_layers)
        if len(set(self.block_layers)) != len(self.block_layers):
            raise ValueError("block_layers must not contain duplicate entries.")
        if min(self.block_layers) < 0 or max(self.block_layers) >= num_layers:
            raise ValueError(
                f"block_layers={self.block_layers} must be inside [0,{num_layers - 1}]."
            )
        self.additional_layer_index = {
            layer: index for index, layer in enumerate(self.block_layers)
        }

        inner_dim = attention_head_dim * num_attention_heads
        self.index_proj = diffusers.models.embeddings.Timesteps(
            inner_dim,
            True,
            0,
        )

        self.rayencoder = PluckerEncoder(out_dim=inner_dim)
        self.camera_crossview_proj = torch.nn.Linear(
            inner_dim,
            inner_dim,
            bias=False,
        )
        torch.nn.init.zeros_(self.camera_crossview_proj.weight)
        self.trajectory_condition_encoder = EgoTrajectoryConditionEncoder(
            out_dim=inner_dim,
            translation_scale=trajectory_translation_scale,
        )
        bbox_config = {} if bbox_config is None else dict(bbox_config)
        self.bbox_condition_encoder = TemporalBBoxConditionEncoder(
            out_dim=inner_dim,
            **bbox_config,
        )
        self.null_condition_token = torch.nn.Parameter(torch.zeros(inner_dim))

        self.time_pos_embeds = torch.nn.ModuleList(
            [
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim,
                    inner_dim * 4,
                    out_dim=inner_dim,
                )
                for _ in self.block_layers
            ]
        )
        self.temporal_transformer_blocks = torch.nn.ModuleList(
            [
                VTSelfAttentionBlock(
                    inner_dim,
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    qk_norm=None,
                )
                for _ in self.block_layers
            ]
        )
        self.time_mixers = torch.nn.ModuleList(
            [
                AlphaBlender(
                    merge_factor,
                    merge_strategy="learned_with_images",
                )
                for _ in self.block_layers
            ]
        )
        self.crossview_transformer_blocks = torch.nn.ModuleList(
            [
                VTSelfAttentionBlock(
                    inner_dim,
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    qk_norm=None,
                )
                for _ in self.block_layers
            ]
        )
        self.view_mixers = torch.nn.ModuleList(
            [
                AlphaBlender(
                    merge_factor,
                    merge_strategy="learned_with_images",
                )
                for _ in self.block_layers
            ]
        )
        self.cond_cross_blocks = torch.nn.ModuleList(
            [
                ConditionCrossAttention(
                    hidden_dim=inner_dim,
                    num_heads=num_attention_heads,
                )
                for _ in self.block_layers
            ]
        )
        self.bev_control = TemporalBEVResidualAdapter(
            in_channels=bev_in_channels,
            out_channels=inner_dim,
            depth=len(self.block_layers),
            hidden_channels=bev_hidden_channels,
        )

    def build_first_frame_plucker_map(
        self,
        camera_intrinsics_norm: torch.Tensor,
        camera_to_ego: torch.Tensor,
        height: int,
        width: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            camera_intrinsics_norm.ndim != 5
            or camera_intrinsics_norm.shape[-2:] != (3, 3)
        ):
            raise ValueError(
                "camera_intrinsics_norm must be "
                "[B,1,V,3,3], "
                f"got {tuple(camera_intrinsics_norm.shape)}."
            )

        if (
            camera_to_ego.ndim != 5
            or camera_to_ego.shape[-2:] != (4, 4)
        ):
            raise ValueError(
                "camera_to_ego must be "
                "[B,1,V,4,4], "
                f"got {tuple(camera_to_ego.shape)}."
            )

        if camera_intrinsics_norm.shape[1] != 1:
            raise ValueError(
                "camera_intrinsics_norm must contain "
                "only the fixed first-frame intrinsic, "
                f"got T={camera_intrinsics_norm.shape[1]}."
            )

        if camera_to_ego.shape[1] != 1:
            raise ValueError(
                "camera_to_ego must contain only "
                "the fixed first-frame rig calibration, "
                f"got T={camera_to_ego.shape[1]}."
            )

        batch_size, _, view_count = camera_to_ego.shape[:3]

        intrinsics = camera_intrinsics_norm.to(
            device=camera_to_ego.device,
            dtype=torch.float32,
        ).clone()

        intrinsics[..., 0, 0] *= width
        intrinsics[..., 1, 1] *= height
        intrinsics[..., 0, 2] *= width
        intrinsics[..., 1, 2] *= height

        ray_origins, ray_directions = get_rays(
            intrinsics.flatten(0, 2),
            camera_to_ego.float().flatten(0, 2),
            (height, width),
        )

        ray_origins = ray_origins[
            :, None, None
        ].expand_as(ray_directions)

        ray_moments = torch.cross(
            ray_origins,
            ray_directions,
            dim=-1,
        )

        plucker_map = self.rayencoder(
            ray_directions.to(output_dtype),
            ray_moments.to(output_dtype),
        )

        return plucker_map.view(
            batch_size,
            view_count,
            height,
            width,
            -1,
        )

    def build_condition_inputs(
        self,
        camera_intrinsics_norm: torch.Tensor,
        camera_to_ego: torch.Tensor,
        ego_to_initial: torch.Tensor,
        bbox_corners: torch.Tensor,
        bbox_classes: torch.Tensor,
        bbox_view_masks: torch.Tensor,
        condition_keep: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        height: int,
        width: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        time_ids = torch.arange(
            sequence_length,
            device=bbox_corners.device,
            dtype=torch.long,
        )[None].expand(batch_size, -1)
        time_embedding = self.index_proj(time_ids.reshape(-1)).to(hidden_dtype)
        time_embedding = time_embedding.reshape(
            batch_size,
            sequence_length,
            -1,
        )

        plucker_map = self.build_first_frame_plucker_map(
            camera_intrinsics_norm,
            camera_to_ego,
            height,
            width,
            hidden_dtype,
        )
        camera_patch_tokens = einops.rearrange(
            plucker_map,
            "b v h w c -> b v (h w) c",
        ).contiguous()

        trajectory_tokens, trajectory_mask = self.trajectory_condition_encoder(
            ego_to_initial.to(hidden_dtype),
            time_embedding,
            condition_keep,
        )
        trajectory_tokens = trajectory_tokens[:, :, None].expand(
            batch_size,
            sequence_length,
            view_count,
            1,
            trajectory_tokens.shape[-1],
        )
        trajectory_mask = trajectory_mask[:, :, None].expand(
            batch_size,
            sequence_length,
            view_count,
            1,
        )

        box_tokens, box_mask = self.bbox_condition_encoder(
            bbox_corners.to(hidden_dtype),
            bbox_classes,
            bbox_view_masks,
            time_embedding,
            condition_keep,
        )

        null_tokens = self.null_condition_token.to(hidden_dtype).view(
            1,
            1,
            1,
            1,
            -1,
        ).expand(
            batch_size,
            sequence_length,
            view_count,
            1,
            -1,
        )
        null_mask = torch.ones(
            batch_size,
            sequence_length,
            view_count,
            1,
            device=bbox_corners.device,
            dtype=torch.bool,
        )

        condition_tokens = torch.cat(
            [null_tokens, trajectory_tokens, box_tokens],
            dim=3,
        )
        condition_mask = torch.cat(
            [null_mask, trajectory_mask, box_mask],
            dim=3,
        )
        condition_tokens = einops.rearrange(
            condition_tokens,
            "b t v l c -> (b t v) l c",
        )
        condition_mask = einops.rearrange(
            condition_mask,
            "b t v l -> (b t v) l",
        )
        return (
            camera_patch_tokens,
            condition_tokens,
            condition_mask,
            time_embedding,
        )

    def forward_temporal_block_and_mix_result(
        self,
        temporal_block: torch.nn.Module,
        mixer: torch.nn.Module,
        hidden_states: torch.Tensor,
        sequence_embedding: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        width: int,
        disable_temporal: torch.BoolTensor,
    ) -> torch.Tensor:
        temporal_hidden_states = hidden_states + sequence_embedding
        temporal_hidden_states = einops.rearrange(
            temporal_hidden_states,
            "(b t v) (h w) c -> (b v h) (t w) c",
            b=batch_size,
            v=view_count,
            w=width,
        )
        temporal_hidden_states = temporal_block(temporal_hidden_states)
        temporal_hidden_states = einops.rearrange(
            temporal_hidden_states,
            "(b v h) (t w) c -> (b t v) (h w) c",
            b=batch_size,
            v=view_count,
            t=sequence_length,
            w=width,
        )
        hidden_states = mixer(
            hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                *hidden_states.shape[1:],
            ),
            temporal_hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                *temporal_hidden_states.shape[1:],
            ),
            image_only_indicator=disable_temporal,
        )
        return hidden_states.flatten(0, 1)

    def forward_crossview_block_and_mix_result(
        self,
        crossview_block: torch.nn.Module,
        mixer: torch.nn.Module,
        hidden_states: torch.Tensor,
        camera_patch_embedding: torch.Tensor,
        condition_keep: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        width: int,
        height: int,
        crossview_attention_mask: torch.Tensor,
        disable_crossview: torch.BoolTensor,
    ) -> torch.Tensor:
        expected_camera_shape = (
            batch_size,
            view_count,
            hidden_states.shape[1],
            hidden_states.shape[2],
        )
        if camera_patch_embedding.shape != expected_camera_shape:
            raise ValueError(
                "camera_patch_embedding must match the DiT patch grid with shape "
                f"{expected_camera_shape}, got {tuple(camera_patch_embedding.shape)}."
            )
        if condition_keep.shape != (batch_size,):
            raise ValueError(
                f"condition_keep must be [{batch_size}], "
                f"got {tuple(condition_keep.shape)}."
            )

        view_cam_emb = camera_patch_embedding[:, None].to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        view_cam_emb = view_cam_emb * condition_keep.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).view(batch_size, 1, 1, 1, 1)
        crossview_hidden_states = einops.rearrange(
            hidden_states,
            "(b t v) n c -> b t v n c",
            b=batch_size,
            t=sequence_length,
            v=view_count,
        )
        crossview_hidden_states = crossview_hidden_states + view_cam_emb
        crossview_hidden_states = einops.rearrange(
            crossview_hidden_states,
            "b t v n c -> (b t v) n c",
        )

        attention_mask = crossview_attention_mask.to(hidden_states.device).bool()
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[None]
        if attention_mask.shape != (batch_size, view_count, view_count):
            raise ValueError(
                "crossview_attention_mask must be [B,V,V], "
                f"got {tuple(attention_mask.shape)}."
            )
        attention_mask = attention_mask.repeat_interleave(width, dim=2)
        attention_mask = attention_mask.repeat_interleave(width, dim=1)
        attention_mask = attention_mask.repeat_interleave(
            sequence_length * height,
            dim=0,
        )
        crossview_hidden_states = einops.rearrange(
            crossview_hidden_states,
            "(b t v) (h w) c -> (b t h) (v w) c",
            b=batch_size,
            t=sequence_length,
            v=view_count,
            w=width,
        )
        crossview_hidden_states = crossview_block(
            crossview_hidden_states,
            self_attention_mask=attention_mask,
        )
        crossview_hidden_states = einops.rearrange(
            crossview_hidden_states,
            "(b t h) (v w) c -> (b t v) (h w) c",
            b=batch_size,
            t=sequence_length,
            v=view_count,
            w=width,
        )
        hidden_states = mixer(
            hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                *hidden_states.shape[1:],
            ),
            crossview_hidden_states.reshape(
                batch_size,
                sequence_length * view_count,
                *crossview_hidden_states.shape[1:],
            ),
            image_only_indicator=disable_crossview,
        )
        return hidden_states.flatten(0, 1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor,
        camera_intrinsics_norm: torch.Tensor,
        camera_to_ego: torch.Tensor,
        ego_to_initial: torch.Tensor,
        bbox_corners: torch.Tensor,
        bbox_classes: torch.Tensor,
        bbox_view_masks: torch.Tensor,
        bev_map: torch.Tensor,
        crossview_attention_mask: torch.Tensor,
        condition_keep: torch.Tensor,
        disable_temporal: torch.BoolTensor,
        return_dict: bool = False,
    ):
        if sample.ndim != 6:
            raise ValueError(
                "sample must be [B,T,V,C,H,W], "
                f"got {tuple(sample.shape)}."
            )
        batch_size, sequence_length, view_count, _, latent_height, latent_width = sample.shape
        patch_size = self.config.patch_size
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError(
                f"latent size {(latent_height, latent_width)} must be divisible "
                f"by patch_size={patch_size}."
            )
        height = latent_height // patch_size
        width = latent_width // patch_size
        token_count = height * width

        hidden_states = sample.flatten(0, 2)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 2)
        pooled_projections = pooled_projections.flatten(0, 2)
        timestep = timestep.flatten(0, 2).to(hidden_states.device)

        hidden_states = self.pos_embed(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)

        (
            camera_patch_tokens,
            condition_tokens,
            condition_mask,
            base_time_embedding,
        ) = self.build_condition_inputs(
            camera_intrinsics_norm,
            camera_to_ego,
            ego_to_initial,
            bbox_corners,
            bbox_classes,
            bbox_view_masks,
            condition_keep,
            batch_size,
            sequence_length,
            view_count,
            height,
            width,
            hidden_states.dtype,
        )
        camera_patch_embedding = self.camera_crossview_proj(
            camera_patch_tokens.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        )
        bev_residuals = self.bev_control(
            bev_map.to(hidden_states.dtype),
            height,
            width,
            condition_keep,
        )
        disable_crossview = torch.zeros_like(disable_temporal, dtype=torch.bool)

        for layer_index, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                )

            if layer_index not in self.additional_layer_index:
                continue
            additional_index = self.additional_layer_index[layer_index]

            bev_residual = bev_residuals[additional_index]
            bev_residual = bev_residual[:, :, None].expand(
                batch_size,
                sequence_length,
                view_count,
                token_count,
                hidden_states.shape[-1],
            )
            hidden_states = hidden_states + einops.rearrange(
                bev_residual,
                "b t v n c -> (b t v) n c",
            )
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.cond_cross_blocks[additional_index],
                    hidden_states,
                    condition_tokens,
                    condition_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = self.cond_cross_blocks[additional_index](
                    hidden_states,
                    condition_tokens,
                    condition_mask,
                )

            temporal_embedding = base_time_embedding[:, :, None].expand(
                batch_size,
                sequence_length,
                view_count,
                hidden_states.shape[-1],
            )
            temporal_embedding = self.time_pos_embeds[additional_index](
                temporal_embedding.reshape(-1, hidden_states.shape[-1])
            ).unsqueeze(1)

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.forward_temporal_block_and_mix_result,
                    self.temporal_transformer_blocks[additional_index],
                    self.time_mixers[additional_index],
                    hidden_states,
                    temporal_embedding,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    disable_temporal,
                    use_reentrant=False,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.forward_crossview_block_and_mix_result,
                    self.crossview_transformer_blocks[additional_index],
                    self.view_mixers[additional_index],
                    hidden_states,
                    camera_patch_embedding,
                    condition_keep,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    height,
                    crossview_attention_mask,
                    disable_crossview,
                    use_reentrant=False,
                )
            else:
                hidden_states = self.forward_temporal_block_and_mix_result(
                    self.temporal_transformer_blocks[additional_index],
                    self.time_mixers[additional_index],
                    hidden_states,
                    temporal_embedding,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    disable_temporal,
                )
                hidden_states = self.forward_crossview_block_and_mix_result(
                    self.crossview_transformer_blocks[additional_index],
                    self.view_mixers[additional_index],
                    hidden_states,
                    camera_patch_embedding,
                    condition_keep,
                    batch_size,
                    sequence_length,
                    view_count,
                    width,
                    height,
                    crossview_attention_mask,
                    disable_crossview,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            batch_size,
            sequence_length,
            view_count,
            self.out_channels,
            latent_height,
            latent_width,
        )
        if return_dict:
            return {"noise_pred": output}
        return [output], encoder_hidden_states, pooled_projections
