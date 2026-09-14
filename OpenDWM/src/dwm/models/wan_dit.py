import math
from typing import Optional

import diffusers
import einops
import torch
import torch.nn.functional as F

from dwm.models.adapters import TemporalConditionImageAdapter, zero_module
from dwm.models.crossview_temporal import AlphaBlender, Mixer, VTSelfAttentionBlock


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_octaves: int = 8, start_octave: int = 0):
        super().__init__()
        self.num_octaves = int(num_octaves)
        self.start_octave = int(start_octave)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(
            self.start_octave,
            self.start_octave + self.num_octaves,
            device=coords.device,
            dtype=torch.float32,
        )
        multipliers = (2.0 ** octaves) * math.pi

        coords = coords.unsqueeze(-1).to(torch.float32)
        while multipliers.ndim < coords.ndim:
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            batch_size,
            num_points,
            dim * self.num_octaves,
        )
        cosines = torch.cos(scaled_coords).reshape(
            batch_size,
            num_points,
            dim * self.num_octaves,
        )

        result = torch.cat([sines, cosines], dim=-1)
        return result


class RayEncoder(torch.nn.Module):
    def __init__(
        self,
        cond_proj_dim: int,
        in_channels: int,
        pos_octaves: int = 8,
        pos_start_octave: int = 0,
        ray_octaves: int = 4,
        ray_start_octave: int = 0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.pos_octaves = int(pos_octaves)
        self.pos_start_octave = int(pos_start_octave)
        self.ray_octaves = int(ray_octaves)
        self.ray_start_octave = int(ray_start_octave)

        self.pos_encoding = PositionalEncoding(
            num_octaves=self.pos_octaves,
            start_octave=self.pos_start_octave,
        )
        self.ray_encoding = PositionalEncoding(
            num_octaves=self.ray_octaves,
            start_octave=self.ray_start_octave,
        )

        expected_cond_proj_dim = 3 * 2 * (self.pos_octaves + self.ray_octaves)
        self.cond_proj_dim = int(cond_proj_dim)

        if self.cond_proj_dim != expected_cond_proj_dim:
            raise ValueError(
                f"RayEncoder cond_proj_dim mismatch: got {self.cond_proj_dim}, "
                f"expected {expected_cond_proj_dim} "
                f"(= 3 * 2 * ({self.pos_octaves} + {self.ray_octaves}))"
            )

        self.proj = torch.nn.Linear(self.cond_proj_dim, self.in_channels, bias=False)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = rays_d.shape

        pos_enc = self.pos_encoding(rays_o.unsqueeze(1))
        pos_enc = pos_enc.view(batch_size, 1, 1, pos_enc.shape[-1])
        pos_enc = pos_enc.expand(batch_size, height, width, pos_enc.shape[-1])

        rays_d_flat = rays_d.view(batch_size, height * width, 3)
        ray_enc = self.ray_encoding(rays_d_flat)
        ray_enc = ray_enc.view(batch_size, height, width, ray_enc.shape[-1])

        x = torch.cat([pos_enc, ray_enc], dim=-1)
        x = self.proj(x)
        return x


def get_rays(
    camera_intrinsics: torch.Tensor,
    camera_transforms: torch.Tensor,
    target_size,
):
    device = camera_transforms.device
    dtype = camera_transforms.dtype

    camera_transforms_f32 = camera_transforms.to(dtype=torch.float32)
    camera_intrinsics_f32 = camera_intrinsics.to(dtype=torch.float32)

    if isinstance(target_size, int):
        height = target_size
        width = target_size
    else:
        height, width = target_size

    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device),
        torch.linspace(0, height - 1, height, device=device),
        indexing="ij",
    )
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5
    zs = torch.ones_like(i)

    points_coord = torch.stack([i, j, zs], dim=0)

    directions = torch.inverse(camera_intrinsics_f32) @ points_coord.unsqueeze(0)
    rays_d = camera_transforms_f32[:, :3, :3] @ directions
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True).clamp(min=1e-8)
    rays_d = rays_d.transpose(1, 2).contiguous().view(-1, height, width, 3).to(dtype=dtype)

    rays_o = camera_transforms[:, :3, 3].to(dtype=dtype)
    return rays_o, rays_d


class TemporalViewEncoder(torch.nn.Module):
    def __init__(
        self,
        inner_dim: int,
        temporal_downsample_factor: int = 4,
        cond_proj_dim: int = 72,
        pos_octaves: int = 8,
        pos_start_octave: int = 0,
        ray_octaves: int = 4,
        ray_start_octave: int = 0,
    ):
        super().__init__()

        self.inner_dim = int(inner_dim)
        self.temporal_downsample_factor = int(temporal_downsample_factor)

        self.rayencoder = RayEncoder(
            cond_proj_dim=cond_proj_dim,
            in_channels=self.inner_dim,
            pos_octaves=pos_octaves,
            pos_start_octave=pos_start_octave,
            ray_octaves=ray_octaves,
            ray_start_octave=ray_start_octave,
        )

        self.temporal_mix = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
            ),
            torch.nn.SiLU(),
            zero_module(
                torch.nn.Conv3d(
                    in_channels=self.inner_dim,
                    out_channels=self.inner_dim,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                )
            ),
        )

    def _resolve_downsample_factor(
        self,
        dense_sequence_length: int,
        target_sequence_length: int,
    ) -> int:
        if target_sequence_length <= 0:
            raise ValueError(
                f"target_sequence_length must be positive, got {target_sequence_length}"
            )

        if dense_sequence_length < target_sequence_length:
            raise ValueError(
                f"dense_sequence_length ({dense_sequence_length}) must be >= "
                f"target_sequence_length ({target_sequence_length})"
            )

        if dense_sequence_length % target_sequence_length == 0:
            return dense_sequence_length // target_sequence_length

        return self.temporal_downsample_factor

    def forward(
        self,
        camera_intrinsics_norm: torch.Tensor,
        camera2referego: torch.Tensor,
        batch_size: int,
        dense_sequence_length: int,
        view_count: int,
        patch_height: int,
        patch_width: int,
        target_sequence_length: int,
    ):
        rays_o, rays_d = get_rays(
            camera_intrinsics_norm.flatten(0, 2),
            camera2referego.flatten(0, 2),
            target_size=(patch_height, patch_width),
        )

        x = self.rayencoder(rays_o, rays_d)

        x = einops.rearrange(
            x,
            "(b t v) h w c -> (b v) c t h w",
            b=batch_size,
            t=dense_sequence_length,
            v=view_count,
            h=patch_height,
            w=patch_width,
        )

        x = self.temporal_mix(x)

        downsample_factor = self._resolve_downsample_factor(
            dense_sequence_length=dense_sequence_length,
            target_sequence_length=target_sequence_length,
        )

        if downsample_factor > 1:
            x = F.avg_pool3d(
                x,
                kernel_size=(downsample_factor, 1, 1),
                stride=(downsample_factor, 1, 1),
                padding=(0, 0, 0),
            )

        if x.shape[2] != target_sequence_length:
            x = F.adaptive_avg_pool3d(
                x,
                output_size=(target_sequence_length, patch_height, patch_width),
            )

        x = einops.rearrange(
            x,
            "(b v) c t h w -> (b v) (t h w) c",
            b=batch_size,
            v=view_count,
            t=target_sequence_length,
            h=patch_height,
            w=patch_width,
        )
        return x



class WanCrossviewBlock(torch.nn.Module):
    def __init__(
        self,
        inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_on_additional_modules=None,
        mixer_type: str = "AlphaBlender",
        merge_factor: float = 2.0,
        merge_strategy: str = "learned_with_images",
    ):
        super().__init__()

        self.attn = VTSelfAttentionBlock(
            inner_dim,
            inner_dim,
            num_attention_heads,
            attention_head_dim,
            qk_norm=qk_norm_on_additional_modules,
        )

        if mixer_type == "AlphaBlender":
            self.mixer = AlphaBlender(
                merge_factor,
                merge_strategy=merge_strategy,
            )
        else:
            self.mixer = Mixer(channel=inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        crossview_hidden_states: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
    ) -> torch.Tensor:
        del sequence_length

        hidden_states = hidden_states.view(
            batch_size,
            view_count,
            hidden_states.shape[1],
            hidden_states.shape[2],
        )
        crossview_hidden_states = crossview_hidden_states.view(
            batch_size,
            view_count,
            crossview_hidden_states.shape[1],
            crossview_hidden_states.shape[2],
        )

        if isinstance(self.mixer, AlphaBlender):
            image_only_indicator = torch.zeros(
                (batch_size, 1),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states = self.mixer(
                hidden_states,
                crossview_hidden_states,
                image_only_indicator=image_only_indicator,
            )
        else:
            hidden_states = self.mixer(
                hidden_states,
                crossview_hidden_states,
            )

        return hidden_states.flatten(0, 1)

class WanOutput(dict):
    @property
    def sample(self):
        return self["sample"]


class WanCrossviewConditionModel(diffusers.WanTransformer3DModel):
    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_channels=48,
        out_channels=48,
        num_layers=30,
        attention_head_dim=128,
        num_attention_heads=24,
        ffn_dim=14336,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        projection_class_embeddings_input_dim: int = 256,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = True,
        crossview_attention_type: str = "rowwise",
        merge_factor: float = 2.0,
        merge_strategy: str = "learned_with_images",
        crossview_block_layers: Optional[list] = None,
        crossview_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        qk_norm_on_additional_modules=None,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            **kwargs,
        )

        self.crossview_gradient_checkpointing = bool(crossview_gradient_checkpointing)
        self.gradient_checkpointing = False

        self.enable_crossview = bool(enable_crossview)
        self.crossview_attention_type = str(crossview_attention_type)
        self.crossview_block_layers = [int(i) for i in (crossview_block_layers or [])]

        self.merge_factor = float(merge_factor)
        self.merge_strategy = str(merge_strategy)
        self.mixer_type = str(mixer_type)
        self.qk_norm_on_additional_modules = qk_norm_on_additional_modules
        self.projection_class_embeddings_input_dim = int(projection_class_embeddings_input_dim)

        self.inner_dim = self.config.attention_head_dim * self.config.num_attention_heads

        adapter_config = dict(condition_image_adapter_config or {})
        adapter_depth = adapter_config.pop("depth", None)
        block_out_channels = adapter_config.pop("block_out_channels", None)
        adapter_in_channels = adapter_config.pop("in_channels", 6)
        adapter_hidden_channels = adapter_config.pop("hidden_channels", self.inner_dim)
        adapter_out_channels = adapter_config.pop("out_channels", None)
        temporal_downsample_factor = adapter_config.pop("temporal_downsample_factor", 4)

        if adapter_depth is None and block_out_channels is not None:
            adapter_depth = len(block_out_channels)
        if adapter_depth is None:
            adapter_depth = self.config.num_layers

        if adapter_out_channels is not None and int(adapter_out_channels) != self.inner_dim:
            raise ValueError(
                f"condition_image_adapter_config['out_channels'] must be {self.inner_dim}, "
                f"but got {adapter_out_channels}"
            )

        if len(adapter_config) > 0:
            raise ValueError(
                f"Unsupported condition_image_adapter_config keys: {sorted(adapter_config.keys())}"
            )

        if condition_image_adapter_config is not None:
            self.condition_image_adapter = TemporalConditionImageAdapter(
                in_channels=int(adapter_in_channels),
                out_channels=self.inner_dim,
                depth=int(adapter_depth),
                hidden_channels=int(adapter_hidden_channels),
                temporal_downsample_factor=int(temporal_downsample_factor),
            )
        else:
            self.condition_image_adapter = None

        self.index_proj = diffusers.models.embeddings.Timesteps(self.inner_dim, True, 0)

        self.view_cam_encoder = TemporalViewEncoder(
            inner_dim=self.inner_dim,
            temporal_downsample_factor=int(temporal_downsample_factor),
            cond_proj_dim=72,
        )

        self.crossview_layer_to_index = {
            layer_id: idx for idx, layer_id in enumerate(self.crossview_block_layers)
        }

        self.view_pos_embeds = torch.nn.ModuleList()
        self.crossview_modules = torch.nn.ModuleList()

        if self.enable_crossview:
            for _ in range(len(self.crossview_block_layers)):
                self.view_pos_embeds.append(
                    diffusers.models.embeddings.TimestepEmbedding(
                        self.inner_dim,
                        self.inner_dim * 4,
                        out_dim=self.inner_dim,
                    )
                )
                self.crossview_modules.append(
                    WanCrossviewBlock(
                        inner_dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                        qk_norm_on_additional_modules=self.qk_norm_on_additional_modules,
                        mixer_type=self.mixer_type,
                        merge_factor=self.merge_factor,
                        merge_strategy=self.merge_strategy,
                    )
                )

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if module is self:
            self.gradient_checkpointing = value
    def _forward_wan_block(
        self,
        block,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        timestep_proj: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        return block(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
        )

    def _run_wan_block(
        self,
        block,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        timestep_proj: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_wan_block,
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                use_reentrant=False,
            )

        return self._forward_wan_block(
            block,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
        )

    def _run_crossview_block(
        self,
        hidden_states: torch.Tensor,
        cross_module,
        view_emb: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        width: int,
        height: int,
        crossview_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.training and self.crossview_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_crossview,
                hidden_states,
                cross_module,
                view_emb,
                batch_size,
                sequence_length,
                view_count,
                width,
                height,
                crossview_attention_mask,
                use_reentrant=False,
            )

        return self._forward_crossview(
            hidden_states=hidden_states,
            cross_module=cross_module,
            view_emb=view_emb,
            batch_size=batch_size,
            sequence_length=sequence_length,
            view_count=view_count,
            width=width,
            height=height,
            crossview_attention_mask=crossview_attention_mask,
        )   
    def _run_condition_embedder_no_fsdp_iter(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        target_dtype: Optional[torch.dtype] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        cond = self.condition_embedder

        timestep = cond.timesteps_proj(timestep)

        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        if target_dtype is None:
            if encoder_hidden_states is not None:
                target_dtype = encoder_hidden_states.dtype
            else:
                target_dtype = timestep.dtype

        if timestep.dtype != target_dtype and target_dtype != torch.int8:
            timestep = timestep.to(target_dtype)

        temb = cond.time_embedder(timestep)

        if encoder_hidden_states is not None:
            temb = temb.type_as(encoder_hidden_states)

        timestep_proj = cond.time_proj(cond.act_fn(temb))

        if encoder_hidden_states is not None:
            encoder_hidden_states = cond.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states
    
    def _build_view_cam_emb(
        self,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        patch_height: int,
        patch_width: int,
        camera_intrinsics_norm: Optional[torch.Tensor],
        camera2referego: Optional[torch.Tensor],
    ):
        if camera_intrinsics_norm is None or camera2referego is None:
            return None

        dense_sequence_length = int(camera_intrinsics_norm.shape[1])

        return self.view_cam_encoder(
            camera_intrinsics_norm=camera_intrinsics_norm,
            camera2referego=camera2referego,
            batch_size=batch_size,
            dense_sequence_length=dense_sequence_length,
            view_count=view_count,
            patch_height=patch_height,
            patch_width=patch_width,
            target_sequence_length=sequence_length,
        )

    def _forward_crossview(
        self,
        hidden_states: torch.Tensor,
        cross_module: WanCrossviewBlock,
        view_emb: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        view_count: int,
        width: int,
        height: int,
        crossview_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        crossview_hidden_states = hidden_states + view_emb

        if self.crossview_attention_type == "full":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states,
                "(b v) (t h w) c -> (b t) (h v w) c",
                b=batch_size,
                t=sequence_length,
                v=view_count,
                h=height,
                w=width,
            )
            crossview_hidden_states = cross_module.attn(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask,
            )
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states,
                "(b t) (h v w) c -> (b v) (t h w) c",
                b=batch_size,
                t=sequence_length,
                v=view_count,
                h=height,
                w=width,
            )

        elif self.crossview_attention_type == "rowwise":
            rowwise_mask = crossview_attention_mask
            if rowwise_mask is not None:
                rowwise_mask = rowwise_mask.repeat_interleave(width, dim=2)
                rowwise_mask = rowwise_mask.repeat_interleave(width, dim=1)
                rowwise_mask = rowwise_mask.repeat_interleave(sequence_length * height, dim=0)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states,
                "(b v) (t h w) c -> (b t h) (v w) c",
                b=batch_size,
                t=sequence_length,
                v=view_count,
                h=height,
                w=width,
            )
            crossview_hidden_states = cross_module.attn(
                crossview_hidden_states,
                self_attention_mask=rowwise_mask,
            )
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states,
                "(b t h) (v w) c -> (b v) (t h w) c",
                b=batch_size,
                t=sequence_length,
                v=view_count,
                h=height,
                w=width,
            )

        else:
            raise ValueError(
                f"Unsupported crossview_attention_type={self.crossview_attention_type}"
            )

        hidden_states = cross_module(
            hidden_states=hidden_states,
            crossview_hidden_states=crossview_hidden_states,
            batch_size=batch_size,
            sequence_length=sequence_length,
            view_count=view_count,
        )
        return hidden_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image=None,
        condition_image_tensor: Optional[torch.Tensor] = None,
        camera_intrinsics_norm: Optional[torch.Tensor] = None,
        camera2referego: Optional[torch.Tensor] = None,
        crossview_attention_mask: Optional[torch.Tensor] = None,
        view_count: int = 1,
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        bs_view, _, num_frames, height, width = hidden_states.shape

        patch_size_t, patch_size_h, patch_size_w = self.config.patch_size
        post_patch_num_frames = num_frames // patch_size_t
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w
        spatial_token_count = patch_height * patch_width

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            timestep_flat = timestep.reshape(-1)
        else:
            timestep_seq_len = None
            timestep_flat = timestep

        embed_target_dtype = hidden_states.dtype
        if encoder_hidden_states is not None:
            embed_target_dtype = encoder_hidden_states.dtype

        temb, timestep_proj, encoder_hidden_states = self._run_condition_embedder_no_fsdp_iter(
            timestep=timestep_flat,
            encoder_hidden_states=encoder_hidden_states,
            target_dtype=embed_target_dtype,
            timestep_seq_len=timestep_seq_len,
        )

        if timestep_seq_len is None:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))

        condition_residuals = None
        if self.condition_image_adapter is not None and condition_image_tensor is not None:
            condition_residuals = self.condition_image_adapter(
                condition_image_tensor=condition_image_tensor,
                target_sequence_length=post_patch_num_frames,
                target_patch_size=(patch_height, patch_width),
            )

        crossview_batch_size = None
        view_cam_emb = None

        if self.enable_crossview and view_count > 1:
            if bs_view % view_count != 0:
                raise ValueError(
                    f"bs_view ({bs_view}) must be divisible by "
                    f"view_count ({view_count}) when enable_crossview=True."
                )

            crossview_batch_size = bs_view // view_count

            view_cam_emb = self._build_view_cam_emb(
                batch_size=crossview_batch_size,
                sequence_length=post_patch_num_frames,
                view_count=view_count,
                patch_height=patch_height,
                patch_width=patch_width,
                camera_intrinsics_norm=camera_intrinsics_norm,
                camera2referego=camera2referego,
            )

        for layer_id, block in enumerate(self.blocks):
            if condition_residuals is not None and layer_id < len(condition_residuals):
                hidden_states = hidden_states + condition_residuals[layer_id]

            hidden_states = self._run_wan_block(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )

            if (
                self.enable_crossview and view_count > 1
                and layer_id in self.crossview_layer_to_index
            ):
                cross_idx = self.crossview_layer_to_index[layer_id]

                view_ids = torch.arange(
                    view_count,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                view_emb = self.view_pos_embeds[cross_idx](
                    self.index_proj(view_ids).to(hidden_states.dtype)
                )
                view_emb = view_emb.unsqueeze(0).unsqueeze(2).repeat(
                    crossview_batch_size * post_patch_num_frames,
                    1,
                    spatial_token_count,
                    1,
                )
                view_emb = einops.rearrange(
                    view_emb,
                    "(b t) v hw c -> (b v) (t hw) c",
                    b=crossview_batch_size,
                    t=post_patch_num_frames,
                    v=view_count,
                )

                if view_cam_emb is not None:
                    view_emb = view_emb + view_cam_emb

                hidden_states = self._run_crossview_block(
                    hidden_states,
                    self.crossview_modules[cross_idx],
                    view_emb,
                    crossview_batch_size,
                    post_patch_num_frames,
                    view_count,
                    patch_width,
                    patch_height,
                    crossview_attention_mask,
                )

        if temb.ndim != 3:
            raise ValueError(
                f"Expected token-level temb with ndim=3, got shape={tuple(temb.shape)}"
            )

        scale_shift = self.scale_shift_table.to(temb.device)
        if scale_shift.ndim == 2:
            scale_shift = scale_shift.unsqueeze(0)

        shift, scale = (scale_shift + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        out_channels = self.config.out_channels

        hidden_states = hidden_states.reshape(
            bs_view,
            post_patch_num_frames,
            patch_height,
            patch_width,
            patch_size_t,
            patch_size_h,
            patch_size_w,
            out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return WanOutput(sample=hidden_states)
