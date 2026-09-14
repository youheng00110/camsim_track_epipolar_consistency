import math
from typing import Optional

import einops
import torch


class FourierFeatures(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_freqs: int = 4,
        include_input: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        frequencies = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", frequencies, persistent=False)
        self.out_dim = self.input_dim * self.num_freqs * 2
        if self.include_input:
            self.out_dim += self.input_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"FourierFeatures expects last dim {self.input_dim}, "
                f"got {tuple(values.shape)}."
            )
        frequencies = self.freq_bands.to(values)
        scaled = values.unsqueeze(-1) * frequencies * math.pi
        sine = torch.sin(scaled).flatten(-2)
        cosine = torch.cos(scaled).flatten(-2)
        if self.include_input:
            return torch.cat([values, sine, cosine], dim=-1)
        return torch.cat([sine, cosine], dim=-1)


class TemporalTokenBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        self.norm_attn = torch.nn.LayerNorm(hidden_dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=True,
        )
        self.norm_mlp = torch.nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_dim, hidden_dim),
        )
        torch.nn.init.zeros_(self.attn.out_proj.weight)
        torch.nn.init.zeros_(self.attn.out_proj.bias)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.norm_attn(hidden_states)
        attention_output = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.norm_mlp(hidden_states))
        return hidden_states


class TemporalBBoxConditionEncoder(torch.nn.Module):
    """
    MagicDrive-style box encoding adapted to stable clip-level slots.

    The geometric/class projection intentionally keeps the parameter names and
    shapes of the previous BBoxTokenEncoder. A legacy checkpoint can therefore
    initialize this base encoder. The new temporal blocks then model every
    stable instance slot across the full clip.
    """

    def __init__(
        self,
        out_dim: int,
        num_classes: int = 10,
        points_per_box: int = 8,
        num_freqs: int = 4,
        hidden_dim: int = 768,
        class_dim: int = 768,
        temporal_depth: int = 1,
        temporal_heads: int = 8,
        position_min=(-80.0, -80.0, -5.0),
        position_range=(160.0, 160.0, 10.0),
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.num_classes = int(num_classes)
        self.points_per_box = int(points_per_box)
        self.register_buffer(
            "position_min",
            torch.tensor(position_min, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "position_range",
            torch.tensor(position_range, dtype=torch.float32),
            persistent=False,
        )

        self.embedder = FourierFeatures(input_dim=3, num_freqs=num_freqs)
        position_dim = self.embedder.out_dim * self.points_per_box
        self.class_embedding = torch.nn.Embedding(self.num_classes, class_dim)
        self.bbox_proj = torch.nn.Linear(position_dim, hidden_dim)
        self.second_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + class_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, self.out_dim),
        )
        self.after_proj = torch.nn.Linear(self.out_dim, self.out_dim, bias=True)

        self.box_type_embedding = torch.nn.Parameter(torch.zeros(self.out_dim))
        torch.nn.init.normal_(self.box_type_embedding, std=0.02)
        self.temporal_blocks = torch.nn.ModuleList(
            [
                TemporalTokenBlock(
                    hidden_dim=self.out_dim,
                    num_heads=temporal_heads,
                )
                for _ in range(int(temporal_depth))
            ]
        )

    def forward(
        self,
        corners: torch.Tensor,
        classes: torch.Tensor,
        view_masks: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if corners.ndim != 6 or corners.shape[-2:] != (self.points_per_box, 3):
            raise ValueError(
                "corners must be [B,T,V,S,8,3], "
                f"got {tuple(corners.shape)}."
            )
        if classes.shape != corners.shape[:4]:
            raise ValueError(
                f"classes must be {tuple(corners.shape[:4])}, "
                f"got {tuple(classes.shape)}."
            )
        if view_masks.shape != corners.shape[:4]:
            raise ValueError(
                f"view_masks must be {tuple(corners.shape[:4])}, "
                f"got {tuple(view_masks.shape)}."
            )

        batch_size, sequence_length, view_count, slot_count = corners.shape[:4]
        if time_embedding.shape != (batch_size, sequence_length, self.out_dim):
            raise ValueError(
                "time_embedding must be [B,T,C], "
                f"got {tuple(time_embedding.shape)}."
            )

        shared_corners = corners[:, :, 0]
        shared_classes = classes[:, :, 0].long().clamp(0, self.num_classes - 1)
        presence = shared_corners.abs().amax(dim=(-1, -2)) > 1e-6

        position_min = self.position_min.to(shared_corners)
        position_range = self.position_range.to(shared_corners)
        normalized = (
            shared_corners - position_min.view(1, 1, 1, 1, 3)
        ) / position_range.view(1, 1, 1, 1, 3)
        normalized = normalized.clamp(0.0, 1.0)

        position_features = self.embedder(normalized).flatten(-2)
        presence_float = presence.unsqueeze(-1).to(position_features.dtype)
        geometry = torch.nn.functional.silu(self.bbox_proj(position_features))
        class_features = self.class_embedding(shared_classes)
        tokens = self.second_linear(torch.cat([geometry, class_features], dim=-1))
        tokens = self.after_proj(tokens)
        tokens = tokens + time_embedding[:, :, None]
        tokens = tokens + self.box_type_embedding.view(1, 1, 1, -1)
        tokens = tokens * presence_float

        slot_sequences = einops.rearrange(tokens, "b t s c -> (b s) t c")
        slot_presence = einops.rearrange(presence, "b t s -> (b s) t")
        valid_slots = slot_presence.any(dim=1)

        temporal_padding_mask = ~slot_presence
        empty_slots = ~valid_slots

        temporal_padding_mask = temporal_padding_mask.clone()
        temporal_padding_mask[empty_slots, 0] = False

        temporal_output = slot_sequences
        for block in self.temporal_blocks:
            temporal_output = block(
                temporal_output,
                temporal_padding_mask,
            )

        temporal_output = temporal_output.masked_fill(
            (~slot_presence).unsqueeze(-1),
            0.0,
        )

        temporal_output = einops.rearrange(
            temporal_output,
            "(b s) t c -> b t s c",
            b=batch_size,
            s=slot_count,
        )
        tokens_by_view = temporal_output[:, :, None].expand(
            batch_size,
            sequence_length,
            view_count,
            slot_count,
            self.out_dim,
        )
        visible = view_masks > 0
        visible = visible & presence[:, :, None]
        visible = visible & condition_keep[:, None, None, None].bool()
        return tokens_by_view, visible


class EgoTrajectoryConditionEncoder(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_freqs: int = 4,
        translation_scale: float = 10.0,
    ):
        super().__init__()
        if translation_scale <= 0:
            raise ValueError("translation_scale must be positive.")
        self.out_dim = int(out_dim)
        self.translation_scale = float(translation_scale)
        self.embedder = FourierFeatures(input_dim=3, num_freqs=num_freqs)
        input_dim = self.embedder.out_dim * 4
        self.pose_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.out_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.out_dim, self.out_dim),
        )
        self.trajectory_type_embedding = torch.nn.Parameter(
            torch.zeros(self.out_dim)
        )
        torch.nn.init.normal_(self.trajectory_type_embedding, std=0.02)

    def forward(
        self,
        ego_to_initial: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ego_to_initial.ndim != 4 or ego_to_initial.shape[-2:] != (4, 4):
            raise ValueError(
                "ego_to_initial must be [B,T,4,4], "
                f"got {tuple(ego_to_initial.shape)}."
            )
        batch_size, sequence_length = ego_to_initial.shape[:2]
        if time_embedding.shape != (batch_size, sequence_length, self.out_dim):
            raise ValueError(
                "time_embedding must be [B,T,C], "
                f"got {tuple(time_embedding.shape)}."
            )

        pose = ego_to_initial[:, :, :3, :4].clone()
        pose[:, :, :3, 3] = pose[:, :, :3, 3] / self.translation_scale
        pose_columns = pose.transpose(-2, -1)
        pose_features = self.embedder(pose_columns).flatten(-2)
        tokens = self.pose_mlp(pose_features)
        tokens = tokens + time_embedding
        tokens = tokens + self.trajectory_type_embedding.view(1, 1, -1)

        tokens = tokens[:, :, None]
        mask = condition_keep[:, None, None].bool().expand(
            batch_size,
            sequence_length,
            1,
        )
        return tokens, mask


class PluckerRigConditionEncoder(torch.nn.Module):
    """Keep first-frame Plucker features aligned with the full DiT patch grid."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.camera_type_embedding = torch.nn.Parameter(
            torch.zeros(self.hidden_dim)
        )
        torch.nn.init.normal_(self.camera_type_embedding, std=0.02)

    def forward(self, plucker_map: torch.Tensor) -> torch.Tensor:
        if plucker_map.ndim != 5:
            raise ValueError(
                "plucker_map must be [B,V,H,W,C], "
                f"got {tuple(plucker_map.shape)}."
            )
        _, _, _, _, hidden_dim = plucker_map.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"plucker hidden dim must be {self.hidden_dim}, got {hidden_dim}."
            )

        tokens = einops.rearrange(
            plucker_map,
            "b v h w c -> b v (h w) c",
        )
        tokens = self.norm(tokens)
        tokens = tokens + self.camera_type_embedding.view(1, 1, 1, -1)
        return tokens


class ConditionCrossAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm_query = torch.nn.LayerNorm(self.hidden_dim)
        self.norm_context = torch.nn.LayerNorm(self.hidden_dim)
        self.query_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        torch.nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or condition_tokens.ndim != 3:
            raise ValueError(
                "hidden_states and condition_tokens must both be 3D tensors."
            )
        if condition_mask.shape != condition_tokens.shape[:2]:
            raise ValueError(
                f"condition_mask must be {tuple(condition_tokens.shape[:2])}, "
                f"got {tuple(condition_mask.shape)}."
            )

        residual = hidden_states
        normalized_context = self.norm_context(condition_tokens)
        query = self.query_proj(self.norm_query(hidden_states))
        key = self.key_proj(normalized_context)
        value = self.value_proj(normalized_context)

        batch_size, query_length = query.shape[:2]
        context_length = key.shape[1]
        query = query.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            context_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_length,
            device=query.device,
            dtype=query.dtype,
        )
        attention_mask = attention_mask.masked_fill(
            (~condition_mask.bool())[:, None, None],
            torch.finfo(query.dtype).min,
        )
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.hidden_dim,
        )
        return residual + self.out_proj(attention_output)


class TemporalBEVResidualAdapter(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.depth = int(depth)
        self.stem = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            torch.nn.SiLU(),
            torch.nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            torch.nn.SiLU(),
            torch.nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
            ),
            torch.nn.SiLU(),
        )
        self.projections = torch.nn.ModuleList(
            [
                torch.nn.Conv3d(
                    hidden_channels,
                    self.out_channels,
                    kernel_size=1,
                )
                for _ in range(self.depth)
            ]
        )
        for projection in self.projections:
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.zeros_(projection.bias)

    def forward(
        self,
        bev_map: torch.Tensor,
        target_height: int,
        target_width: int,
        condition_keep: torch.Tensor,
    ) -> list[torch.Tensor]:
        if bev_map.ndim != 5:
            raise ValueError(
                "bev_map must be [B,T,C,H,W], "
                f"got {tuple(bev_map.shape)}."
            )
        if bev_map.shape[2] != self.in_channels:
            raise ValueError(
                f"bev_map must have {self.in_channels} channels, "
                f"got {bev_map.shape[2]}."
            )

        batch_size, sequence_length = bev_map.shape[:2]
        keep = condition_keep[:, None, None, None, None].to(bev_map.dtype)
        features = bev_map * keep
        features = einops.rearrange(features, "b t c h w -> b c t h w")
        features = torch.nn.functional.adaptive_avg_pool3d(
            features,
            output_size=(sequence_length, target_height, target_width),
        )
        features = self.stem(features)

        outputs = []
        for projection in self.projections:
            residual = projection(features) * keep
            residual = einops.rearrange(
                residual,
                "b c t h w -> b t (h w) c",
            ).contiguous()
            outputs.append(residual)
        return outputs
