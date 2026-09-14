from typing import Optional

import diffusers.models.attention
import torch

from dwm.models.urope.mha import MultiheadAttention
from dwm.models.urope.urope import URoPEDotProductAttention


class VTURoPEAttentionBlock(torch.nn.Module):
    """OpenDWM cross-view self-attention block using URoPE."""

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm=None,
        urope_config: Optional[dict] = None,
    ):
        super().__init__()
        expected_dim = num_attention_heads * attention_head_dim
        if time_mix_inner_dim != expected_dim:
            raise ValueError(
                "time_mix_inner_dim must equal heads * head_dim: {} != {}.".format(
                    time_mix_inner_dim,
                    expected_dim,
                )
            )

        self.norm_in = torch.nn.LayerNorm(dim)
        self.ff_in = diffusers.models.attention.FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )
        self.norm1 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.norm3 = torch.nn.LayerNorm(time_mix_inner_dim)
        self.ff = diffusers.models.attention.FeedForward(
            time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.attn1 = MultiheadAttention(
            embed_dim=time_mix_inner_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm is not None,
        )
        config = {
            "min_depth": 2.0,
            "max_depth": 20.0,
            "freq_base": 100.0,
            "freq_scale": 1.0,
            "group_size": 4,
            "leaveout_head": 0,
            "camera_convention": "opencv",
            **(urope_config or {}),
        }
        self.urope_attention = URoPEDotProductAttention(
            head_num=num_attention_heads,
            head_dim=attention_head_dim,
            **config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        context_hidden_states: torch.Tensor,
        query_viewmats: torch.Tensor,
        query_intrinsics: torch.Tensor,
        source_viewmats: torch.Tensor,
        source_intrinsics: torch.Tensor,
        patch_height: int,
        patch_width: int,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + query_residual

        context_residual = context_hidden_states
        context_hidden_states = self.norm_in(context_hidden_states)
        context_hidden_states = self.ff_in(context_hidden_states)
        context_hidden_states = (
            context_hidden_states + context_residual
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_context_hidden_states = self.norm1(
            context_hidden_states
        )
        attention_output = self.attn1(
            norm_hidden_states,
            norm_context_hidden_states,
            norm_context_hidden_states,
            urope_attention=self.urope_attention,
            query_viewmats=query_viewmats,
            query_intrinsics=query_intrinsics,
            source_viewmats=source_viewmats,
            source_intrinsics=source_intrinsics,
            patch_width=patch_width,
            patch_height=patch_height,
            attention_mask=self_attention_mask,
        )
        hidden_states = hidden_states + attention_output

        norm_hidden_states = self.norm3(hidden_states)
        feed_forward_output = self.ff(norm_hidden_states)
        return hidden_states + feed_forward_output

