from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter

from dwm.models.urope.urope import URoPEDotProductAttention


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(dtype=hidden_states.dtype)
        return normalized * self.weight.to(dtype=hidden_states.dtype)


class MultiheadAttention(torch.nn.Module):
    """Self-attention projection with URoPE applied around standard SDPA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim {} must be divisible by num_heads {}.".format(
                    embed_dim,
                    num_heads,
                )
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qk_norm = qk_norm

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
        if self.out_proj.bias is not None:
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        urope_attention: URoPEDotProductAttention,
        query_viewmats: Tensor,
        query_intrinsics: Tensor,
        source_viewmats: Tensor,
        source_intrinsics: Tensor,
        patch_width: int,
        patch_height: int,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if key.shape != value.shape:
            raise ValueError(
                "URoPE key and value must have the same shape."
            )
        if query.shape[0] != key.shape[0]:
            raise ValueError(
                "URoPE query and context batch sizes must match."
            )

        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(
            3,
            dim=0,
        )
        if self.in_proj_bias is None:
            q_bias = None
            k_bias = None
            v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(
                3,
                dim=0,
            )

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        batch_size, query_length = q.shape[:2]
        key_length = k.shape[1]
        q = q.reshape(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        )
        k = k.reshape(
            batch_size,
            key_length,
            self.num_heads,
            self.head_dim,
        )
        v = v.reshape(
            batch_size,
            key_length,
            self.num_heads,
            self.head_dim,
        )

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        output = urope_attention(
            q,
            k,
            v,
            query_viewmats=query_viewmats,
            query_intrinsics=query_intrinsics,
            source_viewmats=source_viewmats,
            source_intrinsics=source_intrinsics,
            patch_width=patch_width,
            patch_height=patch_height,
            attention_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = output.permute(0, 2, 1, 3).reshape(
            batch_size,
            query_length,
            self.embed_dim,
        )
        return F.linear(
            output,
            self.out_proj.weight,
            self.out_proj.bias,
        )

