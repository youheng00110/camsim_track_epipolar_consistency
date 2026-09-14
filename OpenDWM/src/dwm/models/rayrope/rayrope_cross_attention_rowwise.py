
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, List

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
# from pos_enc.timing_utils import time_block
from torch.profiler import profile, record_function, ProfilerActivity

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
# MAX_LOG_DEPTH = math.log(MAX_DEPTH)
MAX_ASINH_DEPTH = math.asinh(MAX_DEPTH)

MAX_D_F = 10.0
MAX_ASINH_D_F = math.asinh(MAX_D_F)

class RayRoPE_DotProductAttention_Cross_Rowwise(torch.nn.Module):
    """
    Rowwise cross-attention with RayRoPE positional encoding for multi-view patches.

    这个版本不是 full-grid 版，而是 rowwise 版：
    - 一次 attention 只处理 token grid 里的单独一行
    - 因此每个 camera 当前只有 patches_x 个 token
    - 整个序列长度应为 num_cameras * patches_x

    Args:
        head_dim:
            每个 attention head 的维度。
            当前实现支持 partial RoPE：
            - 只对前 rope_head_dim 维做旋转
            - 剩余维度 feats_tail 保持不变
            所以 head_dim 不再要求必须被 rope_coord_dim * 2 整除，
            但必须满足 head_dim >= rope_mat_dim 才能至少容纳 1 组 rope 频率。
        pos_enc_type:
            指定使用哪些几何点、以及如何变到 query frame。
            例如 "d_pj+0_3d" 表示：
            - d_pj: 使用预测深度点，并在 query 相机下做投影坐标表达
            - 0_3d: 使用相机中心，并在 query 相机下做 3D SE(3) 表达
        num_rays_per_patch:
            每个 patch 采样几条 ray。
        depth_type:
            当前默认是 "predict_dsig"。
            如果以后改成 "known+predict_dsig"，需要另外实现 rowwise 的 depth map sampling。
        apply_vo:
            True 时也对 V / output 做 rope；False 时只对 Q/K 做 rope。
    """

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y_total: int,
        image_width: int,
        image_height: int,
        pos_enc_type: str = 'd_pj+0_3d',
        num_rays_per_patch: int = 3,
        depth_type: str = 'predict_dsig',
        denc_type: str = 'd',
        freq_base: float = 3.0,
        apply_vo: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y_total = patches_y_total   # 原始 token grid 的总高度
        self.patches_y = 1                       # rowwise 模式下一次只处理 1 行
        self.num_patches = patches_x             # 当前每个 camera 只含这一行上的 width 个 token
        self.image_width = image_width
        self.image_height = image_height
        self.pos_enc_type = pos_enc_type
        self.depth_type = depth_type
        self.denc_type = denc_type
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base
        self.apply_vo = apply_vo
        # self.last_positions = None

        # parse pos_enc_type
        self.parse_pos_enc_type(pos_enc_type)
        # rope_coord_dim: 几何坐标维度，不是特征维度
        # rope_mat_dim: 一组 rope 旋转需要占用的 head 维度 = 2 * rope_coord_dim
        # num_rope_freqs: 当前 head_dim 最多能容纳多少组频率
        self.rope_coord_dim = 3 * self.use_p0 + self.num_rays_per_patch * 3 * (int(self.use_pd) + int(self.use_pinf))
        self.rope_mat_dim = 2 * self.rope_coord_dim
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim
        if self.num_rope_freqs < 1:
            raise ValueError(
                f"head_dim={self.head_dim} is too small for rope_mat_dim={self.rope_mat_dim}"
            )
        self.rope_head_dim = self.num_rope_freqs * self.rope_mat_dim
        print(f"head_dim: {head_dim}, rope_enc_dim: {self.head_dim}, rope_coord_dim: {self.rope_coord_dim}, num_rope_freqs: {self.num_rope_freqs}")

        self.context_depths = None

        if self.num_rays_per_patch == 3:
            self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.num_rays_per_patch == 2:
            self.offsets = [[0.0, 0.0], [1.0, 1.0]]
        else:
            self.offsets = [[0.5, 0.5]]

    def parse_pos_enc_type(self, pos_enc_type: str):
        self.use_p0 = False
        self.p0_type = 'none'
        self.use_pd = False
        self.pd_type = 'none'
        self.use_pinf = False
        self.pinf_type = 'none'

        for part in pos_enc_type.split('+'):
            assert '_' in part, f"pos_enc_type part {part} is invalid."
            point, ptype = part.split('_', 1)
            if point == '0':
                self.use_p0 = True
                self.p0_type = ptype
            elif point == 'd':
                self.use_pd = True
                self.pd_type = ptype
            elif point == 'inf':
                self.use_pinf = True
                self.pinf_type = ptype

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        predicted_d: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        predicted_d_kv: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        timing_enabled: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        
        # if predicted_d_kv is None:
        #     predicted_d_kv = predicted_d
            
        # with time_block("attention_total", timing_enabled):
            # with time_block("prepare_enc", timing_enabled):
        # positions_debug = {}
        apply_fn_q, all_apply_fns_kv, apply_fn_o = self._prepare_apply_fns(
            predicted_d=predicted_d,
            predicted_d_kv=predicted_d_kv,
            # positions_collector=positions_debug,
        )
        # self.last_positions = positions_debug

        output = self.rayrope_dot_product_attention(
            q,
            k,
            v,
            num_cameras=self.num_cameras,
            apply_fn_q=apply_fn_q,
            all_apply_fns_kv=all_apply_fns_kv,
            apply_fn_o=apply_fn_o,
            apply_vo=self.apply_vo,
            timing_enabled=timing_enabled,
            **kwargs,
        )

        return output

    def _precompute_and_cache_apply_fns(
        self,
        w2cs,
        Ks,
        w2cs_kv,
        Ks_kv,
        row_indices,
        context_depths=None,
    ):
        # row_indices: 当前 batch 中每个 rowwise 样本，对应原始 token grid 的第几行
        self.row_indices = row_indices
        geo_dtype = torch.float32
        w2cs = w2cs.to(dtype=geo_dtype)
        Ks = Ks.to(dtype=geo_dtype)
        w2cs_kv = w2cs_kv.to(dtype=geo_dtype)
        Ks_kv = Ks_kv.to(dtype=geo_dtype)
        (batch, num_cameras, _, _) = w2cs.shape
        (batch_kv, num_cameras_kv, _, _) = w2cs_kv.shape
        assert batch == batch_kv, "Batch size for Q and KV must be the same."

        self.batch = batch
        self.num_cameras = num_cameras
        self.num_cameras_kv = num_cameras_kv

        # 注意：context_depths 是给 KV 视角组的，不是给 Q 组的
        self.context_depths = context_depths

        # Q 组几何
        self.w2cs = w2cs
        self.c2ws = _invert_SE3(w2cs)
        Ks_norm = normalize_K(Ks, self.image_width, self.image_height)

        # KV 组几何
        self.w2cs_kv = w2cs_kv
        self.c2ws_kv = _invert_SE3(w2cs_kv)
        Ks_norm_kv = normalize_K(Ks_kv, self.image_width, self.image_height)

        # PRoPE / RayRoPE 使用的齐次投影矩阵
        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        self.P_T = self.P.transpose(-1, -2)
        self.P_inv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws,
            _lift_K(_invert_K(Ks_norm)),
        )

        self.P_kv = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm_kv), w2cs_kv)
        self.P_T_kv = self.P_kv.transpose(-1, -2)
        self.P_inv_kv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws_kv,
            _lift_K(_invert_K(Ks_norm_kv)),
        )

        # 这里先缓存与 predicted_d 无关的静态几何量
        # pd_world / pd_world_kv 依赖当前 forward 的 predicted_d，
        # 所以要放到 _prepare_apply_fns() 里动态计算
        self.p0_world = _get_cam_centers(self.c2ws, self.num_patches)
        self.p0_world_kv = _get_cam_centers(self.c2ws_kv, self.num_patches)

        self.pinf_world = _get_point_coords_rowwise(
            self.P_inv,
            self.patches_x,
            self.patches_y_total,
            self.row_indices,
            self.offsets,
        )

        self.pinf_world_kv = _get_point_coords_rowwise(
            self.P_inv_kv,
            self.patches_x,
            self.patches_y_total,
            self.row_indices,
            self.offsets,
        )
        if not hasattr(self, "_rayrope_precompute_debug_printed"):
            self._rayrope_precompute_debug_printed = False

        if not self._rayrope_precompute_debug_printed:
            if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print("[RayRoPECore] precompute")
                print("  num_cameras      =", self.num_cameras)
                print("  num_cameras_kv   =", self.num_cameras_kv)
                print("  patches_x        =", self.patches_x)
                print("  patches_y_total  =", self.patches_y_total)
                print("  num_patches      =", self.num_patches)
                print("  head_dim         =", self.head_dim)
                print("  rope_coord_dim   =", self.rope_coord_dim)
                print("  rope_mat_dim     =", self.rope_mat_dim)
                print("  rope_head_dim    =", self.rope_head_dim)
                print("  num_rope_freqs   =", self.num_rope_freqs)
                print("  p0_world         =", tuple(self.p0_world.shape))
                print("  pinf_world       =", tuple(self.pinf_world.shape))
            self._rayrope_precompute_debug_printed = True
        

    def rayrope_dot_product_attention(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        num_cameras: int, 
        apply_fn_q: Callable[[torch.Tensor], torch.Tensor],
        all_apply_fns_kv: Optional[List[Callable[[torch.Tensor], torch.Tensor]]],
        apply_fn_o: Callable[[torch.Tensor], torch.Tensor],
        apply_vo: bool = True,
        timing_enabled: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Similar to torch.nn.functional.scaled_dot_product_attention, but applies
        RayRoPE positional encoding before attention.

        This is the rowwise version.

        Assumptions:
            1. The input sequence is camera-major.
            2. Each camera contributes tokens from exactly one row of the token grid.
            3. Therefore, the sequence length is:

                seqlen = num_cameras * patches_x

            rather than num_cameras * patches_x * patches_y.

        Token layout:
            For the current rowwise sample, the `(seqlen,)` axis is interpreted as

                (num_cameras, patches_x)

            where each contiguous block of length `num_patches` belongs to one camera.

        In particular, for camera `cam_idx`, its query slice is:

            [cam_idx * num_patches : (cam_idx + 1) * num_patches]

        where `num_patches = patches_x` in the current rowwise implementation.
        """
        # We're going to assume self-attention: all inputs are the same shape.
        (batch, num_heads, seqlen, head_dim) = q.shape
        # assert q.shape == k.shape == v.shape
        num_patches = seqlen // num_cameras
        # assert seqlen == num_cameras * num_patches
        

        out = torch.zeros_like(q)
        q = apply_fn_q(q)
        attn_mask = kwargs.pop("attn_mask", None)

        if not hasattr(self, "_rayrope_attn_debug_printed"):
            self._rayrope_attn_debug_printed = False

        for cam_idx, apply_fn_kv in enumerate(all_apply_fns_kv):
            k_idx = apply_fn_kv(k)
            if apply_vo:
                v_idx = apply_fn_kv(v)
            else:
                v_idx = v

            q_start = cam_idx * num_patches
            q_end = (cam_idx + 1) * num_patches
            q_idx = q[:, :, q_start:q_end, :]

            attn_mask_idx = None
            if attn_mask is not None:
                attn_mask_idx = attn_mask[..., q_start:q_end, :]
                if attn_mask_idx.ndim == 3:
                    attn_mask_idx = attn_mask_idx.unsqueeze(1)

            out_idx = F.scaled_dot_product_attention(
                query=q_idx.contiguous(),
                key=k_idx.contiguous(),
                value=v_idx.contiguous(),
                attn_mask=attn_mask_idx,
                **kwargs,
            )
            out[:, :, q_start:q_end, :] = out_idx

            if (cam_idx == 0) and (not self._rayrope_attn_debug_printed):
                if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    print("[RayRoPEAttn] shapes")
                    print("  q            =", tuple(q.shape))
                    print("  k            =", tuple(k.shape))
                    print("  v            =", tuple(v.shape))
                    print("  q_idx        =", tuple(q_idx.shape))
                    print("  k_idx        =", tuple(k_idx.shape))
                    print("  v_idx        =", tuple(v_idx.shape))
                    print("  attn_mask    =", None if attn_mask is None else tuple(attn_mask.shape))
                    print("  attn_mask_idx=", None if attn_mask_idx is None else tuple(attn_mask_idx.shape))
                self._rayrope_attn_debug_printed = True
        # assert not (torch.isinf(q).any()), "Inf values found in encoded q."
            # with time_block("apply_enc", timing_enabled):
           
            # assert not (torch.isinf(k_idx).any()), f"Inf values found in encoded k for cam_idx={cam_idx}."
            # assert not (torch.isinf(v_idx).any()), f"Inf values found in encoded v for cam_idx={cam_idx}."

            # with time_block("attention", timing_enabled):
            # assert not (torch.isinf(out_idx).any()), "Inf values found in attention out_idx."
        if apply_vo:
            # with time_block("apply_enc", timing_enabled):
            out = apply_fn_o(out)

        # assert not (torch.isnan(out).any()), "NaN values found in attention output."
        # assert not (torch.isinf(out).any()), "Inf values found in attention output."
        # assert out.shape == (batch, num_heads, seqlen, head_dim)
        if not hasattr(self, "_rayrope_attn_debug_printed"):
            self._rayrope_attn_debug_printed = False

        
        return out.contiguous()


    @torch.compile()
    def _prepare_apply_fns(
        self,
        predicted_d: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        predicted_d_kv: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        # debug: bool = False,
        # positions_collector: Optional[dict] = None,
    ) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """Prepare transforms for PRoPE-style positional encoding.
        # (batch, num_cameras, _, _) = w2cs.shape
        
        基于当前 forward 的 predicted_d / predicted_d_kv，
        现算与深度相关的几何点，并为每个 query camera 构造一组 apply_fn_kv。
        """

        batch = self.batch
        num_cameras = self.num_cameras
        num_cameras_kv = self.num_cameras_kv
        patches_x = self.patches_x
        patches_y = self.patches_y
        num_rays_per_patch = self.num_rays_per_patch
        num_patches = patches_x * patches_y
        # Q 组 / KV 组的预测深度，shape 会在 _prepare_depths 内整理成
        # (2, batch, cameras, num_patches, num_rays_per_patch, 1)
        depths = _prepare_depths(predicted_d, None, self.depth_type, 
                batch=batch, num_cameras=num_cameras, num_patches=num_patches, 
                patches_x=patches_x, patches_y=patches_y, 
                num_rays_per_patch=num_rays_per_patch, offsets=self.offsets)
        depths_kv = _prepare_depths(predicted_d_kv, self.context_depths, self.depth_type, 
                batch=batch, num_cameras=num_cameras_kv, num_patches=num_patches, 
                patches_x=patches_x, patches_y=patches_y, 
                num_rays_per_patch=num_rays_per_patch, offsets=self.offsets)
        # rowwise 版必须用 row_indices 来恢复当前真实行号
        # 不能再回退到 full-grid 的 _get_point_coords()
        pd_world = _get_point_coords_rowwise(
            self.P_inv,
            self.patches_x,
            self.patches_y_total,
            self.row_indices,
            self.offsets,
            depths,
        )

        pd_world_kv = _get_point_coords_rowwise(
            self.P_inv_kv,
            self.patches_x,
            self.patches_y_total,
            self.row_indices,
            self.offsets,
            depths_kv,
        )

        positions_Q = defaultdict(list)
        # if positions_collector is not None:
        #     positions_collector["KV"] = []
        all_apply_fns_kv = []
        for cam_idx in range(num_cameras):
            positions_KV = {}
            P_q = self.P[:, cam_idx, :, :]  # (batch, 4, 4)
            w2c_q = self.w2cs[:, cam_idx, :, :]  # (batch, 4, 4)

            # cam centers
            if self.p0_type == '3d':
                p0_3d = _transform_to_query_frame(self.p0_world, P_q, w2c_q, self.p0_type, self.denc_type)
                p0_3d = p0_3d.flatten(-2, -1)
                positions_Q['p0'].append(p0_3d[:, cam_idx])

                p0_3d_kv = _transform_to_query_frame(self.p0_world_kv, P_q, w2c_q, self.p0_type, self.denc_type)
                p0_3d_kv = p0_3d_kv.flatten(-2, -1)
                positions_KV['p0'] = p0_3d_kv
                
            elif self.p0_type == 'pj':
                p0_dir, p0_d = _transform_to_query_frame(self.p0_world, P_q, w2c_q, self.p0_type, self.denc_type)
                p0_dir = p0_dir[..., :2].flatten(-2, -1)
                p0_d = p0_d.flatten(-2, -1)
                positions_Q['p0_dir'].append(p0_dir[:, cam_idx])

                p0_dir_kv, p0_d_kv = _transform_to_query_frame(self.p0_world_kv, P_q, w2c_q, self.p0_type, self.denc_type)
                p0_dir_kv = p0_dir_kv[..., :2].flatten(-2, -1)
                p0_d_kv = p0_d_kv.flatten(-2, -1)
                positions_KV['p0_dir'] = p0_dir_kv
                
                if self.denc_type == 'inv_d':
                    positions_KV['p0_disparity'] = p0_d_kv
                    positions_Q['p0_disparity'].append(p0_d[:, cam_idx])
                elif self.denc_type == 'd':
                    positions_KV['p0_depth'] = p0_d_kv
                    positions_Q['p0_depth'].append(p0_d[:, cam_idx])
                elif self.denc_type == 'asinh_d':
                    positions_KV['p0_asinh_depth'] = p0_d_kv
                    positions_Q['p0_asinh_depth'].append(p0_d[:, cam_idx])

            # points at infinity
            if self.pinf_type == '3d':
                pinf_3d = _transform_to_query_frame(self.pinf_world, P_q, w2c_q, self.pinf_type, self.denc_type, norm_by='length')
                pinf_3d = pinf_3d.flatten(-2, -1)
                positions_Q['pinf_dir'].append(pinf_3d[:, cam_idx])

                pinf_3d_kv = _transform_to_query_frame(self.pinf_world_kv, P_q, w2c_q, self.pinf_type, self.denc_type, norm_by='length')
                pinf_3d_kv = pinf_3d_kv.flatten(-2, -1)
                positions_KV['pinf_dir'] = pinf_3d_kv
                
            elif self.pinf_type == 'pj':
                pinf_dir, _ = _transform_to_query_frame(self.pinf_world, P_q, w2c_q, self.pinf_type, self.denc_type)
                pinf_dir = pinf_dir.flatten(-2, -1)
                positions_Q['pinf_dir'].append(pinf_dir[:, cam_idx])

                pinf_dir_kv, _ = _transform_to_query_frame(self.pinf_world_kv, P_q, w2c_q, self.pinf_type, self.denc_type)
                pinf_dir_kv = pinf_dir_kv.flatten(-2, -1)
                positions_KV['pinf_dir'] = pinf_dir_kv
                
                # no need to include depth here

            # points at depth
            if self.pd_type == '3d':
                pd_3d = _transform_to_query_frame(pd_world, P_q, w2c_q, self.pd_type, self.denc_type)
                pd_3d = pd_3d.flatten(0, 1).flatten(-2, -1)
                positions_Q['pd_3d'].append(pd_3d[:, cam_idx])

                pd_3d_kv = _transform_to_query_frame(pd_world_kv, P_q, w2c_q, self.pd_type, self.denc_type)
                pd_3d_kv = pd_3d_kv.flatten(0, 1).flatten(-2, -1)
                positions_KV['pd_3d'] = pd_3d_kv
            elif self.pd_type == 'pj':
                pd_dir, pd_d = _transform_to_query_frame(pd_world, P_q, w2c_q, self.pd_type, self.denc_type)
                pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
                pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
                positions_Q['pd_dir'].append(pd_dir[:, cam_idx])

                pd_dir_kv, pd_d_kv = _transform_to_query_frame(pd_world_kv, P_q, w2c_q, self.pd_type, self.denc_type)
                pd_dir_kv = pd_dir_kv.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
                pd_d_kv = pd_d_kv.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
                positions_KV['pd_dir'] = pd_dir_kv
                
                if self.denc_type == 'inv_d':
                    positions_KV['pd_disparity'] = pd_d_kv
                    positions_Q['pd_disparity'].append(pd_d[:, cam_idx])
                elif self.denc_type == 'd':
                    positions_KV['pd_depth'] = pd_d_kv
                    positions_Q['pd_depth'].append(pd_d[:, cam_idx])
                elif self.denc_type == 'asinh_d':
                    positions_KV['pd_asinh_depth'] = pd_d_kv
                    positions_Q['pd_asinh_depth'].append(pd_d[:, cam_idx])

            # if positions_collector is not None:
            #     positions_collector["KV"].append(positions_KV)

            cos_KV, sin_KV = _prepare_rope_coeff_uniformd(
                positions_KV,
                self.num_rope_freqs,
                self.freq_base,
                batch,
                num_cameras_kv,
                num_patches,
            )

            apply_fn_kv = partial(_apply_rope_coeffs, cos=cos_KV, sin=sin_KV, inverse=True)
            
            all_apply_fns_kv.append(apply_fn_kv)

            # if (not torch.isfinite(cos_KV).all()) or (not torch.isfinite(sin_KV).all()):
            #     debug_info = {
            #         "cos_KV": cos_KV,
            #         "sin_KV": sin_KV,
            #         "positions_KV": positions_KV,
            #         "pd_world": pd_world,
            #         "pd_dir": pd_dir,
            #         "pd_d": pd_d,
            #         "depths": depths,
            #         "P_q": P_q,
            #         "w2c_q": w2c_q,
            #         "w2cs": w2cs,
            #         "cam_idx": cam_idx,
            #     }
            #     jobid = os.environ.get('SLURM_JOB_ID', 'unknown')
            #     rank = os.environ.get('SLURM_PROCID', '0')

            #     debug_path = f"/home/yuwu3/prope/logs/debug_tensor/kvnan_{jobid}_{rank}.pt"
            #     torch.save(debug_info, debug_path)
            #     print(f"Debug information saved to {debug_path}")
            #     raise ValueError(f"NaN/inf values found in cos/sin KV coeffs for pd for cam_idx={cam_idx}.")

        for key, val in positions_Q.items():
            positions_Q[key] = torch.stack(val, dim=1)  # (batch, num_cameras, num_patches, ...)

        # if positions_collector is not None:
        #     positions_collector["Q"] = positions_Q

        cos_Q, sin_Q = _prepare_rope_coeff_uniformd(positions_Q, self.num_rope_freqs, self.freq_base, batch, num_cameras, num_patches)
        apply_fn_q = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=True)
        apply_fn_o = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=False)

        # if (not torch.isfinite(cos_Q).all()) or (not torch.isfinite(sin_Q).all()):
        #     raise ValueError("NaN/inf values found in rope_matrices_Q.")
        if not hasattr(self, "_rayrope_coeff_debug_printed"):
            self._rayrope_coeff_debug_printed = False

        if not self._rayrope_coeff_debug_printed:
            if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print("[RayRoPECoeff] prepared")
                print("  Q keys           =", list(positions_Q.keys()))
                print("  num_apply_fns_kv =", len(all_apply_fns_kv))
                print("  num_patches      =", num_patches)
                print("  cos_Q shape      =", tuple(cos_Q.shape))
                print("  sin_Q shape      =", tuple(sin_Q.shape))
            self._rayrope_coeff_debug_printed = True
        return apply_fn_q, all_apply_fns_kv, apply_fn_o


def _transform_to_query_frame(
    points_world: torch.Tensor,  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)
    P: torch.Tensor,  # (batch, 4, 4)
    w2c: torch.Tensor,  # (batch, 4, 4)
    transform_type: str = 'pj', # pj or 3d
    denc_type: str = 'd', # inv_d, d, asinh_d
    norm_by: str = 'w' # length or w
):
    if transform_type == '3d':
        points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, points_world) # (batch, num_cameras, num_patches, 4)
        if norm_by == 'w':
            points_cam = points_cam / torch.clamp(points_cam[..., -1:], min=1e-4)  # normalize
        elif norm_by == 'length':
            points_cam = points_cam / (torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6)
        return points_cam[..., :3]
    
    elif transform_type == 'pj':
        # points at depth
        points_cam = torch.einsum("bij,...bcprj->...bcpri", P, points_world)
        safe_abs = torch.sqrt(points_cam[..., 2:3].pow(2) + 1e-9)
        z = torch.clamp(safe_abs, min=1e-4)
        # z = _clamp_zero(points_cam[..., 2:3], min_abs_value=1e-4)
        w = torch.clamp(points_cam[..., -1:], min=1e-4)
        pd_dir = points_cam[..., :3] / (torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6)
        if denc_type == 'inv_d':
            pd_disparity = w / z
            return pd_dir, pd_disparity
        elif denc_type == 'd':
            pd_depth = z / w
            return pd_dir, pd_depth
        elif denc_type == 'asinh_d':
            pd_depth = z / w
            pd_depth = torch.asinh(pd_depth)
            return pd_dir, pd_depth
        

def _get_cam_centers(
    c2ws: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    device = c2ws.device
    batches = c2ws.shape[0]
    num_cameras = c2ws.shape[1]

    cam_centers = c2ws[:, :, :, 3]  # (batch, num_cameras, 4)
    cam_centers = cam_centers.view(
        batches, num_cameras, 1, 1, 4
    ).expand(
        batches, num_cameras, num_patches, 1, 4
    )
    return cam_centers
    
def _get_point_coords(
        P_inv: torch.Tensor,  # (batch, num_cameras, 4, 4)
        patches_x: int, 
        patches_y: int, 
        offsets: List[Tuple[float, float]],
        depths: torch.Tensor = None,  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 1)
    ) -> torch.Tensor:
    # return the pixel space 3d homogenous coordinates
    """
    为 rowwise attention 构造当前这一行的 patch 几何点。

    与 full-grid 版本不同：
    - x 仍然在 [0, patches_x) 上展开
    - y 不再自己扫完整张图
    - y 由 row_indices 指定当前 batch 样本对应原始 token grid 的哪一行
    """
    device = P_inv.device
    batches = P_inv.shape[0]
    num_cameras = P_inv.shape[1]
    num_patches = patches_x * patches_y
    num_rays_per_patch = len(offsets)
    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device),
        torch.arange(patches_y, device=device),
        indexing="xy",
    )  # [H, W]

    coords = []
    for offset in offsets:
        u = ((u_base + offset[0]) / patches_x) - 0.5 #Since we assume normalized K where px=py=0
        v = ((v_base + offset[1]) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))  # [num_patches, 2]
    coords = torch.stack(coords, dim=1) # (num_patches, num_rays_per_patch, 2)
    # assert coords.shape == (num_patches, num_rays_per_patch, 2)
    coords = coords.view(1, 1, num_patches, num_rays_per_patch, 2).expand(batches, num_cameras, -1, -1, -1) 
    # (batch, num_cameras, num_patches, num_rays_per_patch, 2)

    if depths is not None:
        # depths = torch.clip(depths, min=1e-2)
        # gives [u, v, 1, 1/d]
        depths = torch.clamp(depths, min=1e-2, max=MAX_DEPTH)
        disparity = 1 / depths
        if depths.ndim == 5:
            assert depths.shape == (batches, num_cameras, num_patches, num_rays_per_patch, 1)
            coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
            coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
        elif depths.ndim == 6: # when two depths along ray is provided
            assert depths.shape == (2, batches, num_cameras, num_patches, num_rays_per_patch, 1)
            coords = coords.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
            coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)
            coords_4d = torch.einsum("bcij,ebcprj->ebcpri", P_inv, coords_4d)
    else:
        # if no depth provided, gives points at infinity (directions)
        depths = torch.ones_like(coords[..., :1])
        scales = torch.zeros_like(coords[..., :1])

        coords_4d = torch.cat([coords, depths, scales], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
    
    return coords_4d    

def _sample_depth_map(
    pixel_depths: torch.Tensor,  # (batch, cameras, image_height, image_width, 1)
    patches_x: int,
    patches_y: int,
    offsets: List[Tuple[float, float]],
) -> torch.Tensor:
    # assert pixel_depths.ndim == 5 and pixel_depths.shape[-1] == 1
    batch, cameras, image_height, image_width, _ = pixel_depths.shape
    # if image_width % patches_x != 0 or image_height % patches_y != 0:
    #     raise ValueError("Image resolution must be divisible by the patch grid dimensions.")

    patch_w = image_width // patches_x
    patch_h = image_height // patches_y
    device = pixel_depths.device
    dtype = pixel_depths.dtype
    pixel_depths = torch.clamp(pixel_depths, min=1e-4, max=MAX_DEPTH)

    num_offsets = len(offsets)

    # Grid over patch indices; match meshgrid ordering used in _get_point_coords.
    x_idx = torch.arange(patches_x, device=device, dtype=dtype)
    y_idx = torch.arange(patches_y, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x_idx, y_idx, indexing="xy")
    base_x = grid_x * float(patch_w)
    base_y = grid_y * float(patch_h)

    depths_flat = pixel_depths.reshape(batch * cameras, 1, image_height, image_width)
    samples = []
    for ox, oy in offsets:
        ox_t = torch.as_tensor(ox, device=device, dtype=dtype)
        oy_t = torch.as_tensor(oy, device=device, dtype=dtype)
        sample_x = base_x + ox_t * patch_w
        sample_y = base_y + oy_t * patch_h

        # Convert to normalized grid coordinates in [-1, 1].
        x_norm = sample_x / image_width * 2.0 - 1.0
        y_norm = sample_y / image_height * 2.0 - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).permute(1, 0, 2)
        grid = grid.unsqueeze(0).expand(batch * cameras, -1, -1, -1)
        sampled = F.grid_sample(
            depths_flat,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        samples.append(sampled.permute(0, 1, 3, 2))

    sampled_depths = torch.stack(samples, dim=-1)
    sampled_depths = sampled_depths.reshape(batch, cameras, patches_x * patches_y, num_offsets, 1)

    return sampled_depths

def _prepare_depths(
    predicted_d: Optional[torch.Tensor],  # (batch, seqlen, 1 or 2)
    context_depths: Optional[torch.Tensor],  # (batch, cameras, image_height, image_width, 1)
    depth_type: str = 'none',
    batch: int = 1,
    num_cameras: int = 2,
    num_patches: int = 1024,
    patches_x: int = 32,
    patches_y: int = 32,
    num_rays_per_patch: int = 3,
    offsets = [[0.0, 0.0]],
):
    predicted_d = predicted_d.reshape(batch, num_cameras, num_patches, 1, 2)
    predicted_logd = predicted_d[..., 0:1]
    # predicted_sigma = torch.exp(predicted_d[..., 1:2])
    predicted_sigma = predicted_d[..., 1:2]

    predicted_d1 = torch.exp(torch.clamp(predicted_logd - predicted_sigma, max=MAX_LOG_DEPTH))
    predicted_d2 = torch.exp(torch.clamp(predicted_logd + predicted_sigma, max=MAX_LOG_DEPTH))

    depths = torch.stack([predicted_d1, predicted_d2], dim=0)  # (2, batch, num_cameras, num_patches, 1)
    depths = depths.expand(-1, -1, -1, -1, num_rays_per_patch, -1)
    
    if torch.isnan(depths).any() or torch.isinf(depths).any():
        raise ValueError("NaN/inf values found in predicted depths.")
    if 'known' in depth_type and context_depths is not None:
        num_contexts = context_depths.shape[1]
        context_depths_sampled = _sample_depth_map(
            context_depths, patches_x, patches_y, offsets=offsets
        ).unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        # (2, batch, num_cameras, num_patches, num_rays_per_patch, 1)
        depths = depths.contiguous()
        depths[:, :, :num_contexts] = context_depths_sampled

    return depths

@torch.compile
def _prepare_rope_coeff_uniformd(
    positions: dict[str, torch.Tensor], # (batch, num_cameras, num_patches, coord_dim)
    num_freqs: int,
    freq_base: float,
    batch: int,
    num_cameras: int,
    num_patches: int,
):
    """
    对特征前缀应用 rope，尾部维度保持不变。

    这是 partial RoPE：
    - feats[..., :rope_dim] 做旋转
    - feats[..., rope_dim:] 原样保留

    这样可以兼容 head_dim 不能整除 rope_mat_dim 的情况。
    """
    coord_dim = 0
    for key, value in positions.items():
        coord_dim += value.shape[-1]

        device = value.device

    cosine_list = []
    sine_list = []
    for pos_name, pos in positions.items():
        if pos_name in ['p0', 'pd_3d']:
            max_period = 1.0 * 4
        elif pos_name in ['pinf_dir', 'pd_dir', 'p0_dir']:
            max_period = 2.0 * 4
        elif pos_name in ['pd_disparity', 'p0_disparity']:
            max_period = 20.0 * 4
            # pos = torch.clamp(pos, min=-20.0, max=20.0)
            pos = torch.clamp(pos, min=0.0, max=20.0)
        elif pos_name in ['pd_depth', 'p0_depth']:
            max_period = MAX_D_F * 2 * 4
            pos = torch.clamp(pos, min=-MAX_D_F, max=MAX_D_F)
        elif pos_name in ['pd_asinh_depth', 'p0_asinh_depth']:
            max_period = MAX_ASINH_D_F * 2 * 4
            pos = torch.clamp(pos, min=-MAX_ASINH_D_F, max=MAX_ASINH_D_F)
        else: 
            raise ValueError(f"Unknown position name: {pos_name}")
        
        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs,
                                max_period=max_period,
                                min_period=min_period).to(device)
        rope_angle = torch.einsum("f,bcpd->bcpfd", freqs, pos)

        if rope_angle.shape[0] == batch * 2:
            rope_angles1 = rope_angle[:batch]
            rope_angles2 = rope_angle[batch:]
            same_mask = torch.isclose(rope_angles1, rope_angles2, atol=1e-2, rtol=0)

            cosine1 = torch.cos(rope_angles1)
            cosine2 = torch.cos(rope_angles2)
            sine1 = torch.sin(rope_angles1)
            sine2 = torch.sin(rope_angles2)
            delta = rope_angles2 - rope_angles1
            delta_safe = torch.where(same_mask, torch.ones_like(delta), delta)
            E_cosine = (sine2 - sine1) / delta_safe
            E_sine = (cosine1 - cosine2) / delta_safe

            cosine_final = torch.where(same_mask, cosine1, E_cosine)
            sine_final = torch.where(same_mask, sine1, E_sine)

            assert cosine_final.abs().max() <= 1.0 + 1e-2
            assert sine_final.abs().max() <= 1.0 + 1e-2
            cosine_list.append(cosine_final)
            sine_list.append(sine_final)

        elif rope_angle.shape[0] == batch:
            cosine_final = torch.cos(rope_angle)
            sine_final = torch.sin(rope_angle)
            cosine_list.append(cosine_final)
            sine_list.append(sine_final)
        else:
            raise ValueError(f"Unexpected rope_angle batch size. rope_angle shape: {rope_angle.shape}")
        
    cosine_out = torch.cat(cosine_list, dim=-1).reshape(batch, num_cameras * num_patches, num_freqs * coord_dim).contiguous()
    sine_out = torch.cat(sine_list, dim=-1).reshape(batch, num_cameras * num_patches, num_freqs * coord_dim).contiguous()
    return cosine_out, sine_out


def _get_frequency(
    num_freqs: int,
    max_period: float,
    min_period: float,
):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    log_freqs = torch.linspace(
        log_min_frequency,
        log_max_frequency,
        num_freqs,
    )
    freqs = torch.exp(log_freqs)
    # ratio = freqs[1] / freqs[0]
    # print(f"max/min: {max_period}, {min_period}, Frequency ratio: {ratio:.4f}, freqs: {freqs}")
    return freqs
def _get_point_coords_rowwise(
    P_inv: torch.Tensor,
    patches_x: int,
    patches_y_total: int,
    row_indices: torch.Tensor,
    offsets: list,
    depths: torch.Tensor = None,
):
    device = P_inv.device
    dtype = P_inv.dtype
    batches = P_inv.shape[0]
    num_cameras = P_inv.shape[1]
    num_patches = patches_x

    x_base = torch.arange(patches_x, device=device, dtype=dtype)
    row_base = row_indices.to(device=device, dtype=dtype)

    coords = []
    for offset in offsets:
        u = ((x_base[None, :] + offset[0]) / patches_x) - 0.5
        v = ((row_base[:, None] + offset[1]) / patches_y_total) - 0.5

        uv = torch.stack(
            [
                u.expand(batches, -1),
                v.expand(-1, patches_x),
            ],
            dim=-1,
        )
        coords.append(uv)

    coords = torch.stack(coords, dim=2)
    coords = coords.unsqueeze(1).expand(-1, num_cameras, -1, -1, -1)

    if depths is None:
        ones = torch.ones(
            batches,
            num_cameras,
            num_patches,
            len(offsets),
            1,
            device=device,
            dtype=dtype,
        )
        zeros = torch.zeros_like(ones)

        # 无穷远点应为 [u, v, 1, 0]
        coords_4d = torch.cat([coords, ones, zeros], dim=-1)
        coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
        return coords_4d

    if depths.dim() == 5:
        disparity = 1.0 / torch.clamp(depths, min=1e-4)
        coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)
        coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
        return coords_4d

    disparity = 1.0 / torch.clamp(depths, min=1e-4)
    coords = coords.unsqueeze(0).expand(depths.shape[0], -1, -1, -1, -1, -1)
    coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)
    coords_4d = torch.einsum("bcij,ebcprj->ebcpri", P_inv, coords_4d)
    return coords_4d
@torch.compile
def _apply_rope_coeffs(
    feats: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
    interleaved: bool = False,
):
    batch, num_heads, seqlen, feat_dim = feats.shape
    rope_half_dim = cos.shape[-1]
    rope_dim = rope_half_dim * 2

    feats_rope = feats[..., :rope_dim]
    feats_tail = feats[..., rope_dim:]

    cos = cos.to(dtype=feats.dtype).unsqueeze(1)
    sin = sin.to(dtype=feats.dtype).unsqueeze(1)

    if interleaved:
        x1 = feats_rope[..., 0::2]
        x2 = feats_rope[..., 1::2]
    else:
        x1, x2 = torch.chunk(feats_rope, 2, dim=-1)

    if inverse:
        x_rotated_1 = x1 * cos + x2 * sin
        x_rotated_2 = -x1 * sin + x2 * cos
    else:
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

    if interleaved:
        out_rope = torch.empty_like(feats_rope)
        out_rope[..., 0::2] = x_rotated_1
        out_rope[..., 1::2] = x_rotated_2
    else:
        out_rope = torch.cat([x_rotated_1, x_rotated_2], dim=-1)

    if feats_tail.numel() > 0:
        return torch.cat([out_rope, feats_tail], dim=-1)

    return out_rope


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(
        Ks.shape[:-2] + (4, 4),
        device=Ks.device,
        dtype=Ks.dtype,
    )
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out

def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out

def normalize_K(Ks: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Normalize camera intrinsics."""
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
    Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
    Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
    Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
    Ks_norm[..., 2, 2] = 1.0
    return Ks_norm

def _clamp_zero(tensor: torch.Tensor, min_abs_value: float = 1e-4) -> torch.Tensor:
    # Get the sign of the input.
    # Note: torch.sign(0) is 0, so we replace 0s with 1s to treat exact 0 as positive.
    sign = torch.sign(tensor)
    sign[sign == 0] = 1.0
    abs_val = torch.abs(tensor)
    clamped_abs = torch.clamp(abs_val, min=min_abs_value)
    return sign * clamped_abs
    