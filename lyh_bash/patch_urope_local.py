#!/usr/bin/env python3
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
         "yanjunchi-24040/songbur/camsim/OpenDWM"
).resolve()

targets = {
    "crossview": root / "src/dwm/models/crossview_temporal_dit_urope.py",
    "block": root / "src/dwm/models/urope/block.py",
    "mha": root / "src/dwm/models/urope/mha.py",
    "urope": root / "src/dwm/models/urope/urope.py",
}

for name, path in targets.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")

print("=== git status before patch ===")
subprocess.run(
    ["git", "status", "--short", "--"] + [str(p.relative_to(root)) for p in targets.values()],
    cwd=root,
    check=False,
)

original = {name: path.read_text() for name, path in targets.items()}
updated = dict(original)

crossview_text = updated["crossview"]
if "context_hidden_states=context_hidden_states" not in crossview_text:
    start_marker = "        del crossview_attention_index\n"
    end_marker = "        if mixer is None:\n"
    start = crossview_text.find(start_marker)
    if start < 0:
        raise RuntimeError(
            "Cannot find old cross-view URoPE body start."
        )
    end = crossview_text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(
            "Cannot find old cross-view URoPE body end."
        )
    crossview_text = (
        crossview_text[:start]
        + '\n        del view_emb\n\n        if camera_intrinsics_norm is None:\n            camera_intrinsics_norm = getattr(\n                self,\n                "_urope_camera_intrinsics_norm",\n                None,\n            )\n\n        if camera2referego is None:\n            camera2referego = getattr(\n                self,\n                "_urope_camera2referego",\n                None,\n            )\n\n        if camera_intrinsics_norm is None or camera2referego is None:\n            raise ValueError(\n                "URoPE requires camera_intrinsics_norm and camera2referego."\n            )\n\n        bt_count = batch_size * sequence_length\n        patch_count = height * width\n        hidden_states_by_view = hidden_states.reshape(\n            bt_count,\n            view_count,\n            patch_count,\n            hidden_states.shape[-1],\n        )\n\n        if crossview_attention_mask is not None:\n            camera_mask = crossview_attention_mask.to(\n                device=hidden_states.device\n            ).bool()\n            if camera_mask.ndim == 2:\n                camera_mask = camera_mask.unsqueeze(0)\n            elif camera_mask.ndim == 4 and camera_mask.shape[1] == 1:\n                camera_mask = camera_mask[:, 0]\n\n            if camera_mask.ndim != 3:\n                raise ValueError(\n                    "crossview_attention_mask must be [V,V], [B,V,V], "\n                    "or [B*T,V,V], got {}.".format(\n                        tuple(camera_mask.shape)\n                    )\n                )\n            if camera_mask.shape[-2:] != (view_count, view_count):\n                raise ValueError(\n                    "crossview_attention_mask ends with {}, expected "\n                    "({}, {}).".format(\n                        tuple(camera_mask.shape[-2:]),\n                        view_count,\n                        view_count,\n                    )\n                )\n\n            mask_batch_size = camera_mask.shape[0]\n            if mask_batch_size == 1 and batch_size > 1:\n                camera_mask = camera_mask.expand(\n                    batch_size,\n                    -1,\n                    -1,\n                )\n                mask_batch_size = batch_size\n\n            if mask_batch_size == batch_size:\n                camera_mask = camera_mask.repeat_interleave(\n                    sequence_length,\n                    dim=0,\n                )\n            elif mask_batch_size != bt_count:\n                raise ValueError(\n                    "Camera mask batch must be B or B*T, got {} "\n                    "for B={} and T={}.".format(\n                        mask_batch_size,\n                        batch_size,\n                        sequence_length,\n                    )\n                )\n\n            diagonal = torch.arange(\n                view_count,\n                device=hidden_states.device,\n            )\n            camera_mask = camera_mask.clone()\n            camera_mask[:, diagonal, diagonal] = True\n\n            local_view_count = int(\n                camera_mask.sum(dim=-1).max().item()\n            )\n            all_view_indices = torch.arange(\n                view_count,\n                device=hidden_states.device,\n            ).reshape(1, 1, view_count)\n            all_view_indices = all_view_indices.expand(\n                bt_count,\n                view_count,\n                view_count,\n            )\n            invalid_index = torch.full_like(\n                all_view_indices,\n                view_count,\n            )\n            source_view_index = torch.where(\n                camera_mask,\n                all_view_indices,\n                invalid_index,\n            )\n            source_view_index = source_view_index.sort(\n                dim=-1\n            ).values[..., :local_view_count]\n            source_view_valid = source_view_index < view_count\n\n            query_view_index = torch.arange(\n                view_count,\n                device=hidden_states.device,\n            ).reshape(1, view_count, 1)\n            query_view_index = query_view_index.expand(\n                bt_count,\n                view_count,\n                local_view_count,\n            )\n            source_view_index = torch.where(\n                source_view_valid,\n                source_view_index,\n                query_view_index,\n            )\n        else:\n            if crossview_attention_index is None:\n                raise ValueError(\n                    "Local URoPE requires crossview_attention_mask or "\n                    "crossview_attention_index."\n                )\n\n            source_view_index = crossview_attention_index.to(\n                device=hidden_states.device,\n                dtype=torch.long,\n            )\n            if source_view_index.ndim == 2:\n                if source_view_index.shape[-1] % view_count != 0:\n                    raise ValueError(\n                        "crossview_attention_index width must be divisible "\n                        "by view_count, got {} for V={}.".format(\n                            source_view_index.shape[-1],\n                            view_count,\n                        )\n                    )\n                source_view_index = source_view_index.reshape(\n                    source_view_index.shape[0],\n                    view_count,\n                    -1,\n                )\n            elif source_view_index.ndim != 3:\n                raise ValueError(\n                    "crossview_attention_index must be [B,V*K] or "\n                    "[B,V,K], got {}.".format(\n                        tuple(source_view_index.shape)\n                    )\n                )\n\n            index_batch_size = source_view_index.shape[0]\n            if index_batch_size == 1 and batch_size > 1:\n                source_view_index = source_view_index.expand(\n                    batch_size,\n                    -1,\n                    -1,\n                )\n                index_batch_size = batch_size\n\n            if index_batch_size == batch_size:\n                source_view_index = source_view_index.repeat_interleave(\n                    sequence_length,\n                    dim=0,\n                )\n            elif index_batch_size != bt_count:\n                raise ValueError(\n                    "Cross-view index batch must be B or B*T, got {} "\n                    "for B={} and T={}.".format(\n                        index_batch_size,\n                        batch_size,\n                        sequence_length,\n                    )\n                )\n\n            if source_view_index.shape[1] != view_count:\n                raise ValueError(\n                    "crossview_attention_index has {} query views, "\n                    "expected {}.".format(\n                        source_view_index.shape[1],\n                        view_count,\n                    )\n                )\n            if (\n                source_view_index.min().item() < 0\n                or source_view_index.max().item() >= view_count\n            ):\n                raise ValueError(\n                    "crossview_attention_index contains an invalid "\n                    "camera index."\n                )\n            local_view_count = source_view_index.shape[-1]\n            source_view_valid = torch.ones_like(\n                source_view_index,\n                dtype=torch.bool,\n            )\n\n        bt_index = torch.arange(\n            bt_count,\n            device=hidden_states.device,\n        ).reshape(bt_count, 1, 1)\n\n        context_hidden_states = hidden_states_by_view[\n            bt_index,\n            source_view_index,\n        ]\n        context_hidden_states = context_hidden_states.reshape(\n            bt_count * view_count,\n            local_view_count * patch_count,\n            hidden_states.shape[-1],\n        )\n        query_hidden_states = hidden_states_by_view.reshape(\n            bt_count * view_count,\n            patch_count,\n            hidden_states.shape[-1],\n        )\n\n        intrinsics = camera_intrinsics_norm.clone().float()\n        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width\n        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height\n        intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width\n        intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height\n        intrinsics = intrinsics.reshape(\n            bt_count,\n            view_count,\n            3,\n            3,\n        ).to(device=hidden_states.device)\n\n        viewmats = torch.linalg.inv(camera2referego.float())\n        viewmats = viewmats.reshape(\n            bt_count,\n            view_count,\n            4,\n            4,\n        ).to(device=hidden_states.device)\n\n        query_intrinsics = intrinsics.reshape(\n            bt_count * view_count,\n            3,\n            3,\n        )\n        query_viewmats = viewmats.reshape(\n            bt_count * view_count,\n            4,\n            4,\n        )\n        source_intrinsics = intrinsics[\n            bt_index,\n            source_view_index,\n        ].reshape(\n            bt_count * view_count,\n            local_view_count,\n            3,\n            3,\n        )\n        source_viewmats = viewmats[\n            bt_index,\n            source_view_index,\n        ].reshape(\n            bt_count * view_count,\n            local_view_count,\n            4,\n            4,\n        )\n\n        source_token_valid = source_view_valid.unsqueeze(-1).expand(\n            bt_count,\n            view_count,\n            local_view_count,\n            patch_count,\n        ).reshape(\n            bt_count * view_count,\n            local_view_count * patch_count,\n        )\n        local_attention_mask = source_token_valid[\n            :,\n            None,\n            None,\n            :,\n        ].expand(\n            -1,\n            1,\n            patch_count,\n            -1,\n        )\n\n        crossview_hidden_states = crossview_block(\n            query_hidden_states,\n            context_hidden_states=context_hidden_states,\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_height=height,\n            patch_width=width,\n            self_attention_mask=local_attention_mask,\n        )\n'.lstrip("\n")
        + "\n"
        + crossview_text[end:]
    )
updated["crossview"] = crossview_text

block_text = updated["block"]
if "context_hidden_states: torch.Tensor" not in block_text:
    class_pos = block_text.find(
        "class VTURoPEAttentionBlock(torch.nn.Module):"
    )
    if class_pos < 0:
        raise RuntimeError(
            "Cannot find VTURoPEAttentionBlock."
        )
    forward_pos = block_text.find(
        "    def forward(\n",
        class_pos,
    )
    if forward_pos < 0:
        raise RuntimeError(
            "Cannot find VTURoPEAttentionBlock.forward."
        )
    block_text = (
        block_text[:forward_pos]
        + '\n    def forward(\n        self,\n        hidden_states: torch.Tensor,\n        *,\n        context_hidden_states: torch.Tensor,\n        query_viewmats: torch.Tensor,\n        query_intrinsics: torch.Tensor,\n        source_viewmats: torch.Tensor,\n        source_intrinsics: torch.Tensor,\n        patch_height: int,\n        patch_width: int,\n        self_attention_mask: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n        query_residual = hidden_states\n        hidden_states = self.norm_in(hidden_states)\n        hidden_states = self.ff_in(hidden_states)\n        hidden_states = hidden_states + query_residual\n\n        context_residual = context_hidden_states\n        context_hidden_states = self.norm_in(context_hidden_states)\n        context_hidden_states = self.ff_in(context_hidden_states)\n        context_hidden_states = (\n            context_hidden_states + context_residual\n        )\n\n        norm_hidden_states = self.norm1(hidden_states)\n        norm_context_hidden_states = self.norm1(\n            context_hidden_states\n        )\n        attention_output = self.attn1(\n            norm_hidden_states,\n            norm_context_hidden_states,\n            norm_context_hidden_states,\n            urope_attention=self.urope_attention,\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_width=patch_width,\n            patch_height=patch_height,\n            attention_mask=self_attention_mask,\n        )\n        hidden_states = hidden_states + attention_output\n\n        norm_hidden_states = self.norm3(hidden_states)\n        feed_forward_output = self.ff(norm_hidden_states)\n        return hidden_states + feed_forward_output\n'.lstrip("\n")
        + "\n"
    )
updated["block"] = block_text

mha_text = updated["mha"]
if "query_viewmats: Tensor" not in mha_text:
    class_pos = mha_text.find(
        "class MultiheadAttention(torch.nn.Module):"
    )
    if class_pos < 0:
        raise RuntimeError(
            "Cannot find MultiheadAttention."
        )
    forward_pos = mha_text.find(
        "    def forward(\n",
        class_pos,
    )
    if forward_pos < 0:
        raise RuntimeError(
            "Cannot find MultiheadAttention.forward."
        )
    mha_text = (
        mha_text[:forward_pos]
        + '\n    def forward(\n        self,\n        query: Tensor,\n        key: Tensor,\n        value: Tensor,\n        *,\n        urope_attention: URoPEDotProductAttention,\n        query_viewmats: Tensor,\n        query_intrinsics: Tensor,\n        source_viewmats: Tensor,\n        source_intrinsics: Tensor,\n        patch_width: int,\n        patch_height: int,\n        attention_mask: Optional[Tensor] = None,\n    ) -> Tensor:\n        if key.shape != value.shape:\n            raise ValueError(\n                "URoPE key and value must have the same shape."\n            )\n        if query.shape[0] != key.shape[0]:\n            raise ValueError(\n                "URoPE query and context batch sizes must match."\n            )\n\n        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(\n            3,\n            dim=0,\n        )\n        if self.in_proj_bias is None:\n            q_bias = None\n            k_bias = None\n            v_bias = None\n        else:\n            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(\n                3,\n                dim=0,\n            )\n\n        q = F.linear(query, q_weight, q_bias)\n        k = F.linear(key, k_weight, k_bias)\n        v = F.linear(value, v_weight, v_bias)\n\n        batch_size, query_length = q.shape[:2]\n        key_length = k.shape[1]\n        q = q.reshape(\n            batch_size,\n            query_length,\n            self.num_heads,\n            self.head_dim,\n        )\n        k = k.reshape(\n            batch_size,\n            key_length,\n            self.num_heads,\n            self.head_dim,\n        )\n        v = v.reshape(\n            batch_size,\n            key_length,\n            self.num_heads,\n            self.head_dim,\n        )\n\n        if self.qk_norm:\n            q = self.q_norm(q)\n            k = self.k_norm(k)\n\n        q = q.permute(0, 2, 1, 3)\n        k = k.permute(0, 2, 1, 3)\n        v = v.permute(0, 2, 1, 3)\n        output = urope_attention(\n            q,\n            k,\n            v,\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_width=patch_width,\n            patch_height=patch_height,\n            attention_mask=attention_mask,\n            dropout_p=self.dropout if self.training else 0.0,\n        )\n\n        output = output.permute(0, 2, 1, 3).reshape(\n            batch_size,\n            query_length,\n            self.embed_dim,\n        )\n        return F.linear(\n            output,\n            self.out_proj.weight,\n            self.out_proj.bias,\n        )\n'.lstrip("\n")
        + "\n"
    )
updated["mha"] = mha_text

urope_text = updated["urope"]
if "def _prepare_local_geometry_coefficients(" not in urope_text:
    old_tail_pos = urope_text.find(
        "def _reshape_camera_mask("
    )
    if old_tail_pos < 0:
        raise RuntimeError(
            "Cannot find old URoPE mask/geometry section."
        )
    urope_text = (
        urope_text[:old_tail_pos]
        + '\ndef _prepare_local_geometry_coefficients(\n    query_viewmats: torch.Tensor,\n    query_intrinsics: torch.Tensor,\n    source_viewmats: torch.Tensor,\n    source_intrinsics: torch.Tensor,\n    patch_width: int,\n    patch_height: int,\n    depth_count: int,\n    group_size: int,\n    min_depth: float,\n    max_depth: float,\n    freq_base: float,\n    freq_scale: float,\n    head_dim: int,\n    camera_convention: str,\n    output_dtype: torch.dtype,\n    clamp_min: float,\n    clamp_max: float,\n) -> Tuple[\n    Tuple[torch.Tensor, torch.Tensor],\n    Tuple[torch.Tensor, torch.Tensor],\n    Tuple[torch.Tensor, torch.Tensor],\n    Tuple[torch.Tensor, torch.Tensor],\n]:\n    """Build URoPE coefficients only for selected source cameras."""\n    query_viewmats = _convert_viewmats(\n        query_viewmats[:, None].float(),\n        camera_convention,\n    )[:, 0]\n    source_viewmats = _convert_viewmats(\n        source_viewmats.float(),\n        camera_convention,\n    )\n    query_intrinsics = query_intrinsics.float()\n    source_intrinsics = source_intrinsics.float()\n\n    batch_size, source_count = source_viewmats.shape[:2]\n    patch_count = patch_width * patch_height\n    source_camera_to_world = _invert_se3(source_viewmats)\n    source_intrinsics_inv = _invert_intrinsics(\n        source_intrinsics\n    )\n\n    grid_x, grid_y = torch.meshgrid(\n        torch.arange(\n            patch_width,\n            device=source_viewmats.device,\n            dtype=torch.float32,\n        ),\n        torch.arange(\n            patch_height,\n            device=source_viewmats.device,\n            dtype=torch.float32,\n        ),\n        indexing="xy",\n    )\n    image_points = torch.stack(\n        [\n            grid_x + 0.5,\n            grid_y + 0.5,\n            torch.ones_like(grid_x),\n        ],\n        dim=-1,\n    ).reshape(\n        1,\n        1,\n        patch_count,\n        3,\n    )\n    image_points = image_points.expand(\n        batch_size,\n        source_count,\n        -1,\n        -1,\n    )\n\n    ray_directions_camera = torch.einsum(\n        "bsij,bspj->bspi",\n        source_intrinsics_inv,\n        image_points,\n    )\n    ray_origins_world = source_camera_to_world[..., :3, 3]\n    ray_directions_world = torch.einsum(\n        "bsij,bspj->bspi",\n        source_camera_to_world[..., :3, :3],\n        ray_directions_camera,\n    )\n\n    world_to_query_rotation = query_viewmats[..., :3, :3]\n    world_to_query_translation = query_viewmats[..., :3, 3]\n    ray_origins_query = torch.einsum(\n        "bij,bsj->bsi",\n        world_to_query_rotation,\n        ray_origins_world,\n    )\n    ray_origins_query = (\n        ray_origins_query\n        + world_to_query_translation[:, None, :]\n    )\n    ray_directions_query = torch.einsum(\n        "bij,bspj->bspi",\n        world_to_query_rotation,\n        ray_directions_world,\n    )\n\n    depth_indices = torch.arange(\n        depth_count,\n        device=source_viewmats.device,\n        dtype=torch.float32,\n    )\n    depth_bin_size = (max_depth - min_depth) / depth_count\n    depths = min_depth + depth_bin_size * depth_indices\n\n    points_query = ray_origins_query[:, None, :, None, :]\n    points_query = points_query + (\n        ray_directions_query[:, None]\n        * depths[None, :, None, None, None]\n    )\n    projected_points = torch.einsum(\n        "bij,bdspj->bdspi",\n        query_intrinsics,\n        points_query,\n    )\n\n    projected_z = projected_points[..., 2].abs() + 1e-5\n    key_x = projected_points[..., 0] / projected_z\n    key_y = projected_points[..., 1] / projected_z\n    key_x = key_x.clamp(\n        min=clamp_min,\n        max=clamp_max,\n    )\n    key_y = key_y.clamp(\n        min=clamp_min,\n        max=clamp_max,\n    )\n\n    key_x = key_x.reshape(\n        batch_size,\n        depth_count,\n        source_count * patch_count,\n    ).repeat_interleave(\n        group_size,\n        dim=1,\n    )\n    key_y = key_y.reshape(\n        batch_size,\n        depth_count,\n        source_count * patch_count,\n    ).repeat_interleave(\n        group_size,\n        dim=1,\n    )\n\n    active_head_count = depth_count * group_size\n    query_x = (grid_x + 0.5).reshape(\n        1,\n        1,\n        patch_count,\n    ).expand(\n        batch_size,\n        active_head_count,\n        patch_count,\n    )\n    query_y = (grid_y + 0.5).reshape(\n        1,\n        1,\n        patch_count,\n    ).expand(\n        batch_size,\n        active_head_count,\n        patch_count,\n    )\n\n    rope_block_dim = head_dim // 2\n    query_x_coefficients = _rope_coefficients(\n        query_x,\n        freq_base,\n        freq_scale,\n        rope_block_dim,\n        output_dtype,\n    )\n    query_y_coefficients = _rope_coefficients(\n        query_y,\n        freq_base,\n        freq_scale,\n        rope_block_dim,\n        output_dtype,\n    )\n    key_x_coefficients = _rope_coefficients(\n        key_x,\n        freq_base,\n        freq_scale,\n        rope_block_dim,\n        output_dtype,\n    )\n    key_y_coefficients = _rope_coefficients(\n        key_y,\n        freq_base,\n        freq_scale,\n        rope_block_dim,\n        output_dtype,\n    )\n    return (\n        query_x_coefficients,\n        query_y_coefficients,\n        key_x_coefficients,\n        key_y_coefficients,\n    )\n\n\nclass URoPEDotProductAttention(torch.nn.Module):\n    """Local cross-view URoPE around PyTorch SDPA."""\n\n    def __init__(\n        self,\n        head_num: int,\n        head_dim: int,\n        min_depth: float = 2.0,\n        max_depth: float = 20.0,\n        freq_base: float = 100.0,\n        freq_scale: float = 1.0,\n        group_size: int = 4,\n        leaveout_head: int = 0,\n        camera_convention: str = "opencv",\n        clamp_min: float = -64.0,\n        clamp_max: float = 96.0,\n    ):\n        super().__init__()\n        if head_dim % 4 != 0:\n            raise ValueError(\n                "URoPE requires head_dim divisible by 4, got {}.".format(\n                    head_dim\n                )\n            )\n        if leaveout_head < 0 or leaveout_head >= head_num:\n            raise ValueError(\n                "leaveout_head must be in [0, head_num), got {}.".format(\n                    leaveout_head\n                )\n            )\n\n        active_head_count = head_num - leaveout_head\n        if active_head_count % group_size != 0:\n            raise ValueError(\n                "head_num - leaveout_head must be divisible by group_size. "\n                "Got {} - {} and group_size {}.".format(\n                    head_num,\n                    leaveout_head,\n                    group_size,\n                )\n            )\n        if not 0.0 < min_depth < max_depth:\n            raise ValueError(\n                "Depth range must satisfy 0 < min_depth < max_depth."\n            )\n\n        self.head_num = head_num\n        self.head_dim = head_dim\n        self.leaveout_head = leaveout_head\n        self.group_size = group_size\n        self.depth_count = active_head_count // group_size\n        self.min_depth = min_depth\n        self.max_depth = max_depth\n        self.freq_base = freq_base\n        self.freq_scale = freq_scale\n        self.camera_convention = camera_convention\n        self.clamp_min = clamp_min\n        self.clamp_max = clamp_max\n\n    def forward(\n        self,\n        query: torch.Tensor,\n        key: torch.Tensor,\n        value: torch.Tensor,\n        *,\n        query_viewmats: torch.Tensor,\n        query_intrinsics: torch.Tensor,\n        source_viewmats: torch.Tensor,\n        source_intrinsics: torch.Tensor,\n        patch_width: int,\n        patch_height: int,\n        attention_mask: Optional[torch.Tensor] = None,\n        dropout_p: float = 0.0,\n        is_causal: bool = False,\n    ) -> torch.Tensor:\n        if key.shape != value.shape:\n            raise ValueError(\n                "URoPE key and value must have the same shape."\n            )\n        if query.ndim != 4 or key.ndim != 4:\n            raise ValueError(\n                "URoPE expects Q/K/V with shape [B,H,L,D]."\n            )\n\n        batch_size, head_count, query_length, head_dim = (\n            query.shape\n        )\n        key_batch_size, key_head_count, key_length, key_head_dim = (\n            key.shape\n        )\n        patch_count = patch_width * patch_height\n        source_count = source_viewmats.shape[1]\n\n        if (\n            key_batch_size != batch_size\n            or key_head_count != head_count\n            or key_head_dim != head_dim\n        ):\n            raise ValueError(\n                "URoPE Q and K batch/head dimensions do not match."\n            )\n        if head_count != self.head_num or head_dim != self.head_dim:\n            raise ValueError(\n                "QKV heads are ({}, {}), configured as ({}, {}).".format(\n                    head_count,\n                    head_dim,\n                    self.head_num,\n                    self.head_dim,\n                )\n            )\n        if query_length != patch_count:\n            raise ValueError(\n                "Local URoPE expects query length H*W = {}, got {}.".format(\n                    patch_count,\n                    query_length,\n                )\n            )\n        if key_length != source_count * patch_count:\n            raise ValueError(\n                "Local URoPE expects key length K*H*W = {}, got {}.".format(\n                    source_count * patch_count,\n                    key_length,\n                )\n            )\n        if query_viewmats.shape != (batch_size, 4, 4):\n            raise ValueError(\n                "query_viewmats has invalid shape {}.".format(\n                    tuple(query_viewmats.shape)\n                )\n            )\n        if query_intrinsics.shape != (batch_size, 3, 3):\n            raise ValueError(\n                "query_intrinsics has invalid shape {}.".format(\n                    tuple(query_intrinsics.shape)\n                )\n            )\n        if source_viewmats.shape != (\n            batch_size,\n            source_count,\n            4,\n            4,\n        ):\n            raise ValueError(\n                "source_viewmats has invalid shape {}.".format(\n                    tuple(source_viewmats.shape)\n                )\n            )\n        if source_intrinsics.shape != (\n            batch_size,\n            source_count,\n            3,\n            3,\n        ):\n            raise ValueError(\n                "source_intrinsics has invalid shape {}.".format(\n                    tuple(source_intrinsics.shape)\n                )\n            )\n\n        (\n            query_x_coefficients,\n            query_y_coefficients,\n            key_x_coefficients,\n            key_y_coefficients,\n        ) = _prepare_local_geometry_coefficients(\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_width=patch_width,\n            patch_height=patch_height,\n            depth_count=self.depth_count,\n            group_size=self.group_size,\n            min_depth=self.min_depth,\n            max_depth=self.max_depth,\n            freq_base=self.freq_base,\n            freq_scale=self.freq_scale,\n            head_dim=self.head_dim,\n            camera_convention=self.camera_convention,\n            output_dtype=query.dtype,\n            clamp_min=self.clamp_min,\n            clamp_max=self.clamp_max,\n        )\n\n        active_head_count = head_count - self.leaveout_head\n        query_active = _apply_xy_rope(\n            query[:, :active_head_count],\n            query_x_coefficients,\n            query_y_coefficients,\n        )\n        key_active = _apply_xy_rope(\n            key[:, :active_head_count],\n            key_x_coefficients,\n            key_y_coefficients,\n        )\n        value_active = _apply_xy_rope(\n            value[:, :active_head_count],\n            key_x_coefficients,\n            key_y_coefficients,\n        )\n\n        if self.leaveout_head > 0:\n            query = torch.cat(\n                [\n                    query_active,\n                    query[:, active_head_count:],\n                ],\n                dim=1,\n            )\n            key = torch.cat(\n                [\n                    key_active,\n                    key[:, active_head_count:],\n                ],\n                dim=1,\n            )\n            value = torch.cat(\n                [\n                    value_active,\n                    value[:, active_head_count:],\n                ],\n                dim=1,\n            )\n        else:\n            query = query_active\n            key = key_active\n            value = value_active\n\n        if attention_mask is not None:\n            attention_mask = attention_mask.to(\n                device=query.device\n            )\n            if attention_mask.shape[-2:] != (\n                query_length,\n                key_length,\n            ):\n                raise ValueError(\n                    "Local URoPE attention mask ends with {}, expected "\n                    "({}, {}).".format(\n                        tuple(attention_mask.shape[-2:]),\n                        query_length,\n                        key_length,\n                    )\n                )\n\n        output = F.scaled_dot_product_attention(\n            query=query,\n            key=key,\n            value=value,\n            attn_mask=attention_mask,\n            dropout_p=dropout_p,\n            is_causal=is_causal,\n        )\n\n        output_active = _apply_xy_rope(\n            output[:, :active_head_count],\n            query_x_coefficients,\n            query_y_coefficients,\n            inverse=True,\n        )\n        if self.leaveout_head > 0:\n            output = torch.cat(\n                [\n                    output_active,\n                    output[:, active_head_count:],\n                ],\n                dim=1,\n            )\n        else:\n            output = output_active\n\n        return output\n'.lstrip("\n")
        + "\n"
    )
updated["urope"] = urope_text

for name, text in updated.items():
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(
            f"Patched {name} has invalid Python syntax: {exc}"
        ) from exc

changed = [
    name
    for name in targets
    if updated[name] != original[name]
]
if not changed:
    print("Nothing to patch: files already look patched.")
    sys.exit(0)

backup_root = Path(
    tempfile.mkdtemp(prefix="urope_local_patch_backup_")
)
for name in changed:
    backup_path = backup_root / targets[name].name
    shutil.copy2(targets[name], backup_path)

print(f"Backup written to: {backup_root}")

for name in changed:
    targets[name].write_text(updated[name])

print("=== py_compile ===")
subprocess.run(
    [
        sys.executable,
        "-m",
        "py_compile",
        *[str(targets[name]) for name in changed],
    ],
    cwd=root,
    check=True,
)

print("=== git diff --stat ===")
subprocess.run(
    [
        "git",
        "diff",
        "--stat",
        "--",
        *[str(targets[name].relative_to(root)) for name in changed],
    ],
    cwd=root,
    check=False,
)

print("=== patched files ===")
for name in changed:
    print(targets[name].relative_to(root))

print()
print("Patch complete.")
print("Review with:")
print(
    "git diff -- "
    + " ".join(str(targets[name].relative_to(root)) for name in changed)
)
