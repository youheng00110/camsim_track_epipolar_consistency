#!/usr/bin/env python3
"""
Create a NEW URoPE-inside-TV implementation under src/dwm/models/lyh/.

The script does not modify the original TV source or dwm/models/urope/*.

Usage:
  python patch_urope_inside_tv.py /path/to/OpenDWM
  python patch_urope_inside_tv.py /path/to/OpenDWM --force
"""

import argparse
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


TARGET_CODE = 'from typing import Optional\n\nimport diffusers\nimport diffusers.models.attention_processor\nimport torch\n\nfrom dwm.models.crossview_temporal_dit_PLUCKER_TVROW import (\n    DiTCrossviewTemporalConditionModel as TVModelBase,\n    build_tv_view_index_from_crossview_mask,\n)\nfrom dwm.models.urope.urope import URoPEDotProductAttention\n\n\nclass VTURoPETVAttentionBlock(torch.nn.Module):\n    """\n    Joint TV attention with URoPE applied inside the same attention operation.\n\n    Query:\n        current (t, v), all HW tokens -> [N, HW, C]\n\n    Context:\n        [t-1, t, t+1] x [left, self, right] x HW\n        -> [N, 9*HW, C]\n\n    Temporal position:\n        supplied by the original TV time embedding before Q/K/V projection.\n\n    Camera geometry:\n        URoPE is applied inside this TV attention using the target camera and\n        each gathered source time-view slot.\n\n    No camera SlotID embedding is added.\n    """\n\n    def __init__(\n        self,\n        inner_dim: int,\n        context_dim: int,\n        num_attention_heads: int,\n        attention_head_dim: int,\n        urope_config: Optional[dict] = None,\n        qk_norm=None,\n        ff_mult: int = 4,\n        dropout: float = 0.0,\n    ):\n        super().__init__()\n        self.inner_dim = inner_dim\n        self.context_dim = context_dim\n        self.num_attention_heads = num_attention_heads\n        self.attention_head_dim = attention_head_dim\n        self.inner_attention_dim = num_attention_heads * attention_head_dim\n        self.dropout = float(dropout)\n\n        self.norm_q = torch.nn.LayerNorm(inner_dim)\n        self.norm_context = torch.nn.LayerNorm(context_dim)\n\n        # Keep the original TV parameter names/shapes for checkpoint reuse.\n        self.q_proj = torch.nn.Linear(\n            inner_dim,\n            self.inner_attention_dim,\n            bias=False,\n        )\n        self.k_proj = torch.nn.Linear(\n            context_dim,\n            self.inner_attention_dim,\n            bias=False,\n        )\n        self.v_proj = torch.nn.Linear(\n            context_dim,\n            self.inner_attention_dim,\n            bias=False,\n        )\n        self.out_proj = torch.nn.Linear(\n            self.inner_attention_dim,\n            inner_dim,\n            bias=False,\n        )\n\n        self.qk_norm_q = None\n        self.qk_norm_k = None\n        if qk_norm is not None:\n            qk_norm_helper = diffusers.models.attention_processor.Attention(\n                query_dim=self.inner_attention_dim,\n                cross_attention_dim=self.inner_attention_dim,\n                heads=num_attention_heads,\n                dim_head=attention_head_dim,\n                qk_norm=qk_norm,\n                bias=False,\n            )\n            self.qk_norm_q = qk_norm_helper.norm_q\n            self.qk_norm_k = qk_norm_helper.norm_k\n\n        self.urope_attention = URoPEDotProductAttention(\n            head_num=num_attention_heads,\n            head_dim=attention_head_dim,\n            **(urope_config or {}),\n        )\n\n        self.norm_ff = torch.nn.LayerNorm(inner_dim)\n        self.ff_in = torch.nn.Linear(\n            inner_dim,\n            inner_dim * ff_mult,\n        )\n        self.ff_act = torch.nn.GELU(approximate="tanh")\n        self.ff_out = torch.nn.Linear(\n            inner_dim * ff_mult,\n            inner_dim,\n        )\n\n    def forward(\n        self,\n        query_hidden_states: torch.Tensor,\n        context_hidden_states: torch.Tensor,\n        *,\n        query_viewmats: torch.Tensor,\n        query_intrinsics: torch.Tensor,\n        source_viewmats: torch.Tensor,\n        source_intrinsics: torch.Tensor,\n        patch_width: int,\n        patch_height: int,\n        context_attention_mask: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:\n        residual = query_hidden_states\n\n        query_hidden_states = self.norm_q(query_hidden_states)\n        context_hidden_states = self.norm_context(context_hidden_states)\n\n        q = self.q_proj(query_hidden_states)\n        k = self.k_proj(context_hidden_states)\n        v = self.v_proj(context_hidden_states)\n\n        batch_size, query_length, _ = q.shape\n        context_length = k.shape[1]\n\n        q = q.view(\n            batch_size,\n            query_length,\n            self.num_attention_heads,\n            self.attention_head_dim,\n        ).transpose(1, 2)\n        k = k.view(\n            batch_size,\n            context_length,\n            self.num_attention_heads,\n            self.attention_head_dim,\n        ).transpose(1, 2)\n        v = v.view(\n            batch_size,\n            context_length,\n            self.num_attention_heads,\n            self.attention_head_dim,\n        ).transpose(1, 2)\n\n        if self.qk_norm_q is not None:\n            q = self.qk_norm_q(q)\n        if self.qk_norm_k is not None:\n            k = self.qk_norm_k(k)\n\n        attention_mask = None\n        if context_attention_mask is not None:\n            if context_attention_mask.shape != (\n                batch_size,\n                context_length,\n            ):\n                raise ValueError(\n                    "context_attention_mask should be [N,K], "\n                    f"but got {tuple(context_attention_mask.shape)} "\n                    f"for N={batch_size}, K={context_length}."\n                )\n            attention_mask = context_attention_mask.to(\n                device=q.device,\n                dtype=torch.bool,\n            )[:, None, None, :].expand(\n                batch_size,\n                1,\n                query_length,\n                context_length,\n            )\n\n        attention_output = self.urope_attention(\n            q,\n            k,\n            v,\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_width=patch_width,\n            patch_height=patch_height,\n            attention_mask=attention_mask,\n            dropout_p=self.dropout if self.training else 0.0,\n            is_causal=False,\n        )\n\n        attention_output = attention_output.transpose(\n            1,\n            2,\n        ).contiguous().view(\n            batch_size,\n            query_length,\n            self.inner_attention_dim,\n        )\n\n        hidden_states = residual + self.out_proj(attention_output)\n        hidden_states = hidden_states + self.ff_out(\n            self.ff_act(\n                self.ff_in(\n                    self.norm_ff(hidden_states)\n                )\n            )\n        )\n        return hidden_states\n\n\nclass DiTCrossviewTemporalConditionModel(TVModelBase):\n    """\n    OpenDWM DiT with URoPE directly inside TV attention.\n\n    This is NOT:\n        cross-view URoPE attention -> TV attention\n\n    It is one joint attention:\n        TV selects the local temporal-view context;\n        URoPE provides camera geometry inside that same attention;\n        the original TV time embedding provides temporal position.\n\n    Independent cross-view attention is disabled, so Camera SlotID /\n    view_pos_embeds are not used.\n    """\n\n    @diffusers.configuration_utils.register_to_config\n    def __init__(\n        self,\n        patch_size: int = 2,\n        num_layers: int = 18,\n        attention_head_dim: int = 64,\n        num_attention_heads: int = 18,\n        projection_class_embeddings_input_dim: int = None,\n        condition_image_adapter_config: Optional[dict] = None,\n        enable_crossview: bool = False,\n        enable_temporal: bool = False,\n        enable_tv: bool = True,\n        urope_config: Optional[dict] = None,\n        crossview_attention_type: str = "full",\n        temporal_attention_type: str = None,\n        tv_attention_type: str = "full",\n        merge_factor: float = 2,\n        merge_strategy: str = "learned_with_images",\n        crossview_block_layers: Optional[dict] = None,\n        temporal_block_layers: Optional[dict] = None,\n        tv_block_layers: Optional[dict] = None,\n        crossview_gradient_checkpointing: bool = False,\n        temporal_gradient_checkpointing: bool = False,\n        tv_gradient_checkpointing: bool = False,\n        mixer_type: str = "AlphaBlender",\n        perspective_modeling_type: str = "urope",\n        disable_view_emb_on_temporal_module: bool = False,\n        qk_norm_on_additional_modules=None,\n        mask_module=None,\n        tv_time_radius: int = 1,\n        tv_view_radius: int = 1,\n        tv_height_chunk_size: int = 0,\n        tv_full_batch_chunk_size: int = 1,\n        **kwargs,\n    ):\n        # Accepted for old config compatibility, but intentionally unused.\n        del enable_crossview\n        del crossview_gradient_checkpointing\n\n        if perspective_modeling_type != "urope":\n            raise ValueError(\n                "URoPE-inside-TV requires "\n                "perspective_modeling_type=\'urope\'."\n            )\n        if not enable_tv:\n            raise ValueError(\n                "URoPE-inside-TV requires enable_tv=True."\n            )\n        if tv_attention_type != "full":\n            raise ValueError(\n                "URoPE-inside-TV currently requires "\n                "tv_attention_type=\'full\'."\n            )\n\n        self.urope_config = {\n            "min_depth": 2.0,\n            "max_depth": 20.0,\n            "freq_base": 100.0,\n            "freq_scale": 1.0,\n            "group_size": 4,\n            "leaveout_head": 0,\n            "camera_convention": "opencv",\n            **(urope_config or {}),\n        }\n\n        # No independent cross-view branch. TV still receives and uses\n        # crossview_attention_mask to choose left/self/right source views.\n        super().__init__(\n            patch_size=patch_size,\n            num_layers=num_layers,\n            attention_head_dim=attention_head_dim,\n            num_attention_heads=num_attention_heads,\n            projection_class_embeddings_input_dim=(\n                projection_class_embeddings_input_dim\n            ),\n            condition_image_adapter_config=condition_image_adapter_config,\n            enable_crossview=False,\n            enable_temporal=enable_temporal,\n            enable_tv=True,\n            crossview_attention_type=crossview_attention_type,\n            temporal_attention_type=temporal_attention_type,\n            tv_attention_type=tv_attention_type,\n            merge_factor=merge_factor,\n            merge_strategy=merge_strategy,\n            crossview_block_layers=crossview_block_layers,\n            temporal_block_layers=temporal_block_layers,\n            tv_block_layers=tv_block_layers,\n            crossview_gradient_checkpointing=False,\n            temporal_gradient_checkpointing=(\n                temporal_gradient_checkpointing\n            ),\n            tv_gradient_checkpointing=tv_gradient_checkpointing,\n            mixer_type=mixer_type,\n            perspective_modeling_type="urope",\n            disable_view_emb_on_temporal_module=(\n                disable_view_emb_on_temporal_module\n            ),\n            qk_norm_on_additional_modules=(\n                qk_norm_on_additional_modules\n            ),\n            mask_module=mask_module,\n            tv_time_radius=tv_time_radius,\n            tv_view_radius=tv_view_radius,\n            tv_height_chunk_size=tv_height_chunk_size,\n            tv_full_batch_chunk_size=tv_full_batch_chunk_size,\n            **kwargs,\n        )\n\n        self.register_to_config(\n            enable_crossview=False,\n            enable_tv=True,\n            perspective_modeling_type="urope",\n        )\n\n        inner_dim = attention_head_dim * num_attention_heads\n        self.tv_transformer_blocks = torch.nn.ModuleList([\n            VTURoPETVAttentionBlock(\n                inner_dim=inner_dim,\n                context_dim=inner_dim,\n                num_attention_heads=num_attention_heads,\n                attention_head_dim=attention_head_dim,\n                urope_config=self.urope_config,\n                qk_norm=qk_norm_on_additional_modules,\n            )\n            for _ in range(len(self.tv_block_layers))\n        ])\n\n    def forward(\n        self,\n        *args,\n        camera_intrinsics_norm=None,\n        camera2referego=None,\n        **kwargs,\n    ):\n        if camera_intrinsics_norm is None or camera2referego is None:\n            raise ValueError(\n                "URoPE-inside-TV requires camera_intrinsics_norm "\n                "and camera2referego."\n            )\n\n        # Keep geometry for checkpoint recomputation during backward.\n        self._urope_tv_camera_intrinsics_norm = camera_intrinsics_norm\n        self._urope_tv_camera2referego = camera2referego\n\n        return super().forward(\n            *args,\n            camera_intrinsics_norm=camera_intrinsics_norm,\n            camera2referego=camera2referego,\n            **kwargs,\n        )\n\n    def forward_tv_full_block_and_mix_result(\n        self,\n        tv_block: torch.nn.Module,\n        mixer,\n        hidden_states: torch.Tensor,\n        tv_emb: torch.Tensor,\n        batch_size: int,\n        sequence_length: int,\n        view_count: int,\n        width: int,\n        height: int,\n        disable_tv: torch.BoolTensor,\n        crossview_attention_mask: torch.Tensor,\n        crossview_attention_index: torch.Tensor,\n    ):\n        del crossview_attention_index\n\n        if self.tv_attention_type != "full":\n            raise ValueError(\n                "forward_tv_full_block_and_mix_result requires "\n                f"tv_attention_type=\'full\', got "\n                f"{self.tv_attention_type!r}."\n            )\n\n        if hidden_states.ndim != 3:\n            raise ValueError(\n                "hidden_states should be [B*T*V, HW, C], "\n                f"but got {tuple(hidden_states.shape)}."\n            )\n\n        camera_intrinsics_norm = getattr(\n            self,\n            "_urope_tv_camera_intrinsics_norm",\n            None,\n        )\n        camera2referego = getattr(\n            self,\n            "_urope_tv_camera2referego",\n            None,\n        )\n        if camera_intrinsics_norm is None or camera2referego is None:\n            raise ValueError(\n                "URoPE TV geometry cache is empty. "\n                "Call model.forward with camera_intrinsics_norm and "\n                "camera2referego."\n            )\n\n        device = hidden_states.device\n        dtype = hidden_states.dtype\n        channel = hidden_states.shape[-1]\n        token_count = height * width\n        expected_flat_batch = (\n            batch_size * sequence_length * view_count\n        )\n\n        if hidden_states.shape[0] != expected_flat_batch:\n            raise ValueError(\n                "Flattened batch mismatch: "\n                f"hidden_states.shape[0]={hidden_states.shape[0]}, "\n                f"expected={expected_flat_batch}."\n            )\n        if hidden_states.shape[1] != token_count:\n            raise ValueError(\n                "Spatial token mismatch: "\n                f"hidden_states.shape[1]={hidden_states.shape[1]}, "\n                f"height*width={token_count}."\n            )\n\n        # With perspective_modeling_type=\'urope\', view_cam_emb in TVModelBase\n        # stays zero, so tv_emb contains temporal position only.\n        tv_hidden_states = hidden_states + tv_emb.to(\n            device=device,\n            dtype=dtype,\n        )\n        tv_hidden_states = tv_hidden_states.reshape(\n            batch_size,\n            sequence_length,\n            view_count,\n            token_count,\n            channel,\n        )\n\n        intrinsics = camera_intrinsics_norm.to(\n            device=device,\n            dtype=torch.float32,\n        ).clone().reshape(\n            batch_size,\n            sequence_length,\n            view_count,\n            3,\n            3,\n        )\n        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width\n        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height\n        intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width\n        intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height\n\n        camera2referego = camera2referego.to(\n            device=device,\n            dtype=torch.float32,\n        ).reshape(\n            batch_size,\n            sequence_length,\n            view_count,\n            4,\n            4,\n        )\n        viewmats = torch.linalg.inv(camera2referego)\n\n        time_base = torch.arange(\n            sequence_length,\n            device=device,\n            dtype=torch.long,\n        )\n        time_offsets = torch.tensor(\n            [-1, 0, 1],\n            device=device,\n            dtype=torch.long,\n        )\n        raw_time_index = (\n            time_base[:, None] + time_offsets[None, :]\n        )\n\n        # Preserve the current TV semantics: boundary time slots clamp and\n        # still participate.\n        time_index = raw_time_index.clamp(\n            0,\n            sequence_length - 1,\n        )\n        time_valid_mask = torch.ones_like(\n            time_index,\n            dtype=torch.bool,\n        )\n\n        # Preserve current TV semantics: missing left/right uses self.\n        view_index, view_valid_mask = (\n            build_tv_view_index_from_crossview_mask(\n                crossview_attention_mask,\n                batch_size,\n                view_count,\n                device,\n            )\n        )\n\n        flat_batch_index = torch.arange(\n            batch_size,\n            device=device,\n            dtype=torch.long,\n        )[:, None, None].expand(\n            batch_size,\n            sequence_length,\n            view_count,\n        ).reshape(-1)\n\n        flat_time_index = torch.arange(\n            sequence_length,\n            device=device,\n            dtype=torch.long,\n        )[None, :, None].expand(\n            batch_size,\n            sequence_length,\n            view_count,\n        ).reshape(-1)\n\n        flat_view_index = torch.arange(\n            view_count,\n            device=device,\n            dtype=torch.long,\n        )[None, None, :].expand(\n            batch_size,\n            sequence_length,\n            view_count,\n        ).reshape(-1)\n\n        target_count = expected_flat_batch\n        chunk_size = min(\n            self.tv_full_batch_chunk_size,\n            target_count,\n        )\n        output_chunks = []\n\n        for chunk_start in range(\n            0,\n            target_count,\n            chunk_size,\n        ):\n            chunk_end = min(\n                chunk_start + chunk_size,\n                target_count,\n            )\n\n            target_b = flat_batch_index[\n                chunk_start:chunk_end\n            ]\n            target_t = flat_time_index[\n                chunk_start:chunk_end\n            ]\n            target_v = flat_view_index[\n                chunk_start:chunk_end\n            ]\n            current_chunk_size = target_b.shape[0]\n\n            query_hidden_states = tv_hidden_states[\n                target_b,\n                target_t,\n                target_v,\n            ]\n\n            source_time_index = time_index[target_t]\n            source_time_valid = time_valid_mask[target_t]\n            source_view_index = view_index[\n                target_b,\n                target_v,\n            ]\n            source_view_valid = view_valid_mask[\n                target_b,\n                target_v,\n            ]\n\n            local_hidden_states = tv_hidden_states[\n                target_b[:, None, None],\n                source_time_index[:, :, None],\n                source_view_index[:, None, :],\n            ]\n\n            expected_local_shape = (\n                current_chunk_size,\n                3,\n                3,\n                token_count,\n                channel,\n            )\n            if tuple(local_hidden_states.shape) != (\n                expected_local_shape\n            ):\n                raise RuntimeError(\n                    "TV-full gather shape mismatch: "\n                    f"got {tuple(local_hidden_states.shape)}, "\n                    f"expected {expected_local_shape}."\n                )\n\n            context_hidden_states = (\n                local_hidden_states.reshape(\n                    current_chunk_size,\n                    9 * token_count,\n                    channel,\n                )\n            )\n\n            context_slot_valid = (\n                source_time_valid[:, :, None]\n                & source_view_valid[:, None, :]\n            )\n            context_attention_mask = (\n                context_slot_valid[\n                    :,\n                    :,\n                    :,\n                    None,\n                ].expand(\n                    current_chunk_size,\n                    3,\n                    3,\n                    token_count,\n                ).reshape(\n                    current_chunk_size,\n                    9 * token_count,\n                )\n            )\n\n            # Target camera geometry.\n            query_viewmats = viewmats[\n                target_b,\n                target_t,\n                target_v,\n            ]\n            query_intrinsics = intrinsics[\n                target_b,\n                target_t,\n                target_v,\n            ]\n\n            # Geometry for each actual (source_time, source_view) TV slot.\n            source_viewmats = viewmats[\n                target_b[:, None, None],\n                source_time_index[:, :, None],\n                source_view_index[:, None, :],\n            ].reshape(\n                current_chunk_size,\n                9,\n                4,\n                4,\n            )\n            source_intrinsics = intrinsics[\n                target_b[:, None, None],\n                source_time_index[:, :, None],\n                source_view_index[:, None, :],\n            ].reshape(\n                current_chunk_size,\n                9,\n                3,\n                3,\n            )\n\n            tv_chunk = tv_block(\n                query_hidden_states,\n                context_hidden_states,\n                query_viewmats=query_viewmats,\n                query_intrinsics=query_intrinsics,\n                source_viewmats=source_viewmats,\n                source_intrinsics=source_intrinsics,\n                patch_width=width,\n                patch_height=height,\n                context_attention_mask=(\n                    context_attention_mask\n                ),\n            )\n\n            if tv_chunk.shape != query_hidden_states.shape:\n                raise RuntimeError(\n                    "URoPE-TV output shape mismatch: "\n                    f"output={tuple(tv_chunk.shape)}, "\n                    f"query={tuple(query_hidden_states.shape)}."\n                )\n\n            output_chunks.append(tv_chunk)\n\n        tv_hidden_states = torch.cat(\n            output_chunks,\n            dim=0,\n        )\n        if tv_hidden_states.shape != hidden_states.shape:\n            raise RuntimeError(\n                "URoPE-TV final shape mismatch: "\n                f"tv={tuple(tv_hidden_states.shape)}, "\n                f"hidden={tuple(hidden_states.shape)}."\n            )\n\n        if mixer is None:\n            return tv_hidden_states\n\n        return mixer(\n            hidden_states.reshape(\n                batch_size,\n                sequence_length * view_count,\n                token_count,\n                channel,\n            ),\n            tv_hidden_states.reshape(\n                batch_size,\n                sequence_length * view_count,\n                token_count,\n                channel,\n            ),\n            image_only_indicator=disable_tv,\n        ).flatten(0, 1)\n'

def git_status(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="OpenDWM repository root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite only the generated lyh target if it already exists",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    models_dir = root / "src" / "dwm" / "models"
    tv_source = models_dir / "crossview_temporal_dit_PLUCKER_TVROW.py"
    urope_core = models_dir / "urope" / "urope.py"
    root_urope_wrapper = models_dir / "crossview_temporal_dit_urope.py"

    lyh_dir = models_dir / "lyh"
    lyh_urope_wrapper = lyh_dir / "crossview_temporal_dit_urope.py"
    target = lyh_dir / "crossview_temporal_dit_urope_tv.py"

    if not models_dir.is_dir():
        raise SystemExit(f"models directory not found: {models_dir}")
    if not tv_source.is_file():
        raise SystemExit(f"TV source not found: {tv_source}")
    if not urope_core.is_file():
        raise SystemExit(f"URoPE core not found: {urope_core}")

    tv_text = tv_source.read_text()
    for symbol in (
        "build_tv_view_index_from_crossview_mask",
        "forward_tv_full_block_and_mix_result",
        "tv_time_pos_embeds",
    ):
        if symbol not in tv_text:
            raise SystemExit(
                f"TV source does not contain required symbol: {symbol}"
            )

    urope_text = urope_core.read_text()
    for symbol in (
        "class URoPEDotProductAttention",
        "query_viewmats",
        "source_viewmats",
        "source_intrinsics",
    ):
        if symbol not in urope_text:
            raise SystemExit(
                "Current dwm.models.urope.urope is not the local "
                f"query/source version; missing: {symbol}"
            )

    print("=== BEFORE git status ===")
    print(git_status(root) or "(clean or git unavailable)")
    print()

    lyh_dir.mkdir(parents=True, exist_ok=True)

    init_file = lyh_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    # Keep an existing lyh pure-URoPE file exactly as-is.
    # If it is absent and a root-level wrapper exists, copy it unchanged.
    if lyh_urope_wrapper.exists():
        print(f"Kept existing file unchanged: {lyh_urope_wrapper}")
    elif root_urope_wrapper.exists():
        shutil.copy2(root_urope_wrapper, lyh_urope_wrapper)
        print(
            "Copied pure URoPE wrapper unchanged:\n"
            f"  {root_urope_wrapper}\n"
            f"  -> {lyh_urope_wrapper}"
        )
    else:
        print(
            "No pure URoPE wrapper copied; source not found:\n"
            f"  {root_urope_wrapper}"
        )

    if target.exists():
        if not args.force:
            raise SystemExit(
                "\nTarget already exists; nothing was overwritten:\n"
                f"  {target}\n"
                "Use --force only if you want to replace this generated file."
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = target.with_name(f"{target.name}.bak.{timestamp}")
        shutil.copy2(target, backup)
        print(f"Backed up old generated target: {backup}")

    target.write_text(TARGET_CODE)
    py_compile.compile(str(target), doraise=True)

    print()
    print("=== CREATED ===")
    print(target)
    print()
    print("Original TV source preserved:")
    print(f"  {tv_source}")
    print("Original URoPE core preserved:")
    print(f"  {urope_core}")
    print()
    print("Use model class:")
    print(
        "  dwm.models.lyh.crossview_temporal_dit_urope_tv."
        "DiTCrossviewTemporalConditionModel"
    )
    print()
    print("Recommended core config:")
    print('  "enable_tv": true')
    print('  "enable_crossview": false')
    print('  "perspective_modeling_type": "urope"')
    print('  "tv_attention_type": "full"')
    print('  "enable_temporal": false  # pure joint TV+URoPE')
    print()
    print("=== AFTER git status ===")
    print(git_status(root) or "(clean or git unavailable)")


if __name__ == "__main__":
    main()
