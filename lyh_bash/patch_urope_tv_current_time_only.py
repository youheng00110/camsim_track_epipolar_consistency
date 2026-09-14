#!/usr/bin/env python3
import argparse
import ast
import json
import py_compile
import shutil
from datetime import datetime
from pathlib import Path

CURRENT_TIME_UROPE = '"""\nSelective URoPE for a 9-slot TV context.\n\nOnly one contiguous K/V token interval receives camera-geometry URoPE.\nAll other K/V tokens remain vanilla. Q remains in its original basis.\n\nFor selected K/V, apply the relative transform R_q^{-1} R_k. This lets\nselected URoPE keys and unselected vanilla temporal keys coexist in a single\nscaled-dot-product attention with one global softmax.\n"""\n\nfrom typing import Optional\n\nimport torch\nimport torch.nn.functional as F\n\nfrom dwm.models.urope.urope import (\n    URoPEDotProductAttention,\n    _apply_xy_rope,\n    _prepare_local_geometry_coefficients,\n)\n\n\nclass CurrentTimeURoPEDotProductAttention(URoPEDotProductAttention):\n    """URoPE only on a selected current-time K/V interval."""\n\n    def forward(\n        self,\n        query: torch.Tensor,\n        key: torch.Tensor,\n        value: torch.Tensor,\n        *,\n        query_viewmats: torch.Tensor,\n        query_intrinsics: torch.Tensor,\n        source_viewmats: torch.Tensor,\n        source_intrinsics: torch.Tensor,\n        patch_width: int,\n        patch_height: int,\n        geometry_key_start: int,\n        attention_mask: Optional[torch.Tensor] = None,\n        dropout_p: float = 0.0,\n        is_causal: bool = False,\n    ) -> torch.Tensor:\n        if key.shape != value.shape:\n            raise ValueError(\n                "URoPE key and value must have the same shape."\n            )\n        if query.ndim != 4 or key.ndim != 4:\n            raise ValueError(\n                "URoPE expects Q/K/V with shape [B,H,L,D]."\n            )\n\n        batch_size, head_count, query_length, head_dim = query.shape\n        (\n            key_batch_size,\n            key_head_count,\n            key_length,\n            key_head_dim,\n        ) = key.shape\n\n        patch_count = patch_width * patch_height\n        source_count = source_viewmats.shape[1]\n        geometry_key_length = source_count * patch_count\n        geometry_key_end = geometry_key_start + geometry_key_length\n\n        if (\n            key_batch_size != batch_size\n            or key_head_count != head_count\n            or key_head_dim != head_dim\n        ):\n            raise ValueError(\n                "URoPE Q and K batch/head dimensions do not match."\n            )\n        if head_count != self.head_num or head_dim != self.head_dim:\n            raise ValueError(\n                "QKV heads are ({}, {}), configured as ({}, {}).".format(\n                    head_count,\n                    head_dim,\n                    self.head_num,\n                    self.head_dim,\n                )\n            )\n        if query_length != patch_count:\n            raise ValueError(\n                "Current-time URoPE expects query length H*W = {}, "\n                "got {}.".format(patch_count, query_length)\n            )\n        if geometry_key_start < 0 or geometry_key_end > key_length:\n            raise ValueError(\n                "URoPE geometry interval [{}, {}) is outside K/V "\n                "length {}.".format(\n                    geometry_key_start,\n                    geometry_key_end,\n                    key_length,\n                )\n            )\n        if query_viewmats.shape != (batch_size, 4, 4):\n            raise ValueError(\n                "query_viewmats has invalid shape {}.".format(\n                    tuple(query_viewmats.shape)\n                )\n            )\n        if query_intrinsics.shape != (batch_size, 3, 3):\n            raise ValueError(\n                "query_intrinsics has invalid shape {}.".format(\n                    tuple(query_intrinsics.shape)\n                )\n            )\n        if source_viewmats.shape != (\n            batch_size,\n            source_count,\n            4,\n            4,\n        ):\n            raise ValueError(\n                "source_viewmats has invalid shape {}.".format(\n                    tuple(source_viewmats.shape)\n                )\n            )\n        if source_intrinsics.shape != (\n            batch_size,\n            source_count,\n            3,\n            3,\n        ):\n            raise ValueError(\n                "source_intrinsics has invalid shape {}.".format(\n                    tuple(source_intrinsics.shape)\n                )\n            )\n\n        (\n            query_x_coefficients,\n            query_y_coefficients,\n            key_x_coefficients,\n            key_y_coefficients,\n        ) = _prepare_local_geometry_coefficients(\n            query_viewmats=query_viewmats,\n            query_intrinsics=query_intrinsics,\n            source_viewmats=source_viewmats,\n            source_intrinsics=source_intrinsics,\n            patch_width=patch_width,\n            patch_height=patch_height,\n            depth_count=self.depth_count,\n            group_size=self.group_size,\n            min_depth=self.min_depth,\n            max_depth=self.max_depth,\n            freq_base=self.freq_base,\n            freq_scale=self.freq_scale,\n            head_dim=self.head_dim,\n            camera_convention=self.camera_convention,\n            output_dtype=query.dtype,\n            clamp_min=self.clamp_min,\n            clamp_max=self.clamp_max,\n        )\n\n        active_head_count = head_count - self.leaveout_head\n        key_active = key[:, :active_head_count]\n        value_active = value[:, :active_head_count]\n\n        selected_key = key_active[\n            :,\n            :,\n            geometry_key_start:geometry_key_end,\n        ]\n        selected_value = value_active[\n            :,\n            :,\n            geometry_key_start:geometry_key_end,\n        ]\n\n        # Relative URoPE on selected current-time K/V only:\n        # R_q^{-1} R_k.\n        selected_key = _apply_xy_rope(\n            selected_key,\n            key_x_coefficients,\n            key_y_coefficients,\n        )\n        selected_key = _apply_xy_rope(\n            selected_key,\n            query_x_coefficients,\n            query_y_coefficients,\n            inverse=True,\n        )\n\n        selected_value = _apply_xy_rope(\n            selected_value,\n            key_x_coefficients,\n            key_y_coefficients,\n        )\n        selected_value = _apply_xy_rope(\n            selected_value,\n            query_x_coefficients,\n            query_y_coefficients,\n            inverse=True,\n        )\n\n        key_active = torch.cat(\n            [\n                key_active[:, :, :geometry_key_start],\n                selected_key,\n                key_active[:, :, geometry_key_end:],\n            ],\n            dim=2,\n        )\n        value_active = torch.cat(\n            [\n                value_active[:, :, :geometry_key_start],\n                selected_value,\n                value_active[:, :, geometry_key_end:],\n            ],\n            dim=2,\n        )\n\n        if self.leaveout_head > 0:\n            key = torch.cat(\n                [key_active, key[:, active_head_count:]],\n                dim=1,\n            )\n            value = torch.cat(\n                [value_active, value[:, active_head_count:]],\n                dim=1,\n            )\n        else:\n            key = key_active\n            value = value_active\n\n        if attention_mask is not None:\n            attention_mask = attention_mask.to(device=query.device)\n            if attention_mask.shape[-2:] != (\n                query_length,\n                key_length,\n            ):\n                raise ValueError(\n                    "Current-time URoPE attention mask ends with {}, "\n                    "expected ({}, {}).".format(\n                        tuple(attention_mask.shape[-2:]),\n                        query_length,\n                        key_length,\n                    )\n                )\n\n        return F.scaled_dot_product_attention(\n            query=query,\n            key=key,\n            value=value,\n            attn_mask=attention_mask,\n            dropout_p=dropout_p,\n            is_causal=is_causal,\n        )\n'

TRACK_WRAPPER = '"""\nPVTrack wrapper for current-time-only URoPE-TV.\n\nPVTrack remains unchanged:\nbox/map/instance-flow -> 9-channel ImageAdapter -> DiT residuals.\n"""\n\nfrom dwm.models.lyh.crossview_temporal_dit_urope_tv_tself import (\n    DiTCrossviewTemporalConditionModel as URoPETVCurrentTimeModel,\n)\n\n\nclass DiTCrossviewTemporalConditionModel(URoPETVCurrentTimeModel):\n    def forward(\n        self,\n        *args,\n        camera_param_token=None,\n        camera_token_mask=None,\n        bbox_token_input=None,\n        bbox_class_input=None,\n        bbox_mask_input=None,\n        map_token_input=None,\n        **kwargs,\n    ):\n        del camera_param_token\n        del camera_token_mask\n        del bbox_token_input\n        del bbox_class_input\n        del bbox_mask_input\n        del map_token_input\n        return super().forward(*args, **kwargs)\n'

def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(path.name + ".bak." + stamp)
    shutil.copy2(path, dst)
    return dst


def refuse_or_backup(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise SystemExit(
            "Generated target already exists; refusing to overwrite:\n"
            f"  {path}\n"
            "Use --force to back it up and regenerate."
        )
    print("backup:", backup(path))


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{label}: expected exactly one match, found {count}."
        )
    return text.replace(old, new, 1)


def patch_tv_source(source: str) -> str:
    text = source

    text = replace_once(
        text,
        "from dwm.models.urope.urope import URoPEDotProductAttention\n",
        "from dwm.models.lyh.urope_current_time import (\n"
        "    CurrentTimeURoPEDotProductAttention,\n"
        ")\n",
        "URoPE import",
    )

    text = replace_once(
        text,
        "self.urope_attention = URoPEDotProductAttention(\n",
        "self.urope_attention = CurrentTimeURoPEDotProductAttention(\n",
        "attention class",
    )

    old_doc = (
        "    Camera geometry:\n"
        "        URoPE is applied inside this TV attention using the target camera and\n"
        "        each gathered source time-view slot.\n"
    )
    new_doc = (
        "    Camera geometry:\n"
        "        TV still attends all 9 time-view slots. URoPE geometry is computed\n"
        "        only for CURRENT time t x [left, self, right]. The t-1/t+1 K/V\n"
        "        slots remain vanilla and do not reuse current-time geometry.\n"
    )
    text = replace_once(text, old_doc, new_doc, "TV block docstring")

    old_sig = (
        "        source_intrinsics: torch.Tensor,\n"
        "        patch_width: int,\n"
        "        patch_height: int,\n"
        "        context_attention_mask: Optional[torch.Tensor] = None,\n"
    )
    new_sig = (
        "        source_intrinsics: torch.Tensor,\n"
        "        patch_width: int,\n"
        "        patch_height: int,\n"
        "        geometry_key_start: int,\n"
        "        context_attention_mask: Optional[torch.Tensor] = None,\n"
    )
    text = replace_once(
        text, old_sig, new_sig, "TV block forward signature"
    )

    old_call = (
        "            patch_width=patch_width,\n"
        "            patch_height=patch_height,\n"
        "            attention_mask=attention_mask,\n"
    )
    new_call = (
        "            patch_width=patch_width,\n"
        "            patch_height=patch_height,\n"
        "            geometry_key_start=geometry_key_start,\n"
        "            attention_mask=attention_mask,\n"
    )
    text = replace_once(
        text, old_call, new_call, "URoPE attention call"
    )

    old_geometry = (
        "            # Geometry for each actual (source_time, source_view) TV slot.\n"
        "            source_viewmats = viewmats[\n"
        "                target_b[:, None, None],\n"
        "                source_time_index[:, :, None],\n"
        "                source_view_index[:, None, :],\n"
        "            ].reshape(\n"
        "                current_chunk_size,\n"
        "                9,\n"
        "                4,\n"
        "                4,\n"
        "            )\n"
        "            source_intrinsics = intrinsics[\n"
        "                target_b[:, None, None],\n"
        "                source_time_index[:, :, None],\n"
        "                source_view_index[:, None, :],\n"
        "            ].reshape(\n"
        "                current_chunk_size,\n"
        "                9,\n"
        "                3,\n"
        "                3,\n"
        "            )\n"
    )
    new_geometry = (
        "            # Current-time-only URoPE geometry.\n"
        "            # TV content remains 3 time x 3 view = 9 slots.\n"
        "            # Geometry exists only at target_t for [left,self,right].\n"
        "            # There is NO geometry reuse onto t-1/t+1.\n"
        "            source_viewmats = viewmats[\n"
        "                target_b[:, None],\n"
        "                target_t[:, None],\n"
        "                source_view_index,\n"
        "            ]\n"
        "            source_intrinsics = intrinsics[\n"
        "                target_b[:, None],\n"
        "                target_t[:, None],\n"
        "                source_view_index,\n"
        "            ]\n"
        "\n"
        "            expected_source_viewmats_shape = (\n"
        "                current_chunk_size, 3, 4, 4,\n"
        "            )\n"
        "            expected_source_intrinsics_shape = (\n"
        "                current_chunk_size, 3, 3, 3,\n"
        "            )\n"
        "            if tuple(source_viewmats.shape) != expected_source_viewmats_shape:\n"
        "                raise RuntimeError(\n"
        "                    \"Current-time URoPE source_viewmats mismatch: \"\n"
        "                    f\"got {tuple(source_viewmats.shape)}, \"\n"
        "                    f\"expected {expected_source_viewmats_shape}.\"\n"
        "                )\n"
        "            if tuple(source_intrinsics.shape) != expected_source_intrinsics_shape:\n"
        "                raise RuntimeError(\n"
        "                    \"Current-time URoPE source_intrinsics mismatch: \"\n"
        "                    f\"got {tuple(source_intrinsics.shape)}, \"\n"
        "                    f\"expected {expected_source_intrinsics_shape}.\"\n"
        "                )\n"
    )
    text = replace_once(
        text, old_geometry, new_geometry, "9-slot geometry gather"
    )

    old_tv_args = (
        "                patch_width=width,\n"
        "                patch_height=height,\n"
        "                context_attention_mask=(\n"
        "                    context_attention_mask\n"
        "                ),\n"
    )
    new_tv_args = (
        "                patch_width=width,\n"
        "                patch_height=height,\n"
        "                # Flatten order is time-major:\n"
        "                # [t-1:L/S/R][t:L/S/R][t+1:L/S/R].\n"
        "                geometry_key_start=3 * token_count,\n"
        "                context_attention_mask=(\n"
        "                    context_attention_mask\n"
        "                ),\n"
    )
    text = replace_once(
        text, old_tv_args, new_tv_args, "TV geometry interval"
    )

    return text


def patch_config(config: dict, tv_chunk_size):
    pipeline = config["pipeline"]
    model = pipeline["model"]

    model["_class_name"] = (
        "dwm.models.lyh."
        "crossview_temporal_dit_urope_tv_tself_track."
        "DiTCrossviewTemporalConditionModel"
    )

    if model.get("enable_temporal") is not True:
        raise SystemExit(
            "Source config should have enable_temporal=true."
        )
    if model.get("enable_tv") is not True:
        raise SystemExit(
            "Source config should have enable_tv=true."
        )
    if model.get("enable_crossview") is not False:
        raise SystemExit(
            "Source config should have enable_crossview=false."
        )

    if tv_chunk_size is not None:
        if tv_chunk_size < 1:
            raise SystemExit(
                "--tv-full-batch-chunk-size must be >= 1"
            )
        model["tv_full_batch_chunk_size"] = tv_chunk_size

    common = pipeline["common_config"]
    ddp = common.get("ddp_wrapper_settings", {})
    auto_wrap = ddp.get("auto_wrap_policy", {})
    module_classes = auto_wrap.get("module_classes", [])

    old_block = (
        "dwm.models.lyh.crossview_temporal_dit_urope_tv."
        "VTURoPETVAttentionBlock"
    )
    new_block = (
        "dwm.models.lyh.crossview_temporal_dit_urope_tv_tself."
        "VTURoPETVAttentionBlock"
    )

    found = False
    for item in module_classes:
        if not isinstance(item, dict):
            continue
        if item.get("class_name") == old_block:
            item["class_name"] = new_block
            found = True

    if not found:
        module_classes.append(
            {
                "_class_name": "get_class",
                "class_name": new_block,
            }
        )

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--source-config",
        default=(
            "configs/lyh/"
            "PV_track_urope_tv_temporal_ring_train.json"
        ),
    )
    parser.add_argument(
        "--target-config",
        default=(
            "configs/lyh/"
            "PV_track_urope_tv_temporal_ring_tself_train.json"
        ),
    )
    parser.add_argument(
        "--tv-full-batch-chunk-size",
        type=int,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    base_urope = root / "src/dwm/models/urope/urope.py"
    base_tv = (
        root
        / "src/dwm/models/lyh/crossview_temporal_dit_urope_tv.py"
    )
    source_config = root / args.source_config

    selective_urope = (
        root / "src/dwm/models/lyh/urope_current_time.py"
    )
    new_tv = (
        root
        / "src/dwm/models/lyh/"
        "crossview_temporal_dit_urope_tv_tself.py"
    )
    new_track = (
        root
        / "src/dwm/models/lyh/"
        "crossview_temporal_dit_urope_tv_tself_track.py"
    )
    target_config = root / args.target_config

    for path in (base_urope, base_tv, source_config):
        if not path.is_file():
            raise SystemExit(f"required source not found: {path}")

    base_urope_text = base_urope.read_text()
    for marker in (
        "def _prepare_local_geometry_coefficients(",
        "def _apply_xy_rope(",
        "class URoPEDotProductAttention",
        "query_viewmats",
        "source_viewmats",
    ):
        if marker not in base_urope_text:
            raise SystemExit(
                "Current URoPE core does not have expected local API: "
                + marker
            )

    base_tv_text = base_tv.read_text()
    for marker in (
        "class VTURoPETVAttentionBlock",
        "9 * token_count",
        "source_time_index",
        "source_view_index",
        "source_viewmats = viewmats[",
    ):
        if marker not in base_tv_text:
            raise SystemExit(
                "Current URoPE-TV source does not match expected version: "
                + marker
            )

    config = json.loads(source_config.read_text())

    targets = (
        selective_urope,
        new_tv,
        new_track,
        target_config,
    )
    for path in targets:
        refuse_or_backup(path, args.force)

    selective_urope.parent.mkdir(parents=True, exist_ok=True)
    target_config.parent.mkdir(parents=True, exist_ok=True)

    ast.parse(CURRENT_TIME_UROPE)
    selective_urope.write_text(CURRENT_TIME_UROPE)

    new_tv_text = patch_tv_source(base_tv_text)
    ast.parse(new_tv_text)
    new_tv.write_text(new_tv_text)

    ast.parse(TRACK_WRAPPER)
    new_track.write_text(TRACK_WRAPPER)

    config = patch_config(
        config,
        args.tv_full_batch_chunk_size,
    )
    target_config.write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n"
    )

    py_compile.compile(str(selective_urope), doraise=True)
    py_compile.compile(str(new_tv), doraise=True)
    py_compile.compile(str(new_track), doraise=True)
    json.loads(target_config.read_text())

    print("=== CREATED ===")
    for path in targets:
        print(path)

    print()
    print("=== SEMANTICS ===")
    print("TV K/V content: 9 slots = 3 time x 3 view")
    print("URoPE geometry: 3 slots = current t x [L,S,R]")
    print("t-1/t+1 geometry reuse: NO")
    print("standalone temporal: preserved")
    print("standalone cross-view: preserved")
    print(
        "tv_full_batch_chunk_size:",
        config["pipeline"]["model"].get(
            "tv_full_batch_chunk_size"
        ),
    )


if __name__ == "__main__":
    main()
