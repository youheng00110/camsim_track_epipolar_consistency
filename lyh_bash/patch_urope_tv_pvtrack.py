#!/usr/bin/env python3
# Create a NEW URoPE-inside-TV + PVTrack experiment without modifying originals.
#
# Creates:
#   src/dwm/models/lyh/crossview_temporal_dit_urope_tv_track.py
#   src/dwm/pipelines/lyh/camsim_track_urope_tv.py
#   configs/lyh/PV_track_urope_tv_train.json
#
# Usage:
#   python patch_urope_tv_pvtrack.py /path/to/OpenDWM
#   python patch_urope_tv_pvtrack.py /path/to/OpenDWM --force

import argparse
import ast
import json
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


MODEL_WRAPPER = '"""\nURoPE-inside-TV + PVTrack model entry.\n\nPVTrack remains an image-condition branch:\n    PV box / HD map / instance flow\n        -> condition_image_tensor\n        -> ImageAdapter\n        -> condition residuals\n        -> DiT hidden states\n\nURoPE-TV remains the joint attention branch:\n    TV local 3x3 temporal-view topology\n        + URoPE camera geometry inside the same attention.\n\nThis wrapper intentionally adds no new trainable module. It only exposes a\nseparate experiment class and absorbs legacy token kwargs that camsim_track\nmay optionally produce in other configurations.\n"""\n\nfrom dwm.models.lyh.crossview_temporal_dit_urope_tv import (\n    DiTCrossviewTemporalConditionModel as URoPETVModel,\n)\n\n\nclass DiTCrossviewTemporalConditionModel(URoPETVModel):\n    """URoPE-TV with the existing PVTrack ImageAdapter conditioning path."""\n\n    def forward(\n        self,\n        *args,\n        camera_param_token=None,\n        camera_token_mask=None,\n        bbox_token_input=None,\n        bbox_class_input=None,\n        bbox_mask_input=None,\n        map_token_input=None,\n        **kwargs,\n    ):\n        del camera_param_token\n        del camera_token_mask\n        del bbox_token_input\n        del bbox_class_input\n        del bbox_mask_input\n        del map_token_input\n\n        return super().forward(\n            *args,\n            **kwargs,\n        )\n'
ANCHOR_RESULT = '            "disable_temporal": torch.tensor(\n                [common_config.get("disable_temporal", False)],\n                device=device).repeat(batch_size),\n\n            "crossview_attention_mask": (\n'
REPLACEMENT_RESULT = '            "disable_temporal": torch.tensor(\n                [common_config.get("disable_temporal", False)],\n                device=device).repeat(batch_size),\n\n            "disable_tv": torch.tensor(\n                [common_config.get(\n                    "disable_tv",\n                    common_config.get("disable_crossview", False)\n                    or common_config.get("disable_temporal", False),\n                )],\n                device=device,\n            ).repeat(batch_size),\n\n            "crossview_attention_mask": (\n'
ANCHOR_TRAIN = '            if additional_conditions is not None:\n                model_conditions.update(additional_conditions)\n            if getattr(self.model_wrapper, "mask_module", None) is not None:\n'
REPLACEMENT_TRAIN = '            if additional_conditions is not None:\n                model_conditions.update(additional_conditions)\n\n            # Joint TV contains both temporal and cross-view interaction.\n            # OR the explicit TV switch with legacy disable controls after\n            # CTSD has dynamically updated them.\n            tv_disable = model_conditions["disable_tv"]\n            for disable_key in ("disable_crossview", "disable_temporal"):\n                current_disable = model_conditions.get(disable_key)\n                if current_disable is None:\n                    continue\n                while tv_disable.ndim < current_disable.ndim:\n                    tv_disable = tv_disable.unsqueeze(-1)\n                while current_disable.ndim < tv_disable.ndim:\n                    current_disable = current_disable.unsqueeze(-1)\n                tv_disable = torch.logical_or(tv_disable, current_disable)\n            model_conditions["disable_tv"] = tv_disable\n\n            if getattr(self.model_wrapper, "mask_module", None) is not None:\n'


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


def backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def ensure_new_target(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise SystemExit(
            "Target already exists; nothing was overwritten:\n"
            f"  {path}\n"
            "Re-run with --force to replace only generated targets."
        )
    print(f"Backup: {backup(path)}")


def patch_pipeline(source_text: str) -> str:
    if source_text.count(ANCHOR_RESULT) != 1:
        raise SystemExit(
            "camsim_track.py layout changed: could not uniquely locate "
            "the get_conditions result dictionary."
        )
    text = source_text.replace(
        ANCHOR_RESULT,
        REPLACEMENT_RESULT,
        1,
    )

    if text.count(ANCHOR_TRAIN) != 1:
        raise SystemExit(
            "camsim_track.py layout changed: could not uniquely locate "
            "the training additional_conditions update."
        )
    return text.replace(
        ANCHOR_TRAIN,
        REPLACEMENT_TRAIN,
        1,
    )


def make_config(source_config: dict) -> dict:
    config = source_config

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        raise SystemExit("Source config has no pipeline dictionary.")

    pipeline["_class_name"] = (
        "dwm.pipelines.lyh.camsim_track_urope_tv.CrossviewTemporalSD"
    )

    common = pipeline.setdefault("common_config", {})
    common["explicit_view_modeling"] = True
    common["disable_tv"] = False

    model = pipeline.get("model")
    if not isinstance(model, dict):
        raise SystemExit("Source config has no pipeline.model dictionary.")

    old_crossview_layers = model.get("crossview_block_layers")
    old_temporal_layers = model.get("temporal_block_layers")
    tv_layers = (
        old_crossview_layers
        if old_crossview_layers
        else old_temporal_layers
    )
    if not tv_layers:
        tv_layers = [1, 5, 9, 13, 17, 21]

    model["_class_name"] = (
        "dwm.models.lyh.crossview_temporal_dit_urope_tv_track."
        "DiTCrossviewTemporalConditionModel"
    )
    model["perspective_modeling_type"] = "urope"

    model["enable_crossview"] = False
    model["crossview_attention_type"] = "full"
    model["crossview_block_layers"] = []
    model["crossview_gradient_checkpointing"] = False

    model["enable_temporal"] = False
    model["temporal_attention_type"] = None
    model["temporal_block_layers"] = []
    model["temporal_gradient_checkpointing"] = False

    model["enable_tv"] = True
    model["tv_attention_type"] = "full"
    model["tv_block_layers"] = list(tv_layers)
    model["tv_gradient_checkpointing"] = True
    model["tv_time_radius"] = 1
    model["tv_view_radius"] = 1
    model["tv_height_chunk_size"] = 0
    model["tv_full_batch_chunk_size"] = 1

    model["urope_config"] = {
        "min_depth": 2.0,
        "max_depth": 20.0,
        "freq_base": 100.0,
        "freq_scale": 1.0,
        "group_size": 4,
        "leaveout_head": 0,
        "camera_convention": "opencv",
    }

    adapter_cfg = model.get("condition_image_adapter_config")
    if not isinstance(adapter_cfg, dict):
        raise SystemExit(
            "Source PVTrack config has no condition_image_adapter_config."
        )
    if int(adapter_cfg.get("in_channels", -1)) != 9:
        raise SystemExit(
            "Expected the current PVTrack 9-channel condition adapter, "
            f"got in_channels={adapter_cfg.get('in_channels')!r}."
        )

    ddp_settings = common.get("ddp_wrapper_settings", {})
    auto_wrap = ddp_settings.get("auto_wrap_policy", {})
    module_classes = auto_wrap.get("module_classes")
    if isinstance(module_classes, list):
        new_class_name = (
            "dwm.models.lyh.crossview_temporal_dit_urope_tv."
            "VTURoPETVAttentionBlock"
        )
        existing_names = {
            item.get("class_name")
            for item in module_classes
            if isinstance(item, dict)
        }
        if new_class_name not in existing_names:
            module_classes.append(
                {
                    "_class_name": "get_class",
                    "class_name": new_class_name,
                }
            )

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="OpenDWM repository root",
    )
    parser.add_argument(
        "--source-config",
        default="configs/lyh/PV_track_train.json",
        help="PVTrack config to copy and modify, relative to repo root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only generated targets, with timestamped backups",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    model_source = (
        root
        / "src/dwm/models/lyh/crossview_temporal_dit_urope_tv.py"
    )
    pipeline_source = root / "src/dwm/pipelines/camsim_track.py"
    config_source = root / args.source_config

    model_target = (
        root
        / "src/dwm/models/lyh/crossview_temporal_dit_urope_tv_track.py"
    )
    pipeline_target = (
        root
        / "src/dwm/pipelines/lyh/camsim_track_urope_tv.py"
    )
    config_target = (
        root
        / "configs/lyh/PV_track_urope_tv_train.json"
    )

    for source in (model_source, pipeline_source, config_source):
        if not source.is_file():
            raise SystemExit(f"Required source not found: {source}")

    model_text = model_source.read_text()
    for marker in (
        "class VTURoPETVAttentionBlock",
        "URoPEDotProductAttention",
        "forward_tv_full_block_and_mix_result",
        'perspective_modeling_type="urope"',
    ):
        if marker not in model_text:
            raise SystemExit(
                f"URoPE-TV source is missing expected marker: {marker}"
            )

    pipeline_text = pipeline_source.read_text()
    for marker in (
        '"instance_flow_images"',
        "condition_image_tensor = torch.cat(condition_image_list, -3)",
        '"camera_intrinsics_norm"',
        '"camera2referego"',
        "try_make_input_for_prediction",
    ):
        if marker not in pipeline_text:
            raise SystemExit(
                f"PVTrack pipeline is missing expected marker: {marker}"
            )

    source_config = json.loads(config_source.read_text())

    print("=== BEFORE git status ===")
    print(git_status(root) or "(clean or git unavailable)")
    print()

    for target in (model_target, pipeline_target, config_target):
        ensure_new_target(target, args.force)

    model_target.parent.mkdir(parents=True, exist_ok=True)
    pipeline_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.parent.mkdir(parents=True, exist_ok=True)

    model_init = model_target.parent / "__init__.py"
    pipeline_init = pipeline_target.parent / "__init__.py"
    if not model_init.exists():
        model_init.write_text("")
    if not pipeline_init.exists():
        pipeline_init.write_text("")

    ast.parse(MODEL_WRAPPER)
    model_target.write_text(MODEL_WRAPPER)

    new_pipeline_text = patch_pipeline(pipeline_text)
    ast.parse(new_pipeline_text)
    pipeline_target.write_text(new_pipeline_text)

    new_config = make_config(source_config)
    config_target.write_text(
        json.dumps(new_config, indent=4, ensure_ascii=False) + "\n"
    )
    json.loads(config_target.read_text())

    py_compile.compile(str(model_target), doraise=True)
    py_compile.compile(str(pipeline_target), doraise=True)

    print("=== CREATED ===")
    print(model_target)
    print(pipeline_target)
    print(config_target)
    print()
    print("=== PRESERVED ORIGINALS ===")
    print(model_source)
    print(pipeline_source)
    print(config_source)
    print()
    print("=== ARCHITECTURE ===")
    print("PV box/map/instance-flow -> 9ch condition -> ImageAdapter residuals")
    print("                                   +")
    print("current (t,v) -> TV 3x3 local context -> URoPE inside TV attention")
    print()
    print("No standalone cross-view attention.")
    print("No standalone temporal attention.")
    print("No Camera SlotID.")
    print()
    print("=== CONFIG ENTRYPOINTS ===")
    print("pipeline:", new_config["pipeline"]["_class_name"])
    print("model:", new_config["pipeline"]["model"]["_class_name"])
    print(
        "tv_block_layers:",
        new_config["pipeline"]["model"]["tv_block_layers"],
    )
    print(
        "condition in_channels:",
        new_config["pipeline"]["model"][
            "condition_image_adapter_config"
        ]["in_channels"],
    )
    print()
    print("=== AFTER git status ===")
    print(git_status(root) or "(clean or git unavailable)")


if __name__ == "__main__":
    main()
