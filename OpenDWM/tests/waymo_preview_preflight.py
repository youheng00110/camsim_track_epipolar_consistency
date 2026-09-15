#!/usr/bin/env python
"""Static and data-loader preflight for the migrated Waymo preview config.

This script deliberately never constructs the diffusion pipeline and never
calls ``preview_pipeline``.  It is therefore safe to run before a preview
job is approved.
"""

import argparse
import importlib
import json
import os
import sys
import traceback

import torch

import dwm.common
from dwm.preview import resolve_preview_item_limit
from dwm.utils.preview import compute_eval_frame_resume_count


REQUIRED_BATCH_KEYS = (
    "vae_images",
    "camera_intrinsics",
    "camera_transforms",
    "ego_transforms",
    "3dbox_records",
    "box_reference_to_camera",
)


def iter_absolute_paths(value, key=""):
    if isinstance(value, dict):
        for name, child in value.items():
            child_key = "{}.{}".format(key, name) if key else name
            yield from iter_absolute_paths(child, child_key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_absolute_paths(child, "{}[{}]".format(key, index))
    elif isinstance(value, str) and value.startswith("/"):
        yield key, value


def import_class(class_name):
    module_name, attribute = class_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


def run_resume_simulation():
    cases = (
        ([0, 0, 0, 0], 0),
        ([3, 3, 3, 3], 3),
        ([4, 3, 4, 4], 3),
    )
    results = []
    for rank_counts, expected in cases:
        actual = compute_eval_frame_resume_count(rank_counts, 250, 1)
        results.append((rank_counts, actual, expected, actual == expected))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Only perform static checks and dataset construction.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    print("JSON: PASS")

    pipeline_config = config["pipeline"]
    pipeline_class = import_class(pipeline_config["_class_name"])
    print("pipeline import: PASS ({})".format(pipeline_class.__name__))

    print("absolute path audit:")
    for key, path in iter_absolute_paths(config):
        status = "EXISTS" if os.path.exists(path) else "MISSING"
        scope = "training/deferred" if key.startswith("training_dataset") else "runtime"
        print("  {} [{}] {} -> {}".format(status, scope, key, path))

    runtime_paths = {
        "pretrained_model_name_or_path": pipeline_config[
            "pretrained_model_name_or_path"
        ],
        "model_checkpoint_path": pipeline_config["model_checkpoint_path"],
        "i3d_checkpoint": pipeline_config["metrics"]["fvd"][
            "inception_3d_checkpoint_path"
        ],
        "waymo_root": config["validation_dataset"]["base_dataset"][
            "datasets"
        ][0]["dataset_root"],
        "validation_info": config["validation_dataset"]["base_dataset"][
            "datasets"
        ][0]["info_dict_path"],
    }
    for name, path in runtime_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError("{} missing: {}".format(name, path))
    print("runtime resources: PASS")

    output_path = pipeline_config["inference_config"]["eval_frame_export_path"]
    output_parent = os.path.dirname(output_path)
    if not os.path.isdir(output_parent) or not os.access(output_parent, os.W_OK):
        raise OSError("eval output parent is not writable: {}".format(output_parent))
    print("eval output parent writable: PASS ({})".format(output_parent))

    mesh_shape = config["global_state"]["device_mesh"]["mesh_shape"]
    print("device mesh: {} (expected [1, 4])".format(mesh_shape))
    if mesh_shape != [1, 4]:
        raise ValueError("device mesh must be [1, 4]")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"]
    )
    dataset_length = len(validation_dataset)
    target_global, target_per_rank = resolve_preview_item_limit(
        pipeline_config["inference_config"],
        dataset_length,
        4,
    )
    print(
        "validation length: {} target_global={} target_per_rank={} meets_target={}".format(
            dataset_length,
            target_global,
            target_per_rank,
            dataset_length >= target_global,
        )
    )

    if not args.skip_batch:
        loader_kwargs = dwm.common.instantiate_config(
            config["preview_dataloader"]
        )
        loader_kwargs.pop("shuffle", None)
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
        loader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_kwargs,
        )
        batch = next(iter(loader))
        print("preview batch keys:", sorted(batch.keys()))
        missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError("required batch keys missing: {}".format(missing))
        print("preview batch required keys: PASS")
    else:
        print("preview batch: SKIPPED")

    print("resume simulations:")
    for rank_counts, actual, expected, passed in run_resume_simulation():
        print(
            "  {} -> {} (expected {}) {}".format(
                rank_counts, actual, expected, "PASS" if passed else "FAIL"
            )
        )
    if not all(item[-1] for item in run_resume_simulation()):
        raise AssertionError("resume simulation failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
