#!/usr/bin/env python3
"""Count OpenDWM video windows without decoding images or point clouds.

Place this file at OpenDWM/tools/count_dataset_windows.py and run it from the
OpenDWM repository. The default mode creates every MotionDataset leaf once.
Use --per-sampling to count every fps/stride configuration separately.
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parents[1] if len(SCRIPT_PATH.parents) > 1 else Path.cwd()
OPENDWM_ROOT = Path(os.environ.get("OPENDWM_ROOT", DEFAULT_ROOT)).resolve()
SRC_ROOT = OPENDWM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dwm.common


DATASET_CLASS_SUFFIX = ".MotionDataset"
INDEX_ONLY_DROP_KEYS = {
    "image_description_settings",
    "_3dbox_image_settings",
    "hdmap_image_settings",
    "_3dbox_bev_settings",
    "hdmap_bev_settings",
    "projected_pc_settings",
    "layout_token_settings",
    "stub_key_data_dict",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count OpenDWM dataset windows without reading media data."
    )
    parser.add_argument("config_path", type=Path)
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["training_dataset", "validation_dataset"],
        help="Dataset sections to inspect.",
    )
    parser.add_argument(
        "--per-sampling",
        action="store_true",
        help="Count every fps/stride tuple separately.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached counts and rebuild metadata indices.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Count cache path. Defaults to OpenDWM/.cache/dataset_window_counts.json.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional detailed JSON report path.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_dataset_leaves(node, section, node_path, leaves):
    if isinstance(node, dict):
        class_name = str(node.get("_class_name", ""))
        if class_name.endswith(DATASET_CLASS_SUFFIX):
            leaves.append(
                {
                    "section": section,
                    "path": node_path,
                    "config": node,
                }
            )
            return
        for key, value in node.items():
            collect_dataset_leaves(value, section, node_path + [str(key)], leaves)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            collect_dataset_leaves(value, section, node_path + [str(index)], leaves)


def collect_state_keys(node, keys):
    if isinstance(node, dict):
        if node.get("_class_name") == "dwm.common.get_state":
            key = node.get("key")
            if key is not None:
                keys.add(str(key))
        for value in node.values():
            collect_state_keys(value, keys)
        return
    if isinstance(node, list):
        for value in node:
            collect_state_keys(value, keys)


def initialize_required_global_state(config, leaves):
    required_keys = set()
    for leaf in leaves:
        collect_state_keys(leaf["config"], required_keys)

    state_config = config.get("global_state", {})
    missing = [key for key in sorted(required_keys) if key not in state_config]
    if missing:
        raise KeyError("Missing global_state entries: {}".format(", ".join(missing)))

    pending = set(required_keys)
    while pending:
        progressed = False
        for key in sorted(pending):
            try:
                dwm.common.global_state[key] = dwm.common.create_instance_from_config(
                    state_config[key]
                )
            except KeyError:
                continue
            pending.remove(key)
            progressed = True
        if not progressed:
            raise RuntimeError(
                "Unable to resolve global_state dependencies: {}".format(
                    ", ".join(sorted(pending))
                )
            )


def sanitize_dataset_config(dataset_config):
    result = copy.deepcopy(dataset_config)
    for key in INDEX_ONLY_DROP_KEYS:
        result.pop(key, None)
    if "enable_camera_transforms" in result:
        result["enable_camera_transforms"] = False
    if "enable_ego_transforms" in result:
        result["enable_ego_transforms"] = False
    if "enable_sample_data" in result:
        result["enable_sample_data"] = False
    return result


def patch_nuplan_map_initialization(class_name):
    if class_name != "dwm.datasets.nuplan.MotionDataset":
        return
    module = importlib.import_module("dwm.datasets.nuplan")
    module.get_maps_db = mock.MagicMock(return_value=None)
    module.NuPlanMapFactory = mock.MagicMock(return_value=None)


def describe_dataset(dataset_config, occurrence):
    class_name = str(dataset_config.get("_class_name", ""))
    parts = class_name.split(".")
    dataset_name = parts[-2] if len(parts) >= 2 else class_name
    split = dataset_config.get("split", dataset_config.get("dataset_name", ""))
    channels = [str(value) for value in dataset_config.get("sensor_channels", [])]
    camera_channels = [
        value
        for value in channels
        if "cam" in value.lower() or "camera" in value.lower()
    ]
    unique_cameras = list(dict.fromkeys(camera_channels))
    return {
        "dataset": dataset_name,
        "occurrence": occurrence,
        "split": str(split),
        "sequence_length": int(dataset_config.get("sequence_length", 1)),
        "camera_slot_count": len(camera_channels),
        "unique_camera_count": len(unique_cameras),
        "camera_layout": camera_channels,
        "unique_cameras": unique_cameras,
    }


def fingerprint_count_request(dataset_config, section, sampling, config_stat):
    payload = {
        "section": section,
        "dataset_config": dataset_config,
        "sampling": sampling,
        "config_mtime_ns": config_stat.st_mtime_ns,
        "config_size": config_stat.st_size,
        "script_version": 2,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path):
    if not cache_path.is_file():
        return {}
    try:
        data = load_json(cache_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(cache_path)


def count_dataset_windows(dataset_config):
    class_name = str(dataset_config.get("_class_name", ""))
    patch_nuplan_map_initialization(class_name)
    start_time = time.perf_counter()
    dataset = dwm.common.create_instance_from_config(dataset_config)
    count = len(dataset)
    elapsed = time.perf_counter() - start_time
    del dataset
    gc.collect()
    return int(count), float(elapsed)


def build_count_requests(leaf, per_sampling):
    base_config = sanitize_dataset_config(leaf["config"])
    sampling_configs = base_config.get("fps_stride_tuples", [])
    if not per_sampling or not sampling_configs:
        return [{"sampling": None, "dataset_config": base_config}]

    requests = []
    for sampling in sampling_configs:
        item_config = copy.deepcopy(base_config)
        item_config["fps_stride_tuples"] = [sampling]
        requests.append(
            {
                "sampling": copy.deepcopy(sampling),
                "dataset_config": item_config,
            }
        )
    return requests


def format_sampling(sampling):
    if sampling is None:
        return "all"
    if len(sampling) == 2:
        return "fps={} stride={}".format(sampling[0], sampling[1])
    if len(sampling) == 3:
        return "fps={} stride={} ratio={}".format(
            sampling[0], sampling[1], sampling[2]
        )
    return json.dumps(sampling, ensure_ascii=False)


def print_table(rows):
    columns = [
        ("section", "section"),
        ("id", "id"),
        ("dataset", "dataset"),
        ("split", "split"),
        ("sampling", "sampling_text"),
        ("T", "sequence_length"),
        ("slots/unique", "views_text"),
        ("windows", "window_count"),
        ("seconds", "seconds_text"),
        ("source", "source"),
    ]
    widths = {}
    for title, key in columns:
        values = [str(row.get(key, "")) for row in rows]
        widths[key] = max([len(title)] + [len(value) for value in values])

    header = "  ".join(title.ljust(widths[key]) for title, key in columns)
    print(header)
    print("  ".join("=" * widths[key] for _, key in columns))
    for row in rows:
        print(
            "  ".join(
                str(row.get(key, "")).ljust(widths[key])
                for _, key in columns
            )
        )


def print_camera_layouts(rows):
    printed = set()
    print("\nCamera layouts")
    for row in rows:
        layout_key = (row["section"], row["id"])
        if layout_key in printed:
            continue
        printed.add(layout_key)
        layout = " | ".join(row["camera_layout"])
        print("{} {}  {}".format(row["section"], row["id"], layout))


def main():
    args = parse_args()
    config_path = args.config_path.resolve()
    config = load_json(config_path)
    cache_path = args.cache_path
    if cache_path is None:
        cache_path = OPENDWM_ROOT / ".cache" / "dataset_window_counts.json"
    cache_path = cache_path.resolve()
    cache = {} if args.refresh else load_cache(cache_path)

    leaves = []
    for section in args.sections:
        if section not in config:
            continue
        collect_dataset_leaves(config[section], section, [section], leaves)
    if not leaves:
        raise RuntimeError("No MotionDataset leaves were found in the selected sections.")

    initialize_required_global_state(config, leaves)
    config_stat = config_path.stat()
    occurrence_counter = {}
    rows = []

    for leaf in leaves:
        class_name = str(leaf["config"].get("_class_name", ""))
        dataset_key = (leaf["section"], class_name)
        occurrence_counter[dataset_key] = occurrence_counter.get(dataset_key, 0) + 1
        occurrence = occurrence_counter[dataset_key]
        description = describe_dataset(leaf["config"], occurrence)
        leaf_id = "{}-{}".format(description["dataset"], occurrence)

        requests = build_count_requests(leaf, args.per_sampling)
        for request in requests:
            fingerprint = fingerprint_count_request(
                request["dataset_config"],
                leaf["section"],
                request["sampling"],
                config_stat,
            )
            cached = cache.get(fingerprint)
            if cached is not None:
                count = int(cached["window_count"])
                elapsed = 0.0
                source = "cache"
            else:
                count, elapsed = count_dataset_windows(request["dataset_config"])
                source = "index"
                cache[fingerprint] = {
                    "window_count": count,
                    "build_seconds": elapsed,
                }
                save_cache(cache_path, cache)

            row = {
                "section": leaf["section"],
                "id": leaf_id,
                "path": ".".join(leaf["path"]),
                "dataset": description["dataset"],
                "split": description["split"],
                "sampling": request["sampling"],
                "sampling_text": format_sampling(request["sampling"]),
                "sequence_length": description["sequence_length"],
                "camera_slot_count": description["camera_slot_count"],
                "unique_camera_count": description["unique_camera_count"],
                "views_text": "{}/{}".format(
                    description["camera_slot_count"],
                    description["unique_camera_count"],
                ),
                "camera_layout": description["camera_layout"],
                "unique_cameras": description["unique_cameras"],
                "window_count": count,
                "build_seconds": elapsed,
                "seconds_text": "{:.2f}".format(elapsed),
                "source": source,
            }
            rows.append(row)

    print_table(rows)
    print_camera_layouts(rows)
    print("\nTotal windows  {}".format(sum(row["window_count"] for row in rows)))
    print("Cache file     {}".format(cache_path))

    if args.json_output is not None:
        output_path = args.json_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config_path": str(config_path),
            "per_sampling": bool(args.per_sampling),
            "rows": rows,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print("JSON report    {}".format(output_path))


if __name__ == "__main__":
    main()
