import argparse
import copy
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import dwm.common


DEFAULT_CONFIG = (
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/"
    "unimlvg/camsim/nuplan/nuplancamtoken.json"
)

CLASS_MAP = {
    "dwm.datasets.nuscenes.MotionDataset":
        "dwm.datasets.bevs.nuscenes.MotionDataset",
    "dwm.datasets.waymo.MotionDataset":
        "dwm.datasets.bevs.waymo.MotionDataset",
    "dwm.datasets.argoverse.MotionDataset":
        "dwm.datasets.bevs.argoverse.MotionDataset",
    "dwm.datasets.nuplan.MotionDataset":
        "dwm.datasets.bevs.nuplan.MotionDataset",
}

NEW_RAW_KEYS = {
    "fps",
    "images",
    "camera_intrinsics",
    "image_size",
    "camera_transforms",
    "ego_transforms",
    "3dbox_images",
    "hdmap_bev_images",
    "image_description",
    "bbox_token_corners",
    "bbox_token_classes",
    "bbox_token_masks",
}

STRICT_TENSOR_KEYS = (
    "vae_images",
    "camera_intrinsics",
    "image_size",
    "camera_transforms",
    "ego_transforms",
    "fps",
    "crossview_mask",
    "dataset_tag",
    "3dbox_images",
)

INTENTIONALLY_REMOVED_KEYS = {
    "hdmap_images",
    "camera_names",
    "distortion",
    "angle",
    "dist",
    "scene",
    "sample_data",
    "pts",
}


# ============================================================
# CLI and configuration
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--samples-per-entry", type=int, default=1)
    parser.add_argument("--indices", default="")
    parser.add_argument(
        "--datasets",
        default="nuscenes,waymo,argoverse,nuplan",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--box-atol", type=float, default=2e-3)
    parser.add_argument("--frame-atol", type=float, default=1e-7)
    parser.add_argument(
        "--report",
        default="/tmp/bevs_vs_original_report.json",
    )
    parser.add_argument(
        "--debug-dir",
        default="/tmp/bevs_vs_original_debug",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_kind(class_name):
    for kind in ("nuscenes", "waymo", "argoverse", "nuplan"):
        if f"datasets.{kind}." in class_name:
            return kind
    raise ValueError(f"Unknown dataset class {class_name!r}")


def collect_entries(config, selected):
    entries = []
    for split_name in ("training_dataset", "validation_dataset"):
        adapter_config = config[split_name]
        base_config = adapter_config["base_dataset"]
        if base_config.get("_class_name") == "torch.utils.data.ConcatDataset":
            datasets = base_config["datasets"]
        else:
            datasets = [base_config]
        kind_count = {}
        for config_index, dataset_config in enumerate(datasets):
            class_name = dataset_config.get("_class_name", "")
            if class_name not in CLASS_MAP:
                continue
            kind = get_kind(class_name)
            if kind not in selected:
                continue
            occurrence = kind_count.get(kind, 0)
            kind_count[kind] = occurrence + 1
            entries.append({
                "split": split_name,
                "config_index": config_index,
                "kind": kind,
                "occurrence": occurrence,
                "adapter_config": adapter_config,
                "dataset_config": dataset_config,
            })
    return entries


def find_state_keys(value, output):
    if isinstance(value, dict):
        if value.get("_class_name") == "dwm.common.get_state":
            output.add(value["key"])
        for child in value.values():
            find_state_keys(child, output)
    elif isinstance(value, list):
        for child in value:
            find_state_keys(child, output)


def init_dataset_global_state(config, entries):
    required = set()
    for entry in entries:
        find_state_keys(entry["dataset_config"], required)
    for key in sorted(required):
        if key in dwm.common.global_state:
            continue
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(
            config["global_state"][key]
        )
        print(f"[global_state] {key}")


def filter_new_dataset_config(original_config):
    new_class_name = CLASS_MAP[original_config["_class_name"]]
    class_type = dwm.common.get_class(new_class_name)
    signature = inspect.signature(class_type.__init__)
    allowed = set(signature.parameters) - {"self"}
    result = {"_class_name": new_class_name}
    dropped = []
    for key, value in original_config.items():
        if key == "_class_name":
            continue
        if key in allowed:
            result[key] = copy.deepcopy(value)
        else:
            dropped.append(key)
    return result, sorted(dropped)


def make_adapter_config(adapter_config, dataset_config, slim):
    result = copy.deepcopy(adapter_config)
    result["base_dataset"] = copy.deepcopy(dataset_config)
    if slim:
        result["transform_list"] = [
            transform
            for transform in result.get("transform_list", [])
            if transform.get("old_key") in NEW_RAW_KEYS
        ]
    return result


# ============================================================
# Deterministic sample acquisition
# ============================================================

def seed_dataset(dataset, seed, visited=None):
    if visited is None:
        visited = set()
    object_id = id(dataset)
    if object_id in visited:
        return
    visited.add(object_id)
    for name in ("random_state", "image_desc_rs"):
        state = getattr(dataset, name, None)
        if isinstance(state, np.random.RandomState):
            state.seed(seed)
    base_dataset = getattr(dataset, "base_dataset", None)
    if base_dataset is not None:
        seed_dataset(base_dataset, seed, visited)
    datasets = getattr(dataset, "datasets", None)
    if datasets is not None:
        for child in datasets:
            seed_dataset(child, seed, visited)


def seed_all(dataset, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_dataset(dataset, seed)


def choose_indices(length, count, explicit):
    if explicit:
        result = []
        for index in explicit:
            normalized = index if index >= 0 else length + index
            if normalized < 0 or normalized >= length:
                raise IndexError(f"index={index}, length={length}")
            result.append(normalized)
        return result
    count = max(1, min(count, length))
    if count == 1:
        return [0]
    return [
        int(round(position * (length - 1) / (count - 1)))
        for position in range(count)
    ]


def unwrap_base_dataset(dataset):
    current = dataset
    visited = set()
    while hasattr(current, "base_dataset"):
        object_id = id(current)
        if object_id in visited:
            break
        visited.add(object_id)
        current = current.base_dataset
    if hasattr(current, "datasets") and len(current.datasets) == 1:
        current = current.datasets[0]
    return current


# ============================================================
# Raw-frame identity diagnostics
# ============================================================

def normalize_json_value(value):
    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(child) for child in value]
    if isinstance(value, np.ndarray):
        return normalize_json_value(value.tolist())
    if torch.is_tensor(value):
        return normalize_json_value(value.detach().cpu().tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compact_frame_record(record):
    if isinstance(record, dict):
        keys = (
            "token",
            "lidarpc_token",
            "timestamp",
            "sensor",
            "channel",
            "sample_token",
        )
        result = {
            key: normalize_json_value(record[key])
            for key in keys
            if key in record
        }
        if result:
            return result
        return normalize_json_value(record)
    if isinstance(record, (list, tuple)):
        return [compact_frame_record(child) for child in record]
    return normalize_json_value(record)


def extract_raw_frame_signature(adapter_dataset, kind, index):
    dataset = unwrap_base_dataset(adapter_dataset)
    if not hasattr(dataset, "items"):
        return {
            "available": False,
            "reason": "dataset has no items attribute",
        }
    item = dataset.items[index]
    signature = {
        "available": True,
        "kind": kind,
        "index": int(index),
        "fps": normalize_json_value(item.get("fps")),
    }

    if kind == "waymo":
        signature["scene"] = normalize_json_value(item.get("scene"))
        signature["frames"] = normalize_json_value(item.get("segment", []))

    elif kind == "nuscenes":
        signature["scene"] = normalize_json_value(item.get("scene"))
        signature["frames"] = compact_frame_record(item.get("segment", []))

    elif kind == "argoverse":
        signature["scene"] = normalize_json_value(
            item.get("scene_id", item.get("scene"))
        )
        signature["frames"] = compact_frame_record(item.get("segment", []))

    elif kind == "nuplan":
        scene = item.get("scene")
        indices = list(item.get("indices", []))
        signature["scene"] = normalize_json_value(scene)
        signature["indices"] = normalize_json_value(indices)
        frames = []
        if hasattr(dataset, "scenes") and scene in dataset.scenes:
            scene_frames = dataset.scenes[scene]
            for frame_index in indices:
                info = scene_frames[frame_index]
                frames.append(compact_frame_record(info))
        signature["frames"] = frames

    normalized = normalize_json_value(signature)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    normalized["sha1"] = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
    return normalized


def summarize_signature(signature):
    if not signature.get("available", False):
        return signature
    frames = signature.get("frames", [])
    return {
        "sha1": signature.get("sha1"),
        "scene": signature.get("scene"),
        "frame_count": len(frames),
        "first": frames[0] if frames else None,
        "last": frames[-1] if frames else None,
        "indices": signature.get("indices"),
    }


# ============================================================
# Output-frame matching
# ============================================================

def get_frame_descriptors(item):
    images = item.get("vae_images")
    if images is None or not torch.is_tensor(images) or images.ndim != 5:
        return None
    images = images.detach().float().cpu()
    time_count, view_count = images.shape[:2]
    pooled = F.adaptive_avg_pool2d(
        images.reshape(-1, *images.shape[-3:]),
        output_size=(8, 8),
    )
    descriptors = pooled.reshape(time_count, view_count, -1)
    mean = descriptors.mean(dim=-1, keepdim=True)
    std = descriptors.std(dim=-1, keepdim=True)
    descriptors = torch.cat([descriptors, mean, std], dim=-1)
    return descriptors.reshape(time_count, -1)


def greedy_unique_frame_mapping(distance):
    old_count, new_count = distance.shape
    pairs = []
    for old_index in range(old_count):
        for new_index in range(new_count):
            pairs.append((
                float(distance[old_index, new_index].item()),
                old_index,
                new_index,
            ))
    pairs.sort(key=lambda item: item[0])
    mapping = [-1] * old_count
    used_new = set()
    for value, old_index, new_index in pairs:
        if mapping[old_index] >= 0 or new_index in used_new:
            continue
        mapping[old_index] = new_index
        used_new.add(new_index)
        if len(used_new) == min(old_count, new_count):
            break
    return mapping


def compare_frame_sequences(original, new, frame_atol):
    old_desc = get_frame_descriptors(original)
    new_desc = get_frame_descriptors(new)
    if old_desc is None or new_desc is None:
        return {
            "status": "unavailable",
            "comparable": True,
            "reorder": None,
        }
    distance = torch.cdist(old_desc, new_desc, p=2)
    distance = distance / max(old_desc.shape[1] ** 0.5, 1.0)
    mapping = greedy_unique_frame_mapping(distance)
    mapped_distance = []
    for old_index, new_index in enumerate(mapping):
        if new_index < 0:
            mapped_distance.append(float("inf"))
        else:
            mapped_distance.append(float(distance[old_index, new_index].item()))
    identity = (
        len(mapping) == old_desc.shape[0]
        and old_desc.shape[0] == new_desc.shape[0]
        and mapping == list(range(old_desc.shape[0]))
        and max(mapped_distance, default=0.0) <= frame_atol
    )
    complete_match = (
        old_desc.shape[0] == new_desc.shape[0]
        and all(value <= frame_atol for value in mapped_distance)
    )
    if identity:
        status = "identical"
        reorder = None
        comparable = True
    elif complete_match:
        status = "same_frames_different_order"
        reorder = mapping
        comparable = True
    else:
        status = "different_frame_set"
        reorder = None
        comparable = False
    return {
        "status": status,
        "comparable": comparable,
        "reorder": reorder,
        "old_count": int(old_desc.shape[0]),
        "new_count": int(new_desc.shape[0]),
        "old_to_new": mapping,
        "mapped_distance": mapped_distance,
        "distance_min": float(distance.min().item()),
        "distance_mean": float(distance.mean().item()),
        "distance_max": float(distance.max().item()),
        "distance_matrix": distance.tolist(),
    }


def reorder_temporal_value(value, order, time_count):
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == time_count:
        index = torch.tensor(order, device=value.device, dtype=torch.long)
        return value.index_select(0, index)
    if isinstance(value, list) and len(value) == time_count:
        return [value[index] for index in order]
    if isinstance(value, tuple) and len(value) == time_count:
        return tuple(value[index] for index in order)
    return value


def reorder_temporal_item(item, old_to_new):
    new_to_old_order = [int(index) for index in old_to_new]
    time_count = len(new_to_old_order)
    return {
        key: reorder_temporal_value(value, new_to_old_order, time_count)
        for key, value in item.items()
    }


# ============================================================
# Strict unchanged-field comparison
# ============================================================

def compare_tensor(original, new, atol):
    if original.shape != new.shape:
        return {
            "equal": False,
            "original_shape": list(original.shape),
            "new_shape": list(new.shape),
        }
    original_cpu = original.detach().cpu()
    new_cpu = new.detach().cpu()
    if not torch.is_floating_point(original_cpu):
        mismatch = int((original_cpu != new_cpu).sum().item())
        return {
            "equal": mismatch == 0,
            "mismatch": mismatch,
            "count": int(original_cpu.numel()),
        }
    difference = (original_cpu.float() - new_cpu.float()).abs()
    nonfinite_old = int((~torch.isfinite(original_cpu.float())).sum().item())
    nonfinite_new = int((~torch.isfinite(new_cpu.float())).sum().item())
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    return {
        "equal": bool(max_abs <= atol and nonfinite_old == 0 and nonfinite_new == 0),
        "max_abs": max_abs,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "mismatch_over_atol": int((difference > atol).sum().item()),
        "nonfinite_original": nonfinite_old,
        "nonfinite_new": nonfinite_new,
    }


def compare_strict_outputs(original, new, atol):
    report = {
        "only_original": sorted(set(original) - set(new)),
        "only_new": sorted(set(new) - set(original)),
        "keys": {},
        "unexpected": False,
    }
    for key in STRICT_TENSOR_KEYS:
        if key not in original or key not in new:
            report["keys"][key] = {
                "equal": False,
                "missing_original": key not in original,
                "missing_new": key not in new,
            }
            report["unexpected"] = True
            continue
        result = compare_tensor(original[key], new[key], atol)
        report["keys"][key] = result
        report["unexpected"] |= not result["equal"]
    if "clip_text" in original or "clip_text" in new:
        equal = (
            "clip_text" in original
            and "clip_text" in new
            and original["clip_text"] == new["clip_text"]
        )
        report["keys"]["clip_text"] = {"equal": bool(equal)}
        report["unexpected"] |= not bool(equal)
    unexpected_removed = set(report["only_original"]) - INTENTIONALLY_REMOVED_KEYS
    unexpected_added = set(report["only_new"])
    if unexpected_removed:
        report["unexpected_removed"] = sorted(unexpected_removed)
        report["unexpected"] = True
    if unexpected_added:
        report["unexpected_added"] = sorted(unexpected_added)
        report["unexpected"] = True
    report["intentional_removed"] = sorted(
        set(report["only_original"]) & INTENTIONALLY_REMOVED_KEYS
    )
    return report


# ============================================================
# Box matching independent of slot and corner ordering
# ============================================================

def active_indices(mask, time_index, view_index):
    return torch.nonzero(
        mask[time_index, view_index] > 0,
        as_tuple=False,
    ).flatten()


def corner_set_distance(new_boxes, old_boxes):
    new_count = new_boxes.shape[0]
    old_count = old_boxes.shape[0]
    output = torch.empty(new_count, old_count, dtype=torch.float32)
    for new_index in range(new_count):
        pairwise = torch.cdist(
            new_boxes[new_index].float(),
            old_boxes.float(),
            p=2,
        )
        forward = pairwise.min(dim=-1).values.max(dim=-1).values
        backward = pairwise.min(dim=-2).values.max(dim=-1).values
        output[new_index] = torch.maximum(forward, backward)
    return output


def greedy_box_pairs(distance, tolerance):
    candidates = []
    for new_index in range(distance.shape[0]):
        for old_index in range(distance.shape[1]):
            value = float(distance[new_index, old_index].item())
            if value <= tolerance:
                candidates.append((value, new_index, old_index))
    candidates.sort(key=lambda item: item[0])
    used_new = set()
    used_old = set()
    pairs = []
    for value, new_index, old_index in candidates:
        if new_index in used_new or old_index in used_old:
            continue
        used_new.add(new_index)
        used_old.add(old_index)
        pairs.append((new_index, old_index, value))
    return pairs, used_new, used_old


def increment_count(counter, key, amount=1):
    key = str(key)
    counter[key] = counter.get(key, 0) + int(amount)


def compare_boxes(original, new, tolerance):
    old_corners = original["bbox_token_corners"].float().cpu()
    old_classes = original["bbox_token_classes"].long().cpu()
    old_masks = original["bbox_token_masks"].float().cpu()
    new_corners = new["bbox_token_corners"].float().cpu()
    new_classes = new["bbox_token_classes"].long().cpu()
    new_masks = new["bbox_token_masks"].float().cpu()
    report = {
        "old_active": int((old_masks > 0).sum().item()),
        "new_active": int((new_masks > 0).sum().item()),
        "matched": 0,
        "new_unmatched": 0,
        "old_unmatched": 0,
        "class_changes": {},
        "old_unmatched_classes": {},
        "new_unmatched_classes": {},
        "max_matched_corner_distance": 0.0,
        "mean_matched_corner_distance": 0.0,
        "unexpected": False,
    }
    if old_corners.shape[:2] != new_corners.shape[:2]:
        report["shape_mismatch"] = {
            "old": list(old_corners.shape),
            "new": list(new_corners.shape),
        }
        report["unexpected"] = True
        return report

    matched_distances = []
    time_count, view_count = new_corners.shape[:2]
    per_frame = []
    for time_index in range(time_count):
        frame_report = {
            "time": time_index,
            "views": [],
        }
        for view_index in range(view_count):
            old_indices = active_indices(old_masks, time_index, view_index)
            new_indices = active_indices(new_masks, time_index, view_index)
            view_report = {
                "view": view_index,
                "old": int(old_indices.numel()),
                "new": int(new_indices.numel()),
                "matched": 0,
                "old_unmatched": 0,
                "new_unmatched": 0,
            }
            if old_indices.numel() == 0 or new_indices.numel() == 0:
                used_new = set()
                used_old = set()
                pairs = []
            else:
                old_boxes = old_corners[time_index, view_index, old_indices]
                new_boxes = new_corners[time_index, view_index, new_indices]
                distance = corner_set_distance(new_boxes, old_boxes)
                pairs, used_new, used_old = greedy_box_pairs(
                    distance,
                    tolerance,
                )

            for local_new, local_old, value in pairs:
                new_slot = int(new_indices[local_new].item())
                old_slot = int(old_indices[local_old].item())
                old_class = int(old_classes[
                    time_index, view_index, old_slot
                ].item())
                new_class = int(new_classes[
                    time_index, view_index, new_slot
                ].item())
                if old_class != new_class:
                    increment_count(
                        report["class_changes"],
                        f"{old_class}->{new_class}",
                    )
                matched_distances.append(value)

            unmatched_old_local = [
                local_index
                for local_index in range(old_indices.numel())
                if local_index not in used_old
            ]
            unmatched_new_local = [
                local_index
                for local_index in range(new_indices.numel())
                if local_index not in used_new
            ]
            for local_index in unmatched_old_local:
                old_slot = int(old_indices[local_index].item())
                old_class = int(old_classes[
                    time_index, view_index, old_slot
                ].item())
                increment_count(report["old_unmatched_classes"], old_class)
            for local_index in unmatched_new_local:
                new_slot = int(new_indices[local_index].item())
                new_class = int(new_classes[
                    time_index, view_index, new_slot
                ].item())
                increment_count(report["new_unmatched_classes"], new_class)

            matched_count = len(pairs)
            old_unmatched_count = len(unmatched_old_local)
            new_unmatched_count = len(unmatched_new_local)
            report["matched"] += matched_count
            report["old_unmatched"] += old_unmatched_count
            report["new_unmatched"] += new_unmatched_count
            view_report["matched"] = matched_count
            view_report["old_unmatched"] = old_unmatched_count
            view_report["new_unmatched"] = new_unmatched_count
            frame_report["views"].append(view_report)
        per_frame.append(frame_report)

    if matched_distances:
        report["max_matched_corner_distance"] = max(matched_distances)
        report["mean_matched_corner_distance"] = (
            sum(matched_distances) / len(matched_distances)
        )
    report["per_frame"] = per_frame
    report["unexpected"] = report["new_unmatched"] > 0
    return report


def check_stable_slots(item, tolerance):
    corners = item["bbox_token_corners"].float().cpu()
    classes = item["bbox_token_classes"].long().cpu()
    masks = item["bbox_token_masks"].float().cpu()
    time_count, view_count, slot_count = corners.shape[:3]
    class_errors = []
    view_errors = []
    used_slots = 0
    for slot in range(slot_count):
        present = masks[:, :, slot] > 0
        frames = torch.nonzero(present.any(dim=1), as_tuple=False).flatten()
        if frames.numel() == 0:
            continue
        used_slots += 1
        observed_classes = set()
        for time_index in frames.tolist():
            views = torch.nonzero(present[time_index], as_tuple=False).flatten()
            reference = corners[time_index, int(views[0].item()), slot]
            for view_index in views.tolist():
                observed_classes.add(int(classes[
                    time_index, view_index, slot
                ].item()))
                error = float((
                    corners[time_index, view_index, slot] - reference
                ).abs().max().item())
                if error > tolerance:
                    view_errors.append({
                        "slot": slot,
                        "time": time_index,
                        "view": view_index,
                        "max_abs": error,
                    })
        if len(observed_classes) > 1:
            class_errors.append({
                "slot": slot,
                "classes": sorted(observed_classes),
            })
    return {
        "valid": not class_errors and not view_errors,
        "used_slots": used_slots,
        "class_errors": class_errors,
        "cross_view_corner_errors": view_errors,
    }


# ============================================================
# BEV comparison with an explicit intentional-difference policy
# ============================================================

def binary_iou(original, new):
    union = torch.logical_or(original, new).sum().item()
    if union == 0:
        return 1.0
    intersection = torch.logical_and(original, new).sum().item()
    return float(intersection / union)


def compare_bev(original, new):
    original = original.float().cpu()
    new = new.float().cpu()
    if original.shape != new.shape:
        return {
            "same_shape": False,
            "original_shape": list(original.shape),
            "new_shape": list(new.shape),
        }
    channel_count = original.shape[-3]
    channels = []
    for channel in range(channel_count):
        old_channel = original.select(-3, channel)
        new_channel = new.select(-3, channel)
        difference = (old_channel - new_channel).abs()
        channels.append({
            "channel": channel,
            "mean_abs": float(difference.mean().item()),
            "max_abs": float(difference.max().item()),
            "mismatch_over_1e-5": int((difference > 1e-5).sum().item()),
            "iou": binary_iou(
                old_channel > 1e-6,
                new_channel > 1e-6,
            ),
            "old_nonzero": int((old_channel > 1e-6).sum().item()),
            "new_nonzero": int((new_channel > 1e-6).sum().item()),
        })
    return {"same_shape": True, "channels": channels}


def affected_dynamic_channels(box_report):
    classes = set()
    for transition in box_report.get("class_changes", {}):
        old_class, new_class = transition.split("->")
        classes.add(int(old_class))
        classes.add(int(new_class))
    for class_id in box_report.get("old_unmatched_classes", {}):
        classes.add(int(class_id))
    for class_id in box_report.get("new_unmatched_classes", {}):
        classes.add(int(class_id))
    return {3 + class_id for class_id in classes if 0 <= class_id < 10}


def evaluate_bev_policy(kind, bev_report, box_report, atol):
    policy = {
        "intentional_channels": [],
        "strict_channels": [],
        "unexpected_channels": [],
        "unexpected": False,
    }
    if not bev_report.get("same_shape", False):
        policy["unexpected"] = True
        policy["reason"] = "shape mismatch"
        return policy
    intentional = set()
    if kind == "waymo":
        intentional.update({0, 1, 2})
    intentional.update(affected_dynamic_channels(box_report))
    for channel_report in bev_report["channels"]:
        channel = channel_report["channel"]
        if channel in intentional:
            policy["intentional_channels"].append(channel)
            continue
        policy["strict_channels"].append(channel)
        if channel_report["max_abs"] > atol:
            policy["unexpected_channels"].append(channel)
    policy["unexpected"] = bool(policy["unexpected_channels"])
    return policy


# ============================================================
# Debug artifact export
# ============================================================

def safe_label(label):
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in label
    )


def first_chw(tensor):
    value = tensor.detach().float().cpu()
    while value.ndim > 3:
        value = value[0]
    return value


def channel_to_uint8(channel):
    channel = channel.detach().float().cpu()
    minimum = float(channel.min().item())
    maximum = float(channel.max().item())
    if maximum - minimum <= 1e-12:
        return np.zeros(channel.shape, dtype=np.uint8)
    normalized = (channel - minimum) / (maximum - minimum)
    return np.clip(normalized.numpy() * 255.0, 0, 255).astype(np.uint8)


def save_bev_debug(original, new, output_dir, changed_channels):
    output_dir.mkdir(parents=True, exist_ok=True)
    old_chw = first_chw(original)
    new_chw = first_chw(new)
    if old_chw.shape != new_chw.shape:
        return
    for channel in changed_channels:
        if channel < 0 or channel >= old_chw.shape[0]:
            continue
        old_np = channel_to_uint8(old_chw[channel])
        new_np = channel_to_uint8(new_chw[channel])
        diff_np = channel_to_uint8((old_chw[channel] - new_chw[channel]).abs())
        panel = np.concatenate([old_np, new_np, diff_np], axis=1)
        Image.fromarray(panel, mode="L").save(
            output_dir / f"bev_c{channel:02d}_old_new_diff.png"
        )


def write_debug_json(output_dir, name, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / name).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


# ============================================================
# Pair construction and per-entry execution
# ============================================================

def instantiate_pair(entry):
    old_dataset_config = copy.deepcopy(entry["dataset_config"])
    new_dataset_config, dropped = filter_new_dataset_config(
        old_dataset_config
    )
    old_adapter_config = make_adapter_config(
        entry["adapter_config"], old_dataset_config, False
    )
    new_adapter_config = make_adapter_config(
        entry["adapter_config"], new_dataset_config, True
    )
    old_dataset = dwm.common.create_instance_from_config(old_adapter_config)
    new_dataset = dwm.common.create_instance_from_config(new_adapter_config)
    return old_dataset, new_dataset, dropped


def print_strict_debug(strict_report):
    for key, result in strict_report.get("keys", {}).items():
        status = "OK" if result.get("equal", False) else "FAIL"
        details = ", ".join(
            f"{name}={value}"
            for name, value in result.items()
            if name != "equal"
        )
        print(f"[strict:{status}] {key} {details}")
    if strict_report.get("intentional_removed"):
        print(
            "[strict:INTENTIONAL_REMOVED] "
            + ",".join(strict_report["intentional_removed"])
        )
    if strict_report.get("unexpected_removed"):
        print(
            "[strict:FAIL_REMOVED] "
            + ",".join(strict_report["unexpected_removed"])
        )
    if strict_report.get("unexpected_added"):
        print(
            "[strict:FAIL_ADDED] "
            + ",".join(strict_report["unexpected_added"])
        )


def compare_entry(entry, args, explicit_indices):
    label = (
        f"{entry['split']}[{entry['config_index']}] "
        f"{entry['kind']}#{entry['occurrence']}"
    )
    print(f"\n[entry] {label}")
    old_dataset, new_dataset, dropped = instantiate_pair(entry)
    old_length = len(old_dataset)
    new_length = len(new_dataset)
    print(f"[length] old={old_length} new={new_length}")
    print(f"[dropped config keys] {dropped}")
    report = {
        "label": label,
        "kind": entry["kind"],
        "old_length": old_length,
        "new_length": new_length,
        "dropped_config_keys": dropped,
        "samples": [],
        "unexpected": old_length != new_length,
        "inconclusive": False,
    }
    length = min(old_length, new_length)
    indices = choose_indices(
        length,
        args.samples_per_entry,
        explicit_indices,
    )
    for index in indices:
        sample_debug_dir = (
            Path(args.debug_dir)
            / safe_label(label)
            / f"index_{index:06d}"
        )
        old_signature = extract_raw_frame_signature(
            old_dataset,
            entry["kind"],
            index,
        )
        new_signature = extract_raw_frame_signature(
            new_dataset,
            entry["kind"],
            index,
        )
        signature_same = old_signature.get("sha1") == new_signature.get("sha1")
        print(
            "[raw-frame] same={} old={} new={}".format(
                signature_same,
                summarize_signature(old_signature),
                summarize_signature(new_signature),
            )
        )

        seed = 104729 + index
        seed_all(old_dataset, seed)
        old_item = old_dataset[index]
        seed_all(new_dataset, seed)
        try:
            new_item = new_dataset[index]
        except (KeyError, ValueError) as error:
            print(f"[STOP] official track ID error at {label} index={index}")
            print(str(error))
            raise

        frame_report = compare_frame_sequences(
            old_item,
            new_item,
            args.frame_atol,
        )
        print(
            "[frame-output] status={} comparable={} mapping={} max_distance={}".format(
                frame_report["status"],
                frame_report["comparable"],
                frame_report.get("old_to_new"),
                max(frame_report.get("mapped_distance", [0.0]), default=0.0),
            )
        )

        aligned_new_item = new_item
        if frame_report.get("reorder") is not None:
            aligned_new_item = reorder_temporal_item(
                new_item,
                frame_report["reorder"],
            )
            print(
                "[frame-output] reordered new temporal axis to old frame order"
            )

        sample_report = {
            "index": index,
            "raw_frame_signature_same": signature_same,
            "old_raw_frame_signature": old_signature,
            "new_raw_frame_signature": new_signature,
            "frame_alignment": frame_report,
            "unexpected": False,
            "inconclusive": False,
        }

        if not frame_report["comparable"]:
            sample_report["inconclusive"] = True
            report["inconclusive"] = True
            write_debug_json(
                sample_debug_dir,
                "frame_mismatch.json",
                {
                    "old": old_signature,
                    "new": new_signature,
                    "alignment": frame_report,
                },
            )
            print(
                "[INCONCLUSIVE] old/new sampled different frame sets. "
                "Strict value comparison skipped."
            )
            report["samples"].append(sample_report)
            continue

        strict = compare_strict_outputs(
            old_item,
            aligned_new_item,
            args.atol,
        )
        print_strict_debug(strict)

        boxes = compare_boxes(
            old_item,
            aligned_new_item,
            args.box_atol,
        )
        stable = check_stable_slots(
            aligned_new_item,
            args.box_atol,
        )
        bev = compare_bev(
            old_item["hdmap_bev_images"],
            aligned_new_item["hdmap_bev_images"],
        )
        bev_policy = evaluate_bev_policy(
            entry["kind"],
            bev,
            boxes,
            args.atol,
        )

        unexpected = (
            strict["unexpected"]
            or boxes["unexpected"]
            or not stable["valid"]
            or bev_policy["unexpected"]
        )
        sample_report.update({
            "strict": strict,
            "boxes": boxes,
            "stable_slots": stable,
            "bev": bev,
            "bev_policy": bev_policy,
            "unexpected": unexpected,
        })
        report["unexpected"] |= unexpected

        print(
            "[boxes] matched={} old_unmatched={} new_unmatched={} "
            "class_changes={} old_unmatched_classes={} max_corner={:.6f}".format(
                boxes["matched"],
                boxes["old_unmatched"],
                boxes["new_unmatched"],
                boxes["class_changes"],
                boxes["old_unmatched_classes"],
                boxes["max_matched_corner_distance"],
            )
        )
        print(
            "[stable-slot] valid={} used={} class_errors={} view_errors={}".format(
                stable["valid"],
                stable["used_slots"],
                len(stable["class_errors"]),
                len(stable["cross_view_corner_errors"]),
            )
        )
        if bev.get("same_shape", False):
            for channel in bev["channels"]:
                channel_id = channel["channel"]
                if channel_id in bev_policy["intentional_channels"]:
                    status = "INTENTIONAL"
                elif channel_id in bev_policy["unexpected_channels"]:
                    status = "FAIL"
                else:
                    status = "OK"
                print(
                    "[bev:{}] c{} iou={:.4f} mean={:.6f} max={:.6f} "
                    "old_nz={} new_nz={}".format(
                        status,
                        channel_id,
                        channel["iou"],
                        channel["mean_abs"],
                        channel["max_abs"],
                        channel["old_nonzero"],
                        channel["new_nonzero"],
                    )
                )
        else:
            print(
                "[bev:FAIL] shape old={} new={}".format(
                    bev.get("original_shape"),
                    bev.get("new_shape"),
                )
            )

        changed_channels = []
        if bev.get("same_shape", False):
            changed_channels = [
                channel["channel"]
                for channel in bev["channels"]
                if channel["max_abs"] > args.atol
            ]
        if unexpected or changed_channels or not signature_same:
            write_debug_json(
                sample_debug_dir,
                "comparison.json",
                sample_report,
            )
            if not args.no_debug_images and changed_channels:
                save_bev_debug(
                    old_item["hdmap_bev_images"],
                    aligned_new_item["hdmap_bev_images"],
                    sample_debug_dir,
                    changed_channels,
                )
            print(f"[debug] {sample_debug_dir}")

        status = "FAIL" if unexpected else "PASS"
        print(
            "[sample:{}] index={} raw_frame_same={} frame_status={}".format(
                status,
                index,
                signature_same,
                frame_report["status"],
            )
        )
        report["samples"].append(sample_report)
    return report


def resolve_report_path(raw_path):
    value = str(raw_path).strip()
    if not value:
        return Path("/tmp/bevs_vs_original_report.json")
    path = Path(value)
    if path.exists() and path.is_dir():
        return path / "bevs_vs_original_report.json"
    if value.endswith("/"):
        return path / "bevs_vs_original_report.json"
    if path.name in ("", ".", ".."):
        return path / "bevs_vs_original_report.json"
    return path


def main():
    args = parse_args()
    selected = {
        value.strip()
        for value in args.datasets.split(",")
        if value.strip()
    }
    explicit_indices = [
        int(value.strip())
        for value in args.indices.split(",")
        if value.strip()
    ]
    config = load_json(args.config)
    entries = collect_entries(config, selected)
    if not entries:
        raise RuntimeError("No matching datasets found in the config")
    init_dataset_global_state(config, entries)
    report = {
        "config": args.config,
        "entries": [],
        "unexpected": False,
        "inconclusive": False,
        "policy": {
            "strict": list(STRICT_TENSOR_KEYS) + ["clip_text"],
            "intentional_removed": sorted(INTENTIONALLY_REMOVED_KEYS),
            "intentional_bev": {
                "waymo_static_channels": [0, 1, 2],
                "dynamic_channels": (
                    "only channels explained by class remapping or "
                    "stable-slot filtering"
                ),
            },
        },
    }
    for entry in entries:
        entry_report = compare_entry(entry, args, explicit_indices)
        report["entries"].append(entry_report)
        report["unexpected"] |= entry_report["unexpected"]
        report["inconclusive"] |= entry_report["inconclusive"]

    report_path = resolve_report_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    print(f"\n[report] {report_path}")

    if report["unexpected"]:
        print("[result] FAIL unexpected differences found")
        return 2
    if report["inconclusive"]:
        print("[result] INCONCLUSIVE some samples used different frame sets")
        return 3
    print("[result] PASS only explicit intentional differences found")
    return 0


if __name__ == "__main__":
    sys.exit(main())