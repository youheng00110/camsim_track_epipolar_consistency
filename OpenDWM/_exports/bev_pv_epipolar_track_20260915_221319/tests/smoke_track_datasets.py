"""One-item Track-ID dataset and geometry smoke test for the four datasets."""

import argparse
import copy
import json
import os
import shutil
import tempfile

import torch

import dwm.common
import dwm.datasets.nuscenes_common
from dwm.utils.track_consistency import (
    box_dimensions_from_corners,
    dimension_log_distance,
    prepare_track_consistency_geometry,
    project_box_to_feature_mask,
    sample_track_consistency_selection,
)


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream a top-level JSON array without loading multi-GB tables."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    end_of_file = False
    with open(path, "r", encoding="utf-8") as file:
        while True:
            if position >= len(buffer) and not end_of_file:
                buffer = file.read(chunk_size)
                position = 0
                end_of_file = not buffer
            while position < len(buffer) and buffer[position] in " \r\n\t[,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, stop = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if end_of_file:
                    raise
                buffer = buffer[position:] + file.read(chunk_size)
                position = 0
                end_of_file = file.tell() == os.fstat(file.fileno()).st_size
                continue
            yield value
            position = stop
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def take_contiguous_records(path, predicate):
    """Take the first contiguous matching block from an ordered nuScenes table."""
    records = []
    found = False
    for record in iter_json_array(path):
        if predicate(record):
            records.append(record)
            found = True
        elif found:
            break
    if not records:
        raise RuntimeError(f"No matching records found in {path}.")
    return records


def take_records_by_token(path, token_key, required_tokens):
    records = []
    remaining = set(required_tokens)
    for record in iter_json_array(path):
        token = record[token_key]
        if token not in remaining:
            continue
        records.append(record)
        remaining.remove(token)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(
            f"Missing {len(remaining)} required records in {path}."
        )
    return records


def build_nuscenes_smoke_root(source_root, temporary_root):
    """Build one-scene metadata while reusing the real sensor files."""
    source_metadata = os.path.join(source_root, "interp_12Hz_trainval")
    mini_name = "track_smoke"
    mini_metadata = os.path.join(temporary_root, mini_name)
    os.makedirs(mini_metadata)
    for directory_name in ("samples", "sweeps", "maps", "expansion"):
        os.symlink(
            os.path.join(source_root, directory_name),
            os.path.join(temporary_root, directory_name),
        )

    with open(
        os.path.join(source_metadata, "scene.json"),
        "r",
        encoding="utf-8",
    ) as file:
        scenes = json.load(file)
    scene = next(
        value
        for value in scenes
        if value["name"] in dwm.datasets.nuscenes_common.train
    )
    scene_token = scene["token"]
    samples = take_contiguous_records(
        os.path.join(source_metadata, "sample.json"),
        lambda value: value["scene_token"] == scene_token,
    )
    sample_tokens = {value["token"] for value in samples}
    sample_data = take_contiguous_records(
        os.path.join(source_metadata, "sample_data.json"),
        lambda value: value["sample_token"] in sample_tokens,
    )
    annotations = take_contiguous_records(
        os.path.join(source_metadata, "sample_annotation.json"),
        lambda value: value["sample_token"] in sample_tokens,
    )
    ego_pose_tokens = {value["ego_pose_token"] for value in sample_data}
    ego_poses = take_records_by_token(
        os.path.join(source_metadata, "ego_pose.json"),
        "token",
        ego_pose_tokens,
    )
    selected_tables = {
        "scene": [scene],
        "sample": samples,
        "sample_data": sample_data,
        "sample_annotation": annotations,
        "ego_pose": ego_poses,
    }
    for table_name in (
        "calibrated_sensor",
        "category",
        "instance",
        "log",
        "map",
        "sensor",
    ):
        with open(
            os.path.join(source_metadata, f"{table_name}.json"),
            "r",
            encoding="utf-8",
        ) as file:
            selected_tables[table_name] = json.load(file)
    for table_name, records in selected_tables.items():
        with open(
            os.path.join(mini_metadata, f"{table_name}.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(records, file)
    return mini_name


def print_pair_diagnostic(
    name,
    pair_name,
    pair,
    track_ids,
    classes,
    masks,
    corners,
    hard_threshold=0.2,
):
    time_a, view_a, time_b, view_b = pair
    valid_a = (
        track_ids[time_a].gt(0)
        & masks[time_a, view_a]
        & classes[time_a, view_a].ge(0)
        & classes[time_a, view_a].le(4)
    )
    valid_b = (
        track_ids[time_b].gt(0)
        & masks[time_b, view_b]
        & classes[time_b, view_b].ge(0)
        & classes[time_b, view_b].le(4)
    )
    slots_a = valid_a.nonzero(as_tuple=False).flatten().tolist()
    slots_b = valid_b.nonzero(as_tuple=False).flatten().tolist()
    objects_a = {int(track_ids[time_a, slot]): slot for slot in slots_a}
    objects_b = {int(track_ids[time_b, slot]): slot for slot in slots_b}
    shared = sorted(set(objects_a).intersection(objects_b))
    query_id = shared[0]
    source_slot = objects_a[query_id]
    source_class = int(classes[time_a, view_a, source_slot])
    source_dimensions = box_dimensions_from_corners(
        corners[time_a, view_a, source_slot]
    )
    negative_ids = sorted(value for value in objects_b if value != query_id)
    hard_ids = []
    for negative_id in negative_ids:
        target_slot = objects_b[negative_id]
        if int(classes[time_b, view_b, target_slot]) != source_class:
            continue
        target_dimensions = box_dimensions_from_corners(
            corners[time_b, view_b, target_slot]
        )
        if float(
            dimension_log_distance(source_dimensions, target_dimensions)
        ) <= hard_threshold:
            hard_ids.append(negative_id)
    print(
        f"[{name}] {pair_name}_pair={pair} shared_positive_ids={shared} "
        f"query_id={query_id} target_negative_ids={negative_ids} "
        f"hard_negative_ids={hard_ids}",
        flush=True,
    )


def summarize_item(name, dataset):
    item = dataset[0]
    corners = item["bbox_token_corners"]
    classes = item["bbox_token_classes"]
    masks = item["bbox_token_masks"].bool()
    track_ids = item["bbox_token_track_ids"]
    valid_ids = torch.unique(track_ids[track_ids > 0]).tolist()
    temporal_shared = []
    for time_id in range(track_ids.shape[0] - 1):
        shared = set(track_ids[time_id][track_ids[time_id] > 0].tolist())
        shared &= set(
            track_ids[time_id + 1][track_ids[time_id + 1] > 0].tolist()
        )
        temporal_shared.append(len(shared))

    print(
        f"[{name}] corners={tuple(corners.shape)} "
        f"classes={tuple(classes.shape)} masks={tuple(masks.shape)} "
        f"track_ids={tuple(track_ids.shape)}/{track_ids.dtype} "
        f"unique_valid={valid_ids[:16]} "
        f"adjacent_shared={temporal_shared[:8]}",
        flush=True,
    )

    tensor_keys = (
        "bbox_token_corners",
        "bbox_token_classes",
        "bbox_token_masks",
        "bbox_token_track_ids",
        "view_consistency_pair_mask",
        "camera_intrinsics",
        "camera_transforms",
        "ego_transforms",
        "reference_ego_transforms",
        "image_size",
    )
    batch = {
        key: item[key].unsqueeze(0)
        for key in tensor_keys
        if key in item
    }
    config = {
        "track_consistency_loss_weight": 1.0,
        "track_consistency_vehicle_class_ids": [0, 1, 2, 3, 4],
        "track_consistency_enable_spatial": True,
        "track_consistency_enable_temporal": True,
        "track_consistency_temporal_stride": 1,
        "track_consistency_max_spatial_pairs_per_sample": 100000,
        "track_consistency_max_temporal_pairs_per_sample": 100000,
    }
    selection_result = sample_track_consistency_selection(
        batch,
        config,
        torch.Generator().manual_seed(0),
        torch.device("cpu"),
    )
    if selection_result is None:
        print(f"[{name}] eligible_spatial=0 eligible_temporal=0")
    else:
        _, selection_cpu, pair_types = selection_result
        print(
            f"[{name}] eligible_spatial={int((pair_types == 0).sum())} "
            f"eligible_temporal={int((pair_types == 1).sum())} "
            f"pair_examples={selection_cpu[0, :4].tolist()}",
            flush=True,
        )
        for pair_type, pair_name in ((0, "spatial"), (1, "temporal")):
            indices = (pair_types[0] == pair_type).nonzero(
                as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            pair = selection_cpu[0, int(indices[0])].tolist()
            print_pair_diagnostic(
                name,
                pair_name,
                pair,
                track_ids,
                classes,
                masks,
                corners,
            )

    intrinsics, camera_to_common, box_reference_to_common = (
        prepare_track_consistency_geometry(batch, torch.device("cpu"))
    )
    region_counts = []
    size_distances = []
    for time_id in range(track_ids.shape[0]):
        for view_id in range(classes.shape[1]):
            valid = (
                track_ids[time_id].gt(0)
                & masks[time_id, view_id]
                & classes[time_id, view_id].ge(0)
                & classes[time_id, view_id].le(4)
            )
            slots = valid.nonzero(as_tuple=False).flatten().tolist()
            dimensions = []
            for slot in slots:
                region = project_box_to_feature_mask(
                    corners[time_id, view_id, slot],
                    intrinsics[0, time_id, view_id],
                    camera_to_common[0, time_id, view_id],
                    box_reference_to_common[0, time_id],
                    18,
                    32,
                )
                region_counts.append(int(region.sum()))
                dimensions.append(
                    (
                        int(classes[time_id, view_id, slot]),
                        int(track_ids[time_id, slot]),
                        box_dimensions_from_corners(
                            corners[time_id, view_id, slot]
                        ),
                    )
                )
            for index_a, object_a in enumerate(dimensions):
                for object_b in dimensions[index_a + 1:]:
                    if object_a[:2] == object_b[:2]:
                        continue
                    if object_a[0] != object_b[0]:
                        continue
                    size_distances.append(
                        float(dimension_log_distance(object_a[2], object_b[2]))
                    )

    region_tensor = torch.tensor(region_counts, dtype=torch.float32)
    size_tensor = torch.tensor(size_distances, dtype=torch.float32)
    region_quantiles = (
        torch.quantile(region_tensor, torch.tensor([0.1, 0.5, 0.9])).tolist()
        if region_tensor.numel()
        else []
    )
    size_quantiles = (
        torch.quantile(size_tensor, torch.tensor([0.1, 0.5, 0.9])).tolist()
        if size_tensor.numel()
        else []
    )
    print(
        f"[{name}] region_count={len(region_counts)} "
        f"region_patch_q10_q50_q90={region_quantiles} "
        f"same_class_diff_id_size_q10_q50_q90={size_quantiles}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/lyh/bev_pv_epipolar_track.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("waymo", "nuscenes", "argoverse", "nuplan"),
        default=("waymo", "nuscenes", "argoverse", "nuplan"),
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    dwm.common.global_state["nuscenes_fs"] = (
        dwm.common.create_instance_from_config(
            config["global_state"]["nuscenes_fs"]
        )
    )
    training = config["training_dataset"]["base_dataset"]["datasets"]
    validation = config["validation_dataset"]["base_dataset"]["datasets"]
    dataset_configs = (
        ("waymo", training[0]["bev_dataset"]),
        ("nuscenes", training[1]["bev_dataset"]),
        ("argoverse", training[2]["bev_dataset"]),
        ("nuplan", validation[0]["bev_dataset"]),
    )
    temporary_directories = []
    for name, dataset_config in dataset_configs:
        if name not in args.datasets:
            continue
        dataset_config = copy.deepcopy(dataset_config)
        if name == "nuscenes":
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="track_nuscenes_smoke_"
            )
            temporary_directories.append(temporary_directory)
            source_root = config["global_state"]["nuscenes_fs"]["path"]
            dataset_config["dataset_name"] = build_nuscenes_smoke_root(
                source_root,
                temporary_directory.name,
            )
            dataset_config["fs"] = {
                "_class_name": "dwm.fs.dirfs.DirFileSystem",
                "path": temporary_directory.name,
            }
        if name == "argoverse":
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="track_argoverse_smoke_"
            )
            temporary_directories.append(temporary_directory)
            mini_index = os.path.join(temporary_directory.name, "index")
            os.makedirs(mini_index)
            with open(
                dataset_config["balanced_json_path"],
                "r",
                encoding="utf-8",
            ) as file:
                interval = json.load(file)[0]
            scene_name = interval["scene_name"]
            source_info = os.path.join(
                dataset_config["index_json_path"],
                f"{scene_name}.info.json",
            )
            shutil.copy(source_info, mini_index)
            mini_balance = os.path.join(
                temporary_directory.name,
                "balanced_windows.json",
            )
            with open(mini_balance, "w", encoding="utf-8") as file:
                json.dump([interval], file)
            dataset_config["index_json_path"] = mini_index
            dataset_config["balanced_json_path"] = mini_balance
        dataset = dwm.common.create_instance_from_config(dataset_config)
        summarize_item(name, dataset)


if __name__ == "__main__":
    main()
