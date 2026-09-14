import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dwm.common


def create_parser():
    parser = argparse.ArgumentParser(
        description="Export real validation multi-view videos and manifest for offline FVD."
    )
    parser.add_argument("-c", "--config-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="real")
    parser.add_argument("--manifest-name", type=str, default="real_manifest.jsonl")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--image-quality", type=int, default=95)
    return parser


def setup_global_state_without_device_mesh(config):
    if "global_state" not in config:
        return

    for key, value in config["global_state"].items():
        class_name = ""
        if isinstance(value, dict):
            class_name = str(value.get("_class_name", ""))

        if key == "device_mesh" or "init_device_mesh" in class_name:
            continue

        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)


def tensor_to_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def find_string_list_by_key(node, target_keys):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in target_keys and isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    return value

        for value in node.values():
            found = find_string_list_by_key(value, target_keys)
            if found is not None:
                return found

    if isinstance(node, list):
        for value in node:
            found = find_string_list_by_key(value, target_keys)
            if found is not None:
                return found

    return None


def normalize_camera_names(value, view_count):
    if isinstance(value, (list, tuple)) and len(value) == view_count:
        if all(isinstance(item, str) for item in value):
            return [str(item) for item in value]

    if isinstance(value, (list, tuple)) and len(value) > 0:
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) == view_count:
            if all(isinstance(item, str) for item in first):
                return [str(item) for item in first]

    return None


def get_camera_names_from_config(config, view_count):
    candidate_keys = {
        "sensor_channels",
        "camera_names",
        "view_names",
        "cameras",
        "camera_channels",
    }
    camera_names = find_string_list_by_key(
        config.get("validation_dataset", {}),
        candidate_keys,
    )

    if camera_names is None:
        camera_names = find_string_list_by_key(config, candidate_keys)

    normalized_names = normalize_camera_names(camera_names, view_count)
    if normalized_names is not None:
        return normalized_names

    return [f"CAM_{index:02d}" for index in range(view_count)]


def get_camera_names_from_batch(batch, config, view_count):
    for key in ["camera_names", "sensor_channels", "view_names", "camera_channels"]:
        if key not in batch:
            continue

        normalized_names = normalize_camera_names(batch[key], view_count)
        if normalized_names is not None:
            return normalized_names

    return get_camera_names_from_config(config, view_count)


def get_dataloader_config(config):
    if "validation_dataloader" in config:
        return config["validation_dataloader"]

    if "preview_dataloader" in config:
        return config["preview_dataloader"]

    raise KeyError("Config must contain validation_dataloader or preview_dataloader.")


def image_size_to_tuple(image_size_tensor, image_tensor):
    if isinstance(image_size_tensor, torch.Tensor):
        image_size_list = image_size_tensor.detach().cpu().tolist()
        width = int(round(float(image_size_list[0])))
        height = int(round(float(image_size_list[1])))
        return width, height

    _, height, width = image_tensor.shape
    return width, height


def save_tensor_image(image_tensor, image_path, image_quality):
    image_tensor = image_tensor.detach().cpu().float().clamp(0, 1)
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    Image.fromarray(image_array, mode="RGB").save(image_path, quality=image_quality)


def get_frame_ego_transform(batch, local_batch_index, time_index):
    if "ego_transforms" not in batch:
        return None

    ego_transform = batch["ego_transforms"][local_batch_index, time_index]

    if ego_transform.ndim == 3:
        ego_transform = ego_transform[0]

    if ego_transform.ndim != 2:
        raise ValueError(
            f"Unexpected ego_transforms shape at frame: {tuple(ego_transform.shape)}"
        )

    return ego_transform


def save_valid_mask(mask_tensor, mask_path):
    mask_array = mask_tensor.detach().cpu().float().numpy()

    while mask_array.ndim > 2:
        mask_array = mask_array[0]

    mask_array = (mask_array > 0.5).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask_array, mode="L").save(mask_path)


def build_view_record(
    relative_image_path,
    relative_mask_path,
    camera_name,
    batch,
    local_batch_index,
    time_index,
    view_index,
):
    view_record = {
        "camera": camera_name,
        "image_path": relative_image_path,
        "valid_mask_path": relative_mask_path,
    }

    if "camera_intrinsics" in batch:
        view_record["K"] = tensor_to_json(
            batch["camera_intrinsics"][local_batch_index, time_index, view_index]
        )

    if "camera_transforms" in batch:
        view_record["T_cam_to_ego"] = tensor_to_json(
            batch["camera_transforms"][local_batch_index, time_index, view_index]
        )

    if "image_size" in batch:
        view_record["image_size"] = tensor_to_json(
            batch["image_size"][local_batch_index, time_index, view_index]
        )

    return view_record


def export_one_batch(
    output_root,
    manifest_file,
    batch,
    config,
    dataset_name,
    exported_count,
    max_videos,
    image_quality,
):
    if "vae_images" not in batch:
        raise KeyError("Missing batch['vae_images'].")

    images = batch["vae_images"]
    batch_size, sequence_length, view_count = images.shape[:3]
    camera_names = get_camera_names_from_batch(batch, config, view_count)
    valid_mask = batch.get("valid_mask", None)

    saved_count = 0

    for local_batch_index in range(batch_size):
        if max_videos is not None and exported_count + saved_count >= max_videos:
            break

        video_index = exported_count + saved_count
        video_id = f"{dataset_name}_video_{video_index:06d}"
        frames = []

        for time_index in range(sequence_length):
            frame_record = {
                "frame_index": time_index,
                "views": [],
            }

            ego_transform = get_frame_ego_transform(batch, local_batch_index, time_index)
            if ego_transform is not None:
                frame_record["T_ego_to_world"] = tensor_to_json(ego_transform)

            for view_index in range(view_count):
                camera_name = camera_names[view_index]
                image_tensor = images[local_batch_index, time_index, view_index]

                relative_image_path = os.path.join(
                    "images",
                    video_id,
                    f"t{time_index:03d}",
                    f"{camera_name}.jpg",
                )
                absolute_image_path = output_root / relative_image_path
                save_tensor_image(image_tensor, str(absolute_image_path), image_quality)

                relative_mask_path = None
                if isinstance(valid_mask, torch.Tensor):
                    relative_mask_path = os.path.join(
                        "masks",
                        video_id,
                        f"t{time_index:03d}",
                        f"{camera_name}.png",
                    )
                    absolute_mask_path = output_root / relative_mask_path
                    save_valid_mask(
                        valid_mask[local_batch_index, time_index, view_index],
                        str(absolute_mask_path),
                    )

                view_record = build_view_record(
                    relative_image_path,
                    relative_mask_path,
                    camera_name,
                    batch,
                    local_batch_index,
                    time_index,
                    view_index,
                )
                frame_record["views"].append(view_record)

            frames.append(frame_record)

        manifest_item = {
            "video_id": video_id,
            "dataset_name": dataset_name,
            "frames": frames,
        }
        manifest_file.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
        manifest_file.flush()

        saved_count += 1
        print(f"[real-export] saved {video_id}", flush=True)

    return saved_count


def main():
    args = create_parser().parse_args()

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    setup_global_state_without_device_mesh(config)

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"]
    )
    dataloader_config = get_dataloader_config(config)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        **dwm.common.instantiate_config(dataloader_config),
    )
    print(f"[real-export] validation dataset loaded: {len(validation_dataset)} items")

    exported_count = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for batch_index, batch in enumerate(validation_dataloader):
            if args.max_videos is not None and exported_count >= args.max_videos:
                break

            saved_count = export_one_batch(
                output_root,
                manifest_file,
                batch,
                config,
                args.dataset_name,
                exported_count,
                args.max_videos,
                args.image_quality,
            )
            exported_count += saved_count

            print(
                f"[real-export] batch={batch_index}, saved_count={saved_count}, total={exported_count}",
                flush=True,
            )

    print(f"[real-export] done: {exported_count} videos")
    print(f"[real-export] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
