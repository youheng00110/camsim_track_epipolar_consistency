import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import dwm.common


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export generated multi-view videos as per-view images plus "
            "a camera metadata manifest for ST-Flow evaluation."
        )
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="Config path for loading pipeline and validation dataset.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output directory for generated images and manifest.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unknown",
        help="Dataset tag written into the manifest, e.g. waymo or nuplan.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="stflow_manifest.jsonl",
        help="Manifest file name.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of exported B-level video clips.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="JPEG quality for generated images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override. If omitted, use config['device'].",
    )
    parser.add_argument(
        "--export-paired-real",
        action="store_true",
        help="Also save real validation images from the same batch and write real_image_path into manifest.",
    )
    return parser


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = rank == 0
    return rank, local_rank, world_size, is_main_process


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank0_print(is_main_process, message):
    if is_main_process:
        print(message, flush=True)


def maybe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_device(config, device_override, local_rank):
    device_name = device_override if device_override is not None else config["device"]

    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    return torch.device(device_name)


def setup_global_state(config):
    if "global_state" not in config:
        return

    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)


def flatten_generated_images(images):
    flat_images = []
    stack = [images]

    while len(stack) > 0:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            flat_images.append(item)

    return flat_images


def tensor_to_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value

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


def get_required_batch_tensor(batch, key):
    if key not in batch:
        raise KeyError(
            f"Missing batch['{key}']. ST-Flow export requires "
            "camera_intrinsics, camera_transforms and image_size."
        )

    if not isinstance(batch[key], torch.Tensor):
        raise TypeError(f"batch['{key}'] must be a torch.Tensor.")

    return batch[key]


def save_valid_mask(mask_tensor, mask_path):
    mask_array = mask_tensor.detach().cpu().float().numpy()

    while mask_array.ndim > 2:
        mask_array = mask_array[0]

    mask_array = (mask_array > 0.5).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask_array, mode="L").save(mask_path)



def save_real_image_tensor(image_tensor, image_path, image_quality, size_tuple=None):
    image_tensor = image_tensor.detach().cpu().float().clamp(0, 1)
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255.0).round().astype(np.uint8)

    image = Image.fromarray(image_array, mode="RGB")
    if size_tuple is not None and image.size != size_tuple:
        image = image.resize(size_tuple)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path, quality=image_quality)


def image_size_to_tuple(image_size_tensor):
    image_size_list = image_size_tensor.detach().cpu().tolist()
    width = int(round(float(image_size_list[0])))
    height = int(round(float(image_size_list[1])))
    return width, height


def get_latent_shape(pipeline, batch):
    batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]

    vae_downsample_factor = 2 ** (len(pipeline.vae.config.down_block_types) - 1)
    latent_height = batch["vae_images"].shape[-2] // vae_downsample_factor
    latent_width = batch["vae_images"].shape[-1] // vae_downsample_factor

    latent_shape = (
        batch_size,
        sequence_length,
        view_count,
        pipeline.vae.config.latent_channels,
        latent_height,
        latent_width,
    )
    return latent_shape


def build_view_record(
    relative_image_path,
    relative_mask_path,
    camera_name,
    camera_intrinsics,
    camera_transforms,
    image_size,
    local_batch_index,
    time_index,
    view_index,
):
    intrinsic = camera_intrinsics[local_batch_index, time_index, view_index]
    transform = camera_transforms[local_batch_index, time_index, view_index]
    image_size_value = image_size[local_batch_index, time_index, view_index]

    view_record = {
        "camera": camera_name,
        "image_path": relative_image_path,
        "valid_mask_path": relative_mask_path,
        "K": tensor_to_json(intrinsic),
        "T_cam_to_ego": tensor_to_json(transform),
        "image_size": tensor_to_json(image_size_value),
    }
    return view_record


def export_one_batch(
    output_root,
    manifest_file,
    generated_images,
    batch,
    config,
    dataset_name,
    exported_count,
    max_videos,
    image_quality,
    export_paired_real,
):
    batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
    camera_names = get_camera_names_from_batch(batch, config, view_count)

    camera_intrinsics = get_required_batch_tensor(batch, "camera_intrinsics")
    camera_transforms = get_required_batch_tensor(batch, "camera_transforms")
    image_size = get_required_batch_tensor(batch, "image_size")
    valid_mask = batch.get("valid_mask", None)

    inference_config = config.get("pipeline", {}).get("inference_config", {})
    reference_frame_count = int(inference_config.get("reference_frame_count", 0))
    generate_frames_for_reference = bool(
        inference_config.get("generate_frames_for_reference", True)
    )

    full_image_count = batch_size * sequence_length * view_count
    predicted_sequence_length = sequence_length
    if not generate_frames_for_reference:
        predicted_sequence_length = max(sequence_length - reference_frame_count, 0)

    predicted_image_count = batch_size * predicted_sequence_length * view_count

    if len(generated_images) == full_image_count:
        output_layout = "full"
    elif len(generated_images) == predicted_image_count:
        output_layout = "without_reference"
    else:
        raise RuntimeError(
            f"Unexpected generated image count: got {len(generated_images)}, "
            f"expected either full={full_image_count} or "
            f"without_reference={predicted_image_count}. "
            f"B={batch_size}, T={sequence_length}, V={view_count}, "
            f"reference_frame_count={reference_frame_count}, "
            f"generate_frames_for_reference={generate_frames_for_reference}."
        )

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

                size_tuple = image_size_to_tuple(
                    image_size[local_batch_index, time_index, view_index]
                )

                real_image_tensor = batch["vae_images"][
                    local_batch_index,
                    time_index,
                    view_index,
                ]

                # Decide whether this fake frame should be copied from GT reference
                # or from model output.
                use_gt_as_fake = (
                    (not generate_frames_for_reference)
                    and time_index < reference_frame_count
                )

                generated_image = None
                if not use_gt_as_fake:
                    if output_layout == "full":
                        flat_index = (
                            local_batch_index * sequence_length * view_count
                            + time_index * view_count
                            + view_index
                        )
                    else:
                        generated_time_index = time_index - reference_frame_count
                        if generated_time_index < 0:
                            raise RuntimeError(
                                "Internal error: reference frame requested from "
                                "without_reference generated output."
                            )
                        flat_index = (
                            local_batch_index * predicted_sequence_length * view_count
                            + generated_time_index * view_count
                            + view_index
                        )

                    generated_image = generated_images[flat_index]

                relative_image_path = os.path.join(
                    "images",
                    video_id,
                    f"t{time_index:03d}",
                    f"{camera_name}.jpg",
                )
                absolute_image_path = output_root / relative_image_path
                absolute_image_path.parent.mkdir(parents=True, exist_ok=True)

                if use_gt_as_fake:
                    save_real_image_tensor(
                        real_image_tensor,
                        str(absolute_image_path),
                        image_quality,
                        size_tuple,
                    )
                else:
                    generated_image.resize(size_tuple).save(
                        absolute_image_path,
                        quality=image_quality,
                    )

                real_relative_image_path = None
                if export_paired_real:
                    real_relative_image_path = os.path.join(
                        "paired_real",
                        video_id,
                        f"t{time_index:03d}",
                        f"{camera_name}.jpg",
                    )
                    real_absolute_image_path = output_root / real_relative_image_path
                    save_real_image_tensor(
                        real_image_tensor,
                        str(real_absolute_image_path),
                        image_quality,
                        size_tuple,
                    )

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
                    camera_intrinsics,
                    camera_transforms,
                    image_size,
                    local_batch_index,
                    time_index,
                    view_index,
                )
                if real_relative_image_path is not None:
                    view_record["real_image_path"] = real_relative_image_path
                if use_gt_as_fake:
                    view_record["is_reference_frame"] = True

                frame_record["views"].append(view_record)

            frames.append(frame_record)

        manifest_item = {
            "video_id": video_id,
            "dataset_name": dataset_name,
            "frames": frames,
            "reference_frame_count": reference_frame_count,
            "generate_frames_for_reference": generate_frames_for_reference,
        }
        manifest_file.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
        manifest_file.flush()

        saved_count += 1
        print(f"[export] saved {video_id}", flush=True)

    return saved_count


def main():
    parser = create_parser()
    args = parser.parse_args()

    rank, local_rank, world_size, is_main_process = get_rank_info()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = setup_device(config, args.device, local_rank)
    setup_global_state(config)

    output_root = Path(args.output_path)
    manifest_path = output_root / args.manifest_name

    if is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    maybe_barrier()

    rank0_print(
        is_main_process,
        f"[export] rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}",
    )

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=args.output_path,
        config=config,
        device=device,
    )
    rank0_print(is_main_process, "[export] pipeline loaded")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"]
    )
    dataloader_config = get_dataloader_config(config)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        **dwm.common.instantiate_config(dataloader_config),
    )
    rank0_print(
        is_main_process,
        f"[export] validation dataset loaded: {len(validation_dataset)} items",
    )

    exported_count = 0
    manifest_file = None

    if is_main_process:
        manifest_file = open(manifest_path, "w", encoding="utf-8")

    try:
        for batch_index, batch in enumerate(validation_dataloader):
            if args.max_videos is not None and exported_count >= args.max_videos:
                break

            if "vae_images" not in batch:
                raise KeyError("Missing batch['vae_images'].")

            latent_shape = get_latent_shape(pipeline, batch)

            with torch.no_grad():
                pipeline_output = pipeline.inference_pipeline(latent_shape, batch, "pil")

            batch_size = batch["vae_images"].shape[0]
            remaining = batch_size
            if args.max_videos is not None:
                remaining = max(args.max_videos - exported_count, 0)
                remaining = min(batch_size, remaining)

            if is_main_process:
                if "images" not in pipeline_output:
                    print(f"[export] batch {batch_index}: no images in pipeline output", flush=True)
                    saved_count = 0
                else:
                    generated_images = flatten_generated_images(pipeline_output["images"])
                    saved_count = export_one_batch(
                        output_root,
                        manifest_file,
                        generated_images,
                        batch,
                        config,
                        args.dataset_name,
                        exported_count,
                        args.max_videos,
                        args.image_quality,
                        args.export_paired_real,
                    )
            else:
                saved_count = remaining

            exported_count += saved_count

            rank0_print(
                is_main_process,
                (
                    f"[export] batch={batch_index}, saved_count={saved_count}, "
                    f"total={exported_count}"
                ),
            )

    finally:
        if manifest_file is not None:
            manifest_file.close()

    maybe_barrier()

    rank0_print(is_main_process, f"[export] done: {exported_count} videos")
    rank0_print(is_main_process, f"[export] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
