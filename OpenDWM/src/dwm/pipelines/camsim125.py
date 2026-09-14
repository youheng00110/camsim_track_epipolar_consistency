import contextlib
import diffusers
import diffusers.image_processor
from diffusers.image_processor import VaeImageProcessor
import dwm.common
import dwm.distributed
import dwm.functional
import dwm.models.crossview_temporal_unet
import dwm.utils.preview
import einops
import itertools
import math
import os
import re
import safetensors.torch
import time
import torch
import torch.amp
import torch.distributed.checkpoint.state_dict
import torch.distributed.fsdp
import torch.distributed.fsdp.sharded_grad_scaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.utils.tensorboard
import torchvision
import transformers

############ ck func #########################

def test_fn(L,S,T,C,start_timestep,stop_timestep,take_time,frozen_ref_count):
    
    base = start_timestep - take_time * S
    j = torch.arange(L)
    j_eff = torch.clamp(j - C, min=0)
    inner = torch.clamp(base - j_eff * S, min=0)
    idx = torch.minimum(inner, torch.full_like(inner, base))
    max_idx = T - 1
    idx = torch.clamp(idx, 0, max_idx)

    if frozen_ref_count > 0:
        clean_idx = max_idx
        idx[:frozen_ref_count] = clean_idx
    print('start_t',idx)
    base = stop_timestep - take_time * S
    j = torch.arange(L)
    j_eff = torch.clamp(j - C, min=0)
    inner = torch.clamp(base - j_eff * S, min=0)
    idx = torch.minimum(inner, torch.full_like(inner, base))
    max_idx = T - 1
    idx = torch.clamp(idx, 0, max_idx)

    if frozen_ref_count > 0:
        clean_idx = max_idx
        idx[:frozen_ref_count] = clean_idx
    print('end_t',idx)

def ck(name, x):
    if not torch.isfinite(x).all():
        print("!!!", name, "non-finite", torch.isnan(x).any().item(), torch.isinf(x).any().item())
        raise RuntimeError(name)

############ ck func #########################
###########debug nan###########################
def tensor_debug_line(name, x):
    if not torch.is_tensor(x):
        return "{}: not tensor, value={}\n".format(name, x)

    x_detach = x.detach()

    if not torch.is_floating_point(x_detach):
        x_float = x_detach.float()
    else:
        x_float = x_detach.float()

    finite_mask = torch.isfinite(x_float)
    bad_count = finite_mask.logical_not().sum().item()

    if finite_mask.any():
        finite_values = x_float[finite_mask]
        min_value = finite_values.min().item()
        mean_value = finite_values.mean().item()
        max_value = finite_values.max().item()
    else:
        min_value = float("nan")
        mean_value = float("nan")
        max_value = float("nan")

    return (
        "{}: shape={}, dtype={}, bad_count={}, min={:.6f}, "
        "mean={:.6f}, max={:.6f}\n"
    ).format(
        name,
        tuple(x_detach.shape),
        x_detach.dtype,
        bad_count,
        min_value,
        mean_value,
        max_value,
    )


def save_nan_batch_visualization(
    batch,
    model_conditions,
    sd_pred,
    sd_pred_latent,
    target,
    noise,
    latents,
    sigmas,
    timesteps,
    reference_frame_indicator,
    global_step,
    reason,
    root="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nan_batch",
):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if "dataset_tag" in batch:
        tag_value = int(batch["dataset_tag"].detach().cpu().flatten()[0].item())
    else:
        tag_value = -1

    save_root = os.path.join(
        root,
        "step{:06d}_rank{}_tag{}_{}".format(
            global_step + 1,
            rank,
            tag_value,
            reason,
        ),
    )
    os.makedirs(save_root, exist_ok=True)

    t_candidates = [0, 1, 2, 3, 5, 10, 19]
    if "vae_images" in batch:
        sequence_length = batch["vae_images"].shape[1]
    else:
        sequence_length = 0

    t_list = [t for t in t_candidates if t < sequence_length]

    for t in t_list:
        if "vae_images" in batch:
            rgb_grid = batch["vae_images"][0, t].detach().float().cpu()
            rgb_grid = torch.nan_to_num(
                rgb_grid, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            torchvision.utils.save_image(
                rgb_grid,
                os.path.join(save_root, "grid_t{:02d}_rgb.png".format(t)),
                nrow=rgb_grid.shape[0],
            )

        if "3dbox_images" in batch:
            box_grid = batch["3dbox_images"][0, t].detach().float().cpu()
            box_grid = torch.nan_to_num(
                box_grid, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            torchvision.utils.save_image(
                box_grid,
                os.path.join(save_root, "grid_t{:02d}_box.png".format(t)),
                nrow=box_grid.shape[0],
            )

        if "hdmap_images" in batch:
            hdmap_grid = batch["hdmap_images"][0, t].detach().float().cpu()
            hdmap_grid = torch.nan_to_num(
                hdmap_grid, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            torchvision.utils.save_image(
                hdmap_grid,
                os.path.join(save_root, "grid_t{:02d}_hdmap.png".format(t)),
                nrow=hdmap_grid.shape[0],
            )

        if model_conditions.get("condition_image_tensor", None) is not None:
            cond_grid = model_conditions["condition_image_tensor"][0, t] \
                .detach().float().cpu()
            cond_grid = torch.nan_to_num(
                cond_grid, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            if cond_grid.shape[1] >= 3:
                cond_box_grid = cond_grid[:, :3]
                torchvision.utils.save_image(
                    cond_box_grid,
                    os.path.join(
                        save_root,
                        "grid_t{:02d}_cond_box.png".format(t),
                    ),
                    nrow=cond_box_grid.shape[0],
                )

            if cond_grid.shape[1] >= 6:
                cond_hdmap_grid = cond_grid[:, 3:6]
                torchvision.utils.save_image(
                    cond_hdmap_grid,
                    os.path.join(
                        save_root,
                        "grid_t{:02d}_cond_hdmap.png".format(t),
                    ),
                    nrow=cond_hdmap_grid.shape[0],
                )

    geo_path = os.path.join(save_root, "debug.txt")
    with open(geo_path, "w") as f:
        f.write("reason={}\n".format(reason))
        f.write("step={}\n".format(global_step + 1))
        f.write("rank={}\n".format(rank))
        f.write("dataset_tag={}\n".format(tag_value))

        if "camera_names" in batch:
            f.write("camera_names={}\n".format(batch["camera_names"]))

        if "crossview_mask" in batch:
            crossview_mask = batch["crossview_mask"].detach().cpu()
            f.write("crossview_mask={}\n".format(crossview_mask.tolist()))
            f.write(
                "crossview_row_sum={}\n".format(
                    crossview_mask.bool().sum(-1).tolist()
                )
            )

        if "image_size" in batch:
            f.write(tensor_debug_line("batch.image_size", batch["image_size"]))

        if "camera_intrinsics" in batch:
            f.write(tensor_debug_line(
                "batch.camera_intrinsics",
                batch["camera_intrinsics"],
            ))

        if "camera_transforms" in batch:
            f.write(tensor_debug_line(
                "batch.camera_transforms",
                batch["camera_transforms"],
            ))

        if "ego_transforms" in batch:
            f.write(tensor_debug_line(
                "batch.ego_transforms",
                batch["ego_transforms"],
            ))

        f.write(tensor_debug_line("latents", latents))
        f.write(tensor_debug_line("noise", noise))
        f.write(tensor_debug_line("timesteps", timesteps.float()))
        f.write(tensor_debug_line("reference_frame_indicator", reference_frame_indicator.float()))

        if sigmas is not None:
            f.write(tensor_debug_line("sigmas", sigmas))

        if sd_pred is not None and len(sd_pred) > 0:
            f.write(tensor_debug_line("sd_pred[0]", sd_pred[0]))

        if sd_pred_latent is not None:
            f.write(tensor_debug_line("sd_pred_latent", sd_pred_latent))

        if target is not None:
            f.write(tensor_debug_line("target", target))

        for key in [
            "added_time_ids",
            "condition_image_tensor",
            "camera_intrinsics_norm",
            "camera2referego",
            "crossview_attention_mask",
        ]:
            if model_conditions.get(key, None) is not None:
                value = model_conditions[key]
                if torch.is_tensor(value):
                    f.write(tensor_debug_line("model_conditions." + key, value))

    print(
        "[NAN_BATCH_SAVED] rank={} step={} reason={} path={}".format(
            rank,
            global_step + 1,
            reason,
            save_root,
        ),
        flush=True,
    )
    ###########################################################
class CrossviewTemporalSD():

    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)

        return state
    @staticmethod
    def zero_missing_crossview_weights(
        model,
        missing_keys,
        include_mixers: bool = False,
    ):
        target_model = model.module if hasattr(model, "module") else model

        param_dict = dict(target_model.named_parameters())
        buffer_dict = dict(target_model.named_buffers())

        crossview_markers = [
            "view_pos_embeds.",
            "crossview_transformer_blocks.",
        ]

        if include_mixers:
            crossview_markers.append("view_mixers.")

        zeroed_keys = []

        with torch.no_grad():
            for key in missing_keys:
                if not any(marker in key for marker in crossview_markers):
                    continue

                tensor = param_dict.get(key, None)

                if tensor is None:
                    tensor = buffer_dict.get(key, None)

                if tensor is None:
                    continue

                if not torch.is_floating_point(tensor):
                    continue

                tensor.zero_()
                zeroed_keys.append(key)

        return zeroed_keys
    @staticmethod
    def flatten_clip_text(
        clip_text, flattened_clip_text: list, parsed_shape: list,
        level: int = 0, text_condition_mask=None,
        do_classifier_free_guidance: bool = False
    ):
        level_count = 0
        if isinstance(clip_text, list) and len(parsed_shape) <= level:
            parsed_shape.append(0)

        if do_classifier_free_guidance:
            if isinstance(clip_text, str):
                flattened_clip_text.append("")
                level_count += 1
            else:
                for i in clip_text:
                    CrossviewTemporalSD.flatten_clip_text(
                        i, flattened_clip_text, parsed_shape, level + 1,
                        text_condition_mask, do_classifier_free_guidance)
                    level_count += 1

        if level == 0 or not do_classifier_free_guidance:
            if isinstance(clip_text, str):
                if text_condition_mask is None or (
                        isinstance(text_condition_mask, bool) and
                        text_condition_mask):
                    flattened_clip_text.append(clip_text)
                else:
                    flattened_clip_text.append("")

                level_count += 1

            else:
                for i_id, i in enumerate(clip_text):
                    CrossviewTemporalSD.flatten_clip_text(
                        i, flattened_clip_text, parsed_shape, level + 1,
                        None if text_condition_mask is None else (
                            text_condition_mask[i_id] if
                            isinstance(text_condition_mask, list) else
                            text_condition_mask),
                        False)
                    level_count += 1

        if isinstance(clip_text, list):
            parsed_shape[level] = level_count
    @staticmethod
    def get_camera_param_token(batch, common_config):
        """
        return: [B, T, V, 3, 7]
        """
        K = batch["camera_intrinsics"].clone()
        image_size = batch["image_size"].to(K)

        # normalize intrinsics
        # image_size[..., 0] = W, image_size[..., 1] = H
        K[..., 0, 0] = K[..., 0, 0] / image_size[..., 0]
        K[..., 1, 1] = K[..., 1, 1] / image_size[..., 1]
        K[..., 0, 2] = K[..., 0, 2] / image_size[..., 0]
        K[..., 1, 2] = K[..., 1, 2] / image_size[..., 1]

        # camera_transforms: camera -> ego / sensor -> ego
        E = batch["camera_transforms"].clone()

        # 只取前 3 行，和 MagicDrive-V2 的 3x7 对齐
        E = E[..., :3, :4]

        # 平移归一化，避免 Fourier embedding 被米制尺度冲爆
        scale = common_config.get("camera_token_translation_scale", 10.0)
        E[..., :3, 3] = E[..., :3, 3] / scale

        cam_param = torch.cat([K[..., :3, :3], E], dim=-1)

        return cam_param
    @staticmethod
    def get_camera_transform_ids(batch, common_config):
        return torch.cat([
            batch["camera_intrinsics"].flatten(-2, -1)[
                ..., common_config["camera_intrinsic_embedding_indices"]
            ] / batch["image_size"][
                ..., common_config["camera_intrinsic_denom_embedding_indices"]
            ],
            batch["camera_transforms"].flatten(-2, -1)[
                ..., common_config["camera_transform_embedding_indices"]
            ]
        ], -1)

    @staticmethod
    def get_action_ids(
        batch, common_config: dict, action_condition_mask=None,
        streaming_mode: bool = False, prev_ego_transforms=None
    ):
        if streaming_mode:
            assert batch["ego_transforms"].shape[1] == 1
            ego_transforms = torch.cat([
                (
                    batch["ego_transforms"]
                    if prev_ego_transforms is None
                    else prev_ego_transforms
                ),
                batch["ego_transforms"]
            ], dim=1)
        else:
            ego_transforms = batch["ego_transforms"]

        current_pose = ego_transforms[
            :, :, common_config["camera_ego_sensor_indices"]
        ]
        uncondition_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if action_condition_mask is None:
            is_conditioned = (current_pose - uncondition_pose)\
                .sum((1, 2, 3, 4)).abs() > 1e-3
        else:
            is_conditioned = torch.logical_and(
                (current_pose - uncondition_pose)
                .sum((1, 2, 3, 4)).abs() > 1e-3,
                action_condition_mask)

        relative_pose = torch.linalg.solve(
            current_pose[:, :-1], current_pose[:, 1:])
        relative_pose = torch.cat([relative_pose[:, :1], relative_pose], 1)

        moving_distance = torch.norm(
            relative_pose[..., :3, 3], dim=-1, keepdim=True)
        mps_to_kmph = 3.6
        speed = mps_to_kmph * moving_distance * \
            batch["fps"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        rotation_angles = torch.atan2(
            relative_pose[..., 1, 0:1] - relative_pose[..., 0, 1:2],
            relative_pose[..., 0, 0:1] + relative_pose[..., 1, 1:2])
        wheel_base = 2.7
        steering_ratio = 14
        steering = torch.where(
            torch.abs(moving_distance) > 0.01,
            rotation_angles / moving_distance * wheel_base * steering_ratio,
            -1000.0 * torch.ones_like(rotation_angles))
        action_ids = torch.cat([speed, steering], -1)

        # mask with unconditional cases
        action_ids = torch.where(
            is_conditioned.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            action_ids, -1000.0 * torch.ones_like(action_ids))
        if streaming_mode:
            action_ids = action_ids.chunk(2, dim=1)[-1]

        return action_ids
    @staticmethod
    def get_camera_param_token(batch, common_config):
        """
        Returns:
            camera_param_token: [B, T, V, 3, 7]
        """
        K = batch["camera_intrinsics"].clone().float()
        image_size = batch["image_size"].to(K)

        K[..., 0, 0] = K[..., 0, 0] / image_size[..., 0]
        K[..., 1, 1] = K[..., 1, 1] / image_size[..., 1]
        K[..., 0, 2] = K[..., 0, 2] / image_size[..., 0]
        K[..., 1, 2] = K[..., 1, 2] / image_size[..., 1]

        E = batch["camera_transforms"].clone().float()
        E = E[..., :3, :4]

        scale = common_config.get("camera_token_translation_scale", 10.0)
        E[..., :3, 3] = E[..., :3, 3] / scale

        camera_param_token = torch.cat([K[..., :3, :3], E], dim=-1)
        return camera_param_token

    @staticmethod
    def get_bbox_token_inputs(batch, common_config, device, dtype,
                              do_classifier_free_guidance=False):
        """
        Expected batch fields:
            bbox_token_corners: [B, T, S, 8, 3]
            bbox_token_classes: [B, T, S]
            bbox_token_masks:   [B, T, S]
        mask meaning:
             1 = keep real box
             0 = null / padded box
            -1 = masked box
        """
        corners_key = common_config.get(
            "bbox_token_corners_key", "bbox_token_corners")
        classes_key = common_config.get(
            "bbox_token_classes_key", "bbox_token_classes")
        masks_key = common_config.get(
            "bbox_token_masks_key", "bbox_token_masks")

        require_bbox = common_config.get("require_bbox_token_input", False)
        if corners_key not in batch:
            if require_bbox:
                raise KeyError(
                    f"bbox_token_modeling=True but `{corners_key}` is missing. "
                    "You need to add raw box token fields in dataset."
                )
            return None, None, None

        if classes_key not in batch:
            raise KeyError(
                f"`{corners_key}` exists but `{classes_key}` is missing."
            )

        bbox_token_input = batch[corners_key].to(device=device, dtype=dtype)
        bbox_class_input = batch[classes_key].to(device=device)

        if masks_key in batch:
            bbox_mask_input = batch[masks_key].to(device=device, dtype=dtype)
        else:
            bbox_mask_input = torch.ones(
                bbox_class_input.shape,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance:
            bbox_token_input = torch.cat(
                [bbox_token_input, bbox_token_input],
                dim=0,
            )
            bbox_class_input = torch.cat(
                [bbox_class_input, bbox_class_input],
                dim=0,
            )
            bbox_mask_input = torch.cat(
                [torch.zeros_like(bbox_mask_input), bbox_mask_input],
                dim=0,
            )

        return bbox_token_input, bbox_class_input, bbox_mask_input

    @staticmethod
    def get_map_token_input(batch, common_config, device, dtype,
                            do_classifier_free_guidance=False):
        """
        Default map source:
            hdmap_bev_images: [B, T, C, H, W]
        This is BEV map, not PV map.
        """
        map_key = common_config.get("map_token_key", "hdmap_bev_images")
        require_map = common_config.get("require_map_token_input", False)

        if map_key not in batch:
            if require_map:
                raise KeyError(
                    f"map_token_modeling=True but `{map_key}` is missing. "
                    "Enable hdmap_bev_settings or provide BEV map tokens."
                )
            return None

        map_token_input = batch[map_key].to(device=device, dtype=dtype)

        if map_token_input.ndim == 6 and map_token_input.shape[2] == 1:
            map_token_input = map_token_input.squeeze(2)

        if map_token_input.ndim == 6 and map_token_input.shape[2] != 1:
            b, t, n, c, h, w = map_token_input.shape
            map_token_input = map_token_input.reshape(b, t, n * c, h, w)

        if map_token_input.ndim != 5:
            raise ValueError(
                f"`{map_key}` should be [B, T, C, H, W] or [B, T, N, C, H, W], "
                f"but got {tuple(map_token_input.shape)}."
            )

        if do_classifier_free_guidance:
            map_token_input = torch.cat(
                [torch.zeros_like(map_token_input), map_token_input],
                dim=0,
            )

        return map_token_input
    @staticmethod
    def get_conditions(
        model, text_encoder, tokenizer, common_config: dict, latent_shape,
        batch: dict, device, dtype, text_condition_mask=None,
        _3dbox_condition_mask=None, hdmap_condition_mask=None,
        action_condition_mask=None, explicit_view_modeling_mask=None,
        streaming_mode: bool = False, prev_ego_transforms=None,
        do_classifier_free_guidance: bool = False,
        latents_shape=None,
    ):
        batch_size, _, view_count = latent_shape[:3]
        #############原逻辑用pts，但我们的batch里没有pts。故用vae
        sequence_length = batch["vae_images"].shape[1]
        if do_classifier_free_guidance:
            batch_size *= 2

        condition_embedding_list = []
        condition_image_list = []

        # text prompt
        if text_encoder is not None:
            flattened_clip_text = []
            parsed_shape = []
            CrossviewTemporalSD.flatten_clip_text(
                batch["clip_text"], flattened_clip_text, parsed_shape,
                text_condition_mask=text_condition_mask,
                do_classifier_free_guidance=do_classifier_free_guidance)

            pooled_text_embeddings = None
            if isinstance(model, diffusers.UNetSpatioTemporalConditionModel):
                text_inputs = tokenizer(
                    flattened_clip_text, padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt")
                text_embeddings = text_encoder(
                    text_inputs.input_ids.to(device))[0]
                if len(parsed_shape) == 1:
                    # all times and views share the same text prompt
                    text_embeddings = text_embeddings.unsqueeze(1)\
                        .unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1, 1)
                else:
                    # all times and views use different text prompts
                    text_embeddings = text_embeddings.unflatten(
                        0, parsed_shape)

                condition_embedding_list.append(text_embeddings)

            elif isinstance(model, diffusers.SD3Transformer2DModel):
                clip_text_embeddings_list = []
                clip_pooled_text_embeddings_list = []
                for clip_tokenizer, clip_text_encoder in zip(
                    tokenizer[:2], text_encoder[:2]
                ):
                    text_embeddings, pooled_text_embeddings =\
                        CrossviewTemporalSD.sd3_encode_prompt_with_clip(
                            clip_text_encoder, clip_tokenizer, common_config,
                            flattened_clip_text, clip_text_encoder.device)
                    clip_text_embeddings_list.append(text_embeddings)
                    clip_pooled_text_embeddings_list.append(
                        pooled_text_embeddings)

                clip_text_embeddings = torch.cat(
                    clip_text_embeddings_list, dim=-1)
                pooled_text_embeddings = torch.cat(
                    clip_pooled_text_embeddings_list, dim=-1)

                t5_prompt_embed = CrossviewTemporalSD.sd3_encode_prompt_with_t5(
                    text_encoder[-1], tokenizer[-1], common_config,
                    prompt=flattened_clip_text, device=device)

                clip_text_embeddings = torch.nn.functional.pad(
                    clip_text_embeddings,
                    (0, t5_prompt_embed.shape[-1] -
                     clip_text_embeddings.shape[-1]),
                )
                text_embeddings = torch.cat(
                    [clip_text_embeddings, t5_prompt_embed], dim=-2)

                if len(parsed_shape) == 1:
                    # all times and views share the same text prompt
                    text_embeddings = text_embeddings.unsqueeze(1)\
                        .unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1, 1)\
                        .to(dtype=dtype)
                    pooled_text_embeddings = pooled_text_embeddings\
                        .unsqueeze(1).unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1)\
                        .to(dtype=dtype)
                else:
                    # all times and views use different text prompts
                    text_embeddings = text_embeddings\
                        .unflatten(0, parsed_shape).to(dtype=dtype)
                    pooled_text_embeddings = pooled_text_embeddings\
                        .unflatten(0, parsed_shape).to(dtype=dtype)

                condition_embedding_list.append(text_embeddings)

        # layout condition
        condition_on_all_frames = common_config.get(
            "condition_on_all_frames", False)
        uncondition_image_color = common_config.get(
            "uncondition_image_color", 0)

        # hard switch for PV layout condition.
        # True: original OpenDWM behavior.
        # False: do not feed 3dbox_images / hdmap_images into condition_image_adapter.
        use_pv_layout_condition = common_config.get(
            "use_pv_layout_condition", True)

        debug_pv_layout_condition = common_config.get(
            "debug_pv_layout_condition", False)

        if debug_pv_layout_condition:
            is_rank0 = (
                (not torch.distributed.is_available()) or
                (not torch.distributed.is_initialized()) or
                (torch.distributed.get_rank() == 0)
            )
            if is_rank0:
                print(
                    "[PV_LAYOUT_DEBUG] "
                    f"use_pv_layout_condition={use_pv_layout_condition}, "
                    f"has_3dbox_images={'3dbox_images' in batch}, "
                    f"has_hdmap_images={'hdmap_images' in batch}",
                    flush=True,
                )

        if use_pv_layout_condition:
            if "3dbox_images" in batch:
                if condition_on_all_frames:
                    _3dbox_images = batch["3dbox_images"].to(device)
                else:
                    _3dbox_images = batch["3dbox_images"][:, :1].to(device)

                if _3dbox_condition_mask is not None:
                    _3dbox_images[
                        _3dbox_condition_mask.logical_not().to(device)
                    ] = uncondition_image_color

                if do_classifier_free_guidance:
                    _3dbox_images = torch.cat([
                        torch.ones_like(_3dbox_images) * uncondition_image_color,
                        _3dbox_images
                    ])

                condition_image_list.append(_3dbox_images)

            if "hdmap_images" in batch:
                if condition_on_all_frames:
                    hdmap_images = batch["hdmap_images"].to(device)
                else:
                    hdmap_images = batch["hdmap_images"][:, :1].to(device)

                if hdmap_condition_mask is not None:
                    hdmap_images[
                        hdmap_condition_mask.logical_not().to(device)
                    ] = uncondition_image_color

                if do_classifier_free_guidance:
                    hdmap_images = torch.cat([
                        torch.ones_like(hdmap_images) * uncondition_image_color,
                        hdmap_images
                    ])

                condition_image_list.append(hdmap_images)

        else:
            if debug_pv_layout_condition:
                is_rank0 = (
                    (not torch.distributed.is_available()) or
                    (not torch.distributed.is_initialized()) or
                    (torch.distributed.get_rank() == 0)
                )
                if is_rank0:
                    print(
                        "[PV_LAYOUT_DEBUG] hard no-PV is enabled. "
                        "Skip 3dbox_images and hdmap_images. "
                        "condition_image_tensor should become None unless other image "
                        "conditions are appended later.",
                        flush=True,
                    )

        if len(condition_embedding_list) > 0:
            encoder_hidden_states = torch.cat(condition_embedding_list, -2)\
                .to(dtype=dtype)
        else:
            encoder_hidden_states = None

        if len(condition_image_list) > 0:
            condition_image_tensor = torch.cat(condition_image_list, -3)
        else:
            condition_image_tensor = None

        if debug_pv_layout_condition:
            is_rank0 = (
                (not torch.distributed.is_available()) or
                (not torch.distributed.is_initialized()) or
                (torch.distributed.get_rank() == 0)
            )
            if is_rank0:
                if condition_image_tensor is None:
                    print(
                        "[PV_LAYOUT_DEBUG] final condition_image_tensor=None",
                        flush=True,
                    )
                else:
                    print(
                        "[PV_LAYOUT_DEBUG] final condition_image_tensor.shape="
                        f"{tuple(condition_image_tensor.shape)}",
                        flush=True,
                    )

        # additional numeric condition
        if "added_time_ids" in common_config:
            if common_config["added_time_ids"] == "fps_camera_transforms":
                added_time_ids = torch.cat([
                    batch["fps"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count, 1),
                    CrossviewTemporalSD.get_camera_transform_ids(
                        batch, common_config)
                ], -1)
                if do_classifier_free_guidance:
                    added_time_ids = torch.cat(
                        [added_time_ids, added_time_ids], 0)

                added_time_ids = added_time_ids.to(device)

            elif (
                common_config["added_time_ids"] ==
                "fps_camera_transforms_action"
            ):
                added_time_ids = torch.cat([
                    batch["fps"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count, 1),
                    CrossviewTemporalSD.get_camera_transform_ids(
                        batch, common_config),
                    CrossviewTemporalSD.get_action_ids(
                        batch, common_config, action_condition_mask,
                        streaming_mode, prev_ego_transforms)
                ], -1)
                if do_classifier_free_guidance:
                    # action is allowed to be guidance scaled
                    uncond_added_time_ids = torch.cat([
                        added_time_ids[..., :-2],
                        -1000 * torch.ones_like(added_time_ids[..., -2:])
                    ], -1)
                    added_time_ids = torch.cat(
                        [uncond_added_time_ids, added_time_ids], 0)

                added_time_ids = added_time_ids.to(device)

            else:
                added_time_ids = None

        # required by the UniMLVG
        if common_config.get("explicit_view_modeling", False):
            assert "camera_intrinsics" in batch and \
                "camera_transforms" in batch

            # without ego_transforms or single frame
            if "ego_transforms" not in batch:
                ego_transforms = torch.eye(4).to(batch["camera_transforms"])
                ego_transforms = ego_transforms.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(
                    batch["camera_transforms"].shape[0],
                    batch["camera_transforms"].shape[1],
                    batch["camera_transforms"].shape[2],
                    -1,
                    -1,
                )
            else:
                ego_transforms = batch["ego_transforms"][
                    :, :, -batch["camera_transforms"].shape[2]:, ...
                ]

            camera2world = ego_transforms @ batch["camera_transforms"]
            camera2referego = torch.linalg.inv(
                ego_transforms[:, 0, 0, :, :].unsqueeze(1).unsqueeze(2)
            ) @ camera2world

            camera_intrinsics_norm = batch["camera_intrinsics"].clone()
            camera_intrinsics_norm[..., 0, 0] = \
                camera_intrinsics_norm[..., 0, 0] / batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 1] = \
                camera_intrinsics_norm[..., 1, 1] / batch["image_size"][..., 1]
            camera_intrinsics_norm[..., 0, 2] = \
                camera_intrinsics_norm[..., 0, 2] / batch["image_size"][..., 0]
            camera_intrinsics_norm[..., 1, 2] = \
                camera_intrinsics_norm[..., 1, 2] / batch["image_size"][..., 1]

            if "is_uncalibrated" in batch:
                camera_intrinsics_norm[batch["is_uncalibrated"], :, :] = \
                    torch.eye(3).to(batch["camera_transforms"])
                camera2referego[batch["is_uncalibrated"], :, :] = \
                    torch.eye(4).to(batch["camera_transforms"])

            if explicit_view_modeling_mask is not None:
                camera_intrinsics_norm[
                    explicit_view_modeling_mask.logical_not().to(device)
                ] = torch.eye(3).to(batch["camera_transforms"])

                camera2referego[
                    explicit_view_modeling_mask.logical_not().to(device)
                ] = torch.eye(4).to(batch["camera_transforms"])

            # ---- fp16 / explicit geometry safety clamp 防止 Nan ----
            explicit_clip_cfg = common_config.get("explicit_geometry_clip", {})

            if explicit_clip_cfg.get("enabled", False):
                trans_clip = float(explicit_clip_cfg.get("translation_clip", 100.0))

                fp16_max = float(torch.finfo(torch.float16).max)
                hard_clip = float(explicit_clip_cfg.get("hard_clip", 4096.0))
                hard_clip = min(hard_clip, fp16_max)

                intrinsics_clip = float(explicit_clip_cfg.get("intrinsics_clip", 8.0))

                trans = camera2referego[..., :3, 3]
                camera2referego[..., :3, 3] = trans.clamp(
                    min=-trans_clip,
                    max=trans_clip,
                )

                camera2referego = torch.nan_to_num(
                    camera2referego,
                    nan=0.0,
                    posinf=hard_clip,
                    neginf=-hard_clip,
                )
                camera2referego = camera2referego.clamp(
                    min=-hard_clip,
                    max=hard_clip,
                )

                camera2referego[..., 3, :] = camera2referego.new_tensor(
                    [0.0, 0.0, 0.0, 1.0]
                )

                camera_intrinsics_norm = torch.nan_to_num(
                    camera_intrinsics_norm,
                    nan=0.0,
                    posinf=intrinsics_clip,
                    neginf=-intrinsics_clip,
                )
                camera_intrinsics_norm = camera_intrinsics_norm.clamp(
                    min=-intrinsics_clip,
                    max=intrinsics_clip,
                )

                camera_intrinsics_norm[..., 2, :] = camera_intrinsics_norm.new_tensor(
                    [0.0, 0.0, 1.0]
                )
            if (
                common_config.get("petr_visualize_geometry", False)
                and latents_shape is not None
            ):
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )

                run_this_rank = (
                    rank == 0
                    or common_config.get("petr_visualize_all_ranks", False)
                )

                if run_this_rank:
                    vis_root = common_config.get(
                        "petr_visualize_output_root",
                        "./debug_petr_vis",
                    )
                    os.makedirs(vis_root, exist_ok=True)

                    done_flag = os.path.join(
                        vis_root,
                        f"_petr_vis_done_rank{rank}.txt",
                    )

                    should_visualize = (
                        not common_config.get("petr_visualize_once", True)
                        or not os.path.exists(done_flag)
                    )

                    if should_visualize:
                        from dwm.utils.petr_visualizer import visualize_petr_geometry

                        visualize_petr_geometry(
                            model=model,
                            batch=batch,
                            camera_intrinsics_norm=camera_intrinsics_norm,
                            camera2referego=camera2referego,
                            latents_shape=latents_shape,
                            common_config=common_config,
                            tag="conditions",
                        )

                        if common_config.get("petr_visualize_once", True):
                            with open(done_flag, "w") as f:
                                f.write("done\n")
            if do_classifier_free_guidance:
                camera_intrinsics_norm = torch.cat(
                    [camera_intrinsics_norm, camera_intrinsics_norm], 0
                )
                camera2referego = torch.cat(
                    [camera2referego, camera2referego], 0
                )

            camera_intrinsics_norm = camera_intrinsics_norm.to(device)
            camera2referego = camera2referego.to(device)

        # required by the HoloDrive
        has_depth_input = "camera_intrinsics" in batch and \
            "camera_transforms" in batch
        if has_depth_input:
            camera_intrinsics = batch["camera_intrinsics"].to(device)
            camera_transforms = batch["camera_transforms"].to(device)
            if do_classifier_free_guidance:
                camera_intrinsics = torch.cat(
                    [camera_intrinsics, camera_intrinsics])
                camera_transforms = torch.cat(
                    [camera_transforms, camera_transforms])
        camera_param_token = None
        camera_token_mask = None
        bbox_token_input = None
        bbox_class_input = None
        bbox_mask_input = None

        map_token_input = None

        if common_config.get("camera_token_modeling", False):
            camera_param_token = CrossviewTemporalSD.get_camera_param_token(
                batch,
                common_config,
            ).to(device=device, dtype=dtype)

            if do_classifier_free_guidance:
                b, t, v = camera_param_token.shape[:3]

                uncond_camera_token_mask = torch.zeros(
                    b,
                    t,
                    v,
                    device=device,
                    dtype=dtype,
                )
                cond_camera_token_mask = torch.ones(
                    b,
                    t,
                    v,
                    device=device,
                    dtype=dtype,
                )

                camera_param_token = torch.cat(
                    [camera_param_token, camera_param_token],
                    dim=0,
                )
                camera_token_mask = torch.cat(
                    [uncond_camera_token_mask, cond_camera_token_mask],
                    dim=0,
                )

        if common_config.get("bbox_token_modeling", False):
            bbox_token_input, bbox_class_input, bbox_mask_input = \
                CrossviewTemporalSD.get_bbox_token_inputs(
                    batch,
                    common_config,
                    device,
                    dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

        if common_config.get("map_token_modeling", False):
            map_token_input = CrossviewTemporalSD.get_map_token_input(
                batch,
                common_config,
                device,
                dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        result = {
            "encoder_hidden_states": encoder_hidden_states,
            "condition_image_tensor": condition_image_tensor,

            "disable_crossview": torch.tensor(
                [common_config.get("disable_crossview", False)],
                device=device).repeat(batch_size),

            "disable_temporal": torch.tensor(
                [common_config.get("disable_temporal", False)],
                device=device).repeat(batch_size),

            "crossview_attention_mask": (
                torch.cat([batch["crossview_mask"], batch["crossview_mask"]])
                    if do_classifier_free_guidance else batch["crossview_mask"]
            ).to(device)
            if "crossview_mask" in batch else None,

            "camera_intrinsics": camera_intrinsics if has_depth_input else
            None,
            "camera_transforms": camera_transforms if has_depth_input else
            None,

            "camera_intrinsics_norm": camera_intrinsics_norm
            if common_config.get("explicit_view_modeling", False) else None,

            "camera2referego": camera2referego
            if common_config.get("explicit_view_modeling", False) else None,

            "added_time_ids": added_time_ids
            if "added_time_ids" in common_config else None,

            "camera_param_token": camera_param_token,
            "camera_token_mask": camera_token_mask,

            "bbox_token_input": bbox_token_input,
            "bbox_class_input": bbox_class_input,
            "bbox_mask_input": bbox_mask_input,

            "map_token_input": map_token_input,
        }
        optional_condition_keys = [
            "camera_param_token",
            "camera_token_mask",
            "bbox_token_input",
            "bbox_class_input",
            "bbox_mask_input",
            "map_token_input",
        ]

        for key in optional_condition_keys:
            if result.get(key, None) is None:
                result.pop(key, None)

        if (
            isinstance(model, diffusers.SD3Transformer2DModel) and
            text_encoder is not None
        ):
            result["pooled_projections"] = pooled_text_embeddings

        # for adopting temporal vae
        if latents_shape is not None and latents_shape[1] != sequence_length:
            pre = 1 if sequence_length % 2 == 1 else 0
            stride = (sequence_length - pre)//(latents_shape[1] - pre)
            for k in result:
                if result[k] is not None and result[k].ndim > 1 and result[k].shape[1] == sequence_length:
                    result[k] = torch.cat(
                        [result[k][:, :pre], result[k][:, pre::stride]], dim=1)

        return result

    @staticmethod
    def make_depth_loss(
        batch_size: int, sequence_length: int, view_count: int,
        depth_frustum_range: list, batch: dict, depth_features: torch.Tensor,
        depth_loss_coef: float, point_count_limit_per_view: int,
        point_bundle_size: int, device
    ):
        predicted_depth = depth_features.float().softmax(-3)
        normalized_intrinsics = dwm.functional.make_homogeneous_matrix(
            dwm.functional.normalize_intrinsic_transform(
                batch["image_size"], batch["camera_intrinsics"]))
        camera_from_lidar = torch.linalg.solve(
            batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"],
            batch["ego_transforms"][:, :, :1] @ batch["lidar_transforms"])
        frustum_from_lidar = (
            normalized_intrinsics @ camera_from_lidar).to(device)

        p_list = []
        fd_list = []
        valid_sample_list = []
        for i in range(batch_size):
            for j in range(sequence_length):
                points = dwm.functional.make_homogeneous_vector(
                    batch["lidar_points"][i][j][:, :3].to(device)).t()
                projected_points = frustum_from_lidar[i, j] @ points
                p = (projected_points[:, 0:2] / projected_points[:, 2:3])\
                    .transpose(-2, -1)
                fd = (
                    (projected_points[:, 2] - depth_frustum_range[0]) /
                    depth_frustum_range[2]).round().long()
                mask = torch.logical_and(
                    p.abs().amax(dim=-1) < 1,
                    torch.logical_and(fd >= 0, fd < predicted_depth.shape[-3]))

                for k in range(view_count):
                    view_mask = mask[k]
                    count = torch.sum(view_mask.long())
                    valid_sample_list.append(
                        torch.arange(
                            0, point_count_limit_per_view,
                            device=p.device) < count)
                    if count > 0:
                        k_p = p[k, view_mask]
                        k_fd = fd[k, view_mask]
                        if count > point_count_limit_per_view:
                            k_p = k_p[:point_count_limit_per_view]
                            k_fd = k_fd[:point_count_limit_per_view]
                        elif count < point_count_limit_per_view:
                            k_p = torch.nn.functional.pad(
                                k_p,
                                (0, 0, 0, point_count_limit_per_view - count))
                            k_fd = torch.nn.functional.pad(
                                k_fd, (0, point_count_limit_per_view - count))

                    else:
                        k_p = torch.zeros(
                            (point_count_limit_per_view, 2), dtype=p.dtype,
                            device=p.device)
                        k_fd = torch.zeros(
                            (point_count_limit_per_view), dtype=fd.dtype,
                            device=fd.device)

                    p_list.append(k_p)
                    fd_list.append(k_fd)

        depth_samples = dwm.functional.grid_sample_sequence(
            predicted_depth, torch.stack(p_list),
            bundle_size=point_bundle_size, padding_mode="border")
        ground_truth_depth = torch.nn.functional\
            .one_hot(torch.stack(fd_list), predicted_depth.shape[-3])\
            .transpose(-2, -1).float()
        valid_sample = torch.stack(valid_sample_list)

        depth_cross_entropy = torch.nn.functional.cross_entropy(
            depth_samples.flatten(0, 2), ground_truth_depth,
            reduction="none")
        depth_estimation_loss = \
            (depth_cross_entropy * valid_sample.float()).sum() / \
            valid_sample.sum()
        return depth_loss_coef * depth_estimation_loss

    @staticmethod
    def enum_depth_preds_and_targets(
        batch_size: int, sequence_length: int, view_count: int,
        depth_frustum_range: list, batch: dict, depth_features: torch.Tensor,
        point_count_limit_per_view: int, point_bundle_size: int, device
    ):
        predicted_depth = depth_features.argmax(-3, keepdim=True).float()

        normalized_intrinsics = dwm.functional.make_homogeneous_matrix(
            dwm.functional.normalize_intrinsic_transform(
                batch["image_size"], batch["camera_intrinsics"]))
        camera_from_lidar = torch.linalg.solve(
            batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"],
            batch["ego_transforms"][:, :, :1] @ batch["lidar_transforms"])
        frustum_from_lidar = (
            normalized_intrinsics @ camera_from_lidar).to(device)

        for i in range(batch_size):
            i_p_list = []
            i_fd_list = []
            valid_count_list = []
            for j in range(sequence_length):
                points = dwm.functional.make_homogeneous_vector(
                    batch["lidar_points"][i][j].to(device)).t()
                projected_points = frustum_from_lidar[i, j] @ points
                p = (projected_points[:, 0:2] / projected_points[:, 2:3])\
                    .transpose(-2, -1)
                fd = (projected_points[:, 2] - depth_frustum_range[0]) / \
                    depth_frustum_range[2]
                mask = torch.logical_and(
                    p.abs().amax(dim=-1) < 1,
                    torch.logical_and(
                        fd >= -0.5, fd < depth_features.shape[-3] - 0.5))

                for k in range(view_count):
                    view_mask = mask[k]
                    count = torch.sum(view_mask.long())
                    valid_count_list.append(count)
                    if count > 0:
                        k_p = p[k, view_mask]
                        k_fd = fd[k, view_mask]
                        if count > point_count_limit_per_view:
                            k_p = k_p[:point_count_limit_per_view]
                            k_fd = k_fd[:point_count_limit_per_view]
                        elif count < point_count_limit_per_view:
                            k_p = torch.nn.functional.pad(
                                k_p,
                                (0, 0, 0, point_count_limit_per_view - count))

                    else:
                        k_p = torch.zeros(
                            (point_count_limit_per_view, 2), dtype=p.dtype,
                            device=p.device)
                        k_fd = None

                    i_p_list.append(k_p)
                    i_fd_list.append(k_fd)

            depth_samples = dwm.functional.grid_sample_sequence(
                predicted_depth[i], torch.stack(i_p_list),
                bundle_size=point_bundle_size, padding_mode="border")
            for j in range(sequence_length):
                for k in range(view_count):
                    valid_count = valid_count_list[j * view_count + k]
                    if valid_count > 0:
                        pred_depth = depth_samples[j, k, 0, :valid_count] * \
                            depth_frustum_range[2]
                        target_depth = i_fd_list[j * view_count + k] * \
                            depth_frustum_range[2]
                        yield pred_depth, target_depth

    @staticmethod
    def try_make_input_for_prediction(
        noisy_input: torch.Tensor, latents, timesteps: torch.Tensor,
        training_config: dict, common_config: dict,
        generator: torch.Generator = None,
        reference_latent_count=None
    ):
        # for the reference frame augmentation
        rf_scale = (
            (
                torch.randn(latents.shape[:2], generator=generator) *
                training_config["reference_frame_scale_std"] + 1
            ).view(*latents.shape[:2], 1, 1, 1, 1).to(latents.device)
            if "reference_frame_scale_std" in training_config else 1
        )
        rf_offset = (
            (
                torch.randn(latents.shape[:2], generator=generator) *
                training_config["reference_frame_offset_std"]
            ).view(*latents.shape[:2], 1, 1, 1, 1).to(latents.device)
            if "reference_frame_offset_std" in training_config else 0
        )
        batch_size, sequence_length, view_count = noisy_input.shape[:3]

        frame_prediction_style = common_config.get(
            "frame_prediction_style", None)
        if (
            frame_prediction_style == None or
            frame_prediction_style == "diffusion_forcing"
        ):
            made_timesteps = timesteps
            reference_frame_indicator = torch.zeros(
                *latents.shape[:3], dtype=torch.bool, device=latents.device)
            if frame_prediction_style == "diffusion_forcing":
                # tasks divided by image_generation_ratio:
                # * Y: image generation with temporal module disabled
                # * N: video generation
                disable_temporal = torch.rand(
                    (batch_size, 1, 1), generator=generator) < \
                    training_config.get("image_generation_ratio", 0.0)
                made_noisy_input = torch.where(
                    disable_temporal.view(batch_size, 1, 1, 1, 1, 1)
                    .to(latents.device),
                    noisy_input, noisy_input * rf_scale + rf_offset)
                additional_condition = {
                    "disable_temporal": disable_temporal.to(latents.device)
                }

            else:
                made_noisy_input = noisy_input
                additional_condition = None

        elif frame_prediction_style == "ctsd":
            # tasks divided by generation_task_ratio:
            # * Y: generation tasks divided by image_generation_ratio
            #   * Y: image generation with temporal module disabled
            #   * N: video generation
            # * N: prediction tasks divided by all_reference_visible_ratio
            #   * Y: all reference frame visible
            #   * N: partial reference frame visible (reference_visible_rate)

            generation_task_indicator = \
                torch.rand((batch_size, 1, 1), generator=generator) < \
                training_config.get("generation_task_ratio", 0.0)
            disable_temporal = torch.logical_and(
                torch.rand((batch_size, 1, 1), generator=generator) <
                training_config.get("image_generation_ratio", 0.0),
                generation_task_indicator)

            # the mask of referenced frame for predictive task
            all_reference_visible_indicator = \
                torch.rand((batch_size, 1, 1), generator=generator) < \
                training_config.get("all_reference_visible_ratio", 0.0)
            reference_visible_rate = training_config.get(
                "reference_visible_rate",
                1.0
            )

            if view_count > 1 and reference_visible_rate > 0.0 and \
                    reference_visible_rate < 1.0:
                clean_view_count = int(view_count * reference_visible_rate + 0.5)
                clean_view_count = max(1, min(clean_view_count, view_count - 1))

                clean_view_scores = torch.rand(
                    (batch_size, sequence_length, view_count),
                    generator=generator
                )

                clean_view_indices = torch.topk(
                    clean_view_scores,
                    k=clean_view_count,
                    dim=2,
                    largest=False
                ).indices

                partial_reference_indicator = torch.zeros(
                    (batch_size, sequence_length, view_count),
                    dtype=torch.bool
                )

                partial_reference_indicator.scatter_(
                    2,
                    clean_view_indices,
                    True
                )

            else:
                partial_reference_indicator = \
                    torch.rand(
                        (batch_size, sequence_length, view_count),
                        generator=generator
                    ) < reference_visible_rate

            if isinstance(reference_latent_count, int):
                reference_latent_count_tensor = reference_latent_count * \
                    torch.ones((batch_size, 1, 1), dtype=torch.int32)
            elif isinstance(reference_latent_count, dict):
                count_list = torch.tensor(
                    [int(i) for i in reference_latent_count.keys()],
                    dtype=torch.int32)
                ratio_cumsum_list = torch.tensor(
                    list(itertools.accumulate(reference_latent_count.values())))
                ratio_indices = torch.searchsorted(
                    ratio_cumsum_list, torch.rand(
                        (batch_size, 1, 1), generator=generator))
                reference_latent_count_tensor = count_list[ratio_indices]
            else:
                raise Exception("Un implemented dynamic reference frame count")

            reference_latent_count_indicator = torch\
                .arange(sequence_length, dtype=torch.int32)\
                .unsqueeze(0).unsqueeze(-1)\
                .repeat(batch_size, 1, view_count) < \
                reference_latent_count_tensor
            reference_frame_indicator = torch.logical_and(
                torch.logical_and(
                    torch.logical_not(generation_task_indicator),
                    torch.logical_or(
                        all_reference_visible_indicator,
                        partial_reference_indicator
                    )),
                reference_latent_count_indicator)

            made_noisy_input = torch.where(
                reference_frame_indicator.view(*latents.shape[:3], 1, 1, 1)
                .to(latents.device),
                latents * rf_scale + rf_offset, noisy_input)
            made_timesteps = torch.where(
                reference_frame_indicator.to(timesteps.device),
                torch.zeros_like(timesteps), timesteps)
            additional_condition = {
                "disable_temporal": disable_temporal.to(latents.device)
            }
        else:
            raise Exception("Unknown frame prediction type")

        return made_noisy_input, made_timesteps, additional_condition, \
            reference_frame_indicator

    @staticmethod
    def sd3_encode_prompt_with_t5(
        text_encoder, tokenizer, common_config: dict,
        max_sequence_length: int = 77, prompt=None,
        num_images_per_prompt: int = 1, device=None,
        joint_attention_dim: int = 4096,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if text_encoder is None:
            return torch.zeros(
                (batch_size, max_sequence_length, joint_attention_dim),
                device=device, dtype=torch.float16)

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per
        # prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    @staticmethod
    def sd3_encode_prompt_with_clip(
        text_encoder, tokenizer, common_config: dict, prompt: str, device,
        num_images_per_prompt: int = 1
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True,
            return_tensors="pt")

        text_input_ids = text_inputs.input_ids
        text_embeddings = text_encoder(
            text_input_ids.to(device), output_hidden_states=True)

        pooled_text_embeddings = text_embeddings[0]
        text_embeddings = text_embeddings.hidden_states[-2]
        text_embeddings = text_embeddings.to(
            dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = text_embeddings.shape
        # duplicate text embeddings for each generation per prompt, using mps
        # friendly method
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1)

        return text_embeddings, pooled_text_embeddings

    @staticmethod
    def sd3_compute_density_for_timestep_sampling(
        weighting_scheme: str, size, logit_mean: float = None,
        logit_std: float = None, mode_scale: float = None
    ):
        """Compute the density for sampling the timesteps when doing SD3
        training.

        Courtesy: This was contributed by Rafie Walker in
        https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(
                mean=logit_mean, std=logit_std, size=size,
                device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=size, device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=size, device="cpu")

        return u

    @staticmethod
    def sd3_get_sigmas(
        noise_scheduler, timestep_indices, n_dim, device, dtype
    ):
        sigmas = noise_scheduler.sigmas[timestep_indices]\
            .to(device=device, dtype=dtype)
        while len(sigmas.shape) < n_dim:
            sigmas = sigmas.unsqueeze(-1)

        return sigmas

    def __init__(
        self, output_path, config: dict, device, common_config: dict,
        training_config: dict, inference_config: dict,
        pretrained_model_name_or_path: str, model, model_dtype=None,
        model_checkpoint_path=None, model_load_state_args: dict = {},
        metrics: dict = {}, resume_from=None
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.config = config
        self.device = device
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config

        self.generator = torch.Generator()
        if "generator_seed" in self.config:
            self.generator.manual_seed(self.config["generator_seed"])
        else:
            self.generator.seed()

        # load the diffusion model
        self.model_dtype = model_dtype or torch.float32
        self.model_wrapper = self.model = model.to(dtype=self.model_dtype)
        self.model.enable_gradient_checkpointing()
        distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if (
            not torch.distributed.is_initialized() or
            distribution_framework == "ddp"
        ):
            self.model.to(self.device)
        elif (
            distribution_framework == "fsdp" and
            "fsdp_ignored_module_pattern" in common_config
        ):
            pattern = re.compile(common_config["fsdp_ignored_module_pattern"])
            for name, module in self.model.named_modules():
                if pattern.match(name) is not None:
                    module.to(self.device)

        # tokenizer & text encoder
        text_encoder_load_args = self.common_config.get(
            "text_encoder_load_args", {})
        if isinstance(self.model, diffusers.UNetSpatioTemporalConditionModel):
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = transformers.CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder",
                **text_encoder_load_args)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.to(self.device)
        elif isinstance(self.model, diffusers.SD3Transformer2DModel):
            tokenizer = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer")
            tokenizer_2 = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer_2")
            tokenizer_3 = transformers.T5TokenizerFast.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer_3")
            self.tokenizers = [tokenizer, tokenizer_2, tokenizer_3]

            text_encoder = transformers.CLIPTextModelWithProjection\
                .from_pretrained(
                    pretrained_model_name_or_path, subfolder="text_encoder",
                    **text_encoder_load_args)
            if text_encoder.device.type != device.type:
                text_encoder.to(device)

            text_encoder.requires_grad_(False)
            text_encoder.eval()

            text_encoder_2 = transformers.CLIPTextModelWithProjection\
                .from_pretrained(
                    pretrained_model_name_or_path, subfolder="text_encoder_2",
                    **text_encoder_load_args)
            if text_encoder_2.device.type != device.type:
                text_encoder_2.to(device)

            text_encoder_2.requires_grad_(False)
            text_encoder_2.eval()

            # FSDP is required for the T5 encoder
            text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder_3",
                **text_encoder_load_args)
            if (
                torch.distributed.is_initialized() and
                distribution_framework == "fsdp"
            ):
                text_encoder_3.to(dtype=torch.float16)

            text_encoder_3.requires_grad_(False)
            text_encoder_3.eval()
            if (
                torch.distributed.is_initialized() and
                distribution_framework == "fsdp" and
                "t5_fsdp_wrapper_settings" in common_config
            ):
                text_encoder_3 = FSDP(
                    text_encoder_3, device_id=torch.cuda.current_device(),
                    **self.common_config["t5_fsdp_wrapper_settings"])
            elif text_encoder_3.device.type != device.type:
                text_encoder_3.to(device)

            self.text_encoders = [text_encoder, text_encoder_2, text_encoder_3]
        else:
            raise Exception("Unsupported diffusion model type.")

        # vae & image processor
        vae_type = dwm.common.get_class(
            self.common_config.get("vae", "diffusers.AutoencoderKL"))
        vae_pretrained_model_name_or_path = self.common_config.get(
            "vae_pretrained_model_name_or_path", pretrained_model_name_or_path)
        self.vae = vae_type.from_pretrained(
            vae_pretrained_model_name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.to(self.device)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1))
        self.is_temporal_vae = isinstance(
            self.vae, diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX)

        # scheduler
        if isinstance(self.model, diffusers.UNetSpatioTemporalConditionModel):
            train_scheduler_type = dwm.common.get_class(
                self.training_config.get("scheduler", "diffusers.DDPMScheduler"))
            self.train_scheduler = train_scheduler_type.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")
            test_scheduler_type = dwm.common.get_class(
                self.inference_config.get(
                    "scheduler", "diffusers.DDIMScheduler"))
            self.test_scheduler = test_scheduler_type.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")
        elif isinstance(self.model, diffusers.SD3Transformer2DModel):
            self.train_scheduler =\
                diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                    pretrained_model_name_or_path, subfolder="scheduler")
            test_scheduler_type = dwm.common.get_class(
                self.inference_config.get(
                    "scheduler", "diffusers.FlowMatchEulerDiscreteScheduler"))
            self.test_scheduler = test_scheduler_type.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")

        # load_state
        if resume_from is not None:
            state_dict = CrossviewTemporalSD.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.model.load_state_dict(state_dict)
        elif model_checkpoint_path is not None:
            state_dict = CrossviewTemporalSD.load_state(model_checkpoint_path)

            # Our CTSD 2.1 model definition is based on SVD, when initializing
            # from SD 2.1 weights, some keys need to be renamed.
            if isinstance(
                self.model, diffusers.UNetSpatioTemporalConditionModel
            ):
                state_dict = dwm.models.crossview_temporal_unet\
                    .UNetCrossviewTemporalConditionModel\
                    .try_to_convert_state_dict(state_dict)

            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, **model_load_state_args)
                        # zero-init cross-view modules that are newly added and missing in ckpt
            zeroed_crossview_keys = []

            if self.common_config.get("zero_missing_crossview_weights", False):
                target_model = self.model.module if hasattr(self.model, "module") else self.model

                param_dict = dict(target_model.named_parameters())
                buffer_dict = dict(target_model.named_buffers())

                crossview_markers = (
                    "view_pos_embeds.",
                    "crossview_transformer_blocks.",
                )

                if self.common_config.get("zero_missing_crossview_include_mixers", False):
                    crossview_markers = crossview_markers + ("view_mixers.",)

                with torch.no_grad():
                    for key in missing_keys:
                        if not any(marker in key for marker in crossview_markers):
                            continue

                        tensor = param_dict.get(key, None)

                        if tensor is None:
                            tensor = buffer_dict.get(key, None)

                        if tensor is None:
                            continue

                        if not torch.is_floating_point(tensor):
                            continue

                        tensor.zero_()
                        zeroed_crossview_keys.append(key)
            reset_crossview_mixer_factor = self.common_config.get(
                "reset_crossview_mixer_factor", None
            )
            if reset_crossview_mixer_factor is not None:
                target_model = self.model.module if hasattr(self.model, "module") else self.model
                reset_value = float(reset_crossview_mixer_factor)

                with torch.no_grad():
                    for mixer in target_model.view_mixers:
                        if hasattr(mixer, "mix_factor"):
                            mixer.mix_factor.fill_(reset_value)

                if self.should_save:
                    crossview_weight = 1.0 - float(
                        torch.sigmoid(torch.tensor(reset_value)).item()
                    )
                    print(
                        "[RESET_CROSSVIEW_MIXER] "
                        "mix_factor={:.4f}, crossview_weight={:.4f}".format(
                            reset_value, crossview_weight
                        )
                    )

            if (
                self.should_save and
                self.common_config.get("print_load_state_info", False)
            ):
                print(f"missing keys: {missing_keys}")
                print(f"unexpected keys: {unexpected_keys}")

        if "freezing_pattern" in training_config:
            pattern = re.compile(training_config["freezing_pattern"])
            frozen_module_count = 0
            for name, module in self.model.named_modules():
                if pattern.match(name) is not None:
                    module.requires_grad_(False)
                    frozen_module_count += 1
                    if self.should_save:
                        print("{} is frozen.".format(name))

            if self.should_save:
                print("{} modules are frozen.".format(frozen_module_count))

        if self.should_save:
            param_count = sum([
                i.numel() for i in self.model.parameters() if i.requires_grad
            ])
            print(
                "{:.1f} M parameters are trainable.".format(param_count / 1e6))

        # setup training parts
        self.loss_report_list = []
        self.step_duration = 0

        self.distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if self.training_config.get("enable_grad_scaler", False):
            if (
                not torch.distributed.is_initialized() or
                self.distribution_framework == "ddp"
            ):
                self.grad_scaler = torch.amp.GradScaler()
            elif self.distribution_framework == "fsdp":
                self.grad_scaler = torch.distributed.fsdp.sharded_grad_scaler\
                    .ShardedGradScaler()

        if torch.distributed.is_initialized():
            if self.distribution_framework == "ddp":
                self.model_wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    **self.common_config["ddp_wrapper_settings"])
            elif self.distribution_framework == "fsdp":
                if "fsdp_ignored_module_pattern" in common_config:
                    pattern = re.compile(
                        common_config["fsdp_ignored_module_pattern"])
                    ignored_named_modules = [
                        (name, module)
                        for name, module in self.model.named_modules()
                        if pattern.match(name) is not None
                    ]
                    ignored_modules = [i[1] for i in ignored_named_modules]
                    if self.should_save:
                        print(
                            "{} modules are ignored by FSDP."
                            .format(len(ignored_named_modules)))
                        print(
                            "These ignored modules are {}."
                            .format([i[0] for i in ignored_named_modules]))
                else:
                    ignored_modules = None

                self.model_wrapper = FSDP(
                    self.model, device_id=torch.cuda.current_device(),
                    ignored_modules=ignored_modules,
                    **self.common_config["ddp_wrapper_settings"])
            else:
                raise Exception(
                    "Unknown data parallel framework {}."
                    .format(self.distribution_framework))

        if self.should_save and output_path is not None:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=self.model_wrapper.parameters()
        ) if "optimizer" in config else None

        if resume_from is not None:
            dwm.distributed.distributed_load_optimizer_state(
                self.model_wrapper, self.optimizer,
                os.path.join(output_path, "optimizer"), str(resume_from))

        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer) \
            if "lr_scheduler" in config else None

        self.metrics = metrics
        for i in self.metrics.values():
            i.to(self.device)

    def get_loss_coef(self, name):
        loss_coef = 1
        if "loss_coef_dict" in self.training_config:
            loss_coef = self.training_config["loss_coef_dict"].get(name, 1.0)

        return loss_coef

    def get_latent_sequence_length(self, sequence_length):
        pre = self.inference_config.get("vae_pre", 0)
        stride = self.inference_config.get("vae_stride", 1)
        assert sequence_length % stride == pre or sequence_length == 0, \
            f"{sequence_length} vs {pre} vs {stride}"
        return (sequence_length-pre)//stride + (1 if pre > 0 else 0)

    def get_reference_latent_count(self):
        reference_frame_count = self.training_config.get(
            "reference_frame_count", 0)
        if isinstance(reference_frame_count, int):
            reference_latent_count = self.get_latent_sequence_length(
                reference_frame_count)
        elif isinstance(reference_frame_count, dict):
            reference_latent_count = {
                str(self.get_latent_sequence_length(int(k))): v
                for k, v in reference_frame_count.items()}
        else:
            raise Exception("Un implemented dynamic reference frame count")
        return reference_latent_count

    def save_checkpoint(self, output_path: str, steps: int):
        if torch.distributed.is_initialized():
            # model is fully saved by rank0 for compatibility
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            model_state_dict = torch.distributed.checkpoint.state_dict\
                .get_model_state_dict(self.model_wrapper, options=options)

        elif self.should_save:
            model_state_dict = self.model.state_dict()

        os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)
        if self.should_save:
            torch.save(
                model_state_dict,
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(steps)))

        dwm.distributed.distributed_save_optimizer_state(
            self.model_wrapper, self.optimizer,
            os.path.join(output_path, "optimizer"), str(steps))

    def log(self, global_step: int, log_steps: int):
        if self.should_save:
            if len(self.loss_report_list) > 0 and \
                    isinstance(self.loss_report_list[0], dict):
                loss_values = {
                    i: sum([j[i] for j in self.loss_report_list]) /
                    len(self.loss_report_list)
                    for i in self.loss_report_list[0].keys()
                }
                loss_message = ", ".join([
                    "{}: {:.4f}".format(k, v) for k, v in loss_values.items()
                ])
                print(
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step, self.step_duration / log_steps,
                        loss_message))
                for key, value in loss_values.items():
                    self.summary.add_scalar(
                        "train/{}".format(key), value, global_step)

            else:
                loss_value = sum(self.loss_report_list) / \
                    len(self.loss_report_list)
                print(
                    "Step {} ({:.1f} s/step), loss: {:.4f}".format(
                        global_step, self.step_duration / log_steps,
                        loss_value))
                self.summary.add_scalar("train/Loss", loss_value, global_step)

        self.loss_report_list.clear()
        self.step_duration = 0

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    def train_step(self, batch: dict, global_step: int):
        self.model_wrapper.train()

        t0 = time.time()

        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(self.device))
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if "dataset_tag" in batch:
            tag_value = int(batch["dataset_tag"].detach().cpu().flatten()[0].item())
        else:
            tag_value = -1

        if tag_value == 0 and global_step < 5 and rank == 0:


            save_root = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nuplan_black"
            os.makedirs(save_root, exist_ok=True)

            for t in [0, 1, 2, 5, 10, 19]:
                if t >= batch["vae_images"].shape[1]:
                    continue

                rgb_grid = batch["vae_images"][0, t].detach().cpu().float()
                torchvision.utils.save_image(
                    rgb_grid,
                    os.path.join(save_root, "step{:04d}_t{:02d}_rgb.png".format(
                        global_step + 1, t
                    )),
                    nrow=8
                )

                if "3dbox_images" in batch:
                    box_grid = batch["3dbox_images"][0, t].detach().cpu().float()
                    torchvision.utils.save_image(
                        box_grid,
                        os.path.join(save_root, "step{:04d}_t{:02d}_box.png".format(
                            global_step + 1, t
                        )),
                        nrow=8
                    )

                if "hdmap_images" in batch:
                    hdmap_grid = batch["hdmap_images"][0, t].detach().cpu().float()
                    torchvision.utils.save_image(
                        hdmap_grid,
                        os.path.join(save_root, "step{:04d}_t{:02d}_hdmap.png".format(
                            global_step + 1, t
                        )),
                        nrow=8
                    )

        # prepare the training target
        # use different shape for 3D and 2D vae
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor, "(b t v) c h w -> (b v) c t h w",
                t=sequence_length, v=view_count)

        shift_factor = self.vae.config.shift_factor\
            if self.vae.config.shift_factor is not None else 0
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))

        if self.is_temporal_vae:
            latents = einops.rearrange(
                latents, "(b v) c t h w -> b t v c h w", v=view_count)
        else:
            latents = einops.rearrange(
                latents, "(b t v) c h w -> b t v c h w",
                t=sequence_length, v=view_count)

        # latents = latents.unflatten(0, batch["vae_images"].shape[:3])
        noise = torch.randn(
            latents.shape, generator=self.generator).to(self.device)

        if (
            "frame_prediction_style" in self.common_config and
            self.common_config["frame_prediction_style"] == "diffusion_forcing"
        ):
            timestep_shape_range = 2
        else:
            timestep_shape_range = 1

        if isinstance(self.model, diffusers.UNetSpatioTemporalConditionModel):
            timesteps = torch.randint(
                0, self.train_scheduler.config.num_train_timesteps,
                latents.shape[:timestep_shape_range], generator=self.generator
            ).to(self.device)
            noisy_latents = self.train_scheduler.add_noise(
                latents, noise, timesteps)
            if self.train_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.train_scheduler.config.prediction_type == "v_prediction":
                target = self.train_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise Exception("Unknown training target of the UNet.")

        elif isinstance(self.model, diffusers.SD3Transformer2DModel):
            u = CrossviewTemporalSD.sd3_compute_density_for_timestep_sampling(
                weighting_scheme=self.training_config
                .get("weighting_scheme", "logit_normal"),
                size=latents.shape[:timestep_shape_range],
                logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
            timestep_indices = (
                u * self.train_scheduler.config.num_train_timesteps
            ).long()
            timesteps = self.train_scheduler.timesteps[timestep_indices]\
                .to(self.device)

            # Add noise according to flow matching.
            sigmas = CrossviewTemporalSD.sd3_get_sigmas(
                self.train_scheduler, timestep_indices, n_dim=latents.ndim,
                dtype=latents.dtype, device=latents.device)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            target = latents

        # make sure the timesteps in the shape of (b, t, v)
        while len(timesteps.shape) < 3:
            timesteps = timesteps.unsqueeze(-1)\
                .repeat_interleave(latents.shape[len(timesteps.shape)], -1)

        # prepare conditions
        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("text_prompt_condition_ratio", 1.0))\
            .tolist()
        _3dbox_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("3dbox_condition_ratio", 1.0))\
            .to(self.device)
        hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("hdmap_condition_ratio", 1.0))\
            .to(self.device)
        action_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("action_condition_ratio", 1.0))
        if self.common_config.get("explicit_view_modeling", False):
            explicit_view_modeling_mask = (
                torch.rand((batch_size,), generator=self.generator) <
                self.training_config.get("explicit_view_modeling_ratio", 1.0))\
                .to(self.device)
        else:
            explicit_view_modeling_mask = None

        loss_dict = {}
        with self.get_autocast_context():
            model_conditions = CrossviewTemporalSD.get_conditions(
                self.model,
                self.text_encoders
                if isinstance(self.model, diffusers.SD3Transformer2DModel)
                else self.text_encoder,
                self.tokenizers
                if isinstance(self.model, diffusers.SD3Transformer2DModel)
                else self.tokenizer,
                self.common_config, batch["vae_images"].shape, batch,
                self.device, self.model_dtype, text_condition_mask,
                _3dbox_condition_mask, hdmap_condition_mask,
                action_condition_mask, explicit_view_modeling_mask,
                latents_shape=latents.shape)

            reference_latent_count = self.get_reference_latent_count()
            if self.training_config.get("extra_reference_infer", 0) > 0:
                # NOTE: infer temporal vae with only reference frames
                # This setting is more aligned with inference stage
                extra_image_tensor = image_tensor[
                    :, :, :self.training_config.get("extra_reference_infer", 0)]
                extra_latents = dwm.functional.memory_efficient_split_call(
                    self.vae, extra_image_tensor,
                    lambda block, tensor: (
                        block.encode(tensor).latent_dist.sample() -
                        shift_factor
                    ) * block.config.scaling_factor,
                    self.common_config.get("memory_efficient_batch", -1))
                if self.is_temporal_vae:
                    extra_latents = einops.rearrange(
                        extra_latents, "(b v) c t h w -> b t v c h w", v=view_count)
                else:
                    raise Exception("Not support now.")
                latents_for_prediction = torch.cat(
                    [extra_latents, latents[:, extra_latents.shape[1]:]], dim=1)
            else:
                latents_for_prediction = latents

            noisy_latents, timesteps, additional_conditions, \
                reference_frame_indicator = \
                CrossviewTemporalSD.try_make_input_for_prediction(
                    noisy_latents, latents_for_prediction, timesteps,
                    self.training_config, self.common_config,
                    generator=self.generator,
                    reference_latent_count=reference_latent_count)
            # ===== add debug here =====
            if global_step < 5:
                rank = torch.distributed.get_rank() \
                    if torch.distributed.is_initialized() else 0

                if rank == 0:
                    print(
                        "[REF_MASK_DEBUG] step={} ref_ratio={:.4f} noisy_ratio={:.4f}".format(
                            global_step + 1,
                            reference_frame_indicator.float().mean().item(),
                            (~reference_frame_indicator).float().mean().item(),
                        ),
                        flush=True,
                    )
            # ===== add debug end =====    
            if additional_conditions is not None:
                model_conditions.update(additional_conditions)
            if getattr(self.model_wrapper, "mask_module", None) is not None:
                model_conditions["noise"] = noise

            # forward and calculate the loss
            sd_pred, _, _ = self.model_wrapper(
                noisy_latents, timesteps, **model_conditions)

            sd_pred_latent = sd_pred[0] * (-sigmas) + noisy_latents \
                if isinstance(self.model, diffusers.SD3Transformer2DModel) \
                else sd_pred[0]
            if global_step < 5:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                if rank == 0:
                    if "camera_order_perm" in batch:
                        print("[CAM_PERM_DEBUG]", batch["camera_order_perm"][0].tolist())

                    if "camera_names" in batch:
                        print("[CAM_NAMES_DEBUG]", batch["camera_names"])

                    if "crossview_mask" in batch:
                        print("[MASK_DEBUG]", batch["crossview_mask"][0].int().tolist())
            # nan_reason = None

            # if not torch.isfinite(latents).all():
            #     nan_reason = "latents"

            # elif not torch.isfinite(noise).all():
            #     nan_reason = "noise"

            # elif not torch.isfinite(noisy_latents).all():
            #     nan_reason = "noisy_latents"

            # elif isinstance(self.model, diffusers.SD3Transformer2DModel) and not torch.isfinite(sigmas).all():
            #     nan_reason = "sigmas"

            # elif not torch.isfinite(sd_pred[0]).all():
            #     nan_reason = "sd_pred0"

            # elif not torch.isfinite(sd_pred_latent).all():
            #     nan_reason = "sd_pred_latent"

            # elif not torch.isfinite(target).all():
            #     nan_reason = "target"

            # if nan_reason is not None:
            #     save_nan_batch_visualization(
            #         batch=batch,
            #         model_conditions=model_conditions,
            #         sd_pred=sd_pred,
            #         sd_pred_latent=sd_pred_latent,
            #         target=target,
            #         noise=noise,
            #         latents=latents,
            #         sigmas=sigmas if isinstance(self.model, diffusers.SD3Transformer2DModel) else None,
            #         timesteps=timesteps,
            #         reference_frame_indicator=reference_frame_indicator,
            #         global_step=global_step,
            #         reason=nan_reason,
            #     )

            #     raise RuntimeError(
            #         "[NAN_DETECTED_BEFORE_LOSS] rank={} step={} reason={}".format(
            #             torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            #             global_step + 1,
            #             nan_reason,
            #         )
            #     )
            if isinstance(self.model, diffusers.SD3Transformer2DModel):
                fm_target = noise - latents
                fm_mse = (sd_pred[0].float() - fm_target.float()).square()

                x0_mse = (sd_pred_latent.float() - target.float()).square()

                if self.training_config.get("disable_reference_frame_loss", False):
                    non_ref_mask = ~reference_frame_indicator.view(
                        *sd_pred_latent.shape[:3], 1, 1, 1
                    ).to(sd_pred_latent.device)

                    non_ref_mask = non_ref_mask.expand_as(fm_mse)

                    if non_ref_mask.any():
                        fm_loss_debug = fm_mse[non_ref_mask].mean()
                        x0_loss_debug = x0_mse[non_ref_mask].mean()
                    else:
                        fm_loss_debug = fm_mse.mean()
                        x0_loss_debug = x0_mse.mean()
                else:
                    fm_loss_debug = fm_mse.mean()
                    x0_loss_debug = x0_mse.mean()
                
                # if global_step < 300:
                #     rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                #     sigma_flat = sigmas.detach().float().flatten()

                #     print(
                #         "[rank={}] [step={}] sigma_min={:.6f}, sigma_mean={:.6f}, sigma_max={:.6f}, "
                #         "x0_loss={:.6f}, fm_loss={:.6f}, ref_ratio={:.4f}".format(
                #             rank,
                #             global_step + 1,
                #             sigma_flat.min().item(),
                #             sigma_flat.mean().item(),
                #             sigma_flat.max().item(),
                #             x0_loss_debug.item(),
                #             fm_loss_debug.item(),
                #             reference_frame_indicator.float().mean().item(),
                #         ),
                #         flush=True,
                #     )
                # if global_step < 300:
                #     rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

                #     dataset_tag = batch.get("dataset_tag", None)
                #     if dataset_tag is not None:
                #         dataset_tag_str = dataset_tag.detach().cpu().flatten().tolist()
                #     else:
                #         dataset_tag_str = None

                #     print(
                #         "[COND_DEBUG] rank={} step={} dataset_tag={} text={} box={} hdmap={} action={} explicit={}".format(
                #             rank,
                #             global_step + 1,
                #             dataset_tag_str,
                #             text_condition_mask,
                #             _3dbox_condition_mask.detach().cpu().tolist(),
                #             hdmap_condition_mask.detach().cpu().tolist(),
                #             action_condition_mask.detach().cpu().tolist(),
                #             None if explicit_view_modeling_mask is None
                #             else explicit_view_modeling_mask.detach().cpu().tolist(),
                #         ),
                #         flush=True,
                #     )

                #     if "added_time_ids" in model_conditions and model_conditions["added_time_ids"] is not None:
                #         a = model_conditions["added_time_ids"].detach().float()
                #         print(
                #             "[ACTION_DEBUG] rank={} step={} added_min={:.3f} added_mean={:.3f} added_max={:.3f} neg1000_count={}".format(
                #                 rank,
                #                 global_step + 1,
                #                 a.min().item(),
                #                 a.mean().item(),
                #                 a.max().item(),
                #                 (a <= -999).sum().item(),
                #             ),
                #             flush=True,
                #         )

                #     if model_conditions.get("condition_image_tensor", None) is not None:
                #         ci = model_conditions["condition_image_tensor"].detach().float()
                #         print(
                #             "[COND_IMG_DEBUG] rank={} step={} cond_img_shape={} min={:.4f} mean={:.4f} max={:.4f}".format(
                #                 rank,
                #                 global_step + 1,
                #                 tuple(ci.shape),
                #                 ci.min().item(),
                #                 ci.mean().item(),
                #                 ci.max().item(),
                #             ),
                #             flush=True,
                #         )
                        
            sd_mse = (
                sd_pred_latent.float() - target.float()
            ).square()

            if self.training_config.get("disable_reference_frame_loss", False):
                pixel_weight = ~reference_frame_indicator.view(
                    *sd_pred_latent.shape[:3],
                    1,
                    1,
                    1
                ).to(sd_pred_latent.device)

                pixel_weight = pixel_weight.to(dtype=torch.float32)
            else:
                pixel_weight = torch.ones(
                    (*sd_pred_latent.shape[:3], 1, sd_pred_latent.shape[-2], sd_pred_latent.shape[-1]),
                    device=sd_pred_latent.device,
                    dtype=torch.float32,
                )

            box_latent_mask = None

            if "3dbox_images" in batch:
                box_mask = batch["3dbox_images"].to(
                    device=sd_pred_latent.device,
                    dtype=torch.float32,
                )

                if bool((box_mask.detach().amax() > 1.0).item()):
                    box_mask = box_mask / 255.0

                # [B, T, V, C, H, W] -> [B, T, V, 1, H, W]
                box_mask = box_mask.amax(dim=3, keepdim=True)

                if _3dbox_condition_mask is not None:
                    box_mask = box_mask * _3dbox_condition_mask.to(
                        device=sd_pred_latent.device,
                        dtype=torch.float32,
                    ).view(batch_size, 1, 1, 1, 1, 1)

                dense_t = int(box_mask.shape[1])
                target_t = int(sd_pred_latent.shape[1])

                if dense_t == target_t:
                    temporal_indices = torch.arange(
                        target_t,
                        device=sd_pred_latent.device,
                        dtype=torch.long,
                    )
                else:
                    pre = 1 if dense_t % 2 == 1 else 0
                    stride = (dense_t - pre) // max(target_t - pre, 1)

                    if pre > 0:
                        temporal_indices = torch.cat(
                            [
                                torch.zeros(
                                    1,
                                    device=sd_pred_latent.device,
                                    dtype=torch.long,
                                ),
                                pre + torch.arange(
                                    target_t - pre,
                                    device=sd_pred_latent.device,
                                    dtype=torch.long,
                                ) * stride,
                            ],
                            dim=0,
                        )
                    else:
                        temporal_indices = torch.arange(
                            target_t,
                            device=sd_pred_latent.device,
                            dtype=torch.long,
                        ) * stride

                    temporal_indices = temporal_indices.clamp(max=dense_t - 1)

                box_mask = box_mask.index_select(1, temporal_indices)

                # 3D box 是细线，阈值不能太高。
                box_mask = (box_mask > 0.005).float()

                box_mask_flat = einops.rearrange(
                    box_mask,
                    "b t v c h w -> (b t v) c h w",
                )

                # 先在原图尺度膨胀，再下采样到 latent 尺度。
                box_mask_flat = torch.nn.functional.max_pool2d(
                    box_mask_flat,
                    kernel_size=63,
                    stride=1,
                    padding=31,
                )

                box_mask_flat = torch.nn.functional.interpolate(
                    box_mask_flat,
                    size=sd_pred_latent.shape[-2:],
                    mode="nearest",
                )

                box_latent_mask = einops.rearrange(
                    box_mask_flat,
                    "(b t v) c h w -> b t v c h w",
                    b=batch_size,
                    t=target_t,
                    v=view_count,
                ).to(
                    device=sd_pred_latent.device,
                    dtype=torch.float32,
                )

                box_region_weight = 8.0
                pixel_weight = pixel_weight * (1.0 + box_region_weight * box_latent_mask)

            sd_loss = (
                sd_mse * pixel_weight
            ).sum() / (
                pixel_weight.sum() * sd_pred_latent.shape[3]
            ).clamp_min(1.0)

            loss_dict["sd_loss"] = sd_loss * self.get_loss_coef("sd")
            # if not torch.isfinite(loss_dict["sd_loss"]).all():
            #     save_nan_batch_visualization(
            #         batch=batch,
            #         model_conditions=model_conditions,
            #         sd_pred=sd_pred,
            #         sd_pred_latent=sd_pred_latent,
            #         target=target,
            #         noise=noise,
            #         latents=latents,
            #         sigmas=sigmas if isinstance(self.model, diffusers.SD3Transformer2DModel) else None,
            #         timesteps=timesteps,
            #         reference_frame_indicator=reference_frame_indicator,
            #         global_step=global_step,
            #         reason="sd_loss",
            #     )

            #     raise RuntimeError(
            #         "[NAN_DETECTED_IN_LOSS] rank={} step={} loss={}".format(
            #             torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            #             global_step + 1,
            #             loss_dict["sd_loss"].item(),
            #         )
            #     )
                
        if len(sd_pred) > 1:
            depth_features = sd_pred[1]
            loss_dict["depth_loss"] = \
                CrossviewTemporalSD.make_depth_loss(
                    batch_size, sequence_length, view_count,
                    self.model.depth_frustum_range, batch, depth_features,
                    self.get_loss_coef("d"),
                    self.common_config["point_count_limit_per_view"],
                    self.common_config["point_bundle_size"], self.device)

        loss = torch.stack([i[1] for i in loss_dict.items()]).sum()
        loss_report = {"loss": loss.item()}

        # debug code
        report_detail = self.common_config.get("report_detail", False)
        if hasattr(self.model, "hidden_states_var") and report_detail:
            loss_report["h_var"] = self.model.hidden_states_var

        if hasattr(self.model, "encoder_hidden_states_var") and report_detail:
            loss_report["eh_var"] = self.model.encoder_hidden_states_var

        if hasattr(self.model, "temporal_embedding_var") and report_detail:
            loss_report["se_var"] = self.model.temporal_embedding_var

        if len(loss_dict.items()) > 1:
            for i in loss_dict.items():
                loss_report[i[0]] = i[1].item()

        self.loss_report_list.append(loss_report)
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        should_optimize = \
            "gradient_accumulation_steps" not in self.training_config or \
            (global_step + 1) % \
            self.training_config["gradient_accumulation_steps"] == 0
        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.unscale_(self.optimizer)

                if (
                    torch.distributed.is_initialized() and
                    self.distribution_framework == "fsdp"
                ):
                    self.model_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_norm_for_grad_clip"])

            if self.training_config.get("enable_grad_scaler", False):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.step_duration += time.time() - t0

    def inference_pipeline(
        self, latent_shape, batch, output_type, image_latents=None,
        reference_frame_count: int = 0, start_timestep: int = 0,
        stop_timestep=None, take_time: int = 0, frozen_ref_count: int = 0,
    ):
        self.model_wrapper.eval()

        diffusion_forcing_mode = (
            "frame_prediction_style" in self.common_config and
            self.common_config["frame_prediction_style"] == "diffusion_forcing"
        )
        if diffusion_forcing_mode:
            clear_reference_frame_count = self.inference_config.get(
                "clear_reference_frame_count", 0)
            assert self.inference_config["inference_steps"] % \
                (latent_shape[1] - clear_reference_frame_count) == 0
            steps_per_inference = \
                self.inference_config["inference_steps"] // \
                (latent_shape[1] - clear_reference_frame_count)

        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        guidance_scale = self.inference_config.get("guidance_scale", 1)

        preview_depth = self.model.depth_net is not None and \
            "camera_intrinsics" in batch and "camera_transforms" in batch
        depth_result = []

        shift_factor = self.vae.config.shift_factor \
            if self.vae.config.shift_factor is not None else 0
        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"], self.device)
        if diffusion_forcing_mode and image_latents is not None:
            latents = image_latents
        else:
            # full sequence denoising
            latents = torch\
                .randn(latent_shape, generator=self.generator).to(self.device) * \
                getattr(self.test_scheduler, "init_noise_sigma", 1)
        
        ###数值稳定#######
        # if not torch.isfinite(latents).all():
        #     fin = latents[torch.isfinite(latents)]
        #     lo, hi = (fin.min(), fin.max()) if fin.numel() else (-1.0, 1.0)
        #     latents = torch.nan_to_num(latents, nan=0.0, posinf=float(hi), neginf=float(lo)).clamp(lo, hi)
        ###数值稳定#######
        
        model_conditions = CrossviewTemporalSD.get_conditions(
            self.model,
            self.text_encoders
            if isinstance(self.model, diffusers.SD3Transformer2DModel)
            else self.text_encoder,
            self.tokenizers
            if isinstance(self.model, diffusers.SD3Transformer2DModel)
            else self.tokenizer,
            self.common_config, latent_shape, batch, self.device,
            self.model_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            latents_shape=latents.shape)
        result = {}
        stop_timestep = (
            self.inference_config["inference_steps"]
            if stop_timestep is None
            else stop_timestep
        )
        for i in range(start_timestep, stop_timestep):
            # make the denoising timesteps
            if diffusion_forcing_mode:
                L = latent_shape[1]
                S = steps_per_inference
                T = self.inference_config["inference_steps"]

                base = i - take_time * S
                j = torch.arange(L, device=self.device)
                j_eff = torch.clamp(j - clear_reference_frame_count, min=0)
                inner = torch.clamp(base - j_eff * S, min=0)
                idx = torch.minimum(inner, torch.full_like(inner, base))
                max_idx = T - 1
                idx = torch.clamp(idx, 0, max_idx)

                if frozen_ref_count > 0:
                    clean_idx = max_idx
                    idx[:frozen_ref_count] = clean_idx

                timestep_indices = idx.view(1, -1, 1).repeat(
                    latent_shape[0], 1, latent_shape[2]
                )
                timesteps = self.test_scheduler.timesteps[timestep_indices]
            else:
                t = self.test_scheduler.timesteps[i]
                timesteps = t.unsqueeze(-1).unsqueeze(-1)\
                    .repeat(*latent_shape[:3])

            latent_model_input = latents
            if not diffusion_forcing_mode and image_latents is not None:
                # the reference frame injection for the full sequence denoising
                latent_model_input = torch.cat([
                    image_latents[:, :reference_frame_count],
                    latent_model_input[:, reference_frame_count:]
                ], 1)
                timesteps = torch.cat([
                    torch.zeros((
                        timesteps.shape[0], reference_frame_count,
                        timesteps.shape[2]
                    ), dtype=timesteps.dtype, device=self.device),
                    timesteps[:, reference_frame_count:]
                ], 1)

            latent_model_input = latent_model_input.to(dtype=self.model_dtype)
            if hasattr(self.test_scheduler, "scale_model_input"):
                latent_model_input = self.test_scheduler\
                    .scale_model_input(
                        latent_model_input,
                        timesteps if diffusion_forcing_mode else t)\
                    .to(dtype=self.model_dtype)

            if do_classifier_free_guidance:
                latent_model_input = torch.cat(
                    [latent_model_input, latent_model_input])
                timesteps_input = torch.cat([timesteps, timesteps])
            else:
                timesteps_input = timesteps

            ck("latent_model_input", latent_model_input)
            ck("timesteps_input", timesteps_input)


            with self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(
                    latent_model_input, timesteps_input, **model_conditions)
                            

            noise_pred = model_output[0]

            # 数值稳定
            if not torch.isfinite(noise_pred).all():
                def _fix(x, name):
                    if (not torch.is_tensor(x)) or torch.isfinite(x).all(): return x
                    fin = x[torch.isfinite(x)]
                    if fin.numel() == 0: return torch.zeros_like(x)
                    lo, hi, mean = fin.min(), fin.max(), fin.mean()
                    print("SAN:", name, "nan", torch.isnan(x).sum().item(),
                        "+inf", torch.isposinf(x).sum().item(),
                        "-inf", torch.isneginf(x).sum().item(),
                        "lo/hi", float(lo), float(hi))
                    x = torch.nan_to_num(x, nan=float(mean), posinf=float(hi), neginf=float(lo))
                    return x.clamp(lo, hi)

                def _hook(name):
                    def _h(m, inp, out):
                        if torch.is_tensor(out): return _fix(out, name)
                        if isinstance(out, (list, tuple)):
                            out = [_fix(o, f"{name}[{i}]") for i, o in enumerate(out)]
                            return type(out)(out)
                        if isinstance(out, dict):
                            return {k: _fix(v, f"{name}.{k}") for k, v in out.items()}
                    return _h

                handles = [m.register_forward_hook(_hook(n)) for n, m in self.model_wrapper.named_modules()]
                try:
                    with torch.cuda.amp.autocast(enabled=False):
                        model_output, _, _ = self.model_wrapper(latent_model_input.float(), timesteps_input, **model_conditions)
                finally:
                    for h in handles: h.remove()

                noise_pred = model_output[0]
                    
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            if diffusion_forcing_mode:
                if hasattr(self.test_scheduler, "step_by_indices"):
                    staging_latents = self.test_scheduler.step_by_indices(
                        noise_pred, timestep_indices.cpu(), latents
                    ).prev_sample
                else:
                    staging_latents = self.test_scheduler\
                        .step(noise_pred, timesteps.cpu(), latents).prev_sample
                
                # only update the latents in the schedule range so finally the
                # timesteps in the queue become progrssive
                # fix 6
                j = torch.arange(latent_shape[1], device=self.device)

                j_eff = torch.clamp(j - clear_reference_frame_count, min=0)
                raw_unclipped = base - j_eff * steps_per_inference

                in_schedule_1d = (j >= frozen_ref_count) & (raw_unclipped >= 0)
                in_schedule_range = in_schedule_1d.view(1, -1, 1, 1, 1, 1)

                latents = torch.where(in_schedule_range, staging_latents, latents)
            else:
                latents = self.test_scheduler.step(noise_pred, t, latents)\
                    .prev_sample

            # update the depth visualization
            if preview_depth and len(model_output) > 1:
                depth_features = model_output[1].chunk(2)[1] \
                    if do_classifier_free_guidance else model_output[1]

                noisy_image_tensor = dwm.functional.memory_efficient_split_call(
                    self.vae, latents.flatten(0, 2).to(dtype=self.vae.dtype),
                    lambda block, tensor: block.decode(
                        tensor / block.config.scaling_factor + shift_factor,
                        return_dict=False
                    )[0],
                    self.common_config.get("memory_efficient_batch", -1))
                noisy_images = self.image_processor.postprocess(
                    noisy_image_tensor, output_type="pt")

                depth_images = (
                    1 - depth_features.argmax(-3) / depth_features.shape[-3]
                ).flatten(0, 2).unsqueeze(1)
                depth_images = torch.nn.functional.interpolate(
                    depth_images, noisy_images.shape[-2:])
                depth_images = depth_images.unflatten(0, latents.shape[:3])\
                    .repeat_interleave(3, dim=-3).permute(3, 0, 1, 4, 2, 5)
                noisy_images = noisy_images.unflatten(0, latents.shape[:3])\
                    .permute(3, 0, 1, 4, 2, 5)
                depth_result.append(
                    torch.cat([noisy_images, depth_images], -3).flatten(-2)
                    .flatten(-4, -2))

        if diffusion_forcing_mode:
            # NOTE if is_temporal_vae, the tranport of different duration is not straightforward
            # which mean: for time-window_pos, [0-0, 1-1, 2-2] --(out 0-0)--> [1-0, 2-1, 3-2] is required
            # this coming from that temporal vae is not position independent
            cur_latents = latents[:, take_time].flatten(
                0, 1).to(dtype=self.vae.dtype)
            if self.is_temporal_vae:
                view_count = latents.shape[2]
                cur_latents = torch.cat(
                    [cur_latents[:, :, None], cur_latents[:, :, None]*0], dim=2)
            image_tensor = self.vae.decode(
                cur_latents / self.vae.config.scaling_factor + shift_factor,
                return_dict=False)[0]
            if self.is_temporal_vae:
                image_tensor = image_tensor.chunk(2, dim=2)[0]
                image_tensor = einops.rearrange(
                    image_tensor, "(b v) c t h w -> (b t v) c h w", v=view_count)
            
            ####数值稳定###########
            # if not torch.isfinite(latents).all():
            #     fin = latents[torch.isfinite(latents)]
            #     lo, hi = (fin.min(), fin.max()) if fin.numel() else (-1.0, 1.0)
            #     latents = torch.nan_to_num(latents, nan=0.0, posinf=float(hi), neginf=float(lo)).clamp(lo, hi)
            ####数值稳定###########
          
        else:
            if image_latents is not None:
                latents = torch.cat([
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:]
                ], 1)
            if self.is_temporal_vae:
                view_count = latents.shape[2]
                cur_latents = einops.rearrange(
                    latents, "b t v c h w -> (b v) c t h w")
            else:
                cur_latents = latents.flatten(0, 2)
            image_tensor = dwm.functional.memory_efficient_split_call(
                self.vae, cur_latents.to(dtype=self.vae.dtype),
                lambda block, tensor: block.decode(
                    tensor / block.config.scaling_factor + shift_factor,
                    return_dict=False
                )[0],
                self.common_config.get("memory_efficient_batch", -1))
            if self.is_temporal_vae:
                image_tensor = einops.rearrange(
                    image_tensor, "(b v) c t h w -> (b t v) c h w", v=view_count)

        result = {
            "images": self.image_processor.postprocess(
                image_tensor, output_type=output_type),
            "latents": latents
        }
        if preview_depth and len(model_output) > 1:
            result["depth"] = depth_result
            result["depth_features"] = depth_features
        
        return result

    def autoregressive_inference_pipeline(
        self, latent_shape, batch, output_type
    ):
        total_frame_count = batch["vae_images"].shape[1]
        diffusion_forcing_mode = (
            "frame_prediction_style" in self.common_config and
            self.common_config["frame_prediction_style"] == "diffusion_forcing"
        )
        if diffusion_forcing_mode:
            assert total_frame_count > \
                self.inference_config["sequence_length_per_iteration"]
            clear_reference_frame_count = self.inference_config.get(
                "clear_reference_frame_count", 0)
            steps_per_inference = \
                self.inference_config["inference_steps"] // \
                (latent_shape[1] - clear_reference_frame_count) 
            print("clear_reference_frame_count:",clear_reference_frame_count, 
                  "  steps_per_inference:", steps_per_inference)


        reference_frame_count = self.inference_config.get(
            "reference_frame_count", 1)
        if self.inference_config.get("generate_frames_for_reference", True):
            image_latents = None
            dataset_ref_left = 0 
        else:
            raw_image_tensor = \
                batch["vae_images"][:, :reference_frame_count]
            image_tensor = self.image_processor.preprocess(
                raw_image_tensor.flatten(0, 2).to(self.device))
            shift_factor = self.vae.config.shift_factor \
                if self.vae.config.shift_factor is not None else 0
            # use different shape for 3D and 2D vae
            if self.is_temporal_vae:
                image_tensor = einops.rearrange(image_tensor, "(b t v) c h w -> (b v) c t h w",
                                                t=raw_image_tensor.shape[1], v=raw_image_tensor.shape[2])
            image_latents = dwm.functional.memory_efficient_split_call(
                self.vae, image_tensor,
                lambda block, tensor: (
                    block.encode(tensor).latent_dist.mode() - shift_factor
                ) * block.config.scaling_factor,
                self.common_config.get("memory_efficient_batch", -1))
            if self.is_temporal_vae:
                image_latents = einops.rearrange(
                    image_latents, "(b v) c t h w -> b t v c h w", v=raw_image_tensor.shape[2])
            else:
                image_latents = image_latents.unflatten(
                    0, raw_image_tensor.shape[:3])
            # fix 4
            dataset_ref_left = reference_frame_count 

        result = {
            "images": []
        }
        iteration_sequence_length = self.inference_config["sequence_length_per_iteration"]
        exception_for_take_sequence = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [])
        if diffusion_forcing_mode:
            if image_latents is None:
                iteration_batch = {
                    k: (
                        v
                        if k in exception_for_take_sequence
                        else dwm.functional.take_sequence_clip(
                            v, 0, iteration_sequence_length)
                    )
                    for k, v in batch.items()
                }
                stop_timestep = self.inference_config["inference_steps"] - \
                    steps_per_inference
                iteration_output = self.inference_pipeline(
                    latent_shape, iteration_batch, output_type, image_latents,
                    start_timestep=0,
                    stop_timestep=stop_timestep)
                image_latents = iteration_output["latents"]
            else:
                b = image_latents.shape[0]
                ref_len = image_latents.shape[1]
                assert ref_len == reference_frame_count
                assert ref_len <= latent_shape[1]

                if ref_len < latent_shape[1]:
                    tail_noise = torch.randn(
                        (b, latent_shape[1] - ref_len) + latent_shape[2:],
                        generator=self.generator,
                    ).to(self.device) * getattr(
                        self.test_scheduler, "init_noise_sigma", 1
                    )
                    image_latents = torch.cat([image_latents, tail_noise], dim=1)
                    
            
        for i in range(
            0, total_frame_count - iteration_sequence_length + 1,
            iteration_sequence_length - reference_frame_count
        ):
            iteration_batch = {
                k: (
                    v
                    if k in exception_for_take_sequence
                    else dwm.functional.take_sequence_clip(
                        v, i, i + iteration_sequence_length)
                )
                for k, v in batch.items()
            }

            this_ref_frame_count = 0 if image_latents is None \
                else reference_frame_count
            if diffusion_forcing_mode:
                test_fn(latent_shape[1],
                        self.inference_config["inference_steps"] // (latent_shape[1] - clear_reference_frame_count),
                        self.inference_config["inference_steps"],
                        clear_reference_frame_count,
                        self.inference_config["inference_steps"]-steps_per_inference,
                        self.inference_config["inference_steps"],
                        0,
                        dataset_ref_left)
                iteration_output = self.inference_pipeline(
                    latent_shape, iteration_batch, output_type, image_latents,
                    # fix 2 
                    start_timestep=(
                        self.inference_config["inference_steps"] -
                        steps_per_inference
                    ),
                    stop_timestep=(
                        self.inference_config["inference_steps"]
                    ),
                    frozen_ref_count=dataset_ref_left)
                if self.is_temporal_vae and i == 0:
                    result["images"].append(
                        iteration_output["images"].chunk(4)[-1])
                else:
                    result["images"].append(iteration_output["images"])
                # for each step df
                image_latents = iteration_output["latents"]
                
                if (
                    i + iteration_sequence_length - reference_frame_count <
                    total_frame_count - iteration_sequence_length + 1
                ):
                    # dequeue and enqueue
                    image_latents = torch.cat([
                        image_latents[:, 1:],
                        torch.randn(
                            (latent_shape[0], 1) + latent_shape[2:],
                            generator=self.generator).to(self.device) *
                        getattr(self.test_scheduler, "init_noise_sigma", 1)
                    ], 1)
                    if dataset_ref_left > clear_reference_frame_count:      
                        dataset_ref_left -= 1
            else:
                iteration_output = self.inference_pipeline(
                    latent_shape, iteration_batch, output_type, image_latents,
                    reference_frame_count=self.get_latent_sequence_length(
                        this_ref_frame_count))
                result["images"].append(
                    iteration_output["images"]
                    [latent_shape[0] * this_ref_frame_count * latent_shape[2]:])
                if (  
                    i + iteration_sequence_length - reference_frame_count <
                    total_frame_count - iteration_sequence_length + 1
                ):
                    reference_latent_count = self.get_latent_sequence_length(
                        reference_frame_count)
                    image_latents = iteration_output["latents"][
                        :, -reference_latent_count:
                    ]

        if diffusion_forcing_mode:
            # flushing the tailing frames by reusing the last iteration_batch
            flush_steps = latent_shape[1] - clear_reference_frame_count
            for i in range(1, flush_steps):
                frozen_i = min(dataset_ref_left + i, latent_shape[1])
                print('flushing!!')
                test_fn(latent_shape[1],
                        self.inference_config["inference_steps"] // (latent_shape[1] - clear_reference_frame_count),
                        self.inference_config["inference_steps"],
                        clear_reference_frame_count,
                        self.inference_config["inference_steps"] + (2 * i - 1 ) * steps_per_inference,
                        self.inference_config["inference_steps"] + (2 * i ) * steps_per_inference,
                        i,
                        dataset_ref_left)
                
                iteration_output = self.inference_pipeline(
                    latent_shape, iteration_batch, output_type, image_latents,
                    reference_frame_count=latent_shape[1],
                    start_timestep=(
                        self.inference_config["inference_steps"] +
                        (2 * i - 1 ) * steps_per_inference
                    ),
                    stop_timestep=(
                        self.inference_config["inference_steps"] +
                        (2 * i ) * steps_per_inference
                    ),
                    take_time=i,
                    frozen_ref_count=frozen_i)
                result["images"].append(iteration_output["images"])
                is_finished = (torch.arange(latent_shape[1],
                    device=self.device) < frozen_i) \
                    .unflatten(0, (1, -1, 1, 1, 1, 1)) 
                image_latents = torch.where(
                    is_finished, image_latents, iteration_output["latents"])

        if output_type == "pt":
            result["images"] = torch.cat(result["images"])

        return result

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // \
            (2 ** (len(self.vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // \
            (2 ** (len(self.vae.config.down_block_types) - 1))
        if "sequence_length_per_iteration" in self.inference_config:
            latent_shape = (
                batch_size,
                self.get_latent_sequence_length(
                    self.inference_config["sequence_length_per_iteration"]),
                view_count, self.vae.config.latent_channels, latent_height,
                latent_width
            )
            pipeline_output = self.autoregressive_inference_pipeline(
                latent_shape, batch, "pt")
        else:
            latent_shape = (
                batch_size, self.get_latent_sequence_length(
                    sequence_length), view_count,
                self.vae.config.latent_channels, latent_height,
                latent_width
            )
            pipeline_output = self.inference_pipeline(
                latent_shape, batch, "pt")

        dist_on = (
            torch.distributed.is_available() and
            torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if dist_on else 0
        all_rank_preview = bool(
            self.inference_config.get("all_rank_preview", False)
        )
        save_this_rank = self.should_save or (dist_on and all_rank_preview)

        if save_this_rank:
            os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
            filename = (
                "{}_{}".format(global_step, rank)
                if all_rank_preview
                else str(global_step)
            )
            preview_images = pipeline_output["images"]

            eval_frame_export_path = self.inference_config.get(
                "eval_frame_export_path", None
            )
            if eval_frame_export_path is not None:
                if all_rank_preview and dist_on:
                    eval_frame_export_path = os.path.join(
                        eval_frame_export_path,
                        "rank_{:02d}".format(rank),
                    )

                dwm.utils.preview.save_ctsd_eval_frames_for_preview(
                    preview_images,
                    batch,
                    self.inference_config,
                    output_dir=eval_frame_export_path,
                    dataset_name=self.inference_config.get(
                        "eval_frame_dataset_name", "unknown"
                    ),
                    manifest_name=self.inference_config.get(
                        "eval_frame_manifest_name", "stflow_manifest.jsonl"
                    ),
                    image_quality=self.inference_config.get(
                        "eval_frame_image_quality", 95
                    ),
                    export_paired_real=self.inference_config.get(
                        "eval_frame_export_paired_real", True
                    ),
                )

            if preview_images.ndim == 4:
                preview_btvc = preview_images.cpu().unflatten(
                    0, (batch_size, -1, view_count)
                )
            elif preview_images.ndim == 6:
                preview_btvc = preview_images.cpu()
            else:
                raise ValueError(
                    f"unexpected pipeline_output['images'] shape: {tuple(preview_images.shape)}"
                )

            if "sequence_length_per_iteration" in self.inference_config:
                generate_frames_for_reference = self.inference_config.get(
                    "generate_frames_for_reference", True
                )
                reference_frame_count = min(
                    int(self.inference_config.get("reference_frame_count", 1)),
                    batch["vae_images"].shape[1]
                )

                if not generate_frames_for_reference and reference_frame_count > 0:
                    prefix_images = batch["vae_images"][:, :reference_frame_count].cpu()
                    preview_btvc = torch.cat([prefix_images, preview_btvc], dim=1)

            preview_frame_count = preview_btvc.shape[1]
            preview_images = preview_btvc.flatten(0, 2)

            preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
                preview_images, batch, self.inference_config
            )
            if  preview_frame_count == 1:
                image_output_path = os.path.join(
                    output_path, "preview", "{}.png".format(filename))
                torchvision.transforms.functional.to_pil_image(preview_tensor)\
                    .save(image_output_path)
            else:
                video_output_path = os.path.join(
                    output_path, "preview", "{}.mp4".format(filename))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", batch["fps"][0].item(),
                    preview_tensor)

            preview_depth = self.model.depth_net is not None and \
                "camera_intrinsics" in batch and "camera_transforms" in batch
            if preview_depth and "depth" in pipeline_output:
                os.makedirs(os.path.join(
                    output_path, "preview_depth"), exist_ok=True)
                video_output_path = os.path.join(
                    output_path, "preview_depth", "{}.mp4".format(filename))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", 5, pipeline_output["depth"])

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ):
        if torch.distributed.is_initialized():
            validation_datasampler.set_epoch(0)

        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        iteration_count = \
            self.inference_config["evaluation_item_count"] // world_size \
            if "evaluation_item_count" in self.inference_config else None
        for i, batch in enumerate(validation_dataloader):
            batch_size, sequence_length, view_count = \
                batch["vae_images"].shape[:3]
            if (
                iteration_count is not None and
                i * batch_size >= iteration_count
            ):
                break

            latent_height = batch["vae_images"].shape[-2] // \
                (2 ** (len(self.vae.config.down_block_types) - 1))
            latent_width = batch["vae_images"].shape[-1] // \
                (2 ** (len(self.vae.config.down_block_types) - 1))
            if "sequence_length_per_iteration" in self.inference_config:
                latent_shape = (
                    batch_size,
                    self.get_latent_sequence_length(
                        self.inference_config["sequence_length_per_iteration"]),
                    view_count, self.vae.config.latent_channels, latent_height,
                    latent_width
                )
                pipeline_output = self.autoregressive_inference_pipeline(
                    latent_shape, batch, "pt")
            else:
                latent_shape = (
                    batch_size, self.get_latent_sequence_length(
                        sequence_length), sequence_length, view_count,
                    self.vae.config.latent_channels, latent_height,
                    latent_width
                )
                pipeline_output = self.inference_pipeline(
                    latent_shape, batch, "pt")

            if "fid" in self.metrics:
                fake_images = pipeline_output["images"]\
                    .unflatten(0, (batch_size, -1, view_count))
                start = 0 \
                    if "sequence_length_per_iteration" \
                    not in self.inference_config \
                    or self.inference_config.get(
                        "generate_frames_for_reference", True) \
                    else self.inference_config.get("reference_frame_count", 1)
                stop = start + fake_images.shape[1]
                self.metrics["fid"].update(
                    batch["vae_images"][:, start:stop].flatten(0, 2)
                    .to(self.device),
                    real=True)
                self.metrics["fid"].update(
                    fake_images.flatten(0, 2), real=False)

            if "fvd" in self.metrics:
                fake_images = pipeline_output["images"]\
                    .unflatten(0, (batch_size, -1, view_count))
                start = 0 \
                    if "sequence_length_per_iteration" \
                    not in self.inference_config \
                    or self.inference_config.get(
                        "generate_frames_for_reference", True) \
                    else self.inference_config.get("reference_frame_count", 1)
                stop = start + fake_images.shape[1]
                self.metrics["fvd"].update(
                    einops.rearrange(
                        batch["vae_images"][:, start:stop].to(self.device),
                        "b t v c h w -> (b v) t c h w"),
                    real=True)
                self.metrics["fvd"].update(
                    einops.rearrange(
                        fake_images, "b t v c h w -> (b v) t c h w"),
                    real=False)

            if "rmse" in self.metrics:
                # the depth evaluation only available for non-temporal mode
                assert "sequence_length_per_iteration" \
                    not in self.inference_config

                preds_target_generator = CrossviewTemporalSD\
                    .enum_depth_preds_and_targets(
                        *batch["vae_images"].shape[:3],
                        self.model.depth_frustum_range, batch,
                        pipeline_output["depth_features"],
                        self.common_config["point_count_limit_per_view"],
                        self.common_config["point_bundle_size"], self.device)
                for preds, target in preds_target_generator:
                    self.metrics["rmse"].update(preds, target)

        text = "Step {},".format(global_step)
        for k, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            text += " {}: {:.3f}".format(k, value)
            if self.should_save:
                self.summary.add_scalar(
                    "evaluation/{}".format(k), value, global_step)

        if self.should_save:
            print(text)


class StreamingCrossviewTemporalSD(CrossviewTemporalSD):

    def reset_streaming(self, latent_shape, output_type):
        assert (
            "frame_prediction_style" in self.common_config and
            self.common_config["frame_prediction_style"] == "diffusion_forcing"
        )

        self.conditions = {}
        self.condition_count = 0
        self.latents = None
        self.text_prompt_counter = 0
        self.frames = []
        self.latent_shape = latent_shape
        self.output_type = output_type
        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"], self.device)

        # cache the last ego transform to calculate the action
        self.prev_ego_transforms = None

    def inference_pipeline(
        self, latent_shape, start_timestep: int = 0, stop_timestep=None,
        take_time: int = 0
    ):
        self.model_wrapper.eval()

        assert self.inference_config["inference_steps"] % latent_shape[1] == 0

        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        guidance_scale = self.inference_config.get("guidance_scale", 1)
        steps_per_inference = \
            self.inference_config["inference_steps"] // latent_shape[1]
        latents = self.latents
        stop_timestep = \
            stop_timestep or self.inference_config["inference_steps"]
        for i in range(start_timestep, stop_timestep):
            # make the denoising timesteps
            timestep_indices = torch\
                .tensor([
                    min(
                        i - take_time * steps_per_inference,
                        max(0, i - j * steps_per_inference))
                    for j in range(latent_shape[1])
                ], dtype=torch.int32, device=self.device).unsqueeze(0)\
                .unsqueeze(-1).repeat(latent_shape[0], 1, latent_shape[2])
            timesteps = self.test_scheduler.timesteps[timestep_indices]

            latent_model_input = latents.to(dtype=self.model_dtype)
            if do_classifier_free_guidance:
                latent_model_input = torch.cat(
                    [latent_model_input, latent_model_input])
                timesteps_input = torch.cat([timesteps, timesteps])
            else:
                timesteps_input = timesteps

            with self.get_autocast_context():
                model_output, _, _ = self.model_wrapper(
                    latent_model_input, timesteps_input, **self.conditions)

            # update the noisy latent
            noise_pred = model_output[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            staging_latents = self.test_scheduler.step_by_indices(
                noise_pred, timestep_indices.cpu(), latents
            ).prev_sample

            # only update the latents in the schedule range so finally the
            # timesteps in the queue become progrssive
            in_schedule_range = torch\
                .tensor([
                    i - j * steps_per_inference >= 0
                    for j in range(latent_shape[1])
                ], device=self.device).unsqueeze(0).unsqueeze(-1)\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            latents = torch.where(
                in_schedule_range, staging_latents, latents)

        if stop_timestep >= self.inference_config["inference_steps"]:
            shift_factor = self.vae.config.shift_factor \
                if self.vae.config.shift_factor is not None else 0
            image_tensor = self.vae.decode(
                latents[:, take_time].flatten(0, 1).to(dtype=self.vae.dtype) /
                self.vae.config.scaling_factor + shift_factor,
                return_dict=False)[0]
            self.frames.append(
                self.image_processor.postprocess(
                    image_tensor, self.output_type))

        return latents

    @torch.no_grad()
    def send_frame_condition(self, frame_condition_data):
        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        steps_per_inference = \
            self.inference_config["inference_steps"] // self.latent_shape[1]

        if frame_condition_data is None:
            # flushing
            assert (
                self.condition_count ==
                self.inference_config["sequence_length_per_iteration"]
            )
            for i in range(1, self.latent_shape[1]):
                latents = self.inference_pipeline(
                    self.latent_shape,
                    start_timestep=(
                        self.inference_config["inference_steps"] +
                        (i - 1) * steps_per_inference
                    ),
                    stop_timestep=(
                        self.inference_config["inference_steps"] +
                        i * steps_per_inference
                    ),
                    take_time=i)
                is_finished = torch.tensor(
                    [j <= i for j in range(self.latent_shape[1])],
                    device=self.device
                ).unflatten(0, (1, -1, 1, 1, 1, 1))
                self.latents = torch.where(
                    is_finished, self.latents, latents)

        else:
            frame_conditions = CrossviewTemporalSD.get_conditions(
                self.model,
                self.text_encoders if self.text_prompt_counter == 0 else None,
                self.tokenizers
                if isinstance(self.model, diffusers.SD3Transformer2DModel)
                else self.tokenizer,
                self.common_config,
                (self.latent_shape[0], 1) + self.latent_shape[2:],
                frame_condition_data, self.device, self.model_dtype,
                streaming_mode=True,
                prev_ego_transforms=self.prev_ego_transforms,
                do_classifier_free_guidance=do_classifier_free_guidance)
            if "ego_transforms" in frame_condition_data:
                self.prev_ego_transforms = frame_condition_data["ego_transforms"]

            if self.text_prompt_counter > 0:
                frame_conditions["encoder_hidden_states"] = \
                    self.conditions["encoder_hidden_states"][:, -1:]
                frame_conditions["pooled_projections"] = \
                    self.conditions["pooled_projections"][:, -1:]

            self.text_prompt_counter = (
                (self.text_prompt_counter + 1) %
                self.inference_config.get("text_prompt_interval", 1)
            )

            if (
                self.condition_count <
                self.inference_config["sequence_length_per_iteration"]
            ):
                # gathering conditions
                for k, v in frame_conditions.items():
                    if (k not in self.conditions or k in self.inference_config[
                        "autoregression_condition_exception_for_take_sequence"
                    ]):
                        self.conditions[k] = v
                    else:
                        self.conditions[k] = torch.cat(
                            [self.conditions[k], v], dim=1)

                self.condition_count += 1
                if (
                    self.condition_count ==
                    self.inference_config["sequence_length_per_iteration"]
                ):
                    # transfer to streaming state
                    self.latents = (
                        torch.randn(
                            self.latent_shape, generator=self.generator
                        ).to(self.device) *
                        getattr(self.test_scheduler, "init_noise_sigma", 1)
                    )
                    self.latents = self.inference_pipeline(
                        self.latent_shape, start_timestep=0,
                        stop_timestep=self.inference_config["inference_steps"])

            else:
                # streaming
                for k, v in frame_conditions.items():
                    if (k not in self.conditions or k in self.inference_config[
                        "autoregression_condition_exception_for_take_sequence"
                    ]):
                        self.conditions[k] = v
                    else:
                        self.conditions[k] = torch.cat(
                            [self.conditions[k][:, 1:], v], dim=1)

                self.latents = torch.cat([
                    self.latents[:, 1:],
                    torch.randn(
                        (self.latent_shape[0], 1) + self.latent_shape[2:],
                        generator=self.generator).to(self.device) *
                    getattr(self.test_scheduler, "init_noise_sigma", 1)
                ], 1)
                self.latents = self.inference_pipeline(
                    self.latent_shape,
                    start_timestep=(
                        self.inference_config["inference_steps"] -
                        steps_per_inference
                    ),
                    stop_timestep=(
                        self.inference_config["inference_steps"]
                    ))

    def receive_frame(self):
        if len(self.frames) == 0:
            return None

        return (
            [
                i.resize(self.inference_config["preview_image_size"])
                for i in self.frames.pop(0)
            ]
            if self.output_type == "pil"
            else self.frames.pop(0)
        )

    def fifo_inference_pipeline(
        self, latent_shape, batch, output_type
    ):
        total_frame_count = batch["pts"].shape[1]
        assert (
            "frame_prediction_style" in self.common_config and
            self.common_config["frame_prediction_style"] == "diffusion_forcing"
        )
        assert (
            total_frame_count >
            self.inference_config["sequence_length_per_iteration"]
        )
        exception_for_take_sequence = self.inference_config[
            "autoregression_data_exception_for_take_sequence"
        ]
        result = {
            "images": []
        }
        self.reset_streaming(latent_shape, output_type)
        for i in range(total_frame_count):
            self.send_frame_condition({
                k: (
                    v
                    if k in exception_for_take_sequence
                    else dwm.functional.take_sequence_clip(v, i, i + 1)
                )
                for k, v in batch.items()
            })
            image = self.receive_frame()
            if image is not None:
                result["images"].append(image)

        self.send_frame_condition(None)
        while True:
            image = self.receive_frame()
            if image is not None:
                result["images"].append(image)
            else:
                break

        if output_type == "pt":
            result["images"] = torch.cat(result["images"])

        return result

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // \
            (2 ** (len(self.vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // \
            (2 ** (len(self.vae.config.down_block_types) - 1))
        assert "sequence_length_per_iteration" in self.inference_config
        latent_shape = (
            batch_size,
            self.inference_config["sequence_length_per_iteration"],
            view_count, self.vae.config.latent_channels, latent_height,
            latent_width
        )
        pipeline_output = self.fifo_inference_pipeline(
            latent_shape, batch, "pt")

        if self.should_save or (
            torch.distributed.is_initialized() and
            self.inference_config.get("all_rank_preview", False)
        ):
            os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
            filename = (
                "{}_{}".format(global_step, torch.distributed.get_rank())
                if self.inference_config.get("all_rank_preview", False)
                else str(global_step)
            )
            preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
                pipeline_output["images"], batch, self.inference_config)
            if sequence_length == 1:
                image_output_path = os.path.join(
                    output_path, "preview", "{}.png".format(filename))
                torchvision.transforms.functional.to_pil_image(preview_tensor)\
                    .save(image_output_path)
            else:
                video_output_path = os.path.join(
                    output_path, "preview", "{}.mp4".format(filename))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", batch["fps"][0].item(),
                    preview_tensor)

            preview_depth = self.model.depth_net is not None and \
                "camera_intrinsics" in batch and "camera_transforms" in batch
            if preview_depth and "depth" in pipeline_output:
                os.makedirs(os.path.join(
                    output_path, "preview_depth"), exist_ok=True)
                video_output_path = os.path.join(
                    output_path, "preview_depth", "{}.mp4".format(filename))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", 5, pipeline_output["depth"])