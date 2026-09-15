import gc
import os
import time
from typing import Optional

import diffusers
import einops
import safetensors.torch
import torch
import torch.distributed.checkpoint.state_dict
import torch.distributed.fsdp
import torch.distributed.fsdp.sharded_grad_scaler
import torch.utils.tensorboard
import torchvision
import transformers
from diffusers.image_processor import VaeImageProcessor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

import dwm.common
import dwm.distributed
import dwm.functional
import dwm.utils.preview
import dwm.utils.track_consistency
import dwm.utils.view_consistency
from dwm.models.bev_models.condition import (
    ConditionCrossAttention,
    TemporalTokenBlock,
)
from dwm.models.crossview_temporal import VTSelfAttentionBlock
from diffusers.models.attention import JointTransformerBlock


LEGACY_BOX_PREFIX = "bbox_token_encoder."
NEW_BOX_PREFIX = "bbox_condition_encoder."
REMOVED_LEGACY_PREFIXES = (
    "condition_image_adapter.",
    "map_residual_encoder.",
    "map_token_encoder.",
    "view_pos_embeds.",
)


def load_checkpoint_state(path: str) -> dict:
    if path.endswith(".safetensors"):
        state = safetensors.torch.load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path!r} does not contain a state dictionary.")
    return state


def remap_legacy_state_dict(state_dict, target_state_dict):
    remapped = dict(state_dict)
    mapped_keys = []
    skipped_keys = []
    removed_keys = []
    for old_key in list(state_dict.keys()):
        if (
            old_key.startswith(REMOVED_LEGACY_PREFIXES)
            and old_key not in target_state_dict
        ):
            remapped.pop(old_key, None)
            removed_keys.append(old_key)
            continue
        if not old_key.startswith(LEGACY_BOX_PREFIX):
            continue
        suffix = old_key[len(LEGACY_BOX_PREFIX):]
        new_key = NEW_BOX_PREFIX + suffix
        remapped.pop(old_key, None)
        if (
            new_key in target_state_dict
            and tuple(state_dict[old_key].shape)
            == tuple(target_state_dict[new_key].shape)
        ):
            remapped[new_key] = state_dict[old_key]
            mapped_keys.append((old_key, new_key))
        else:
            skipped_keys.append(old_key)
    return remapped, mapped_keys, skipped_keys, removed_keys


class BEVPipeline:
    def __init__(
        self,
        output_path,
        config: dict,
        device,
        common_config: dict,
        training_config: dict,
        inference_config: dict,
        pretrained_model_name_or_path: str,
        model,
        model_dtype=None,
        model_checkpoint_path=None,
        model_load_state_args: Optional[dict] = None,
        metrics: Optional[dict] = None,
        resume_from=None,
    ):
        self.should_save = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        self.output_path = output_path
        self.config = config
        self.device = device
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.model_dtype = model_dtype or torch.float32
        self.generator = torch.Generator()
        self.generator.manual_seed(int(config.get("generator_seed", 0)))
        self.memory_efficient_batch = int(
            common_config.get("memory_efficient_batch", 12)
        )

        empty_hidden, empty_pooled = self.load_empty_sd3_prompt(
            pretrained_model_name_or_path
        )
        self.empty_encoder_hidden_states = empty_hidden.to(
            device=self.device,
            dtype=self.model_dtype,
        )
        self.empty_pooled_projections = empty_pooled.to(
            device=self.device,
            dtype=self.model_dtype,
        )

        self.model = model.to(dtype=self.model_dtype)
        self.model.enable_gradient_checkpointing()
        self.load_model_weights(
            model_checkpoint_path=model_checkpoint_path,
            model_load_state_args=model_load_state_args,
            resume_from=resume_from,
        )
        self.model_wrapper = self.wrap_model_with_fsdp()

        self.vae = diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.device)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1)
        )

        self.train_scheduler = (
            diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="scheduler",
            )
        )
        scheduler_class = dwm.common.get_class(
            inference_config[
                "scheduler"
            ]
        )
        self.test_scheduler = scheduler_class.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=self.model_wrapper.parameters(),
        )
        self.optimizer.zero_grad(set_to_none=True)
        if resume_from is not None:
            dwm.distributed.distributed_load_optimizer_state(
                self.model_wrapper,
                self.optimizer,
                os.path.join(output_path, "optimizer"),
                str(resume_from),
            )

        if torch.distributed.is_initialized():
            self.grad_scaler = (
                torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler()
            )
        else:
            self.grad_scaler = torch.amp.GradScaler()

        self.metrics = {} if metrics is None else metrics
        for metric in self.metrics.values():
            metric.to(self.device)

        self.summary = None
        if self.should_save and output_path is not None:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log")
            )
        self.loss_report_list = []
        self.step_duration = 0.0

    def load_empty_sd3_prompt(
        self,
        pretrained_model_name_or_path: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        load_args = {
            "variant": "fp16",
            "torch_dtype": torch.float16,
        }
        clip_hidden_states = []
        clip_pooled_states = []
        clip_subfolders = (
            ("tokenizer", "text_encoder"),
            ("tokenizer_2", "text_encoder_2"),
        )
        for tokenizer_subfolder, encoder_subfolder in clip_subfolders:
            tokenizer = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=tokenizer_subfolder,
            )
            encoder = (
                transformers.CLIPTextModelWithProjection.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=encoder_subfolder,
                    **load_args,
                )
            )
            encoder.requires_grad_(False)
            encoder.eval()
            encoder.to(self.device)
            text_inputs = tokenizer(
                [""],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                encoded = encoder(
                    text_inputs.input_ids.to(self.device),
                    output_hidden_states=True,
                )
            clip_pooled_states.append(encoded[0].detach().cpu())
            clip_hidden_states.append(
                encoded.hidden_states[-2].detach().cpu()
            )
            del encoded
            del encoder
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t5_tokenizer = transformers.T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_3",
        )
        t5_encoder = transformers.T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            **load_args,
        )
        t5_encoder.requires_grad_(False)
        t5_encoder.eval()
        t5_encoder.to(self.device)
        t5_inputs = t5_tokenizer(
            [""],
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            t5_hidden = t5_encoder(
                t5_inputs.input_ids.to(self.device)
            )[0].detach().cpu()
        del t5_encoder
        del t5_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clip_hidden = torch.cat(clip_hidden_states, dim=-1)
        clip_hidden = torch.nn.functional.pad(
            clip_hidden,
            (0, t5_hidden.shape[-1] - clip_hidden.shape[-1]),
        )
        encoder_hidden_states = torch.cat(
            [clip_hidden, t5_hidden],
            dim=1,
        )
        pooled_projections = torch.cat(clip_pooled_states, dim=-1)
        return encoder_hidden_states, pooled_projections

    def load_model_weights(
        self,
        model_checkpoint_path,
        model_load_state_args,
        resume_from,
    ):
        if resume_from is not None:
            checkpoint_path = os.path.join(
                self.output_path,
                "checkpoints",
                f"{resume_from}.pth",
            )
            state_dict = load_checkpoint_state(checkpoint_path)
            self.model.load_state_dict(state_dict, strict=True)
            return
        if model_checkpoint_path is None:
            return

        state_dict = load_checkpoint_state(model_checkpoint_path)
        (
            state_dict,
            mapped_box_keys,
            skipped_box_keys,
            removed_legacy_keys,
        ) = remap_legacy_state_dict(
            state_dict,
            self.model.state_dict(),
        )
        # Expand legacy PV ImageAdapter input weights when the checkpoint
        # was trained with fewer image-condition channels (e.g. 6ch box+map)
        # than the current model (9ch box+map+instance-flow).
        #
        # Keep all existing channel weights exactly and zero-initialize only
        # the newly added input channels. This matches the existing PVTrack
        # checkpoint-loading behavior.
        target_state_dict = self.model.state_dict()
        expanded_condition_keys = []
        for key, value in list(state_dict.items()):
            if not key.startswith("condition_image_adapter."):
                continue
            if key not in target_state_dict:
                continue

            target_value = target_state_dict[key]
            if tuple(value.shape) == tuple(target_value.shape):
                continue

            if value.ndim < 2 or value.ndim != target_value.ndim:
                continue

            same_non_channel_shape = (
                value.shape[0] == target_value.shape[0]
                and tuple(value.shape[2:]) == tuple(target_value.shape[2:])
            )
            if (
                same_non_channel_shape
                and value.shape[1] < target_value.shape[1]
            ):
                expanded_value = torch.zeros_like(
                    target_value,
                    device="cpu",
                )
                expanded_value[:, :value.shape[1]] = value.to(
                    dtype=expanded_value.dtype,
                    device="cpu",
                )
                state_dict[key] = expanded_value
                expanded_condition_keys.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(target_value.shape),
                    )
                )

        if self.should_save and expanded_condition_keys:
            print(
                "expanded condition adapter input weights:",
                expanded_condition_keys,
                flush=True,
            )

        load_args = {"strict": False}
        if model_load_state_args is not None:
            load_args.update(model_load_state_args)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            **load_args,
        )
        if self.should_save and bool(
            self.common_config.get("print_load_state_info", True)
        ):
            print(
                "[BEV legacy load] "
                f"mapped_box={len(mapped_box_keys)} "
                f"skipped_box={len(skipped_box_keys)} "
                f"removed_old={len(removed_legacy_keys)} "
                f"missing_new={len(missing_keys)} "
                f"unexpected_old={len(unexpected_keys)}",
                flush=True,
            )
            print(
                "[BEV legacy load] new prefixes: "
                + summarize_prefixes(missing_keys),
                flush=True,
            )
            print(
                "[BEV legacy load] removed prefixes: "
                + summarize_prefixes(unexpected_keys),
                flush=True,
            )

    def wrap_model_with_fsdp(self):
        if not torch.distributed.is_initialized():
            self.model.to(self.device)
            return self.model
        framework = self.common_config.get("distribution_framework", "fsdp")
        if framework != "fsdp":
            raise ValueError(
                f"BEVPipeline only supports FSDP, got {framework!r}."
            )
        fsdp_settings = dict(self.common_config["fsdp_settings"])
        fsdp_settings["auto_wrap_policy"] = ModuleWrapPolicy(
            {
                JointTransformerBlock,
                VTSelfAttentionBlock,
                ConditionCrossAttention,
                TemporalTokenBlock,
            }
        )
        return FSDP(
            self.model,
            device_id=torch.cuda.current_device(),
            **fsdp_settings,
        )

    def encode_images(
        self,
        images: torch.Tensor,
        use_mode: bool,
    ) -> torch.Tensor:
        shift_factor = self.vae.config.shift_factor
        if shift_factor is None:
            shift_factor = 0.0
        latent_chunks = []
        for start in range(0, images.shape[0], self.memory_efficient_batch):
            stop = min(start + self.memory_efficient_batch, images.shape[0])
            image_chunk = images[start:stop].to(
                device=self.device,
                dtype=self.vae.dtype,
            )
            with torch.no_grad():
                posterior = self.vae.encode(image_chunk).latent_dist
                latent_chunk = posterior.mode() if use_mode else posterior.sample()
                latent_chunk = (
                    latent_chunk - shift_factor
                ) * self.vae.config.scaling_factor
            latent_chunks.append(latent_chunk)
        return torch.cat(latent_chunks, dim=0)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        shift_factor = self.vae.config.shift_factor
        if shift_factor is None:
            shift_factor = 0.0
        image_chunks = []
        for start in range(0, latents.shape[0], self.memory_efficient_batch):
            stop = min(start + self.memory_efficient_batch, latents.shape[0])
            latent_chunk = latents[start:stop].to(
                device=self.device,
                dtype=self.vae.dtype,
            )
            latent_chunk = (
                latent_chunk / self.vae.config.scaling_factor + shift_factor
            )
            with torch.no_grad():
                image_chunk = self.vae.decode(
                    latent_chunk,
                    return_dict=False,
                )[0]
            image_chunks.append(image_chunk)
        return torch.cat(image_chunks, dim=0)

    def expand_empty_text(
        self,
        batch_size: int,
        sequence_length: int,
        view_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states = self.empty_encoder_hidden_states[:, None, None]
        encoder_hidden_states = encoder_hidden_states.expand(
            batch_size,
            sequence_length,
            view_count,
            -1,
            -1,
        )
        pooled_projections = self.empty_pooled_projections[:, None, None]
        pooled_projections = pooled_projections.expand(
            batch_size,
            sequence_length,
            view_count,
            -1,
        )
        return encoder_hidden_states, pooled_projections

    def prepare_model_conditions(
        self,
        batch: dict,
        latent_shape,
        condition_keep: Optional[torch.Tensor] = None,
        pv_condition_keep: Optional[torch.Tensor] = None,
        classifier_free_guidance: bool = False,
    ) -> dict:
        batch_size, sequence_length, view_count = latent_shape[:3]
        required_keys = (
            "camera_intrinsics",
            "image_size",
            "camera_transforms",
            "reference_ego_transforms",
            "bbox_token_corners",
            "bbox_token_classes",
            "bbox_token_masks",
            "hdmap_bev_images",
            "3dbox_images",
            "hdmap_images",
            "instance_flow_images",
            "crossview_mask",
        )
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise KeyError(f"Missing required BEV batch keys: {missing}")

        camera_intrinsics_norm = batch["camera_intrinsics"].clone().float()
        image_size = batch["image_size"].to(camera_intrinsics_norm)
        camera_intrinsics_norm[..., 0, 0] /= image_size[..., 0]
        camera_intrinsics_norm[..., 1, 1] /= image_size[..., 1]
        camera_intrinsics_norm[..., 0, 2] /= image_size[..., 0]
        camera_intrinsics_norm[..., 1, 2] /= image_size[..., 1]
        camera_intrinsics_norm = camera_intrinsics_norm[:, :1]

        camera_to_ego = batch["camera_transforms"][:, :1].float()

        world_from_reference_ego = (
            batch["reference_ego_transforms"].double()
        )
        ego_to_initial = torch.linalg.solve(
            world_from_reference_ego[:, :1],
            world_from_reference_ego,
        ).float()
        
        bbox_corners = batch["bbox_token_corners"].float()
        bbox_classes = batch["bbox_token_classes"].long()
        bbox_view_masks = batch["bbox_token_masks"].float()
        bev_map = batch["hdmap_bev_images"].float()
        crossview_mask = batch["crossview_mask"].bool()

        # PV branch: exact 3-image-stream concatenation used by PVTrack.
        pv_condition_parts = [
            batch["3dbox_images"].float(),
            batch["hdmap_images"].float(),
            batch["instance_flow_images"].float(),
        ]
        expected_pv_prefix = (batch_size, sequence_length, view_count)
        for name, value in zip(
            ("3dbox_images", "hdmap_images", "instance_flow_images"),
            pv_condition_parts,
        ):
            if value.ndim != 6 or tuple(value.shape[:3]) != expected_pv_prefix:
                raise ValueError(
                    f"{name} must be [B,T,V,C,H,W] with prefix "
                    f"{expected_pv_prefix}, got {tuple(value.shape)}."
                )
        pv_hw = pv_condition_parts[0].shape[-2:]
        if any(value.shape[-2:] != pv_hw for value in pv_condition_parts[1:]):
            raise ValueError(
                "PV condition images must share the same spatial size: "
                + ", ".join(
                    f"{name}={tuple(value.shape[-2:])}"
                    for name, value in zip(
                        ("3dbox_images", "hdmap_images", "instance_flow_images"),
                        pv_condition_parts,
                    )
                )
            )
        condition_image_tensor = torch.cat(pv_condition_parts, dim=-3)

        if condition_keep is None:
            condition_keep = torch.ones(batch_size, dtype=torch.bool)
        condition_keep = condition_keep.bool()

        # condition_keep controls the original BEV-side conditioning.
        #
        # pv_condition_keep independently controls only the PV raster bundle:
        #   3dbox_images + hdmap_images + instance_flow_images.
        #
        # RGB / vae_images / noisy latents are NOT affected.
        if pv_condition_keep is None:
            pv_condition_keep = condition_keep
        pv_condition_keep = pv_condition_keep.bool()

        if pv_condition_keep.shape != condition_keep.shape:
            raise ValueError(
                "pv_condition_keep must have the same shape as "
                f"condition_keep, got {tuple(pv_condition_keep.shape)} "
                f"vs {tuple(condition_keep.shape)}."
            )

        # Never allow PV conditioning when the global condition is dropped.
        pv_condition_keep = pv_condition_keep & condition_keep

        uncondition_image_color = float(
            self.common_config.get("uncondition_image_color", 0.1255)
        )
        pv_keep = pv_condition_keep[:, None, None, None, None, None]
        condition_image_tensor = torch.where(
            pv_keep,
            condition_image_tensor,
            torch.full_like(condition_image_tensor, uncondition_image_color),
        )

        if classifier_free_guidance:
            camera_intrinsics_norm = torch.cat(
                [camera_intrinsics_norm, camera_intrinsics_norm],
                dim=0,
            )
            camera_to_ego = torch.cat(
                [camera_to_ego, camera_to_ego],
                dim=0,
            )
            ego_to_initial = torch.cat([ego_to_initial, ego_to_initial], dim=0)
            bbox_corners = torch.cat([bbox_corners, bbox_corners], dim=0)
            bbox_classes = torch.cat([bbox_classes, bbox_classes], dim=0)
            bbox_view_masks = torch.cat(
                [bbox_view_masks, bbox_view_masks],
                dim=0,
            )
            bev_map = torch.cat([bev_map, bev_map], dim=0)
            crossview_mask = torch.cat(
                [crossview_mask, crossview_mask],
                dim=0,
            )
            condition_image_tensor = torch.cat(
                [
                    torch.full_like(
                        condition_image_tensor,
                        uncondition_image_color,
                    ),
                    condition_image_tensor,
                ],
                dim=0,
            )
            condition_keep = torch.cat(
                [
                    torch.zeros_like(condition_keep),
                    torch.ones_like(condition_keep),
                ],
                dim=0,
            )
            batch_size *= 2

        encoder_hidden_states, pooled_projections = self.expand_empty_text(
            batch_size,
            sequence_length,
            view_count,
        )
        return {
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "camera_intrinsics_norm": camera_intrinsics_norm.to(self.device),
            "camera_to_ego": camera_to_ego.to(self.device),
            "ego_to_initial": ego_to_initial.to(self.device),
            "bbox_corners": bbox_corners.to(self.device),
            "bbox_classes": bbox_classes.to(self.device),
            "bbox_view_masks": bbox_view_masks.to(self.device),
            "bev_map": bev_map.to(self.device),
            "condition_image_tensor": condition_image_tensor.to(
                device=self.device,
                dtype=self.model_dtype,
            ),
            "crossview_attention_mask": crossview_mask.to(self.device),
            "condition_keep": condition_keep.to(self.device),
        }

    def make_ctsd_training_input(
        self,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, view_count = noisy_latents.shape[:3]
        generation_task = torch.rand(
            (batch_size, 1, 1),
            generator=self.generator,
        ) < float(self.training_config["generation_task_ratio"])
        disable_temporal = generation_task & (
            torch.rand(
                (batch_size, 1, 1),
                generator=self.generator,
            ) < float(self.training_config["image_generation_ratio"])
        )
        all_reference_visible = torch.rand(
            (batch_size, 1, 1),
            generator=self.generator,
        ) < float(self.training_config["all_reference_visible_ratio"])
        visible_rate = float(self.training_config["reference_visible_rate"])
        clean_view_count = int(view_count * visible_rate + 0.5)
        clean_view_count = max(1, min(clean_view_count, view_count - 1))
        view_scores = torch.rand(
            (batch_size, sequence_length, view_count),
            generator=self.generator,
        )
        visible_indices = torch.topk(
            view_scores,
            k=clean_view_count,
            dim=2,
            largest=False,
        ).indices
        partial_reference = torch.zeros(
            (batch_size, sequence_length, view_count),
            dtype=torch.bool,
        )
        partial_reference.scatter_(2, visible_indices, True)

        reference_count = int(self.training_config["reference_frame_count"])
        reference_range = torch.arange(sequence_length)[None, :, None]
        reference_range = reference_range < reference_count
        reference_indicator = (
            ~generation_task
            & (all_reference_visible | partial_reference)
            & reference_range
        )
        reference_indicator = reference_indicator.to(noisy_latents.device)
        noisy_latents = torch.where(
            reference_indicator[..., None, None, None],
            clean_latents,
            noisy_latents,
        )
        timesteps = torch.where(
            reference_indicator,
            torch.zeros_like(timesteps),
            timesteps,
        )
        return (
            noisy_latents,
            timesteps,
            disable_temporal.to(noisy_latents.device),
            reference_indicator,
        )

    def make_box_loss_mask(
        self,
        batch: dict,
        target_shape,
    ) -> torch.Tensor:
        batch_size, sequence_length, view_count = target_shape[:3]
        box_mask = batch["3dbox_images"].to(
            device=self.device,
            dtype=torch.float32,
        )
        if box_mask.amax() > 1.0:
            box_mask = box_mask / 255.0
        box_mask = box_mask.amax(dim=3, keepdim=True)
        box_mask = (box_mask > 0.005).float()
        box_mask = einops.rearrange(
            box_mask,
            "b t v c h w -> (b t v) c h w",
        )
        box_mask = torch.nn.functional.max_pool2d(
            box_mask,
            kernel_size=63,
            stride=1,
            padding=31,
        )
        box_mask = torch.nn.functional.interpolate(
            box_mask,
            size=target_shape[-2:],
            mode="nearest",
        )
        return einops.rearrange(
            box_mask,
            "(b t v) c h w -> b t v c h w",
            b=batch_size,
            t=sequence_length,
            v=view_count,
        )


    def prepare_view_consistency_geometry(
        self,
        batch: dict,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return dwm.utils.view_consistency.prepare_view_consistency_geometry(
            batch, sequence_length, self.device
        )

    def train_step(self, batch: dict, global_step: int):
        self.model_wrapper.train()
        start_time = time.time()
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(self.device)
        )
        latents = self.encode_images(image_tensor, use_mode=False)
        latents = einops.rearrange(
            latents,
            "(b t v) c h w -> b t v c h w",
            b=batch_size,
            t=sequence_length,
            v=view_count,
        )
        noise = torch.randn(
            latents.shape,
            generator=self.generator,
        ).to(self.device)
        density = torch.sigmoid(
            torch.normal(
                mean=0.0,
                std=1.0,
                size=(batch_size,),
                device="cpu",
            )
        )
        timestep_indices = (
            density * self.train_scheduler.config.num_train_timesteps
        ).long()
        timesteps = self.train_scheduler.timesteps[timestep_indices].to(
            self.device
        )
        sigmas = self.train_scheduler.sigmas[timestep_indices].to(
            device=self.device,
            dtype=latents.dtype,
        )
        while sigmas.ndim < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        timesteps = timesteps[:, None, None].expand(
            batch_size,
            sequence_length,
            view_count,
        )

        condition_keep = torch.rand(
            (batch_size,),
            generator=self.generator,
        ) >= float(self.training_config["condition_dropout_ratio"])

        # Extra dropout applied only to the PV raster conditions:
        #   3dbox_images + hdmap_images + instance_flow_images.
        #
        # This does NOT remove RGB / vae_images / noisy latents.
        # It also does NOT additionally drop the BEV branch.
        pv_condition_dropout_ratio = float(
            self.training_config.get(
                "pv_condition_dropout_ratio",
                0.0,
            )
        )
        if not 0.0 <= pv_condition_dropout_ratio <= 1.0:
            raise ValueError(
                "pv_condition_dropout_ratio must be inside [0, 1], "
                f"got {pv_condition_dropout_ratio}."
            )

        pv_extra_keep = torch.rand(
            (batch_size,),
            generator=self.generator,
        ) >= pv_condition_dropout_ratio

        # If global condition is absent, PV must also be absent.
        # Otherwise PV is independently dropped according to
        # pv_condition_dropout_ratio.
        pv_condition_keep = condition_keep & pv_extra_keep

        model_conditions = self.prepare_model_conditions(
            batch,
            latents.shape,
            condition_keep=condition_keep,
            pv_condition_keep=pv_condition_keep,
        )
        noisy_latents, timesteps, disable_temporal, reference_indicator = (
            self.make_ctsd_training_input(
                noisy_latents,
                latents,
                timesteps,
            )
        )
        model_conditions["disable_temporal"] = disable_temporal

        view_consistency_selection_result = (
            dwm.utils.view_consistency.sample_view_consistency_selection(
                batch=batch,
                training_config=self.training_config,
                generator=self.generator,
                device=self.device,
            )
        )
        if view_consistency_selection_result is None:
            view_consistency_selection = None
            view_consistency_selection_cpu = None
        else:
            (
                view_consistency_selection,
                view_consistency_selection_cpu,
            ) = view_consistency_selection_result
            model_conditions["view_consistency_selection"] = (
                view_consistency_selection
            )

        track_consistency_selection_result = (
            dwm.utils.track_consistency.sample_track_consistency_selection(
                batch=batch,
                training_config=self.training_config,
                generator=self.generator,
                device=self.device,
            )
        )
        if track_consistency_selection_result is None:
            track_consistency_selection = None
            track_consistency_selection_cpu = None
            track_consistency_pair_types_cpu = None
        else:
            (
                track_consistency_selection,
                track_consistency_selection_cpu,
                track_consistency_pair_types_cpu,
            ) = track_consistency_selection_result
            model_conditions["track_consistency_selection"] = (
                track_consistency_selection
            )

        model_output, _, _ = self.model_wrapper(
            noisy_latents.to(self.model_dtype),
            timesteps,
            **model_conditions,
        )
        auxiliary_output_index = 1
        projected_features = None
        if view_consistency_selection is not None:
            projected_features = model_output[auxiliary_output_index]
            auxiliary_output_index += 1
        track_projected_features = None
        if track_consistency_selection is not None:
            track_projected_features = model_output[auxiliary_output_index]
        predicted_latents = model_output[0] * (-sigmas) + noisy_latents
        squared_error = (
            predicted_latents.float() - latents.float()
        ).square()

        if bool(self.training_config["disable_reference_frame_loss"]):
            pixel_weight = (~reference_indicator)[..., None, None, None].float()
        else:
            pixel_weight = torch.ones(
                (*latents.shape[:3], 1, 1, 1),
                device=self.device,
                dtype=torch.float32,
            )
        box_mask = self.make_box_loss_mask(batch, predicted_latents.shape)
        pixel_weight = pixel_weight * (1.0 + 8.0 * box_mask)
        denominator = (
            pixel_weight.sum() * predicted_latents.shape[3]
        ).clamp_min(1.0)
        sd_loss = (squared_error * pixel_weight).sum() / denominator

        view_consistency_loss = sd_loss.new_zeros(())
        consistency_weight = float(
            self.training_config.get("view_consistency_loss_weight", 0.0)
        )
        if (
            consistency_weight > 0.0
            and view_consistency_selection is not None
        ):
            (
                camera_intrinsics_norm,
                camera2referego,
            ) = self.prepare_view_consistency_geometry(
                batch,
                sequence_length,
            )
            sigmas_cpu = sigmas.reshape(batch_size, -1)[:, 0].detach().cpu()
            view_consistency_loss = (
                dwm.utils.view_consistency.compute_view_consistency_loss(
                    batch=batch,
                    projected_features=projected_features,
                    selection=view_consistency_selection,
                    selection_cpu=view_consistency_selection_cpu,
                    camera_intrinsics_norm=camera_intrinsics_norm,
                    camera2referego=camera2referego,
                    sigmas_cpu=sigmas_cpu,
                    training_config=self.training_config,
                    generator=self.generator,
                    device=self.device,
                )
            )

        consistency_warmup_steps = int(
            self.training_config.get(
                "view_consistency_loss_warmup_steps",
                0,
            )
        )
        if consistency_warmup_steps > 0:
            warmup_ratio = min(
                float(global_step + 1) / float(consistency_warmup_steps),
                1.0,
            )
        else:
            warmup_ratio = 1.0
        effective_consistency_weight = consistency_weight * warmup_ratio

        track_weight = float(
            self.training_config.get(
                "track_consistency_loss_weight",
                0.0,
            )
        )
        track_statistics = {
            "track_loss": sd_loss.new_zeros(()),
            "track_spatial_loss": sd_loss.new_zeros(()),
            "track_temporal_loss": sd_loss.new_zeros(()),
            "track_valid_pair_count": 0.0,
            "track_valid_query_count": 0.0,
            "track_skipped_small_region_count": 0.0,
            "track_hard_negative_count": 0.0,
            "track_positive_similarity": sd_loss.new_zeros(()),
            "track_negative_similarity": sd_loss.new_zeros(()),
            "track_similarity_gap": sd_loss.new_zeros(()),
            "track_retrieval_at_1": 0.0,
            "track_spatial_retrieval_at_1": 0.0,
            "track_temporal_retrieval_at_1": 0.0,
            "hard_negative_mean_similarity": sd_loss.new_zeros(()),
        }
        if track_weight > 0.0 and track_consistency_selection is not None:
            track_statistics = (
                dwm.utils.track_consistency.compute_track_consistency_loss(
                    batch=batch,
                    projected_features=track_projected_features,
                    selection_cpu=track_consistency_selection_cpu,
                    pair_types_cpu=track_consistency_pair_types_cpu,
                    training_config=self.training_config,
                    device=self.device,
                )
            )
        track_warmup_steps = int(
            self.training_config.get(
                "track_consistency_loss_warmup_steps",
                0,
            )
        )
        if track_warmup_steps > 0:
            track_warmup_ratio = min(
                float(global_step + 1) / float(track_warmup_steps),
                1.0,
            )
        else:
            track_warmup_ratio = 1.0
        effective_track_weight = track_weight * track_warmup_ratio
        loss = (
            sd_loss
            + effective_consistency_weight * view_consistency_loss
            + effective_track_weight * track_statistics["track_loss"]
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite BEV training loss at step "
                f"{global_step + 1}."
            )

        loss_report = {
            "loss": float(loss.detach()),
            "sd_loss": float(sd_loss.detach()),
            "view_consistency_loss": float(
                view_consistency_loss.detach()
            ),
            "view_consistency_weight": float(
                effective_consistency_weight
            ),
            "bev_condition_keep_rate": float(
                condition_keep.float().mean()
            ),
            "pv_condition_keep_rate": float(
                pv_condition_keep.float().mean()
            ),
            "pv_extra_keep_rate": float(
                pv_extra_keep.float().mean()
            ),
        }
        if track_weight > 0.0:
            loss_report["track_consistency_weight"] = float(
                effective_track_weight
            )
            loss_report.update(
                {
                    key: (
                        float(value.detach())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    for key, value in track_statistics.items()
                }
            )
        self.loss_report_list.append(loss_report)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_duration += time.time() - start_time

    def inference_pipeline(
        self,
        latent_shape,
        batch: dict,
        output_type: str,
        image_latents: Optional[torch.Tensor] = None,
        reference_frame_count: int = 0,
    ) -> dict:
        self.model_wrapper.eval()
        batch_size, sequence_length, view_count = latent_shape[:3]
        guidance_scale = float(self.inference_config["guidance_scale"])
        use_guidance = guidance_scale > 1.0
        self.test_scheduler.set_timesteps(
            int(self.inference_config["inference_steps"]),
            self.device,
        )
        latents = torch.randn(
            latent_shape,
            generator=self.generator,
        ).to(self.device)
        latents = latents * getattr(
            self.test_scheduler,
            "init_noise_sigma",
            1.0,
        )
        model_conditions = self.prepare_model_conditions(
            batch,
            latent_shape,
            classifier_free_guidance=use_guidance,
        )
        disable_temporal = torch.zeros(
            batch_size,
            1,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        if use_guidance:
            disable_temporal = torch.cat(
                [disable_temporal, disable_temporal],
                dim=0,
            )
        model_conditions["disable_temporal"] = disable_temporal

        for timestep in self.test_scheduler.timesteps:
            latent_model_input = latents
            model_timesteps = timestep.expand(
                batch_size,
                sequence_length,
                view_count,
            ).clone()
            if image_latents is not None and reference_frame_count > 0:
                latent_model_input = torch.cat(
                    [
                        image_latents[:, :reference_frame_count],
                        latent_model_input[:, reference_frame_count:],
                    ],
                    dim=1,
                )
                model_timesteps[:, :reference_frame_count] = 0
            latent_model_input = latent_model_input.to(self.model_dtype)
            if hasattr(self.test_scheduler, "scale_model_input"):
                latent_model_input = self.test_scheduler.scale_model_input(
                    latent_model_input,
                    timestep,
                ).to(self.model_dtype)
            if use_guidance:
                latent_model_input = torch.cat(
                    [latent_model_input, latent_model_input],
                    dim=0,
                )
                model_timesteps = torch.cat(
                    [model_timesteps, model_timesteps],
                    dim=0,
                )
            model_output, _, _ = self.model_wrapper(
                latent_model_input,
                model_timesteps,
                **model_conditions,
            )
            noise_prediction = model_output[0]
            if use_guidance:
                unconditioned, conditioned = noise_prediction.chunk(2)
                noise_prediction = unconditioned + guidance_scale * (
                    conditioned - unconditioned
                )
            latents = self.test_scheduler.step(
                noise_prediction,
                timestep,
                latents,
            ).prev_sample

        if image_latents is not None and reference_frame_count > 0:
            latents = torch.cat(
                [
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:],
                ],
                dim=1,
            )
        decoded = self.decode_latents(latents.flatten(0, 2))
        images = self.image_processor.postprocess(
            decoded,
            output_type=output_type,
        )
        return {"images": images, "latents": latents}

    def autoregressive_inference_pipeline(
        self,
        latent_shape,
        batch: dict,
        output_type: str,
    ) -> dict:
        total_frame_count = batch["vae_images"].shape[1]
        iteration_length = int(
            self.inference_config["sequence_length_per_iteration"]
        )
        reference_frame_count = int(
            self.inference_config["reference_frame_count"]
        )
        stride = int(self.inference_config["autoregressive_stride"])
        if stride != iteration_length - reference_frame_count:
            raise ValueError(
                "autoregressive_stride must equal "
                "sequence_length_per_iteration - reference_frame_count."
            )

        image_latents = None
        if not bool(
            self.inference_config["generate_frames_for_reference"]
        ):
            reference_images = batch["vae_images"][:, :reference_frame_count]
            image_tensor = self.image_processor.preprocess(
                reference_images.flatten(0, 2).to(self.device)
            )
            image_latents = self.encode_images(image_tensor, use_mode=True)
            image_latents = einops.rearrange(
                image_latents,
                "(b t v) c h w -> b t v c h w",
                b=latent_shape[0],
                t=reference_frame_count,
                v=latent_shape[2],
            )

        result_images = []
        static_keys = set(
            self.inference_config[
                "autoregression_data_exception_for_take_sequence"
            ]
        )
        for start in range(
            0,
            total_frame_count - iteration_length + 1,
            stride,
        ):
            iteration_batch = {}
            for key, value in batch.items():
                if key in static_keys:
                    iteration_batch[key] = value
                else:
                    iteration_batch[key] = dwm.functional.take_sequence_clip(
                        value,
                        start,
                        start + iteration_length,
                    )
            current_reference_count = (
                0 if image_latents is None else reference_frame_count
            )
            iteration_output = self.inference_pipeline(
                latent_shape,
                iteration_batch,
                output_type,
                image_latents=image_latents,
                reference_frame_count=current_reference_count,
            )
            image_start = (
                latent_shape[0]
                * current_reference_count
                * latent_shape[2]
            )
            result_images.append(iteration_output["images"][image_start:])
            image_latents = iteration_output["latents"][
                :, -reference_frame_count:
            ]

        if output_type == "pt":
            images = torch.cat(result_images, dim=0)
        else:
            images = []
            for image_list in result_images:
                images.extend(image_list)
        return {"images": images, "latents": image_latents}

    def _get_eval_frame_export_path(self) -> Optional[str]:
        return dwm.utils.preview.get_eval_frame_export_path(
            self.inference_config
        )

    def _get_eval_frame_manifest_path(self) -> Optional[str]:
        return dwm.utils.preview.get_eval_frame_manifest_path(
            self.inference_config
        )

    def _prepare_eval_frame_resume(
        self,
        item_limit: int,
        loader_batch_size: int,
    ) -> int:
        return dwm.utils.preview.prepare_eval_frame_resume(
            inference_config=self.inference_config,
            device=self.device,
            item_limit=item_limit,
            loader_batch_size=loader_batch_size,
        )

    def _export_eval_frames(
        self,
        output_images: torch.Tensor,
        batch: dict,
    ):
        eval_frame_export_path = self._get_eval_frame_export_path()
        if eval_frame_export_path is None:
            return

        dist_on = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if dist_on else 0

        all_rank_preview = bool(
            self.inference_config.get(
                "all_rank_preview",
                False,
            )
        )

        save_this_rank = self.should_save or (
            dist_on and all_rank_preview
        )

        if not save_this_rank:
            return

        # RGB exporter只负责生成图/真实图。
        # 即使JSON里残留box字段，也强制不导出box。
        rgb_export_config = dict(
            self.inference_config
        )

        rgb_export_config[
            "eval_frame_export_box_image"
        ] = False

        rgb_export_config[
            "eval_frame_export_box_params"
        ] = False

        dwm.utils.preview.save_ctsd_eval_frames_for_preview(
            output_images,
            batch,
            rgb_export_config,
            output_dir=eval_frame_export_path,
            dataset_name=self.inference_config.get(
                "eval_frame_dataset_name",
                "unknown",
            ),
            manifest_name=self.inference_config.get(
                "eval_frame_manifest_name",
                "stflow_manifest.jsonl",
            ),
            image_quality=int(
                self.inference_config.get(
                    "eval_frame_image_quality",
                    95,
                )
            ),
            export_paired_real=bool(
                self.inference_config.get(
                    "eval_frame_export_paired_real",
                    True,
                )
            ),
        )

        print(
            "[BEV_EVAL_FRAME_EXPORT] "
            "rank={} saved frames to {}".format(
                rank,
                eval_frame_export_path,
            ),
            flush=True,
        )

    @torch.no_grad()
    def preview_pipeline(
        self,
        batch: dict,
        output_path: str,
        global_step: int,
    ):
        batch_size, _, view_count = batch["vae_images"].shape[:3]

        latent_height = batch["vae_images"].shape[-2] // 8
        latent_width = batch["vae_images"].shape[-1] // 8

        latent_shape = (
            batch_size,
            int(
                self.inference_config[
                    "sequence_length_per_iteration"
                ]
            ),
            view_count,
            self.vae.config.latent_channels,
            latent_height,
            latent_width,
        )

        pipeline_output = self.autoregressive_inference_pipeline(
            latent_shape,
            batch,
            "pt",
        )

        preview_images = pipeline_output["images"]

        # ====================================================
        # 原preview视频保存逻辑
        # ====================================================

        dist_on = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        rank = (
            torch.distributed.get_rank()
            if dist_on
            else 0
        )

        all_rank_preview = bool(
            self.inference_config.get(
                "all_rank_preview",
                False,
            )
        )

        save_this_rank = self.should_save or (
            dist_on and all_rank_preview
        )

        if not save_this_rank:
            return

        preview_btvc = preview_images.unflatten(
            0,
            (
                batch_size,
                -1,
                view_count,
            ),
        )

        if not bool(
            self.inference_config[
                "generate_frames_for_reference"
            ]
        ):
            reference_frame_count = int(
                self.inference_config[
                    "reference_frame_count"
                ]
            )

            preview_btvc = torch.cat(
                [
                    batch["vae_images"][
                        :, :reference_frame_count
                    ].cpu(),
                    preview_btvc.cpu(),
                ],
                dim=1,
            )

        else:
            preview_btvc = preview_btvc.cpu()

        preview_frame_count = preview_btvc.shape[1]

        preview_tensor = (
            dwm.utils.preview.make_ctsd_preview_tensor(
                preview_btvc.flatten(0, 2),
                batch,
                self.inference_config,
            )
        )

        preview_dir = os.path.join(
            output_path,
            "preview",
        )

        os.makedirs(
            preview_dir,
            exist_ok=True,
        )

        filename = (
            "{}_{}".format(
                global_step,
                rank,
            )
            if all_rank_preview
            else str(global_step)
        )

        if preview_frame_count == 1:
            torchvision.transforms.functional.to_pil_image(
                preview_tensor
            ).save(
                os.path.join(
                    preview_dir,
                    "{}.png".format(filename),
                )
            )

        else:
            dwm.utils.preview.save_tensor_to_video(
                os.path.join(
                    preview_dir,
                    "{}.mp4".format(filename),
                ),
                "libx264",
                batch["fps"][0].item(),
                preview_tensor,
            )

        # Export after the preview image/video succeeds. The manifest line is
        # therefore the resume commit marker for all outputs of this batch.
        self._export_eval_frames(
            preview_images,
            batch,
        )

    @torch.no_grad()
    def evaluate_pipeline(
        self,
        global_step: int,
        dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
    ):
        if (
            torch.distributed.is_initialized()
            and validation_datasampler is not None
        ):
            # Keep the distributed sampling order stable across restarts.
            validation_datasampler.set_epoch(0)

        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        item_limit = int(
            self.inference_config.get(
                "evaluation_item_count",
                dataset_length,
            )
        ) // world_size

        loader_batch_size = validation_dataloader.batch_size
        if loader_batch_size is None:
            loader_batch_size = 1
        loader_batch_size = int(loader_batch_size)

        resume_count = self._prepare_eval_frame_resume(
            item_limit=item_limit,
            loader_batch_size=loader_batch_size,
        )

        if resume_count >= item_limit:
            return

        if resume_count > 0:
            dist_on = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            rank = torch.distributed.get_rank() if dist_on else 0
            print(
                "[BEV_EVAL_RESUME] "
                f"rank={rank} skip first {resume_count} local samples "
                f"and continue until {item_limit}.",
                flush=True,
            )

        seen_items = 0
        for batch_index, batch in enumerate(validation_dataloader):
            batch_size, _, view_count = batch["vae_images"].shape[:3]
            batch_start = seen_items
            batch_stop = batch_start + batch_size
            seen_items = batch_stop

            if batch_start >= item_limit:
                break

            if batch_stop <= resume_count:
                continue

            if batch_start < resume_count:
                raise RuntimeError(
                    "Resume point falls inside a validation batch: "
                    f"batch_index={batch_index}, "
                    f"batch_range=[{batch_start}, {batch_stop}), "
                    f"resume_count={resume_count}. "
                    "This should have been aligned to a batch boundary."
                )

            latent_height = batch["vae_images"].shape[-2] // 8
            latent_width = batch["vae_images"].shape[-1] // 8
            latent_shape = (
                batch_size,
                int(
                    self.inference_config[
                        "sequence_length_per_iteration"
                    ]
                ),
                view_count,
                self.vae.config.latent_channels,
                latent_height,
                latent_width,
            )
            pipeline_output = self.autoregressive_inference_pipeline(
                latent_shape,
                batch,
                "pt",
            )

            raw_fake_images = pipeline_output["images"]

            # Export only after the full sample inference succeeds.
            # One manifest line therefore acts as the resume commit marker.
            self._export_eval_frames(
                raw_fake_images,
                batch,
            )

            fake_images = raw_fake_images.unflatten(
                0,
                (
                    batch_size,
                    -1,
                    view_count,
                ),
            )
            real_start = (
                int(
                    self.inference_config[
                        "reference_frame_count"
                    ]
                )
                if not bool(
                    self.inference_config[
                        "generate_frames_for_reference"
                    ]
                )
                else 0
            )
            real_stop = real_start + fake_images.shape[1]
            if "fid" in self.metrics:
                self.metrics["fid"].update(
                    batch["vae_images"][:, real_start:real_stop]
                    .flatten(0, 2)
                    .to(self.device),
                    real=True,
                )
                self.metrics["fid"].update(
                    fake_images.flatten(0, 2),
                    real=False,
                )
            if "fvd" in self.metrics:
                self.metrics["fvd"].update(
                    einops.rearrange(
                        batch["vae_images"][:, real_start:real_stop].to(
                            self.device
                        ),
                        "b t v c h w -> (b v) t c h w",
                    ),
                    real=True,
                )
                self.metrics["fvd"].update(
                    einops.rearrange(
                        fake_images,
                        "b t v c h w -> (b v) t c h w",
                    ),
                    real=False,
                )

        message = f"Step {global_step},"
        for name, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            message += f" {name}: {value:.3f}"
            if self.should_save and self.summary is not None:
                self.summary.add_scalar(
                    f"evaluation/{name}",
                    value,
                    global_step,
                )
        if self.should_save:
            print(message, flush=True)


    def save_checkpoint(self, output_path: str, steps: int):
        if torch.distributed.is_initialized():
            options = (
                torch.distributed.checkpoint.state_dict.StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                )
            )
            model_state = (
                torch.distributed.checkpoint.state_dict.get_model_state_dict(
                    self.model_wrapper,
                    options=options,
                )
            )
        elif self.should_save:
            model_state = self.model.state_dict()
        else:
            model_state = None

        checkpoint_dir = os.path.join(output_path, "checkpoints")
        optimizer_dir = os.path.join(output_path, "optimizer")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(optimizer_dir, exist_ok=True)
        if self.should_save and model_state is not None:
            torch.save(
                model_state,
                os.path.join(checkpoint_dir, f"{steps}.pth"),
            )
        dwm.distributed.distributed_save_optimizer_state(
            self.model_wrapper,
            self.optimizer,
            optimizer_dir,
            str(steps),
        )

    def log(self, global_step: int, log_steps: int):
        if not self.loss_report_list:
            return
        keys = self.loss_report_list[0].keys()
        mean_values = {
            key: sum(item[key] for item in self.loss_report_list)
            / len(self.loss_report_list)
            for key in keys
        }
        if self.should_save:
            loss_message = ", ".join(
                f"{key}: {value:.4f}"
                for key, value in mean_values.items()
            )
            print(
                f"Step {global_step} "
                f"({self.step_duration / log_steps:.1f} s/step), "
                f"{loss_message}",
                flush=True,
            )
            if self.summary is not None:
                for key, value in mean_values.items():
                    self.summary.add_scalar(
                        f"train/{key}",
                        value,
                        global_step,
                    )
        self.loss_report_list.clear()
        self.step_duration = 0.0


def summarize_prefixes(keys) -> str:
    counts = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    if not counts:
        return "none"
    return ", ".join(
        f"{prefix}={count}"
        for prefix, count in sorted(counts.items())
    )
