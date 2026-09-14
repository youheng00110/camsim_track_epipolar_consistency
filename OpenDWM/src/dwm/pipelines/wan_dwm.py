
import contextlib
import math
import os
import re
import time
from typing import Optional

import diffusers
import dwm
import dwm.common
import dwm.distributed
import dwm.functional
import dwm.utils.preview
import einops
import safetensors.torch
import torch
import torch.distributed
import torch.distributed.checkpoint.state_dict
import torch.utils.tensorboard
import torchvision
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def normalize_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = image_tensor.float()
    if image_tensor.max().item() > 1.0:
        image_tensor = image_tensor / 255.0
    image_tensor = image_tensor * 2.0 - 1.0
    return image_tensor.clamp(-1.0, 1.0)


def postprocess_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = image_tensor.float()
    image_tensor = (image_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
    return image_tensor


def collapse_text_item(text_item) -> str:
    if isinstance(text_item, str):
        return text_item
    if isinstance(text_item, (list, tuple)):
        parts = []
        for item in text_item:
            value = collapse_text_item(item)
            if isinstance(value, str) and len(value) > 0:
                parts.append(value)
        return ", ".join(parts)
    return ""


def encode_prompt_t5(
    text_encoder,
    tokenizer,
    prompts,
    device,
    max_sequence_length: int,
    dtype: torch.dtype,
):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    prompt_embed_list = []
    for i in range(prompt_embeds.shape[0]):
        valid_len = int(seq_lens[i].item())
        valid_tokens = prompt_embeds[i, :valid_len]
        if valid_len < max_sequence_length:
            pad = valid_tokens.new_zeros(
                (max_sequence_length - valid_len, valid_tokens.shape[1])
            )
            valid_tokens = torch.cat([valid_tokens, pad], dim=0)
        prompt_embed_list.append(valid_tokens)

    prompt_embeds = torch.stack(prompt_embed_list, dim=0)
    return prompt_embeds


def get_wan_latent_stats(vae, device, dtype):
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype)
    latents_std = 1.0 / latents_std.view(1, vae.config.z_dim, 1, 1, 1)
    return latents_mean, latents_std


def encode_video_with_wan_vae(vae, image_tensor_btvc):
    batch_size, _, view_count = image_tensor_btvc.shape[:3]
    video_tensor = einops.rearrange(
        image_tensor_btvc,
        "b t v c h w -> (b v) c t h w",
    )
    posterior = vae.encode(video_tensor).latent_dist
    latents = posterior.sample()
    latents_mean, latents_std = get_wan_latent_stats(vae, latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    latents = einops.rearrange(
        latents,
        "(b v) c t h w -> b t v c h w",
        b=batch_size,
        v=view_count,
    )
    return latents


def encode_video_with_wan_vae_mode(vae, image_tensor_btvc):
    batch_size, _, view_count = image_tensor_btvc.shape[:3]
    video_tensor = einops.rearrange(
        image_tensor_btvc,
        "b t v c h w -> (b v) c t h w",
    )
    posterior = vae.encode(video_tensor).latent_dist
    latents = posterior.mode()
    latents_mean, latents_std = get_wan_latent_stats(vae, latents.device, latents.dtype)
    latents = (latents - latents_mean) * latents_std
    latents = einops.rearrange(
        latents,
        "(b v) c t h w -> b t v c h w",
        b=batch_size,
        v=view_count,
    )
    return latents


def decode_video_with_wan_vae(vae, latents_btvc):
    batch_size, _, view_count = latents_btvc.shape[:3]
    latents = einops.rearrange(
        latents_btvc,
        "b t v c h w -> (b v) c t h w",
    )
    latents_mean, latents_std = get_wan_latent_stats(vae, latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    video = einops.rearrange(
        video,
        "(b v) c t h w -> b t v c h w",
        b=batch_size,
        v=view_count,
    )
    return video


def _extract_model_prediction(model_result):
    if isinstance(model_result, torch.Tensor):
        return model_result
    if hasattr(model_result, "sample"):
        return model_result.sample
    if isinstance(model_result, (list, tuple)):
        if len(model_result) == 0:
            raise ValueError("Empty model output.")
        first = model_result[0]
        if isinstance(first, torch.Tensor):
            return first
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], torch.Tensor):
            return first[0]
    raise TypeError(f"Unsupported model output type: {type(model_result)!r}")


class WanDWM:

    @staticmethod
    def fm_compute_density_for_timestep_sampling(
        weighting_scheme: str,
        size,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
    ):
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=size, device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=size, device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=size, device="cpu")
        return u

    @staticmethod
    def fm_get_sigmas(noise_scheduler, timestep_indices, n_dim, device, dtype):
        timestep_indices_cpu = timestep_indices.to("cpu")
        sigmas = noise_scheduler.sigmas[timestep_indices_cpu].to(device=device, dtype=dtype)
        while sigmas.ndim < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas
    
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            return safetensors.torch.load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict):
        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod."):]
            normalized_state_dict[new_key] = value
        return normalized_state_dict

    def _load_backbone_from_pretrained(self, pretrained_model_name_or_path: str):
        backbone = diffusers.WanTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=self.model_dtype,
        )
        backbone_state_dict = backbone.state_dict()
        del backbone

        missing_keys, unexpected_keys = self.model.load_state_dict(
            backbone_state_dict,
            strict=False,
        )

        if self.should_save and self.common_config.get("print_load_state_info", True):
            print("[load pretrained backbone]")
            print(f"missing keys: {missing_keys}")
            print(f"unexpected keys: {unexpected_keys}")

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
        model_load_state_args: dict = {},
        metrics: dict = {},
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

        self.generator = torch.Generator()
        if "generator_seed" in self.config:
            self.generator.manual_seed(self.config["generator_seed"])
        else:
            self.generator.seed()

        self.model_dtype = model_dtype or torch.bfloat16
        self.distribution_framework = self.common_config.get("distribution_framework", "ddp")

        if (
            torch.distributed.is_initialized()
            and self.distribution_framework == "fsdp"
            and torch.cuda.is_available()
        ):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)

        self.model = model.to(device=self.device, dtype=self.model_dtype)
        self.model_wrapper = self.model

        self._load_backbone_from_pretrained(pretrained_model_name_or_path)

        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

        text_encoder_load_args = self.common_config.get("text_encoder_load_args", {})

        self.tokenizer = transformers.T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.text_encoder = transformers.UMT5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            **text_encoder_load_args,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.to(self.device)

        self.vae = diffusers.AutoencoderKLWan.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.device)

        num_train_timesteps = self.common_config.get("wan_num_train_timesteps", 1000)
        shift = float(self.common_config.get("wan_shift", 5.0))
        self.train_scheduler = diffusers.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
        scheduler_cls = getattr(diffusers, "UniPCMultistepScheduler")
        self.test_scheduler = scheduler_cls.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        
        if resume_from is not None:
            state_dict = self.load_state(
                os.path.join(output_path, "checkpoints", f"{resume_from}.pth")
            )
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            state_dict = self._normalize_state_dict_keys(state_dict)
            self.model.load_state_dict(state_dict, strict=False)

        elif model_checkpoint_path is not None:
            model_load_state_args = dict(model_load_state_args or {})
            raw_state = self.load_state(model_checkpoint_path)

            if isinstance(raw_state, dict) and "model" in raw_state:
                state_dict = raw_state["model"]
            elif isinstance(raw_state, dict) and "state_dict" in raw_state:
                state_dict = raw_state["state_dict"]
            else:
                state_dict = raw_state

            state_dict = self._normalize_state_dict_keys(state_dict)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict,
                **model_load_state_args,
            )

            if self.should_save and self.common_config.get("print_load_state_info", False):
                print("[load finetune checkpoint]")
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
                        print(f"{name} is frozen.")
            if self.should_save:
                print(f"{frozen_module_count} modules are frozen.")

        if self.should_save:
            param_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print("{:.1f} M parameters are trainable.".format(param_count / 1e6))

        self.loss_report_list = []
        self.step_duration = 0.0
        self.optimizer = None
        self.lr_scheduler = None

        if torch.distributed.is_initialized():
            if self.distribution_framework == "ddp":
                self.model_wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[int(os.environ["LOCAL_RANK"])],
                    **self.common_config["ddp_wrapper_settings"],
                )
            elif self.distribution_framework == "fsdp":
                ignored_modules = None
                if "fsdp_ignored_module_pattern" in self.common_config:
                    pattern = re.compile(self.common_config["fsdp_ignored_module_pattern"])
                    ignored_named_modules = []
                    for name, module in self.model.named_modules():
                        if pattern.match(name) is not None:
                            ignored_named_modules.append((name, module))
                    ignored_modules = [x[1] for x in ignored_named_modules]

                    if self.should_save:
                        print("{} modules are ignored by FSDP.".format(len(ignored_named_modules)))
                        print(
                            "These ignored modules are {}.".format(
                                [x[0] for x in ignored_named_modules]
                            )
                        )
                        
                fsdp_settings = dict(self.common_config["ddp_wrapper_settings"])
                if self.device.type == "cuda" and "device_id" not in fsdp_settings:
                    fsdp_settings["device_id"] = self.device

                self.model_wrapper = FSDP(
                    self.model,
                    ignored_modules=ignored_modules,
                    **fsdp_settings,
                )
                
        self.summary = None
        if self.should_save and output_path is not None:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log")
            )

        self.optimizer = (
            dwm.common.create_instance_from_config(
                config["optimizer"],
                params=self.model_wrapper.parameters(),
            )
            if "optimizer" in config
            else None
        )

        self.lr_scheduler = (
            dwm.common.create_instance_from_config(
                config["lr_scheduler"],
                optimizer=self.optimizer,
            )
            if "lr_scheduler" in config and self.optimizer is not None
            else None
        )

        self.metrics = {}
        for key, value in (metrics or {}).items():
            self.metrics[key] = value.to(self.device)

        if resume_from is not None and self.optimizer is not None:
            try:
                dwm.distributed.distributed_load_optimizer_state(
                    self.model_wrapper,
                    self.optimizer,
                    os.path.join(output_path, "optimizer"),
                    str(resume_from),
                )
            except Exception as exc:
                if self.should_save:
                    print(f"[resume optimizer] skipped: {exc}")

    def get_loss_coef(self, name):
        if "loss_coef_dict" in self.training_config:
            return self.training_config["loss_coef_dict"].get(name, 1.0)
        return 1.0

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        if self.device.type == "cuda" and self.model_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.model_dtype)
        return contextlib.nullcontext()

    def get_latent_sequence_length(self, sequence_length):
        temporal_scale = getattr(self.vae.config, "scale_factor_temporal", 4)
        return (sequence_length - 1) // temporal_scale + 1

    def build_prompt_batch(
        self,
        batch,
        view_count: int,
        text_condition_mask=None,
        negative_prompt_text: str = "",
    ):
        prefix = "A Video from driving Vehicle Onboard Camera."
        prompts = []
        clip_text = batch["clip_text"]
        batch_size = len(clip_text)

        for i in range(batch_size):
            if text_condition_mask is not None and not bool(text_condition_mask[i]):
                for _ in range(view_count):
                    prompts.append(negative_prompt_text)
                continue

            sample_text = clip_text[i]
            mid_t = sample_text[len(sample_text) // 2]
            for v in range(view_count):
                text = collapse_text_item(mid_t[v])
                prompts.append(f"{prefix} {text}")

        return prompts
    
    def get_conditions(
        self,
        latent_shape,
        batch: dict,
        text_condition_mask=None,
        _3dbox_condition_mask=None,
        hdmap_condition_mask=None,
        explicit_view_modeling_mask=None,
        negative_prompt_text: str = "",
        do_classifier_free_guidance: bool = False,
    ):
        del do_classifier_free_guidance

        batch_size, latent_sequence_length, view_count = latent_shape[:3]
        #############原逻辑用pts，但我们的batch里没有pts。故用vae
        sequence_length = batch["vae_images"].shape[1]

        prompt_batch = self.build_prompt_batch(
            batch,
            view_count=view_count,
            text_condition_mask=text_condition_mask,
            negative_prompt_text=negative_prompt_text,
        )
        max_sequence_length = self.common_config.get("wan_max_sequence_length", 512)
        encoder_hidden_states = encode_prompt_t5(
            self.text_encoder,
            self.tokenizer,
            prompt_batch,
            self.device,
            max_sequence_length,
            self.model_dtype,
        )

        condition_on_all_frames = self.common_config.get("condition_on_all_frames", True)
        uncondition_image_color = self.common_config.get("uncondition_image_color", 0)
        condition_image_list = []

        if "3dbox_images" in batch:
            bbox_images = batch["3dbox_images"].to(self.device)
            if not condition_on_all_frames:
                bbox_images = bbox_images[:, :1]
            if _3dbox_condition_mask is not None:
                bbox_images[
                    _3dbox_condition_mask.logical_not().to(self.device)
                ] = uncondition_image_color
            condition_image_list.append(bbox_images)

        if "hdmap_images" in batch:
            hdmap_images = batch["hdmap_images"].to(self.device)
            if not condition_on_all_frames:
                hdmap_images = hdmap_images[:, :1]
            if hdmap_condition_mask is not None:
                hdmap_images[
                    hdmap_condition_mask.logical_not().to(self.device)
                ] = uncondition_image_color
            condition_image_list.append(hdmap_images)

        condition_image_tensor = None
        if len(condition_image_list) > 0:
            condition_image_tensor = torch.cat(condition_image_list, dim=-3)
            condition_image_tensor = einops.rearrange(
                condition_image_tensor,
                "b t v c h w -> (b v) c t h w",
            )

        camera_intrinsics_norm = None
        camera2referego = None
        if self.common_config.get("explicit_view_modeling", False):
            if "ego_transforms" not in batch:
                ego_transforms = torch.eye(
                    4,
                    device=self.device,
                    dtype=batch["camera_transforms"].dtype,
                ).unsqueeze(0).unsqueeze(1).unsqueeze(2)
                ego_transforms = ego_transforms.expand(
                    batch["camera_transforms"].shape[0],
                    batch["camera_transforms"].shape[1],
                    batch["camera_transforms"].shape[2],
                    -1,
                    -1,
                )
            else:
                ego_transforms = batch["ego_transforms"][
                    :, :, -batch["camera_transforms"].shape[2]:
                ].to(self.device)

            camera_transforms = batch["camera_transforms"].to(self.device)
            camera2world = ego_transforms @ camera_transforms
            camera2referego = torch.linalg.inv(
                ego_transforms[:, 0, 0].unsqueeze(1).unsqueeze(2)
            ) @ camera2world

            camera_intrinsics_norm = batch["camera_intrinsics"].clone().to(self.device)
            image_size = batch["image_size"].to(self.device)

            camera_intrinsics_norm[..., 0, 0] = (
                camera_intrinsics_norm[..., 0, 0] / image_size[..., 0]
            )
            camera_intrinsics_norm[..., 1, 1] = (
                camera_intrinsics_norm[..., 1, 1] / image_size[..., 1]
            )
            camera_intrinsics_norm[..., 0, 2] = (
                camera_intrinsics_norm[..., 0, 2] / image_size[..., 0]
            )
            camera_intrinsics_norm[..., 1, 2] = (
                camera_intrinsics_norm[..., 1, 2] / image_size[..., 1]
            )

            if "is_uncalibrated" in batch:
                eye3 = torch.eye(
                    3,
                    device=self.device,
                    dtype=camera_intrinsics_norm.dtype,
                )
                eye4 = torch.eye(
                    4,
                    device=self.device,
                    dtype=camera2referego.dtype,
                )
                camera_intrinsics_norm[batch["is_uncalibrated"]] = eye3
                camera2referego[batch["is_uncalibrated"]] = eye4

            if explicit_view_modeling_mask is not None:
                eye3 = torch.eye(
                    3,
                    device=self.device,
                    dtype=camera_intrinsics_norm.dtype,
                )
                eye4 = torch.eye(
                    4,
                    device=self.device,
                    dtype=camera2referego.dtype,
                )
                camera_intrinsics_norm[
                    explicit_view_modeling_mask.logical_not().to(self.device)
                ] = eye3
                camera2referego[
                    explicit_view_modeling_mask.logical_not().to(self.device)
                ] = eye4

        crossview_attention_mask = None
        if "crossview_mask" in batch:
            crossview_attention_mask = batch["crossview_mask"].to(self.device)

        result = {
            "encoder_hidden_states": encoder_hidden_states,
            "condition_image_tensor": condition_image_tensor,
            "disable_temporal": torch.tensor(
                [self.common_config.get("disable_temporal", False)],
                device=self.device,
            ).repeat(encoder_hidden_states.shape[0]),
            "crossview_attention_mask": crossview_attention_mask,
            "camera_intrinsics_norm": camera_intrinsics_norm,
            "camera2referego": camera2referego,
            "view_count": int(view_count),
        }

        if latent_sequence_length != sequence_length:
            pre = 1 if sequence_length % 2 == 1 else 0
            stride = (sequence_length - pre) // (latent_sequence_length - pre)

            dense_keys = {
                "condition_image_tensor",
                "camera_intrinsics_norm",
                "camera2referego",
            }

            for key, value in result.items():
                if key in dense_keys:
                    continue
                if (
                    value is not None
                    and hasattr(value, "ndim")
                    and value.ndim > 1
                    and value.shape[1] == sequence_length
                ):
                    result[key] = torch.cat(
                        [value[:, :pre], value[:, pre::stride]],
                        dim=1,
                    )

        return result
    
    
    def sample_reference_frame_counts(
                self,
                batch_size: int,
                frame_sequence_length: int,
            ):
        token_count_distribution = self.training_config.get(
            "reference_token_count_distribution",
            {
                "0": 0.4,
                "1": 0.2,
                "2": 0.2,
                "3": 0.2,
            },
        )

        temporal_scale = int(getattr(self.vae.config, "scale_factor_temporal", 4))

        valid_token_choices = []
        valid_token_probs = []

        for key, value in token_count_distribution.items():
            token_count = int(key)
            if token_count == 0:
                frame_count = 0
            else:
                frame_count = 1 + (token_count - 1) * temporal_scale

            if frame_count <= frame_sequence_length and float(value) > 0:
                valid_token_choices.append(token_count)
                valid_token_probs.append(float(value))

        if len(valid_token_choices) == 0:
            sampled_frame_count = 0
        else:
            token_choice_tensor = torch.tensor(valid_token_choices, dtype=torch.long)
            token_prob_tensor = torch.tensor(valid_token_probs, dtype=torch.float32)
            token_prob_tensor = token_prob_tensor / token_prob_tensor.sum().clamp(min=1e-8)

            sampled_token_index = torch.multinomial(
                token_prob_tensor,
                num_samples=1,
                replacement=True,
                generator=self.generator,
            )
            sampled_token_count = int(token_choice_tensor[sampled_token_index].item())

            if sampled_token_count == 0:
                sampled_frame_count = 0
            else:
                sampled_frame_count = 1 + (sampled_token_count - 1) * temporal_scale

        sampled_reference_frame_counts = torch.full(
            (batch_size,),
            sampled_frame_count,
            dtype=torch.long,
        )

        return sampled_reference_frame_counts


    def make_training_latent_input(
        self,
        batch: dict,
        latents: torch.Tensor,
        noise: torch.Tensor,
    ):
        batch_size, frame_sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_sequence_length = latents.shape[1]

        sampled_reference_frame_counts_cpu = self.sample_reference_frame_counts(
            batch_size=batch_size,
            frame_sequence_length=frame_sequence_length,
        )
        reference_frame_count = int(sampled_reference_frame_counts_cpu[0].item())
        reference_token_count = self.get_latent_sequence_length(reference_frame_count)

        sampled_reference_token_counts_cpu = torch.full(
            (batch_size,),
            reference_token_count,
            dtype=torch.long,
        )

        condition_latents = torch.zeros(
            latents.shape,
            device=self.device,
            dtype=self.model_dtype,
        )

        reference_frame_indicator = torch.zeros(
            batch_size,
            latent_sequence_length,
            view_count,
            dtype=torch.bool,
            device=self.device,
        )

        if reference_token_count > 0:
            reference_images = batch["vae_images"][:, :reference_frame_count]
            reference_images = normalize_image_tensor(reference_images.to(self.device))
            reference_latents = encode_video_with_wan_vae_mode(self.vae, reference_images)

            reference_token_count = min(
                int(reference_latents.shape[1]),
                int(latent_sequence_length),
            )

            condition_latents[:, :reference_token_count] = reference_latents[:, :reference_token_count].to(
                device=self.device,
                dtype=self.model_dtype,
            )
            reference_frame_indicator[:, :reference_token_count] = True

        if (
            "reference_frame_scale_std" in self.training_config
            or "reference_frame_offset_std" in self.training_config
        ) and reference_token_count > 0:
            perturb_ratio = float(
                self.training_config.get("reference_frame_perturb_ratio", 0.6)
            )

            apply_reference_perturb = (
                torch.rand((batch_size,), generator=self.generator) < perturb_ratio
            ).to(self.device)

            reference_scale = torch.ones(
                (batch_size, latent_sequence_length, 1, 1, 1, 1),
                device=self.device,
                dtype=condition_latents.dtype,
            )
            reference_offset = torch.zeros(
                (batch_size, latent_sequence_length, 1, 1, 1, 1),
                device=self.device,
                dtype=condition_latents.dtype,
            )

            if "reference_frame_scale_std" in self.training_config:
                reference_scale = (
                    torch.randn(
                        (batch_size, latent_sequence_length, 1, 1, 1, 1),
                        generator=self.generator,
                        device="cpu",
                        dtype=torch.float32,
                    ).to(device=self.device, dtype=condition_latents.dtype)
                    * float(self.training_config["reference_frame_scale_std"])
                    + 1.0
                )

            if "reference_frame_offset_std" in self.training_config:
                reference_offset = (
                    torch.randn(
                        (batch_size, latent_sequence_length, 1, 1, 1, 1),
                        generator=self.generator,
                        device="cpu",
                        dtype=torch.float32,
                    ).to(device=self.device, dtype=condition_latents.dtype)
                    * float(self.training_config["reference_frame_offset_std"])
                )

            apply_reference_perturb = apply_reference_perturb.view(batch_size, 1, 1, 1, 1, 1)

            reference_scale = torch.where(
                apply_reference_perturb,
                reference_scale,
                torch.ones_like(reference_scale),
            )
            reference_offset = torch.where(
                apply_reference_perturb,
                reference_offset,
                torch.zeros_like(reference_offset),
            )

            condition_latents[:, :reference_token_count] = (
                condition_latents[:, :reference_token_count]
                * reference_scale[:, :reference_token_count]
                + reference_offset[:, :reference_token_count]
            )

        timestep_indices_cpu = (
            self.fm_compute_density_for_timestep_sampling(
                weighting_scheme=self.training_config.get("weighting_scheme", "logit_normal"),
                size=(batch_size, latent_sequence_length, view_count),
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            * self.train_scheduler.config.num_train_timesteps
        ).long().clamp(
            0,
            self.train_scheduler.config.num_train_timesteps - 1,
        )

        sigmas = self.fm_get_sigmas(
            self.train_scheduler,
            timestep_indices_cpu,
            n_dim=latents.ndim,
            dtype=latents.dtype,
            device=latents.device,
        )
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents

        timestep_scalar_btv = self.train_scheduler.timesteps[timestep_indices_cpu].to(self.device)
        timestep_scalar_btv = torch.where(
            reference_frame_indicator,
            torch.zeros_like(timestep_scalar_btv),
            timestep_scalar_btv,
        )

        latent_model_input = torch.where(
            reference_frame_indicator.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            condition_latents.to(noisy_latents.dtype),
            noisy_latents,
        )

        patch_t = int(self.model.config.patch_size[0])
        patch_h = int(self.model.config.patch_size[1])
        patch_w = int(self.model.config.patch_size[2])

        if patch_t != 1:
            raise ValueError(
                f"Current train timestep expansion assumes patch_size_t == 1, got {patch_t}."
            )

        patch_token_count_per_time = (
            (latents.shape[-2] // patch_h) *
            (latents.shape[-1] // patch_w)
        )

        timestep_tokens = timestep_scalar_btv.permute(0, 2, 1).contiguous().view(
            batch_size * view_count,
            latent_sequence_length,
            1,
        )
        timestep_tokens = timestep_tokens.repeat(
            1,
            1,
            patch_token_count_per_time,
        ).view(
            batch_size * view_count,
            latent_sequence_length * patch_token_count_per_time,
        )
        timestep_tokens = timestep_tokens.to(self.model_dtype)

        return (
            latent_model_input,
            timestep_tokens,
            reference_frame_indicator,
            sampled_reference_frame_counts_cpu,
            sampled_reference_token_counts_cpu,
        )


    def train_step(self, batch: dict, global_step: int):
        self.model_wrapper.train()
        t0 = time.time()

        batch_size, frame_sequence_length, view_count = batch["vae_images"].shape[:3]

        image_tensor = normalize_image_tensor(batch["vae_images"].to(self.device))
        latents = encode_video_with_wan_vae(self.vae, image_tensor)

        noise = torch.randn(
            latents.shape,
            device=self.device,
            dtype=latents.dtype,
        )
        target = noise - latents

        latent_model_input, timestep_tokens, reference_frame_indicator, sampled_reference_frame_counts, sampled_reference_token_counts = self.make_training_latent_input(
            batch=batch,
            latents=latents,
            noise=noise,
        )

        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator)
            < self.training_config.get("text_prompt_condition_ratio", 1.0)
        ).tolist()
        _3dbox_condition_mask = (
            torch.rand((batch_size,), generator=self.generator)
            < self.training_config.get("3dbox_condition_ratio", 1.0)
        ).to(self.device)
        hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator)
            < self.training_config.get("hdmap_condition_ratio", 1.0)
        ).to(self.device)

        explicit_view_modeling_mask = None
        if self.common_config.get("explicit_view_modeling", False):
            explicit_view_modeling_mask = (
                torch.rand((batch_size,), generator=self.generator)
                < self.training_config.get("explicit_view_modeling_ratio", 1.0)
            ).to(self.device)

        with self.get_autocast_context():
            model_conditions = self.get_conditions(
                latents.shape,
                batch,
                text_condition_mask=text_condition_mask,
                _3dbox_condition_mask=_3dbox_condition_mask,
                hdmap_condition_mask=hdmap_condition_mask,
                explicit_view_modeling_mask=explicit_view_modeling_mask,
                do_classifier_free_guidance=False,
            )

            latent_model_input = einops.rearrange(
                latent_model_input.to(dtype=self.model_dtype),
                "b t v c h w -> (b v) c t h w",
            )

            model_result = self.model_wrapper(
                hidden_states=latent_model_input,
                timestep=timestep_tokens,
                return_dict=False,
                **model_conditions,
            )
            model_pred = _extract_model_prediction(model_result)
            model_pred = einops.rearrange(
                model_pred,
                "(b v) c t h w -> b t v c h w",
                b=batch_size,
                v=view_count,
            )

            gen_mask = (~reference_frame_indicator).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
                device=model_pred.device,
                dtype=model_pred.dtype,
            )

            sq_error = (model_pred.float() - target.float()).pow(2)
            sq_error = sq_error * gen_mask.float()

            valid_count = gen_mask.sum() * model_pred.shape[-1] * model_pred.shape[-2] * model_pred.shape[-3]
            valid_count = valid_count.clamp(min=1.0)

            loss = sq_error.sum() / valid_count
            loss = loss * self.get_loss_coef("sd")

        self.loss_report_list.append(
            {
                "loss": loss.item(),
                "sd_loss": loss.item(),
                "ref_frame_mean": sampled_reference_frame_counts.float().mean().item(),
                "ref_token_mean": sampled_reference_token_counts.float().mean().item(),
                "ref_zero_ratio": (sampled_reference_token_counts == 0).float().mean().item(),
            }
        )

        loss.backward()

        should_optimize = (
            "gradient_accumulation_steps" not in self.training_config
            or (global_step + 1) % self.training_config["gradient_accumulation_steps"] == 0
        )

        if should_optimize and self.optimizer is not None:
            if "max_norm_for_grad_clip" in self.training_config:
                if (
                    torch.distributed.is_initialized()
                    and self.distribution_framework == "fsdp"
                ):
                    self.model_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"]
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_norm_for_grad_clip"],
                    )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.step_duration += time.time() - t0
        return loss

    @torch.no_grad()
    def inference_pipeline(
        self,
        latent_shape,
        batch,
        output_type="image",
        reference_latents=None,
    ):
        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        guidance_scale = float(self.inference_config.get("guidance_scale", 1.0))
        batch_size, latent_sequence_length, view_count = latent_shape[:3]
        bs_view = batch_size * view_count
        latent_channels = latent_shape[3]
        latent_height = latent_shape[4]
        latent_width = latent_shape[5]

        condition_latents = torch.zeros(
            (
                bs_view,
                latent_channels,
                latent_sequence_length,
                latent_height,
                latent_width,
            ),
            device=self.device,
            dtype=self.model_dtype,
        )

        reference_latent_count = 0
        if reference_latents is not None and reference_latents.shape[1] > 0:
            reference_latents_flat = einops.rearrange(
                reference_latents,
                "b t v c h w -> (b v) c t h w",
            ).to(device=self.device, dtype=self.model_dtype)
            reference_latent_count = min(
                int(reference_latents_flat.shape[2]),
                int(latent_sequence_length),
            )
            condition_latents[:, :, :reference_latent_count] = reference_latents_flat[
                :, :, :reference_latent_count
            ]

        first_frame_mask = torch.ones(
            bs_view,
            1,
            latent_sequence_length,
            latent_height,
            latent_width,
            device=self.device,
            dtype=self.model_dtype,
        )
        if reference_latent_count > 0:
            first_frame_mask[:, :, :reference_latent_count] = 0

        latents = torch.randn(
            (
                bs_view,
                latent_channels,
                latent_sequence_length,
                latent_height,
                latent_width,
            ),
            generator=self.generator,
            device="cpu",
            dtype=self.model_dtype,
        ).to(device=self.device)

        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"],
            self.device,
        )

        model_conditions = self.get_conditions(
            latent_shape,
            batch,
            do_classifier_free_guidance=False,
        )

        model_conditions_uncond = None
        if do_classifier_free_guidance:
            model_conditions_uncond = self.get_conditions(
                latent_shape,
                batch,
                text_condition_mask=[False for _ in range(batch_size)],
                do_classifier_free_guidance=True,
            )

        patch_h = self.model.config.patch_size[1]
        patch_w = self.model.config.patch_size[2]

        for t in self.test_scheduler.timesteps:
            if reference_latent_count > 0:
                latent_model_input = (1 - first_frame_mask) * condition_latents + first_frame_mask * latents
                temp_ts = (first_frame_mask[0, 0][:, ::patch_h, ::patch_w] * t).flatten()
                timestep_tokens = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = latents
                seq_len = latent_sequence_length * (latent_height // patch_h) * (latent_width // patch_w)
                timestep_tokens = t.expand(bs_view, seq_len)

            latent_model_input = latent_model_input.to(dtype=self.model_dtype)

            with self.get_autocast_context():
                model_result = self.model_wrapper(
                    hidden_states=latent_model_input,
                    timestep=timestep_tokens,
                    return_dict=False,
                    **model_conditions,
                )
                model_pred = _extract_model_prediction(model_result)

                if do_classifier_free_guidance:
                    model_result_uncond = self.model_wrapper(
                        hidden_states=latent_model_input,
                        timestep=timestep_tokens,
                        return_dict=False,
                        **model_conditions_uncond,
                    )
                    model_pred_uncond = _extract_model_prediction(model_result_uncond)
                    model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)

            latents = self.test_scheduler.step(
                model_pred,
                t,
                latents,
                return_dict=False,
            )[0]

        if reference_latent_count > 0:
            latents = (1 - first_frame_mask) * condition_latents + first_frame_mask * latents

        final_latents = latents
        result = {
            "latents": einops.rearrange(
                final_latents,
                "(b v) c t h w -> b t v c h w",
                b=batch_size,
                v=view_count,
            )
        }

        if output_type != "latent":
            video = decode_video_with_wan_vae(
                self.vae,
                result["latents"].to(dtype=self.vae.dtype),
            )
            result["images"] = postprocess_image_tensor(video)

        return result

    @torch.no_grad()
    def autoregressive_inference_pipeline(
        self,
        batch: dict,
        output_type="image",
    ):
        total_frame_count = int(batch["pts"].shape[1])

        window_length = int(
            self.inference_config.get(
                "sequence_length_per_iteration",
                total_frame_count,
            )
        )
        reference_frame_count = int(
            self.inference_config.get("reference_frame_count", 0)
        )
        autoregressive_stride = int(
            self.inference_config.get(
                "autoregressive_stride",
                max(window_length - reference_frame_count, 1),
            )
        )

        if window_length <= 0:
            raise ValueError(f"window_length must be > 0, got {window_length}.")
        if reference_frame_count < 0:
            raise ValueError(
                f"reference_frame_count must be >= 0, got {reference_frame_count}."
            )
        if autoregressive_stride <= 0:
            raise ValueError(
                f"autoregressive_stride must be > 0, got {autoregressive_stride}."
            )

        use_reference = reference_frame_count > 0

        if use_reference and reference_frame_count >= window_length:
            raise ValueError(
                "reference_frame_count must be < window_length when reference is used, "
                f"but got R={reference_frame_count}, L={window_length}."
            )

        if use_reference and autoregressive_stride + reference_frame_count > window_length:
            raise ValueError(
                "Invalid AR setting: need S + R <= L so that the next window's "
                "reference [S, S+R) lies inside the previous window [0, L). "
                f"Got S={autoregressive_stride}, R={reference_frame_count}, L={window_length}."
            )

        exception_for_take_sequence = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence",
            [],
        )

        internal_output_type = "image"
        if (not use_reference) and output_type == "latent":
            internal_output_type = "latent"

        frame_cache = None
        output_video = None
        if use_reference or output_type != "latent":
            sample_video = batch["vae_images"][:, :1].float()
            if sample_video.max().item() > 1.0:
                sample_video = sample_video / 255.0
            sample_video = sample_video.clamp(0.0, 1.0).to(self.device)

            frame_cache = torch.zeros(
                (
                    sample_video.shape[0],
                    total_frame_count,
                    sample_video.shape[2],
                    sample_video.shape[3],
                    sample_video.shape[4],
                    sample_video.shape[5],
                ),
                device=self.device,
                dtype=sample_video.dtype,
            )
            output_video = frame_cache

        generated_until = 0
        last_iteration_output = None
        iteration_index = 0

        if use_reference:
            init_reference_images = batch["vae_images"][:, :reference_frame_count].to(
                self.device
            ).float()
            if init_reference_images.max().item() > 1.0:
                init_reference_images = init_reference_images / 255.0
            init_reference_images = init_reference_images.clamp(0.0, 1.0)

            frame_cache[:, :reference_frame_count] = init_reference_images
            generated_until = reference_frame_count

        while True:
            window_start_frame = iteration_index * autoregressive_stride
            window_end_frame = window_start_frame + window_length

            if window_end_frame > total_frame_count:
                break

            current_reference_latents = None
            if use_reference:
                if iteration_index == 0:
                    reference_images = batch["vae_images"][:, :reference_frame_count].to(
                        self.device
                    ).float()
                    if reference_images.max().item() > 1.0:
                        reference_images = reference_images / 255.0
                    reference_images = reference_images.clamp(0.0, 1.0)
                else:
                    required_reference_end = window_start_frame + reference_frame_count
                    if required_reference_end > generated_until:
                        raise ValueError(
                            "AR schedule asks for reference frames that are not yet cached. "
                            f"window_start={window_start_frame}, "
                            f"required_reference_end={required_reference_end}, "
                            f"generated_until={generated_until}."
                        )

                    reference_images = frame_cache[
                        :,
                        window_start_frame:required_reference_end,
                    ]
                    if reference_images.shape[1] != reference_frame_count:
                        raise ValueError(
                            "Invalid reference length from frame cache. "
                            f"Expected {reference_frame_count}, "
                            f"got {reference_images.shape[1]}."
                        )

                reference_images = normalize_image_tensor(reference_images.to(self.device))
                current_reference_latents = encode_video_with_wan_vae_mode(
                    self.vae,
                    reference_images,
                )

            iteration_batch = {
                key: (
                    value
                    if key in exception_for_take_sequence
                    else dwm.functional.take_sequence_clip(
                        value,
                        window_start_frame,
                        window_end_frame,
                    )
                )
                for key, value in batch.items()
            }

            latent_shape = (
                batch["vae_images"].shape[0],
                self.get_latent_sequence_length(window_length),
                batch["vae_images"].shape[2],
                self.vae.config.z_dim,
                batch["vae_images"].shape[-2] // self.vae.config.scale_factor_spatial,
                batch["vae_images"].shape[-1] // self.vae.config.scale_factor_spatial,
            )

            iteration_output = self.inference_pipeline(
                latent_shape=latent_shape,
                batch=iteration_batch,
                output_type=internal_output_type,
                reference_latents=current_reference_latents,
            )
            last_iteration_output = iteration_output

            if frame_cache is not None:
                chunk_images = iteration_output.get("images")
                if chunk_images is None:
                    decoded_video = decode_video_with_wan_vae(
                        self.vae,
                        iteration_output["latents"].to(dtype=self.vae.dtype),
                    )
                    chunk_images = postprocess_image_tensor(decoded_video)
                chunk_images = chunk_images.to(self.device)

                if chunk_images.shape[1] != window_length:
                    raise ValueError(
                        "Decoded chunk length does not match window_length. "
                        f"decoded={chunk_images.shape[1]}, window_length={window_length}. "
                        "For Wan temporal VAE AR, use full windows only."
                    )

                local_write_start = reference_frame_count if use_reference else 0
                global_write_start = window_start_frame + local_write_start
                global_write_end = window_end_frame

                if local_write_start < window_length:
                    frame_cache[:, global_write_start:global_write_end] = chunk_images[
                        :,
                        local_write_start:window_length,
                    ]

                generated_until = max(generated_until, global_write_end)

            iteration_index += 1

        result = {}

        if output_type != "latent" and output_video is not None:
            result["images"] = output_video[:, :generated_until]

        if last_iteration_output is not None:
            result["latents"] = last_iteration_output.get("latents")

        return result

    @torch.no_grad()
    def preview_pipeline(
        self,
        batch: dict,
        output_path: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        self.model_wrapper.eval()

        total_frame_count = int(batch["vae_images"].shape[1])
        sequence_length_per_iteration = int(
            self.inference_config.get(
                "sequence_length_per_iteration",
                total_frame_count,
            )
        )
        ar_stride = int(
            self.inference_config.get(
                "autoregressive_stride",
                4,
            )
        )
        use_ar = total_frame_count > (sequence_length_per_iteration + ar_stride)

        if use_ar:
            pipeline_output = self.autoregressive_inference_pipeline(
                batch=batch,
                output_type="image",
            )
        else:
            latent_shape = (
                batch["vae_images"].shape[0],
                self.get_latent_sequence_length(total_frame_count),
                batch["vae_images"].shape[2],
                self.vae.config.z_dim,
                batch["vae_images"].shape[-2] // self.vae.config.scale_factor_spatial,
                batch["vae_images"].shape[-1] // self.vae.config.scale_factor_spatial,
            )

            generate_frames_for_reference = self.inference_config.get(
                "generate_frames_for_reference",
                False,
            )

            reference_latents = None
            if not generate_frames_for_reference:
                reference_frame_count = int(self.inference_config.get("reference_frame_count", 0))
                if reference_frame_count > 0:
                    reference_images = batch["vae_images"][:, :reference_frame_count]
                    if reference_images.shape[1] > 0:
                        reference_images = normalize_image_tensor(reference_images.to(self.device))
                        reference_latents = encode_video_with_wan_vae_mode(
                            self.vae,
                            reference_images,
                        )

            pipeline_output = self.inference_pipeline(
                latent_shape=latent_shape,
                batch=batch,
                output_type="image",
                reference_latents=reference_latents,
            )

        if output_path is None:
            output_path = self.output_path

        if output_path is not None and (
            self.should_save or (
                torch.distributed.is_initialized()
                and self.inference_config.get("all_rank_preview", False)
            )
        ):
            os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "preview_sampletok"), exist_ok=True)
            filename = str(0 if global_step is None else global_step)

            preview_tensor = dwm.utils.preview.make_ctsd_preview_wan(
                pipeline_output["images"], batch, self.inference_config
            )
            if total_frame_count == 1:
                image_output_path = os.path.join(output_path, "preview", f"{filename}.png")
                torchvision.transforms.functional.to_pil_image(preview_tensor).save(image_output_path)
            else:
                seq_label = None
                if "seq_label" in batch:
                    seq_label = batch["seq_label"]
                    if isinstance(seq_label, (list, tuple)):
                        seq_label = seq_label[0]
                    seq_label = str(seq_label)

                video_dir = os.path.join(output_path, "preview")
                txt_dir = os.path.join(output_path, "preview_sampletok")
                if seq_label is not None:
                    video_dir = os.path.join(video_dir, seq_label)
                    txt_dir = os.path.join(txt_dir, seq_label)

                os.makedirs(video_dir, exist_ok=True)
                os.makedirs(txt_dir, exist_ok=True)

                video_output_path = os.path.join(video_dir, f"{filename}.mp4")
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", batch["fps"][0].item(), preview_tensor
                )

                if "segment_samples" in batch:
                    tok_list = [t[0] if isinstance(t, tuple) else t for t in batch["segment_samples"]]
                    txt_path = os.path.join(txt_dir, f"{filename}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(tok_list) + "\n")

            pipeline_output["saved_output_path"] = output_path

        return pipeline_output

    def save_checkpoint(self, output_path: Optional[str], steps: int):
        if output_path is None:
            output_path = self.output_path

        if output_path is None:
            return

        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
            model_state_dict = torch.distributed.checkpoint.state_dict.get_model_state_dict(
                self.model_wrapper,
                options=options,
            )
        elif self.should_save:
            model_state_dict = self.model.state_dict()
        else:
            model_state_dict = None

        os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)

        if self.should_save and model_state_dict is not None:
            torch.save(
                model_state_dict,
                os.path.join(output_path, "checkpoints", f"{steps}.pth"),
            )

        # dwm.distributed.distributed_save_optimizer_state(
        #     self.model_wrapper, self.optimizer,
        #     os.path.join(output_path, "optimizer"), str(steps))


    def log(self, global_step: int, log_steps: int):
        if len(self.loss_report_list) == 0:
            self.step_duration = 0.0
            return

        if self.should_save:
            if isinstance(self.loss_report_list[0], dict):
                loss_values = {}
                for key in self.loss_report_list[0].keys():
                    loss_values[key] = sum(item[key] for item in self.loss_report_list) / len(self.loss_report_list)

                loss_message = ", ".join(
                    f"{key}: {value:.4f}" for key, value in loss_values.items()
                )
                print(
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step,
                        self.step_duration / max(log_steps, 1),
                        loss_message,
                    )
                )

                if self.summary is not None:
                    for key, value in loss_values.items():
                        self.summary.add_scalar(f"train/{key}", value, global_step)
            else:
                loss_value = sum(self.loss_report_list) / len(self.loss_report_list)
                print(
                    "Step {} ({:.1f} s/step), loss: {:.4f}".format(
                        global_step,
                        self.step_duration / max(log_steps, 1),
                        loss_value,
                    )
                )
                if self.summary is not None:
                    self.summary.add_scalar("train/Loss", loss_value, global_step)

        self.loss_report_list.clear()
        self.step_duration = 0.0

    @torch.no_grad()
    def evaluate_pipeline(
        self,
        global_step: int,
        dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
    ):
        del dataset_length

        self.model_wrapper.eval()

        if torch.distributed.is_initialized() and validation_datasampler is not None:
            validation_datasampler.set_epoch(0)

        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        iteration_count = (
            self.inference_config["evaluation_item_count"] // world_size
            if "evaluation_item_count" in self.inference_config
            else None
        )

        for i, batch in enumerate(validation_dataloader):
            batch_size, total_frame_count, view_count = batch["vae_images"].shape[:3]
            if iteration_count is not None and i * batch_size >= iteration_count:
                break

            sequence_length_per_iteration = int(
                self.inference_config.get(
                    "sequence_length_per_iteration",
                    total_frame_count,
                )
            )
            ar_stride = int(
                self.inference_config.get(
                    "autoregressive_stride",
                    4,
                )
            )
            use_ar = total_frame_count > (sequence_length_per_iteration + ar_stride)

            if use_ar:
                pipeline_output = self.autoregressive_inference_pipeline(
                    batch=batch,
                    output_type="image",
                )
            else:
                latent_shape = (
                    batch_size,
                    self.get_latent_sequence_length(total_frame_count),
                    view_count,
                    self.vae.config.z_dim,
                    batch["vae_images"].shape[-2] // self.vae.config.scale_factor_spatial,
                    batch["vae_images"].shape[-1] // self.vae.config.scale_factor_spatial,
                )

                reference_latents = None
                if not self.inference_config.get("generate_frames_for_reference", False):
                    reference_frame_count = int(
                        self.inference_config.get("reference_frame_count", 0)
                    )
                    if reference_frame_count > 0:
                        reference_images = batch["vae_images"][:, :reference_frame_count]
                        if reference_images.shape[1] > 0:
                            reference_images = normalize_image_tensor(
                                reference_images.to(self.device)
                            )
                            reference_latents = encode_video_with_wan_vae_mode(
                                self.vae,
                                reference_images,
                            )

                pipeline_output = self.inference_pipeline(
                    latent_shape=latent_shape,
                    batch=batch,
                    output_type="image",
                    reference_latents=reference_latents,
                )

            if len(self.metrics) == 0:
                continue

            fake_images = pipeline_output["images"].to(self.device).float()

            real_images = batch["vae_images"].to(self.device).float()
            if real_images.max().item() > 1.0:
                real_images = real_images / 255.0
            real_images = real_images.clamp(0.0, 1.0)

            start = 0
            if not self.inference_config.get("generate_frames_for_reference", False):
                start = int(self.inference_config.get("reference_frame_count", 0))

            if start >= fake_images.shape[1]:
                continue

            fake_eval_images = fake_images[:, start:]
            real_eval_images = real_images[:, start:start + fake_eval_images.shape[1]]

            if fake_eval_images.shape[1] == 0 or real_eval_images.shape[1] == 0:
                continue

            if "fid" in self.metrics:
                self.metrics["fid"].update(
                    real_eval_images.flatten(0, 2),
                    real=True,
                )
                self.metrics["fid"].update(
                    fake_eval_images.flatten(0, 2),
                    real=False,
                )

            if "fvd" in self.metrics:
                self.metrics["fvd"].update(
                    einops.rearrange(
                        real_eval_images,
                        "b t v c h w -> (b v) t c h w",
                    ),
                    real=True,
                )
                self.metrics["fvd"].update(
                    einops.rearrange(
                        fake_eval_images,
                        "b t v c h w -> (b v) t c h w",
                    ),
                    real=False,
                )

        if len(self.metrics) == 0:
            return

        text = f"Step {global_step},"
        for key, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            text += f" {key}: {value:.3f}"
            if self.should_save and self.summary is not None:
                self.summary.add_scalar(f"evaluation/{key}", value, global_step)

        if self.should_save:
            print(text)

