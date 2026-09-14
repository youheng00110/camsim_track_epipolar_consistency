import os

import einops
import torch
import torchvision

import dwm.utils.preview
from dwm.pipelines.bev import BEVPipeline


class BEVPreviewPipeline(BEVPipeline):
    def _export_eval_frames(
        self,
        output_images: torch.Tensor,
        batch: dict,
    ):
        eval_frame_export_path = self.inference_config.get(
            "eval_frame_export_path",
            None,
        )
        if eval_frame_export_path is None:
            return

        dist_on = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if dist_on else 0
        all_rank_preview = bool(
            self.inference_config.get("all_rank_preview", False)
        )
        save_this_rank = self.should_save or (dist_on and all_rank_preview)
        if not save_this_rank:
            return

        if all_rank_preview and dist_on:
            eval_frame_export_path = os.path.join(
                eval_frame_export_path,
                "rank_{:02d}".format(rank),
            )

        dwm.utils.preview.save_ctsd_eval_frames_for_preview(
            output_images,
            batch,
            self.inference_config,
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
            int(self.inference_config["sequence_length_per_iteration"]),
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

        self._export_eval_frames(
            preview_images,
            batch,
        )

        dist_on = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if dist_on else 0
        all_rank_preview = bool(
            self.inference_config.get("all_rank_preview", False)
        )
        save_this_rank = self.should_save or (dist_on and all_rank_preview)
        if not save_this_rank:
            return

        preview_btvc = preview_images.unflatten(
            0,
            (batch_size, -1, view_count),
        )
        if not bool(
            self.inference_config["generate_frames_for_reference"]
        ):
            reference_frame_count = int(
                self.inference_config["reference_frame_count"]
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
        preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
            preview_btvc.flatten(0, 2),
            batch,
            self.inference_config,
        )

        preview_dir = os.path.join(output_path, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        filename = (
            "{}_{}".format(global_step, rank)
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

        for batch_index, batch in enumerate(validation_dataloader):
            batch_size, _, view_count = batch["vae_images"].shape[:3]
            if batch_index * batch_size >= item_limit:
                break

            latent_height = batch["vae_images"].shape[-2] // 8
            latent_width = batch["vae_images"].shape[-1] // 8
            latent_shape = (
                batch_size,
                int(self.inference_config["sequence_length_per_iteration"]),
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

            self._export_eval_frames(
                raw_fake_images,
                batch,
            )

            fake_images = raw_fake_images.unflatten(
                0,
                (batch_size, -1, view_count),
            )
            real_start = (
                int(self.inference_config["reference_frame_count"])
                if not bool(
                    self.inference_config["generate_frames_for_reference"]
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

        message = "Step {},".format(global_step)
        for name, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            message += " {}: {:.3f}".format(name, value)

            if self.should_save and self.summary is not None:
                self.summary.add_scalar(
                    "evaluation/{}".format(name),
                    value,
                    global_step,
                )

        if self.should_save:
            print(message, flush=True)
