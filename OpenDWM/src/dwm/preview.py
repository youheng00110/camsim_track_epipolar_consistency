import os
import cv2
import torch
import numpy as np
if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
    import debugpy

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        debugpy.listen(("0.0.0.0", 9876))
        print(
            "[debugpy] listening on 0.0.0.0:9876, waiting for VS Code to attach...",
            flush=True,
        )
        debugpy.wait_for_client()

import argparse
import json
import torch
import dwm.common


def resolve_preview_item_limit(inference_config, dataset_length, world_size=1):
    """Resolve the per-rank preview limit from a global target.

    ``preview_item_count`` is the explicit preview target.  The legacy
    ``evaluation_item_count`` remains a fallback for older configurations.
    A ceiling split avoids silently dropping a remainder when the target is
    not divisible by the distributed world size.
    """
    target_global = int(
        inference_config.get(
            "preview_item_count",
            inference_config.get("evaluation_item_count", dataset_length),
        )
    )
    if target_global < 0:
        raise ValueError("preview item count must be non-negative")
    world_size = max(1, int(world_size))
    target_per_rank = (target_global + world_size - 1) // world_size
    return target_global, target_per_rank


def customize_text(clip_text, preview_config):

    # text
    if preview_config["text"] is not None:
        text_config = preview_config["text"]

        if text_config["type"] == "add":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"] + k
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "replace":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"]
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "template":
            time = text_config["time"]
            weather = text_config["weather"]
            new_clip_text = \
                [
                    [
                        [
                            text_config["template"][time][weather][idx][0]
                            for idx, k in enumerate(j)
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        else:
            raise NotImplementedError(
                f"{text_config['type']}has not been implemented yet.")

        return new_clip_text

    else:

        return clip_text


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "-pc", "--preview-config-path", default=None, type=str,
        help="The config for preview setting")
    parser.add_argument(
        "-eic", "--export-item-config", default=False, type=bool,
        help="The flag to export the item config as JSON")
    return parser


def main():  # ========= 你要的 main 函数 + debug 在这里 =========
    # ========= 下面是你原来的全部代码，原封不动放进来 =========
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.preview_config_path is not None:
        with open(args.preview_config_path, "r", encoding="utf-8") as f:
            preview_config = json.load(f)
    else:
        preview_config = None

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])

    preview_datasampler = None
    if "preview_dataloader" in config:
        preview_loader_kwargs = dwm.common.instantiate_config(
            config["preview_dataloader"]
        )
        preview_loader_kwargs.pop("shuffle", None)

        if ddp:
            preview_datasampler = torch.utils.data.distributed.DistributedSampler(
                validation_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
                drop_last=False,
            )
            preview_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                **preview_loader_kwargs,
                sampler=preview_datasampler,
            )
            preview_datasampler.set_epoch(0)
        else:
            preview_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                **preview_loader_kwargs,
                shuffle=False,
            )
    else:
        preview_dataloader = None

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    export_batch_except = ["vae_images"]
    output_path = args.output_path
    global_step = 0

    inference_config = config["pipeline"].get("inference_config", {})
    preview_resume_enabled = bool(
        inference_config.get("eval_frame_resume", False)
    )
    preview_resume_count = 0
    preview_item_limit = None
    if preview_resume_enabled:
        if not hasattr(pipeline, "_prepare_eval_frame_resume"):
            raise RuntimeError(
                "eval_frame_resume=true requires a pipeline with "
                "_prepare_eval_frame_resume()."
            )
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        preview_target_global, preview_item_limit = resolve_preview_item_limit(
            inference_config,
            len(validation_dataset),
            world_size,
        )
        if should_log:
            print(
                "[PREVIEW_RESUME] target_global={} target_per_rank={} world_size={}".format(
                    preview_target_global,
                    preview_item_limit,
                    world_size,
                ),
                flush=True,
            )
        loader_batch_size = preview_dataloader.batch_size
        if loader_batch_size is None:
            loader_batch_size = 1
        preview_resume_count = pipeline._prepare_eval_frame_resume(
            item_limit=preview_item_limit,
            loader_batch_size=int(loader_batch_size),
        )

    seen_items = 0
    for i, batch in enumerate(preview_dataloader):
        batch_size = int(batch["vae_images"].shape[0])
        batch_start = seen_items
        batch_stop = batch_start + batch_size
        seen_items = batch_stop

        if (
            preview_item_limit is not None
            and batch_start >= preview_item_limit
        ):
            break

        if preview_resume_enabled and batch_stop <= preview_resume_count:
            continue

        if preview_resume_enabled and batch_start < preview_resume_count:
            raise RuntimeError(
                "Preview resume point falls inside a dataloader batch: "
                "batch_index={}, batch_range=[{}, {}), resume_count={}."
                .format(
                    i,
                    batch_start,
                    batch_stop,
                    preview_resume_count,
                )
            )

        output_step = batch_start if preview_resume_enabled else global_step
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if "pts" in batch:
            preview_id = batch["pts"][0, 0, 0].item()
        else:
            preview_id = -1

        print(
            "[PREVIEW_DIST] rank={}/{} i={} preview_id={} vae_shape={}".format(
                rank,
                world_size,
                i,
                preview_id,
                tuple(batch["vae_images"].shape) if "vae_images" in batch else None,
            ),
            flush=True,
        )

        ####调试#####################
        print("\n========== DEBUG BATCH ==========")
        print("keys:", batch.keys())
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}")
        print("image_size sample:", batch["image_size"][0, 0, 0])
        #print("K_before:\n", batch["camera_intrinsics_before_resize_crop"][0, 0, 0])
        print("K_after:\n", batch["camera_intrinsics"][0, 0, 0])        
        #############检查crossview#############
        if "crossview_mask" in batch:
            print("\n--- crossview_mask sample ---")
            print(batch["crossview_mask"][0].int())  # 打印第一个
        #############检查相机数###########
        if "vae_images" in batch:
            print("\n--- camera check ---")
            print("vae_images shape:", batch["vae_images"].shape)
        ############检查cliptext#######
        if "clip_text" in batch:
            print("\n--- clip_text check ---")
            print(type(batch["clip_text"]))
            print("example:", batch["clip_text"][0][0])
        ################################
        if ddp:
            torch.distributed.barrier()

        if preview_config is not None:
            new_clip_text = customize_text(batch["clip_text"], preview_config)
            batch["clip_text"] = new_clip_text

        pipeline.preview_pipeline(
            batch, output_path, output_step)

        if args.export_item_config:
            with open(
                os.path.join(
                    output_path, "preview",
                    "{}_rank{}.json".format(
                        output_step,
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )),
                "w", encoding="utf-8"
            ) as f:
                json.dump({
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k not in export_batch_except
                }, f, indent=4)

        global_step = batch_stop if preview_resume_enabled else global_step + 1
        if should_log:
            print(f"preview: {global_step}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()  # 统一入口
