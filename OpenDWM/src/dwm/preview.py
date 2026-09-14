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
    
    
    for i, batch in enumerate(preview_dataloader):
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
            batch, output_path, global_step)

        if args.export_item_config:
            with open(
                os.path.join(
                    output_path, "preview",
                    "{}_rank{}.json".format(
                        global_step,
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

        global_step += 1
        if should_log:
            print(f"preview: {global_step}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()  # 统一入口