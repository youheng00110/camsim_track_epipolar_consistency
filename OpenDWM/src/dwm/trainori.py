import argparse
import dwm.common
import json
import os
import time
import torch

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
from tqdm import tqdm
from dwm.utils.sampler import VariableVideoBatchSampler, SameDatasetDistributedSampler
def check_tensor_finite(name, x, step, local_rank):
    if not torch.is_tensor(x):
        return

    if torch.isfinite(x).all():
        return

    finite_mask = torch.isfinite(x)
    bad_count = finite_mask.logical_not().sum().item()

    x_float = x.detach().float()
    finite_values = x_float[finite_mask]

    if finite_values.numel() > 0:
        finite_min = finite_values.min().item()
        finite_max = finite_values.max().item()
    else:
        finite_min = None
        finite_max = None

    raise RuntimeError(
        "[rank={}] [step={}] non-finite in {}, "
        "shape={}, dtype={}, bad_count={}, finite_min={}, finite_max={}".format(
            local_rank,
            step,
            name,
            tuple(x.shape),
            x.dtype,
            bad_count,
            finite_min,
            finite_max,
        )
    )


def check_batch_finite(batch, step, local_rank):
    keys = [
        "vae_images",
        "camera_intrinsics",
        "camera_transforms",
        "ego_transforms",
        "image_size",
        "fps",
        "crossview_mask",
    ]

    for k in keys:
        if k in batch:
            check_tensor_finite(k, batch[k], step, local_rank)

    if "image_size" in batch:
        image_size = batch["image_size"].float()
        if not (image_size > 0).all():
            raise RuntimeError(
                "[rank={}] [step={}] non-positive image_size: {}".format(
                    local_rank,
                    step,
                    image_size,
                )
            )

    if "camera_intrinsics" in batch:
        K = batch["camera_intrinsics"].float()
        if not (K[..., 0, 0] > 0).all():
            raise RuntimeError(
                "[rank={}] [step={}] bad fx: {}".format(
                    local_rank,
                    step,
                    K[..., 0, 0],
                )
            )
        if not (K[..., 1, 1] > 0).all():
            raise RuntimeError(
                "[rank={}] [step={}] bad fy: {}".format(
                    local_rank,
                    step,
                    K[..., 1, 1],
                )
            )

    if "crossview_mask" in batch:
        m = batch["crossview_mask"].bool()
        if not m.any(dim=-1).all():
            bad_rows = m.any(dim=-1).logical_not().nonzero()
            raise RuntimeError(
                "[rank={}] [step={}] all-false crossview_mask rows: {}".format(
                    local_rank,
                    step,
                    bad_rows[:20],
                )
            )


def check_latest_loss_finite(pipeline, step, local_rank):
    if len(pipeline.loss_report_list) == 0:
        return

    latest = pipeline.loss_report_list[-1]

    if isinstance(latest, dict):
        for k, v in latest.items():
            if not torch.isfinite(torch.tensor(float(v))):
                raise RuntimeError(
                    "[rank={}] [step={}] non-finite loss {}: {}".format(
                        local_rank,
                        step,
                        k,
                        v,
                    )
                )
    else:
        if not torch.isfinite(torch.tensor(float(latest))):
            raise RuntimeError(
                "[rank={}] [step={}] non-finite loss: {}".format(
                    local_rank,
                    step,
                    latest,
                )
            )

def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "--log-steps", default=200, type=int,
        help="The step count to print log and update the tensorboard.")
    parser.add_argument(
        "--preview-steps", default=500, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--checkpointing-steps", default=10000, type=int,
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--evaluation-steps", default=10000, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--resume-from", default=None, type=int,
        help="The step to resume from")
    parser.add_argument(
        "--wandb", action="store_true",
        help="Use wandb to log the training process.")
    parser.add_argument(
        "--wandb-project", type=str, default="dwm",
        help="The wandb project name.")
    parser.add_argument(
        "--wandb-run-name", type=str, default="train",
        help="The wandb run name.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    ddp = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ["LOCAL_RANK"]) if ddp else 0


    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print("config path =", args.config_path)
    print("training_dataloader cfg =", config["training_dataloader"])
    print("has mix_config =", "mix_config" in config)   
    torch.manual_seed(config["generator_seed"])

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
    output_path = config["output_path"] if args.output_path is None else args.output_path
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=output_path, config=config,
        device=device, resume_from=args.resume_from)

    if should_log:
        print("The pipeline is loaded.")

    if args.wandb and should_save:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    # load the dataset
    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:
        if config.get("same_dataset_per_global_batch", False):
            process_group = torch.distributed.group.WORLD
            train_loader_kwargs = dwm.common.instantiate_config(
                config["training_dataloader"]
            )

            train_batch_size = train_loader_kwargs.get("batch_size", 1)

            training_datasampler = SameDatasetDistributedSampler(
                training_dataset,
                batch_size=train_batch_size,
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=config["data_shuffle"],
                seed=config["generator_seed"],
                drop_last=False,
            )

            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **train_loader_kwargs,
                sampler=training_datasampler,
            )

        elif "mix_config" in config.keys():
            process_group = torch.distributed.group.WORLD
            training_datasampler = VariableVideoBatchSampler(
                training_dataset,
                config["mix_config"],
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=config["data_shuffle"],
                seed=config["generator_seed"]
            )
            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                batch_sampler=training_datasampler)

        else:
            training_datasampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset,
                shuffle=config["data_shuffle"],
                seed=config["generator_seed"])

            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                sampler=training_datasampler)

        # make equal sample count for each process to simplify the result
        # gathering
        total_batch_size = int(os.environ["WORLD_SIZE"]) * \
            config["validation_dataloader"]["batch_size"]
        dataset_length = len(validation_dataset) // \
            total_batch_size * total_batch_size
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(0, dataset_length))
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            **dwm.common.instantiate_config(config["training_dataloader"]),
            shuffle=config["data_shuffle"])
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["preview_dataloader"])) if \
        "preview_dataloader" in config else None
    if preview_dataloader is not None:
        preview_data_iterator = iter(preview_dataloader)

    if should_log:
        print("The training dataset is loaded with {} items.".format(
            len(training_dataset)))
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    # train loop
    global_step = 0 if args.resume_from is None else args.resume_from
    for epoch in range(config["train_epochs"]):

        if ddp:
            # Fixing training data order reduces the accessed objects per rank,
            # therefore reduces the upper-bound of memory usage consumed by the
            # Python reference counting of objects.
            sampler_epoch = 0 if config.get("fix_training_data_order", False) \
                else epoch
            training_datasampler.set_epoch(sampler_epoch)

        epoch_start_time = time.time()
        loader = training_dataloader

        if should_log:
            loader = tqdm(
                training_dataloader,
                total=len(training_dataloader),
                desc="Epoch {}".format(epoch),
                dynamic_ncols=True,
                leave=True
            )

        for batch_idx, batch in enumerate(loader, start=1):
            step_start_time = time.time()
            debug_step = global_step + 1

            #check_batch_finite(batch, debug_step, local_rank)

            pipeline.train_step(batch, global_step)

            #check_latest_loss_finite(pipeline, debug_step, local_rank)

            global_step += 1

            step_time = time.time() - step_start_time
            elapsed_epoch_time = time.time() - epoch_start_time
            avg_step_time = elapsed_epoch_time / batch_idx
            remaining_steps = len(training_dataloader) - batch_idx
            eta_seconds = avg_step_time * remaining_steps

            if should_log:
                loader.set_postfix({
                    "step": global_step,
                    "step_s": "{:.2f}".format(step_time),
                    "avg_s": "{:.2f}".format(avg_step_time),
                    "eta_min": "{:.1f}".format(eta_seconds / 60.0)
                })

            # log
            if global_step % args.log_steps == 0:
                pipeline.log(global_step, args.log_steps)

            # preview
            if global_step % args.preview_steps == 0:
                if preview_dataloader is None:
                    pipeline.preview_pipeline(batch, output_path, global_step)
                else:
                    try:
                        preview_batch = next(preview_data_iterator)
                    except StopIteration:
                        preview_data_iterator = iter(preview_dataloader)
                        preview_batch = next(preview_data_iterator)

                    pipeline.preview_pipeline(
                        preview_batch, output_path, global_step)

            # save step checkpoint
            if global_step % args.checkpointing_steps == 0:
                pipeline.save_checkpoint(output_path, global_step)

            # evaluation
            if (
                args.evaluation_steps > 0 and
                global_step % args.evaluation_steps == 0
            ):
                pipeline.evaluate_pipeline(
                    global_step, len(validation_dataset),
                    validation_dataloader, validation_datasampler)

        if should_log:
            epoch_total_time = time.time() - epoch_start_time
            print(
                "Epoch {} done. total_time={:.1f} min, avg_step={:.2f} s".format(
                    epoch,
                    epoch_total_time / 60.0,
                    epoch_total_time / len(training_dataloader)
                )
            )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
