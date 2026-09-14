#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import dwm.common
import dwm.utils.view_consistency as vc


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Load an existing BEV-PV checkpoint and visualize the raw "
            "layer feature immediately BEFORE the view-consistency projector."
        )
    )
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--sample-index", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--max-pairs", type=int, default=4)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


def collate_one(sample):
    return torch.utils.data.default_collate([sample])


def tensor_to_pil(image_tensor):
    image = image_tensor.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    elif image.shape[0] > 3:
        image = image[:3]
    image = image.clamp(0, 1)
    array = (
        image.permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def patch_box(flat_index, feature_width, scale_x, scale_y):
    y = int(flat_index) // int(feature_width)
    x = int(flat_index) % int(feature_width)
    return (
        x * scale_x,
        y * scale_y,
        (x + 1) * scale_x,
        (y + 1) * scale_y,
    )


def line_segment_on_image(
    line,
    feature_height,
    feature_width,
    scale_x,
    scale_y,
):
    a, b, c = [float(v) for v in line]
    eps = 1e-8
    points = []

    def add_point(x, y):
        if (
            -1e-6 <= x <= feature_width + 1e-6
            and -1e-6 <= y <= feature_height + 1e-6
        ):
            p = (x * scale_x, y * scale_y)
            if all(
                abs(p[0] - q[0]) > 1e-4
                or abs(p[1] - q[1]) > 1e-4
                for q in points
            ):
                points.append(p)

    if abs(b) > eps:
        add_point(0.0, -c / b)
        add_point(
            float(feature_width),
            -(a * feature_width + c) / b,
        )
    if abs(a) > eps:
        add_point(-c / a, 0.0)
        add_point(
            -(b * feature_height + c) / a,
            float(feature_height),
        )

    if len(points) < 2:
        return None
    return points[0], points[1]


def make_similarity_overlay(
    image,
    similarities,
    feature_height,
    feature_width,
):
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    width, height = base.size
    scale_x = width / float(feature_width)
    scale_y = height / float(feature_height)

    sim = similarities.reshape(feature_height, feature_width).float()
    lo = float(sim.min())
    hi = float(sim.max())
    denom = max(hi - lo, 1e-8)
    norm = (sim - lo) / denom

    for y in range(feature_height):
        for x in range(feature_width):
            alpha = int(190 * float(norm[y, x]))
            if alpha <= 0:
                continue
            index = y * feature_width + x
            draw.rectangle(
                patch_box(index, feature_width, scale_x, scale_y),
                fill=(255, 64, 0, alpha),
            )

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_query(image, query_index, feature_width, feature_height):
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    scale_x = width / float(feature_width)
    scale_y = height / float(feature_height)
    box = patch_box(query_index, feature_width, scale_x, scale_y)
    draw.rectangle(box, fill=(255, 0, 0, 70))
    draw.rectangle(box, outline=(255, 0, 0, 255), width=4)
    return out.convert("RGB")


def draw_epipolar_topk(
    image,
    line,
    positive_indices,
    topk_indices,
    feature_height,
    feature_width,
):
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    scale_x = width / float(feature_width)
    scale_y = height / float(feature_height)

    for index in positive_indices:
        box = patch_box(index, feature_width, scale_x, scale_y)
        draw.rectangle(box, fill=(255, 215, 0, 55))
        draw.rectangle(box, outline=(255, 165, 0, 100), width=1)

    segment = line_segment_on_image(
        line,
        feature_height,
        feature_width,
        scale_x,
        scale_y,
    )
    if segment is not None:
        draw.line(
            [segment[0], segment[1]],
            fill=(0, 255, 255, 255),
            width=3,
        )

    for rank, index in enumerate(topk_indices, start=1):
        box = patch_box(index, feature_width, scale_x, scale_y)
        draw.rectangle(box, outline=(0, 255, 0, 255), width=4)
        draw.text(
            (box[0] + 2, box[1] + 1),
            str(rank),
            fill=(0, 0, 0, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    return out.convert("RGB")


def make_triptych(image_a, image_heat, image_epi, labels):
    width, height = image_a.size
    header = 36
    canvas = Image.new(
        "RGB",
        (width * 3, height + header),
        "white",
    )
    canvas.paste(image_a, (0, header))
    canvas.paste(image_heat, (width, header))
    canvas.paste(image_epi, (width * 2, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), labels[0], fill="black")
    draw.text((width + 8, 10), labels[1], fill="black")
    draw.text((width * 2 + 8, 10), labels[2], fill="black")
    return canvas


def choose_scheduler_point(scheduler, requested_sigma):
    sigmas = scheduler.sigmas.detach().float().cpu()
    timesteps = scheduler.timesteps.detach().float().cpu()

    usable = min(int(sigmas.numel()), int(timesteps.numel()))
    if usable <= 0:
        raise RuntimeError("Training scheduler has no usable sigma/timestep entries.")

    sigmas = sigmas[:usable]
    timesteps = timesteps[:usable]
    index = int(
        torch.argmin(torch.abs(sigmas - float(requested_sigma))).item()
    )
    return index, float(sigmas[index]), float(timesteps[index])


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(config.get("device", "cuda"))

    # Reproduce train.py global-state initialization.
    for key, value in config.get("global_state", {}).items():
        dwm.common.global_state[key] = (
            dwm.common.create_instance_from_config(value)
        )

    print("=== DATASET ===", flush=True)
    dataset = dwm.common.create_instance_from_config(
        config["training_dataset"]
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} outside dataset length {len(dataset)}"
        )
    batch = collate_one(dataset[args.sample_index])

    batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
    if batch_size != 1:
        raise ValueError(f"This debug tool expects B=1, got B={batch_size}")

    dataset_tag = None
    if "dataset_tag" in batch:
        dataset_tag = int(batch["dataset_tag"].reshape(-1)[0].item())

    print("sample_index =", args.sample_index, flush=True)
    print("dataset_tag  =", dataset_tag, flush=True)
    print("T,V          =", sequence_length, view_count, flush=True)

    # Use an independent RNG for pair selection so changing the noise path
    # does not silently change the selected camera/time pairs.
    selection_generator = torch.Generator(device="cpu")
    selection_generator.manual_seed(args.seed)

    selection_result = vc.sample_view_consistency_selection(
        batch=batch,
        training_config=config["pipeline"]["training_config"],
        generator=selection_generator,
        device=device,
    )
    if selection_result is None:
        raise RuntimeError("No view-consistency pair was sampled.")

    selection, selection_cpu = selection_result
    print(
        "selection     =",
        selection_cpu[0].tolist(),
        flush=True,
    )

    # Avoid constructing evaluation metrics in this one-off debug pipeline.
    pipeline_config = dict(config["pipeline"])
    pipeline_config["metrics"] = {}

    print("=== LOAD PIPELINE / CHECKPOINT ===", flush=True)
    print(
        "checkpoint =",
        pipeline_config.get("model_checkpoint_path"),
        flush=True,
    )
    pipeline = dwm.common.create_instance_from_config(
        pipeline_config,
        output_path=None,
        config=config,
        device=device,
        resume_from=None,
    )
    pipeline.model.eval()

    projector = pipeline.model.view_consistency_projector
    if projector is None:
        raise RuntimeError("view_consistency_projector is disabled/missing.")

    captured = {}

    def capture_projector_input(module, inputs):
        if not inputs:
            raise RuntimeError("Projector pre-hook received no positional input.")
        captured["feature_pair"] = inputs[0].detach()

    hook = projector.register_forward_pre_hook(capture_projector_input)

    try:
        with torch.no_grad():
            images = batch["vae_images"].flatten(0, 2).to(device)
            image_tensor = pipeline.image_processor.preprocess(images)
            latents = pipeline.encode_images(
                image_tensor,
                use_mode=True,
            )
            latents = latents.reshape(
                batch_size,
                sequence_length,
                view_count,
                *latents.shape[1:],
            )

            scheduler_index, sigma_value, timestep_value = (
                choose_scheduler_point(
                    pipeline.train_scheduler,
                    args.sigma,
                )
            )
            print("scheduler_idx =", scheduler_index, flush=True)
            print("sigma         =", sigma_value, flush=True)
            print("timestep      =", timestep_value, flush=True)

            noise_generator = torch.Generator(device="cpu")
            noise_generator.manual_seed(args.seed + 1)
            noise = torch.randn(
                latents.shape,
                generator=noise_generator,
                dtype=latents.dtype,
                device="cpu",
            ).to(device)

            noisy_latents = (
                sigma_value * noise
                + (1.0 - sigma_value) * latents
            )

            model_timesteps = torch.full(
                (batch_size, sequence_length, view_count),
                timestep_value,
                device=device,
                dtype=torch.float32,
            )

            # Diagnostic mode: full conditioning, temporal branch enabled,
            # no random condition dropout.
            condition_keep = torch.ones(
                batch_size,
                dtype=torch.bool,
            )
            model_conditions = pipeline.prepare_model_conditions(
                batch,
                latents.shape,
                condition_keep=condition_keep,
            )
            model_conditions["disable_temporal"] = torch.zeros(
                batch_size,
                1,
                1,
                device=device,
                dtype=torch.bool,
            )
            model_conditions["view_consistency_selection"] = selection

            pipeline.model_wrapper(
                noisy_latents.to(pipeline.model_dtype),
                model_timesteps,
                **model_conditions,
            )
    finally:
        hook.remove()

    if "feature_pair" not in captured:
        raise RuntimeError(
            "Projector pre-hook did not fire. "
            "Check layer_id and view_consistency_selection."
        )

    raw_pair = captured["feature_pair"]
    pair_count = int(selection.shape[1])
    expected_first_dim = 2 * batch_size * pair_count
    if raw_pair.ndim != 4 or raw_pair.shape[0] != expected_first_dim:
        raise RuntimeError(
            "Unexpected projector input shape: "
            f"{tuple(raw_pair.shape)}, expected first dim "
            f"{expected_first_dim}."
        )

    raw_a, raw_b = raw_pair.chunk(2, dim=0)
    raw_a = raw_a.reshape(
        batch_size,
        pair_count,
        *raw_a.shape[1:],
    )
    raw_b = raw_b.reshape(
        batch_size,
        pair_count,
        *raw_b.shape[1:],
    )

    feature_channels = int(raw_a.shape[2])
    feature_height = int(raw_a.shape[3])
    feature_width = int(raw_a.shape[4])

    print("=== RAW FEATURE CAPTURED ===", flush=True)
    print(
        "projector_input =",
        tuple(raw_pair.shape),
        flush=True,
    )
    print(
        "per_pair_raw    =",
        (feature_channels, feature_height, feature_width),
        flush=True,
    )
    print(
        "NOTE: similarities below use projector INPUT only; "
        "the random projector output is ignored.",
        flush=True,
    )

    camera_intrinsics_norm, camera2referego = (
        pipeline.prepare_view_consistency_geometry(
            batch,
            sequence_length,
        )
    )

    training_config = config["pipeline"]["training_config"]
    lower_half_start_ratio = float(
        training_config.get(
            "view_consistency_lower_half_start_ratio",
            0.5,
        )
    )
    band_width = float(
        training_config.get(
            "view_consistency_epipolar_band_width",
            2.5,
        )
    )

    rows = torch.arange(feature_height).unsqueeze(1).expand(
        feature_height,
        feature_width,
    )
    lower_half_flat = rows.ge(
        int(feature_height * lower_half_start_ratio)
    ).reshape(-1)

    all_indices = torch.arange(
        feature_height * feature_width,
        dtype=torch.long,
    )
    all_coordinates = vc.build_patch_homogeneous_coordinates(
        all_indices,
        feature_width,
    )

    box_masks = None
    if "3dbox_images" in batch:
        box_masks = vc.build_selected_box_patch_masks(
            box_images=batch["3dbox_images"].to(device),
            selection=selection,
            feature_height=feature_height,
            feature_width=feature_width,
            dilation_kernel=int(
                training_config.get(
                    "view_consistency_box_dilation_kernel",
                    31,
                )
            ),
        ).cpu()

    output_dir = Path(
        args.output
        or (
            Path(config["output_path"])
            / "raw_feature_debug"
            / f"sample_{args.sample_index:06d}_seed_{args.seed}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    query_generator = torch.Generator(device="cpu")
    query_generator.manual_seed(args.seed + 2)

    records = []
    max_pairs = min(pair_count, int(args.max_pairs))

    for pair_index in range(max_pairs):
        time_a, view_a, time_b, view_b = selection_cpu[
            0, pair_index
        ].tolist()
        if min(time_a, view_a, time_b, view_b) < 0:
            continue

        pair_type = (
            "crossview"
            if time_a == time_b
            else "crossframe"
        )

        if box_masks is not None:
            foreground_a = box_masks[
                0, pair_index, 0
            ].reshape(-1)
        else:
            foreground_a = torch.zeros_like(lower_half_flat)

        query_indices, _ = vc.sample_patch_indices(
            foreground_mask_flat=foreground_a,
            valid_mask_flat=lower_half_flat,
            max_foreground_patches=int(
                training_config.get(
                    "view_consistency_max_foreground_patches",
                    128,
                )
            ),
            max_background_patches=int(
                training_config.get(
                    "view_consistency_max_background_patches",
                    128,
                )
            ),
            generator=query_generator,
            device=torch.device("cpu"),
        )
        if query_indices.numel() == 0:
            print(
                f"[skip] pair {pair_index}: no query patch",
                flush=True,
            )
            continue

        query_index = int(query_indices[0].item())
        query_is_foreground = bool(
            foreground_a[query_index].item()
        )

        feature_a = raw_a[
            0, pair_index
        ].float().cpu()
        feature_b = raw_b[
            0, pair_index
        ].float().cpu()

        flat_a = feature_a.flatten(1).transpose(0, 1)
        flat_b = feature_b.flatten(1).transpose(0, 1)

        query_feature = torch.nn.functional.normalize(
            flat_a[query_index],
            dim=0,
        )
        target_features = torch.nn.functional.normalize(
            flat_b,
            dim=1,
        )
        similarities = torch.mv(
            target_features,
            query_feature,
        )

        fundamental = vc.build_fundamental_matrix_on_feature_grid(
            camera_intrinsics_a_norm=camera_intrinsics_norm[
                0, time_a, view_a
            ],
            camera_intrinsics_b_norm=camera_intrinsics_norm[
                0, time_b, view_b
            ],
            camera2referego_a=camera2referego[
                0, time_a, view_a
            ],
            camera2referego_b=camera2referego[
                0, time_b, view_b
            ],
            feature_height=feature_height,
            feature_width=feature_width,
        ).detach().cpu()

        query_coord = vc.build_patch_homogeneous_coordinates(
            torch.tensor([query_index], dtype=torch.long),
            feature_width,
        )[0]

        line = fundamental @ query_coord
        denominator = torch.sqrt(
            line[0].square() + line[1].square()
        ).clamp(min=1e-6)
        distances = torch.abs(
            all_coordinates @ line
        ) / denominator

        positive_mask = (
            lower_half_flat
            & distances.le(band_width)
        )
        positive_indices_tensor = positive_mask.nonzero(
            as_tuple=False
        ).flatten()
        positive_indices = positive_indices_tensor.tolist()

        if positive_indices_tensor.numel() > 0:
            positive_scores = similarities.index_select(
                0,
                positive_indices_tensor,
            )
            k = min(
                int(args.topk),
                int(positive_scores.numel()),
            )
            top_local = torch.topk(
                positive_scores,
                k=k,
                largest=True,
            ).indices
            top_indices_tensor = positive_indices_tensor.index_select(
                0,
                top_local,
            )
            top_indices = top_indices_tensor.tolist()
            top_scores = similarities.index_select(
                0,
                top_indices_tensor,
            ).tolist()
        else:
            top_indices = []
            top_scores = []

        image_a = tensor_to_pil(
            batch["vae_images"][0, time_a, view_a]
        )
        image_b = tensor_to_pil(
            batch["vae_images"][0, time_b, view_b]
        )

        panel_a = draw_query(
            image_a,
            query_index,
            feature_width,
            feature_height,
        )
        panel_heat = make_similarity_overlay(
            image_b,
            similarities,
            feature_height,
            feature_width,
        )
        panel_epi = draw_epipolar_topk(
            image_b,
            line,
            positive_indices,
            top_indices,
            feature_height,
            feature_width,
        )

        labels = (
            (
                f"A {pair_type}: t={time_a},v={view_a}, "
                f"query={query_index},fg={query_is_foreground}"
            ),
            (
                f"B raw layer feature cosine heatmap "
                f"(before projector), t={time_b},v={view_b}"
            ),
            (
                f"B epipolar band + raw-feature Top-{len(top_indices)}"
            ),
        )
        canvas = make_triptych(
            panel_a,
            panel_heat,
            panel_epi,
            labels,
        )

        filename = (
            f"pair_{pair_index:02d}_{pair_type}_"
            f"t{time_a:02d}_v{view_a}_to_"
            f"t{time_b:02d}_v{view_b}_raw1536.png"
        )
        canvas.save(output_dir / filename)

        record = {
            "pair_index": pair_index,
            "type": pair_type,
            "time_a": time_a,
            "view_a": view_a,
            "time_b": time_b,
            "view_b": view_b,
            "query_patch": query_index,
            "query_is_foreground": query_is_foreground,
            "positive_patch_count": len(positive_indices),
            "topk_patch_indices": top_indices,
            "topk_cosine_scores": top_scores,
            "image": filename,
        }
        records.append(record)

        print(
            f"[saved] {filename}",
            flush=True,
        )
        print(
            f"        query={query_index} fg={query_is_foreground} "
            f"positive={len(positive_indices)}",
            flush=True,
        )
        print(
            f"        topk={list(zip(top_indices, [round(x, 4) for x in top_scores]))}",
            flush=True,
        )

    metadata = {
        "config": str(config_path),
        "checkpoint": pipeline_config.get("model_checkpoint_path"),
        "sample_index": args.sample_index,
        "dataset_tag": dataset_tag,
        "seed": args.seed,
        "requested_sigma": args.sigma,
        "actual_sigma": sigma_value,
        "timestep": timestep_value,
        "view_consistency_layer_id": int(
            pipeline.model.view_consistency_layer_id
        ),
        "raw_feature_channels": feature_channels,
        "feature_height": feature_height,
        "feature_width": feature_width,
        "selection": selection_cpu[0].tolist(),
        "pairs": records,
        "note": (
            "Cosine similarities use the raw feature captured by a "
            "forward_pre_hook on view_consistency_projector. "
            "The projector output is not used for visualization."
        ),
    }
    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print("output_dir =", output_dir, flush=True)


if __name__ == "__main__":
    main()
