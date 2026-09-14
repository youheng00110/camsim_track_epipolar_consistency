#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
import dwm.common
import dwm.utils.view_consistency as vc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the exact epipolar selections used by BEV-PV view consistency."
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-height", type=int, default=18)
    parser.add_argument("--feature-width", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def collate_one(sample):
    return torch.utils.data.default_collate([sample])


def prepare_geometry_like_pipeline(batch, sequence_length):
    return vc.prepare_view_consistency_geometry(
        batch, sequence_length, torch.device("cpu")
    )


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


def line_segment_on_image(line, feature_height, feature_width, scale_x, scale_y):
    a, b, c = [float(v) for v in line]
    points = []
    eps = 1e-8

    def add_point(x, y):
        if (
            -1e-6 <= x <= feature_width + 1e-6
            and -1e-6 <= y <= feature_height + 1e-6
        ):
            px = x * scale_x
            py = y * scale_y
            candidate = (px, py)
            if all(
                abs(candidate[0] - old[0]) > 1e-4
                or abs(candidate[1] - old[1]) > 1e-4
                for old in points
            ):
                points.append(candidate)

    if abs(b) > eps:
        add_point(0.0, -c / b)
        add_point(float(feature_width), -(a * feature_width + c) / b)
    if abs(a) > eps:
        add_point(-c / a, 0.0)
        add_point(-(b * feature_height + c) / a, float(feature_height))

    if len(points) < 2:
        return None
    return points[0], points[1]


def draw_pair(
    image_a,
    image_b,
    query_index,
    positive_indices,
    line,
    feature_height,
    feature_width,
    title_a,
    title_b,
):
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")

    width, height = image_a.size
    if image_b.size != (width, height):
        raise ValueError(
            f"Paired images must have same size, got {image_a.size} and {image_b.size}"
        )

    scale_x = width / float(feature_width)
    scale_y = height / float(feature_height)

    draw_a = ImageDraw.Draw(image_a, "RGBA")
    draw_b = ImageDraw.Draw(image_b, "RGBA")

    # Query patch.
    qbox = patch_box(query_index, feature_width, scale_x, scale_y)
    draw_a.rectangle(qbox, outline=(255, 0, 0, 255), width=4)
    draw_a.rectangle(qbox, fill=(255, 0, 0, 55))

    # Positive epipolar-band candidates.
    for index in positive_indices:
        box = patch_box(index, feature_width, scale_x, scale_y)
        draw_b.rectangle(box, fill=(255, 215, 0, 65))
        draw_b.rectangle(box, outline=(255, 165, 0, 150), width=1)

    segment = line_segment_on_image(
        line,
        feature_height,
        feature_width,
        scale_x,
        scale_y,
    )
    if segment is not None:
        draw_b.line([segment[0], segment[1]], fill=(0, 255, 255, 255), width=3)

    header_h = 34
    canvas = Image.new("RGB", (width * 2, height + header_h), "white")
    canvas.paste(image_a.convert("RGB"), (0, header_h))
    canvas.paste(image_b.convert("RGB"), (width, header_h))
    header = ImageDraw.Draw(canvas)
    header.text((8, 9), title_a, fill="black")
    header.text((width + 8, 9), title_b, fill="black")
    return canvas


def main():
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    # The dataset config uses global_state (for example nuscenes_fs).
    for key, value in config.get("global_state", {}).items():
        # debug config intentionally removed device_mesh
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)

    dataset = dwm.common.create_instance_from_config(config["training_dataset"])
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} outside dataset length {len(dataset)}"
        )

    batch = collate_one(dataset[args.sample_index])
    if "vae_images" not in batch:
        raise KeyError("training sample has no vae_images")

    batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
    if batch_size != 1:
        raise ValueError(f"This debug tool expects B=1, got {batch_size}")

    training_config = config["pipeline"]["training_config"]
    selection_result = vc.sample_view_consistency_selection(
        batch=batch,
        training_config=training_config,
        generator=generator,
        device=torch.device("cpu"),
    )
    if selection_result is None:
        raise RuntimeError("No view-consistency pairs were sampled.")

    selection, selection_cpu = selection_result
    camera_intrinsics_norm, camera2referego = prepare_geometry_like_pipeline(
        batch,
        sequence_length,
    )

    feature_height = int(args.feature_height)
    feature_width = int(args.feature_width)
    band_width = float(
        training_config.get("view_consistency_epipolar_band_width", 2.5)
    )
    lower_half_start_ratio = float(
        training_config.get("view_consistency_lower_half_start_ratio", 0.5)
    )
    dilation_kernel = int(
        training_config.get("view_consistency_box_dilation_kernel", 31)
    )
    max_fg = int(
        training_config.get("view_consistency_max_foreground_patches", 128)
    )
    max_bg = int(
        training_config.get("view_consistency_max_background_patches", 128)
    )

    box_masks = None
    if "3dbox_images" in batch:
        box_masks = vc.build_selected_box_patch_masks(
            box_images=batch["3dbox_images"],
            selection=selection,
            feature_height=feature_height,
            feature_width=feature_width,
            dilation_kernel=dilation_kernel,
        )

    rows = torch.arange(feature_height).unsqueeze(1).expand(
        feature_height, feature_width
    )
    lower_half_flat = rows.ge(
        int(feature_height * lower_half_start_ratio)
    ).reshape(-1)
    target_indices = lower_half_flat.nonzero(as_tuple=False).flatten()
    target_coordinates = vc.build_patch_homogeneous_coordinates(
        target_indices,
        feature_width,
    )

    output_dir = Path(
        args.output
        or (
            Path(config["output_path"])
            / "geometry"
            / f"sample_{args.sample_index:06d}_seed_{args.seed}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_records = []
    pair_count = min(int(selection_cpu.shape[1]), int(args.max_pairs))

    print("=== VIEW CONSISTENCY GEOMETRY DEBUG ===")
    print("sample_index =", args.sample_index)
    print("seed         =", args.seed)
    print("T,V          =", sequence_length, view_count)
    print("feature_grid =", (feature_height, feature_width))
    print("band_width   =", band_width)
    print("pairs        =", selection_cpu[0, :pair_count].tolist())

    for pair_index in range(pair_count):
        time_a, view_a, time_b, view_b = selection_cpu[
            0, pair_index
        ].tolist()
        if min(time_a, view_a, time_b, view_b) < 0:
            continue

        pair_type = "crossview" if time_a == time_b else "crossframe"

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
        )

        if not torch.isfinite(fundamental).all():
            print(f"[skip] pair {pair_index}: non-finite F")
            continue

        if box_masks is not None:
            foreground_a = box_masks[0, pair_index, 0].reshape(-1)
        else:
            foreground_a = torch.zeros_like(lower_half_flat)

        query_indices, _ = vc.sample_patch_indices(
            foreground_mask_flat=foreground_a,
            valid_mask_flat=lower_half_flat,
            max_foreground_patches=max_fg,
            max_background_patches=max_bg,
            generator=generator,
            device=torch.device("cpu"),
        )
        if query_indices.numel() == 0:
            print(f"[skip] pair {pair_index}: no query patch")
            continue

        # sample_patch_indices puts foreground before background.
        query_index = int(query_indices[0].item())
        query_coordinate = vc.build_patch_homogeneous_coordinates(
            torch.tensor([query_index], dtype=torch.long),
            feature_width,
        )[0]

        line = fundamental @ query_coordinate
        denominator = torch.sqrt(
            line[0].square() + line[1].square()
        ).clamp(min=1e-6)
        distances = torch.abs(target_coordinates @ line) / denominator
        positive_mask = distances.le(band_width)
        positive_indices = target_indices[positive_mask].tolist()

        image_a = tensor_to_pil(batch["vae_images"][0, time_a, view_a])
        image_b = tensor_to_pil(batch["vae_images"][0, time_b, view_b])

        title_a = (
            f"A {pair_type}: t={time_a}, v={view_a}, "
            f"query_patch={query_index}"
        )
        title_b = (
            f"B: t={time_b}, v={view_b}, positives={len(positive_indices)}"
        )
        canvas = draw_pair(
            image_a=image_a,
            image_b=image_b,
            query_index=query_index,
            positive_indices=positive_indices,
            line=line,
            feature_height=feature_height,
            feature_width=feature_width,
            title_a=title_a,
            title_b=title_b,
        )

        filename = (
            f"pair_{pair_index:02d}_{pair_type}_"
            f"t{time_a:02d}_v{view_a}_to_t{time_b:02d}_v{view_b}.png"
        )
        canvas.save(output_dir / filename)

        pair_records.append(
            {
                "pair_index": pair_index,
                "type": pair_type,
                "time_a": time_a,
                "view_a": view_a,
                "time_b": time_b,
                "view_b": view_b,
                "query_patch": query_index,
                "positive_patch_count": len(positive_indices),
                "fundamental_matrix": fundamental.tolist(),
                "image": filename,
            }
        )
        print(
            f"[saved] {filename}: query={query_index}, "
            f"positive_count={len(positive_indices)}"
        )

    metadata = {
        "config": str(config_path),
        "sample_index": args.sample_index,
        "seed": args.seed,
        "feature_height": feature_height,
        "feature_width": feature_width,
        "epipolar_band_width": band_width,
        "selection": selection_cpu[0].tolist(),
        "pairs": pair_records,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("output_dir =", output_dir)
    print("NOTE: this stage validates geometry only; it does not load the model checkpoint.")


if __name__ == "__main__":
    main()
