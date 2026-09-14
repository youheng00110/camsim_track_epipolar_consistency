from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(images_root: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}

    if not images_root.is_dir():
        return image_map

    for current_root, _, file_names in os.walk(images_root):
        current_path = Path(current_root)

        for file_name in file_names:
            image_path = current_path / file_name

            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            relative_key = image_path.relative_to(images_root).as_posix()
            image_map[relative_key] = image_path

    return image_map


def make_overlay(rgb_image: Image.Image, box_image: Image.Image) -> Image.Image:
    rgb = rgb_image.convert("RGB")
    box = box_image.convert("RGB").resize(
        rgb.size,
        Image.Resampling.BILINEAR,
    )

    rgb_array = __import__("numpy").asarray(rgb).astype("float32")
    box_array = __import__("numpy").asarray(box).astype("float32")

    box_strength = box_array.max(axis=2)
    mask = box_strength > 12

    output = rgb_array.copy()
    alpha = 0.90
    output[mask] = (
        (1.0 - alpha) * output[mask]
        + alpha * box_array[mask]
    )

    output = output.clip(0, 255).astype("uint8")
    return Image.fromarray(output)


def fit_with_label(
    image: Image.Image,
    title: str,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (tile_width, tile_height + 34),
        (20, 20, 20),
    )

    fitted = ImageOps.contain(
        image.convert("RGB"),
        (tile_width, tile_height),
        method=Image.Resampling.LANCZOS,
    )

    paste_x = (tile_width - fitted.width) // 2
    paste_y = (tile_height - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, tile_height + 10),
        title,
        fill=(255, 255, 255),
        font=font,
    )

    return canvas


parser = argparse.ArgumentParser(
    description=(
        "Randomly sample matching generated images and 3D-box projection "
        "images from one OpenDWM preview method, then save one montage."
    )
)
parser.add_argument(
    "--method-root",
    type=Path,
    required=True,
    help="Method directory containing rank_XX and box_projection/rank_XX.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Default: <method-root>/box_alignment_check",
)
parser.add_argument(
    "--samples",
    type=int,
    default=12,
    help="Number of matching samples in the montage.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=3407,
    help="Random seed.",
)
parser.add_argument(
    "--tile-width",
    type=int,
    default=420,
    help="Width of each generated/box/overlay tile.",
)
parser.add_argument(
    "--tile-height",
    type=int,
    default=240,
    help="Height of each generated/box/overlay tile.",
)
parser.add_argument(
    "--ranks",
    nargs="*",
    default=["rank_00", "rank_01", "rank_02", "rank_03"],
    help="Rank directories to search.",
)
parser.add_argument(
    "--camera",
    default=None,
    help="Optional camera filter, for example CAM_03.",
)
parser.add_argument(
    "--frame",
    default=None,
    help="Optional frame filter, for example t005.",
)
args = parser.parse_args()

method_root = args.method_root.expanduser().resolve()

if not method_root.is_dir():
    raise FileNotFoundError(
        f"Method directory does not exist: {method_root}"
    )

output_dir = (
    args.output_dir.expanduser().resolve()
    if args.output_dir is not None
    else method_root / "box_alignment_check"
)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[START] method_root={method_root}", flush=True)

all_candidates: list[dict[str, str]] = []

for rank_name in args.ranks:
    rgb_root = method_root / rank_name / "images"
    box_root = method_root / "box_projection" / rank_name / "images"

    print(
        f"[SCAN] {rank_name}\n"
        f"  rgb={rgb_root}\n"
        f"  box={box_root}",
        flush=True,
    )

    rgb_map = collect_images(rgb_root)
    box_map = collect_images(box_root)

    if not rgb_map:
        print(
            f"[SKIP] No RGB images found under {rgb_root}",
            flush=True,
        )
        continue

    if not box_map:
        print(
            f"[SKIP] No box images found under {box_root}",
            flush=True,
        )
        continue

    common_keys = sorted(set(rgb_map) & set(box_map))

    for relative_key in common_keys:
        key_parts = Path(relative_key).parts
        camera_name = Path(relative_key).stem

        if args.camera and camera_name != args.camera:
            continue

        if args.frame and args.frame not in key_parts:
            continue

        all_candidates.append(
            {
                "rank": rank_name,
                "relative_key": relative_key,
                "rgb_path": str(rgb_map[relative_key]),
                "box_path": str(box_map[relative_key]),
            }
        )

    print(
        f"[FOUND] {rank_name}: "
        f"rgb={len(rgb_map)} "
        f"box={len(box_map)} "
        f"common={len(common_keys)} "
        f"selected_total={len(all_candidates)}",
        flush=True,
    )

if not all_candidates:
    raise RuntimeError(
        "No matching generated/box image paths were found. "
        "Check whether both sides use the same relative layout below images/."
    )

sample_count = min(args.samples, len(all_candidates))
random_generator = random.Random(args.seed)
selected_samples = random_generator.sample(
    all_candidates,
    sample_count,
)

tile_width = args.tile_width
tile_height = args.tile_height
header_height = 42
row_height = tile_height + 34
canvas_width = tile_width * 3
canvas_height = header_height + row_height * sample_count

montage = Image.new(
    "RGB",
    (canvas_width, canvas_height),
    (12, 12, 12),
)
draw = ImageDraw.Draw(montage)
font = ImageFont.load_default()

draw.text(
    (8, 14),
    "Generated image | Box projection | Overlay",
    fill=(255, 255, 255),
    font=font,
)

saved_records: list[dict[str, str]] = []

for sample_index, sample in enumerate(selected_samples):
    rgb_path = Path(sample["rgb_path"])
    box_path = Path(sample["box_path"])

    rgb_image = Image.open(rgb_path).convert("RGB")
    box_image = Image.open(box_path).convert("RGB")
    overlay_image = make_overlay(rgb_image, box_image)

    sample_title = (
        f"{sample_index:02d}  "
        f"{sample['rank']}/{sample['relative_key']}"
    )

    rgb_tile = fit_with_label(
        rgb_image,
        f"RGB | {sample_title}",
        tile_width,
        tile_height,
    )
    box_tile = fit_with_label(
        box_image,
        f"BOX | {sample_title}",
        tile_width,
        tile_height,
    )
    overlay_tile = fit_with_label(
        overlay_image,
        f"OVERLAY | {sample_title}",
        tile_width,
        tile_height,
    )

    y0 = header_height + sample_index * row_height
    montage.paste(rgb_tile, (0, y0))
    montage.paste(box_tile, (tile_width, y0))
    montage.paste(overlay_tile, (tile_width * 2, y0))

    saved_records.append(
        {
            "rank": sample["rank"],
            "relative_key": sample["relative_key"],
            "rgb_path": str(rgb_path),
            "box_path": str(box_path),
        }
    )

    print(
        f"[{sample_index + 1:02d}/{sample_count:02d}] "
        f"{sample['rank']}/{sample['relative_key']}",
        flush=True,
    )

montage_path = output_dir / "comparison_montage.jpg"
montage.save(montage_path, quality=95)

index_path = output_dir / "comparison_samples.json"
index_path.write_text(
    json.dumps(
        {
            "method_root": str(method_root),
            "seed": args.seed,
            "candidate_count": len(all_candidates),
            "sample_count": sample_count,
            "samples": saved_records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"[DONE] montage={montage_path}", flush=True)
print(f"[DONE] index={index_path}", flush=True)
