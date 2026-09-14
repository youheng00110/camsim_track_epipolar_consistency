from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_method_images(
    root: Path,
    requested_methods: list[str] | None,
) -> dict[str, dict[str, Path]]:
    method_maps: dict[str, dict[str, Path]] = {}

    if requested_methods:
        method_dirs = [root / name for name in requested_methods]
    else:
        method_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    for method_dir in method_dirs:
        box_root = method_dir / "box_projection"
        if not box_root.is_dir():
            continue

        image_map: dict[str, Path] = {}
        image_dirs = sorted(
            path for path in box_root.rglob("images") if path.is_dir()
        )

        for image_dir in image_dirs:
            prefix = image_dir.relative_to(box_root).parent

            for image_path in sorted(image_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                suffix = image_path.relative_to(image_dir)
                key_path = suffix if prefix == Path(".") else prefix / suffix
                key = key_path.as_posix()

                if key in image_map:
                    raise RuntimeError(
                        f"Duplicate comparison key in {method_dir.name}: {key}\n"
                        f"  first:  {image_map[key]}\n"
                        f"  second: {image_path}"
                    )

                image_map[key] = image_path

        if image_map:
            method_maps[method_dir.name] = image_map

    if len(method_maps) < 2:
        raise RuntimeError(
            "Fewer than two methods with box_projection images were found. "
            f"Root: {root}"
        )

    return method_maps


def filter_common_keys(
    method_maps: dict[str, dict[str, Path]],
    camera: str | None,
    frame: str | None,
    video: str | None,
) -> list[str]:
    key_sets = [set(image_map) for image_map in method_maps.values()]
    common_keys = set.intersection(*key_sets)

    selected: list[str] = []
    for key in sorted(common_keys):
        parts = Path(key).parts
        filename = Path(key).name
        camera_name = Path(filename).stem

        if camera and camera_name != camera:
            continue
        if frame and frame not in parts:
            continue
        if video and video not in parts:
            continue

        selected.append(key)

    return selected


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_montage(
    key: str,
    method_maps: dict[str, dict[str, Path]],
    output_path: Path,
    columns: int,
    tile_width: int,
) -> dict:
    method_names = list(method_maps)
    loaded_images: list[tuple[str, Path, Image.Image, str]] = []

    for method_name in method_names:
        image_path = method_maps[method_name][key]
        image = Image.open(image_path).convert("RGB")
        digest = file_md5(image_path)
        loaded_images.append((method_name, image_path, image, digest))

    aspect_heights = [
        max(1, round(tile_width * image.height / image.width))
        for _, _, image, _ in loaded_images
    ]
    image_height = max(aspect_heights)
    label_height = 48
    title_height = 54
    tile_height = image_height + label_height

    columns = max(1, min(columns, len(loaded_images)))
    rows = math.ceil(len(loaded_images) / columns)
    canvas_width = columns * tile_width
    canvas_height = title_height + rows * tile_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (12, 16),
        key,
        fill=(255, 255, 255),
        font=font,
    )

    for index, (method_name, image_path, image, digest) in enumerate(loaded_images):
        row = index // columns
        column = index % columns
        x0 = column * tile_width
        y0 = title_height + row * tile_height

        fitted = ImageOps.contain(
            image,
            (tile_width, image_height),
            method=Image.Resampling.LANCZOS,
        )
        paste_x = x0 + (tile_width - fitted.width) // 2
        paste_y = y0 + (image_height - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

        label = f"{method_name}\nmd5={digest[:10]}"
        draw.multiline_text(
            (x0 + 8, y0 + image_height + 5),
            label,
            fill=(255, 255, 255),
            font=font,
            spacing=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)

    digests = [item[3] for item in loaded_images]
    return {
        "key": key,
        "output": str(output_path),
        "all_identical": len(set(digests)) == 1,
        "methods": {
            method_name: {
                "path": str(image_path),
                "md5": digest,
                "size": [image.width, image.height],
            }
            for method_name, image_path, image, digest in loaded_images
        },
    }


parser = argparse.ArgumentParser(
    description=(
        "Randomly sample identical rank/video/frame/camera keys from "
        "different methods and create box-projection montages."
    )
)
parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="Evaluation root containing method directories.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output directory. Default: <root>/box_projection_compare",
)
parser.add_argument(
    "--samples",
    type=int,
    default=20,
    help="Number of common image keys to sample.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=3407,
    help="Random seed.",
)
parser.add_argument(
    "--columns",
    type=int,
    default=4,
    help="Number of method tiles per montage row.",
)
parser.add_argument(
    "--tile-width",
    type=int,
    default=480,
    help="Width of each method tile.",
)
parser.add_argument(
    "--methods",
    nargs="*",
    default=None,
    help="Optional exact method directory names to compare.",
)
parser.add_argument(
    "--camera",
    default=None,
    help="Optional camera filter, for example CAM_03.",
)
parser.add_argument(
    "--frame",
    default=None,
    help="Optional frame-directory filter, for example t005.",
)
parser.add_argument(
    "--video",
    default=None,
    help="Optional video-directory filter, for example nuplan_video_000123.",
)
args = parser.parse_args()

root = args.root.expanduser().resolve()
if not root.is_dir():
    raise FileNotFoundError(f"Evaluation root does not exist: {root}")

output_dir = (
    args.output.expanduser().resolve()
    if args.output is not None
    else root / "box_projection_compare"
)
output_dir.mkdir(parents=True, exist_ok=True)

method_maps = discover_method_images(root, args.methods)
common_keys = filter_common_keys(
    method_maps,
    camera=args.camera,
    frame=args.frame,
    video=args.video,
)

if not common_keys:
    raise RuntimeError(
        "No common rank/video/frame/camera image key exists across all "
        f"selected methods: {list(method_maps)}"
    )

sample_count = min(args.samples, len(common_keys))
random_generator = random.Random(args.seed)
sampled_keys = random_generator.sample(common_keys, sample_count)

print(f"Root: {root}")
print(f"Methods ({len(method_maps)}):")
for method_name, image_map in method_maps.items():
    print(f"  {method_name}: {len(image_map)} images")
print(f"Common keys: {len(common_keys)}")
print(f"Sampled keys: {sample_count}")
print(f"Output: {output_dir}")

records: list[dict] = []
for sample_index, key in enumerate(sampled_keys):
    safe_key = key.replace("/", "__")
    output_path = output_dir / f"{sample_index:03d}__{safe_key}.jpg"
    record = build_montage(
        key=key,
        method_maps=method_maps,
        output_path=output_path,
        columns=args.columns,
        tile_width=args.tile_width,
    )
    records.append(record)
    state = "IDENTICAL" if record["all_identical"] else "DIFFERENT"
    print(f"[{sample_index + 1:03d}/{sample_count:03d}] {state}: {key}")

index_path = output_dir / "comparison_index.json"
index_path.write_text(
    json.dumps(
        {
            "root": str(root),
            "seed": args.seed,
            "methods": list(method_maps),
            "common_key_count": len(common_keys),
            "samples": records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Index: {index_path}")
