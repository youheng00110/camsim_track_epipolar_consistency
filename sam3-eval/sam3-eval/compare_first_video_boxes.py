from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_methods(
    eval_root: Path,
    rank_name: str,
    requested_methods: list[str] | None,
) -> dict[str, Path]:
    if requested_methods:
        method_dirs = [eval_root / name for name in requested_methods]
    else:
        method_dirs = sorted(path for path in eval_root.iterdir() if path.is_dir())

    discovered: dict[str, Path] = {}

    for method_dir in method_dirs:
        images_root = (
            method_dir
            / "box_projection"
            / rank_name
            / "images"
        )

        if images_root.is_dir():
            discovered[method_dir.name] = images_root

    if len(discovered) < 2:
        raise RuntimeError(
            "至少需要找到两个包含 "
            f"box_projection/{rank_name}/images 的方法，实际找到："
            f"{list(discovered)}"
        )

    return discovered


def find_video_names(images_root: Path) -> set[str]:
    return {
        path.name
        for path in images_root.iterdir()
        if path.is_dir()
    }


def collect_video_images(video_root: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}

    for image_path in sorted(video_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        key = image_path.relative_to(video_root).as_posix()
        image_map[key] = image_path

    return image_map


def image_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def compare_pixels(reference_path: Path, candidate_path: Path) -> dict:
    reference = Image.open(reference_path).convert("RGB")
    candidate = Image.open(candidate_path).convert("RGB")

    if reference.size != candidate.size:
        return {
            "pixel_equal": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "reference_size": list(reference.size),
            "candidate_size": list(candidate.size),
        }

    difference = ImageChops.difference(reference, candidate)
    pixel_equal = difference.getbbox() is None

    reference_array = np.asarray(reference, dtype=np.int16)
    candidate_array = np.asarray(candidate, dtype=np.int16)
    absolute_difference = np.abs(reference_array - candidate_array)

    return {
        "pixel_equal": pixel_equal,
        "mean_abs_diff": float(absolute_difference.mean()),
        "max_abs_diff": int(absolute_difference.max()),
        "reference_size": list(reference.size),
        "candidate_size": list(candidate.size),
    }


def choose_montage_keys(
    common_keys: list[str],
    difference_keys: list[str],
    sample_count: int,
) -> list[str]:
    if sample_count <= 0:
        return []

    selected = difference_keys[:sample_count]

    if len(selected) >= sample_count:
        return selected

    remaining = [
        key
        for key in common_keys
        if key not in set(selected)
    ]

    needed = sample_count - len(selected)

    if len(remaining) <= needed:
        selected.extend(remaining)
        return selected

    step = len(remaining) / needed

    for index in range(needed):
        selected.append(remaining[int(index * step)])

    return selected


def make_montage(
    selected_keys: list[str],
    method_images: dict[str, dict[str, Path]],
    output_path: Path,
    tile_width: int,
    tile_height: int,
) -> None:
    if not selected_keys:
        return

    method_names = list(method_images)
    columns = len(method_names)
    header_height = 54
    label_height = 34
    row_height = tile_height + label_height
    canvas_width = columns * tile_width
    canvas_height = header_height + len(selected_keys) * row_height

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, method_name in enumerate(method_names):
        draw.text(
            (column * tile_width + 8, 10),
            method_name,
            fill=(255, 255, 255),
            font=font,
        )

    for row, key in enumerate(selected_keys):
        y0 = header_height + row * row_height

        for column, method_name in enumerate(method_names):
            image_path = method_images[method_name][key]
            image = Image.open(image_path).convert("RGB")
            fitted = ImageOps.contain(
                image,
                (tile_width, tile_height),
                method=Image.Resampling.NEAREST,
            )

            x0 = column * tile_width
            paste_x = x0 + (tile_width - fitted.width) // 2
            paste_y = y0 + (tile_height - fitted.height) // 2
            canvas.paste(fitted, (paste_x, paste_y))

            draw.text(
                (x0 + 8, y0 + tile_height + 8),
                key,
                fill=(255, 255, 255),
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


parser = argparse.ArgumentParser(
    description="比较不同方法第一个视频的 Box 投影图是否一致。"
)
parser.add_argument(
    "--root",
    type=Path,
    required=True,
    help="nuplanhard1000 评测根目录。",
)
parser.add_argument(
    "--rank",
    default="rank_00",
    help="默认比较 rank_00，因为全局第一个视频位于该 rank。",
)
parser.add_argument(
    "--video",
    default="nuplan_video_000000",
    help="要比较的视频目录名；设为空字符串时自动选公共第一个视频。",
)
parser.add_argument(
    "--methods",
    nargs="*",
    default=None,
    help="可选：只比较指定方法目录。",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
)
parser.add_argument(
    "--montage-samples",
    type=int,
    default=16,
    help="拼图展示的 frame-camera 数量；优先展示不一致项。",
)
parser.add_argument(
    "--tile-width",
    type=int,
    default=360,
)
parser.add_argument(
    "--tile-height",
    type=int,
    default=210,
)
args = parser.parse_args()

eval_root = args.root.expanduser().resolve()

if not eval_root.is_dir():
    raise FileNotFoundError(f"评测根目录不存在：{eval_root}")

output_dir = (
    args.output_dir.expanduser().resolve()
    if args.output_dir is not None
    else eval_root / "first_video_box_compare"
)
output_dir.mkdir(parents=True, exist_ok=True)

method_roots = discover_methods(
    eval_root=eval_root,
    rank_name=args.rank,
    requested_methods=args.methods,
)

print(f"[FOUND] methods={len(method_roots)}", flush=True)
for method_name in method_roots:
    print(f"  {method_name}", flush=True)

video_sets = {
    method_name: find_video_names(images_root)
    for method_name, images_root in method_roots.items()
}
common_videos = set.intersection(*video_sets.values())

if not common_videos:
    raise RuntimeError("不同方法之间没有公共视频目录。")

if args.video:
    video_name = args.video

    missing_video_methods = [
        method_name
        for method_name, videos in video_sets.items()
        if video_name not in videos
    ]

    if missing_video_methods:
        raise RuntimeError(
            f"{video_name} 在以下方法中不存在：{missing_video_methods}"
        )
else:
    video_name = sorted(common_videos)[0]

print(f"[VIDEO] {video_name}", flush=True)

method_images: dict[str, dict[str, Path]] = {}

for method_name, images_root in method_roots.items():
    video_root = images_root / video_name
    image_map = collect_video_images(video_root)
    method_images[method_name] = image_map
    print(
        f"[LOAD] {method_name}: {len(image_map)} images",
        flush=True,
    )

key_sets = [set(image_map) for image_map in method_images.values()]
common_keys = sorted(set.intersection(*key_sets))
all_keys = set.union(*key_sets)

if not common_keys:
    raise RuntimeError("该视频在不同方法间没有公共 frame-camera 图片。")

missing_report: dict[str, list[str]] = {}

for method_name, image_map in method_images.items():
    missing_report[method_name] = sorted(all_keys - set(image_map))

reference_method = next(iter(method_images))
comparison_methods = [
    method_name
    for method_name in method_images
    if method_name != reference_method
]

difference_keys: list[str] = []
per_method_summary: dict[str, dict] = {}

for method_name in comparison_methods:
    pixel_equal_count = 0
    byte_equal_count = 0
    method_difference_keys: list[str] = []
    mean_diffs: list[float] = []

    for key in common_keys:
        reference_path = method_images[reference_method][key]
        candidate_path = method_images[method_name][key]

        byte_equal = image_md5(reference_path) == image_md5(candidate_path)
        result = compare_pixels(reference_path, candidate_path)

        if byte_equal:
            byte_equal_count += 1

        if result["pixel_equal"]:
            pixel_equal_count += 1
        else:
            method_difference_keys.append(key)
            difference_keys.append(key)

        if result["mean_abs_diff"] is not None:
            mean_diffs.append(result["mean_abs_diff"])

    per_method_summary[method_name] = {
        "compared_with": reference_method,
        "common_image_count": len(common_keys),
        "byte_equal_count": byte_equal_count,
        "pixel_equal_count": pixel_equal_count,
        "different_count": len(method_difference_keys),
        "mean_abs_diff_over_images": (
            float(sum(mean_diffs) / len(mean_diffs))
            if mean_diffs
            else None
        ),
        "different_keys": method_difference_keys,
    }

difference_keys = sorted(set(difference_keys))
all_pixel_identical = not difference_keys
all_paths_complete = all(not values for values in missing_report.values())

print(
    f"[RESULT] common_images={len(common_keys)} "
    f"all_paths_complete={all_paths_complete} "
    f"all_pixel_identical={all_pixel_identical} "
    f"different_keys={len(difference_keys)}",
    flush=True,
)

for method_name, summary in per_method_summary.items():
    print(
        f"  {method_name}: "
        f"pixel_equal={summary['pixel_equal_count']}/"
        f"{summary['common_image_count']}, "
        f"different={summary['different_count']}, "
        f"mean_abs_diff={summary['mean_abs_diff_over_images']}",
        flush=True,
    )

selected_keys = choose_montage_keys(
    common_keys=common_keys,
    difference_keys=difference_keys,
    sample_count=min(args.montage_samples, len(common_keys)),
)

montage_path = output_dir / f"{video_name}_box_compare.jpg"
make_montage(
    selected_keys=selected_keys,
    method_images=method_images,
    output_path=montage_path,
    tile_width=args.tile_width,
    tile_height=args.tile_height,
)

summary_path = output_dir / f"{video_name}_summary.json"
summary_path.write_text(
    json.dumps(
        {
            "eval_root": str(eval_root),
            "rank": args.rank,
            "video": video_name,
            "reference_method": reference_method,
            "methods": list(method_images),
            "common_image_count": len(common_keys),
            "all_paths_complete": all_paths_complete,
            "all_pixel_identical": all_pixel_identical,
            "difference_key_count": len(difference_keys),
            "difference_keys": difference_keys,
            "missing_keys": missing_report,
            "per_method_summary": per_method_summary,
            "montage_keys": selected_keys,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"[SAVE] montage={montage_path}", flush=True)
print(f"[SAVE] summary={summary_path}", flush=True)
