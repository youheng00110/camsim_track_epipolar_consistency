from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Line {line_number} in {path} is not a JSON object."
                )

            videos.append(item)

    return videos


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)

    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(item)
            for key, item in sorted(value.items())
        }

    return value


def video_pose_signature(video: dict[str, Any]) -> str:
    frames = video.get("frames", [])
    identity: list[dict[str, Any]] = []

    for frame in frames:
        identity.append(
            {
                "frame_index": frame.get("frame_index"),
                "timestamp": frame.get("timestamp"),
                "T_ego_to_world": normalize_json_value(
                    frame.get("T_ego_to_world")
                ),
            }
        )

    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def resolve_image_path(
    raw_path: str,
    manifest_path: Path,
) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path

    rank_root = manifest_path.parent
    box_root = rank_root.parent
    method_root = box_root.parent

    candidates = (
        rank_root / path,
        box_root / path,
        method_root / path,
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve image path {raw_path!r} from {manifest_path}"
    )


def extract_video_images(
    video: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Path]:
    image_map: dict[str, Path] = {}

    for frame_order, frame in enumerate(video.get("frames", [])):
        frame_index = frame.get("frame_index", frame_order)

        for view_order, view in enumerate(frame.get("views", [])):
            camera = str(
                view.get("camera", f"CAM_{view_order:02d}")
            )
            raw_path = (
                view.get("image_path")
                or view.get("box_image_path")
                or view.get("rgb_path")
            )

            if not raw_path:
                continue

            key = f"t{int(frame_index):03d}/{camera}"
            image_map[key] = resolve_image_path(
                str(raw_path),
                manifest_path,
            )

    return image_map


def decoded_pixel_hash(path: Path) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(str(image.size).encode("utf-8"))
        digest.update(image.tobytes())
        return digest.hexdigest(), image.size


def compare_video_images(
    video_a: dict[str, Any],
    manifest_a: Path,
    video_b: dict[str, Any],
    manifest_b: Path,
) -> dict[str, Any]:
    images_a = extract_video_images(video_a, manifest_a)
    images_b = extract_video_images(video_b, manifest_b)

    keys_a = set(images_a)
    keys_b = set(images_b)
    common_keys = sorted(keys_a & keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    missing_in_b = sorted(keys_a - keys_b)

    identical_count = 0
    different_keys: list[str] = []

    for key in common_keys:
        hash_a, size_a = decoded_pixel_hash(images_a[key])
        hash_b, size_b = decoded_pixel_hash(images_b[key])

        if hash_a == hash_b and size_a == size_b:
            identical_count += 1
        else:
            different_keys.append(key)

    all_identical = (
        not missing_in_a
        and not missing_in_b
        and not different_keys
        and len(common_keys) > 0
    )

    return {
        "images_a": len(images_a),
        "images_b": len(images_b),
        "common_images": len(common_keys),
        "identical_images": identical_count,
        "different_keys": different_keys,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "all_images_identical": all_identical,
        "image_paths_a": {
            key: str(images_a[key])
            for key in common_keys
        },
        "image_paths_b": {
            key: str(images_b[key])
            for key in common_keys
        },
    }


def save_pair_montage(
    image_a_path: Path,
    image_b_path: Path,
    method_a: str,
    method_b: str,
    title: str,
    output_path: Path,
) -> None:
    with Image.open(image_a_path) as image_a_file:
        image_a = image_a_file.convert("RGB")

    with Image.open(image_b_path) as image_b_file:
        image_b = image_b_file.convert("RGB")

    tile_width = 640
    tile_height = max(
        round(tile_width * image_a.height / image_a.width),
        round(tile_width * image_b.height / image_b.width),
    )
    label_height = 42
    title_height = 42

    canvas = Image.new(
        "RGB",
        (tile_width * 2, title_height + tile_height + label_height),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 12), title, fill=(255, 255, 255), font=font)

    images = (
        (method_a, image_a),
        (method_b, image_b),
    )

    for column, (method_name, image) in enumerate(images):
        fitted = ImageOps.contain(
            image,
            (tile_width, tile_height),
            method=Image.Resampling.NEAREST,
        )
        x0 = column * tile_width
        paste_x = x0 + (tile_width - fitted.width) // 2
        paste_y = title_height + (tile_height - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        draw.text(
            (x0 + 8, title_height + tile_height + 10),
            method_name,
            fill=(255, 255, 255),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


parser = argparse.ArgumentParser(
    description=(
        "Compare the first and last complete 8-camera Box methods. "
        "For every rank, compare all Box images in the first and last video."
    )
)
parser.add_argument(
    "--check-report",
    type=Path,
    required=True,
    help="box_manifest_check.json generated by the previous checker.",
)
parser.add_argument(
    "--method-a",
    default=None,
    help="Optional exact first method name.",
)
parser.add_argument(
    "--method-b",
    default=None,
    help="Optional exact second method name.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("./box_complete_edge_compare"),
)
args = parser.parse_args()

report_path = args.check_report.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve()

report = json.loads(report_path.read_text(encoding="utf-8"))
if not isinstance(report, list):
    raise RuntimeError("The check report must contain a JSON list.")

ok_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

for item in report:
    if item.get("status") != "OK":
        continue

    method = str(item.get("method", ""))
    rank = str(item.get("rank", ""))

    if method and rank:
        ok_by_method[method].append(item)

complete_methods = sorted(
    method
    for method, items in ok_by_method.items()
    if len({str(item.get("rank")) for item in items}) == 4
)

if len(complete_methods) < 2:
    raise RuntimeError(
        "Fewer than two methods have four OK Box ranks. "
        f"Found: {complete_methods}"
    )

method_a = args.method_a or complete_methods[0]
method_b = args.method_b or complete_methods[-1]

if method_a not in ok_by_method:
    raise RuntimeError(f"Method A is not complete/OK: {method_a}")

if method_b not in ok_by_method:
    raise RuntimeError(f"Method B is not complete/OK: {method_b}")

rank_map_a = {
    str(item["rank"]): Path(str(item["box_manifest"]))
    for item in ok_by_method[method_a]
}
rank_map_b = {
    str(item["rank"]): Path(str(item["box_manifest"]))
    for item in ok_by_method[method_b]
}
common_ranks = sorted(set(rank_map_a) & set(rank_map_b))

print(f"[COMPLETE_METHODS] {complete_methods}", flush=True)
print(f"[COMPARE] A={method_a}", flush=True)
print(f"[COMPARE] B={method_b}", flush=True)
print(f"[RANKS] {common_ranks}", flush=True)

results: list[dict[str, Any]] = []
all_identical = True

for rank in common_ranks:
    manifest_a = rank_map_a[rank]
    manifest_b = rank_map_b[rank]
    videos_a = load_jsonl(manifest_a)
    videos_b = load_jsonl(manifest_b)

    if not videos_a or not videos_b:
        raise RuntimeError(f"Empty manifest at {rank}")

    if len(videos_a) != len(videos_b):
        print(
            f"[{rank}] count mismatch: "
            f"{len(videos_a)} != {len(videos_b)}",
            flush=True,
        )
        all_identical = False

    selected_pairs = (
        ("first", videos_a[0], videos_b[0]),
        ("last", videos_a[-1], videos_b[-1]),
    )

    for edge_name, video_a, video_b in selected_pairs:
        pose_same = (
            video_pose_signature(video_a)
            == video_pose_signature(video_b)
        )
        image_comparison = compare_video_images(
            video_a,
            manifest_a,
            video_b,
            manifest_b,
        )
        edge_identical = (
            pose_same
            and image_comparison["all_images_identical"]
        )
        all_identical = all_identical and edge_identical

        print(
            f"[{rank} {edge_name}] "
            f"pose_same={pose_same} "
            f"images={image_comparison['identical_images']}/"
            f"{image_comparison['common_images']} "
            f"all_identical={edge_identical}",
            flush=True,
        )

        if image_comparison["different_keys"]:
            preview_keys = image_comparison["different_keys"][:8]
            print(
                f"  different_keys={preview_keys}",
                flush=True,
            )

        common_keys = sorted(
            set(image_comparison["image_paths_a"])
            & set(image_comparison["image_paths_b"])
        )

        if common_keys:
            representative_key = (
                common_keys[0]
                if edge_name == "first"
                else common_keys[-1]
            )
            save_pair_montage(
                Path(image_comparison["image_paths_a"][representative_key]),
                Path(image_comparison["image_paths_b"][representative_key]),
                method_a,
                method_b,
                f"{rank} {edge_name} {representative_key}",
                output_dir
                / f"{rank}_{edge_name}_{representative_key.replace('/', '_')}.jpg",
            )

        results.append(
            {
                "rank": rank,
                "edge": edge_name,
                "method_a": method_a,
                "method_b": method_b,
                "manifest_a": str(manifest_a),
                "manifest_b": str(manifest_b),
                "pose_same": pose_same,
                "edge_identical": edge_identical,
                **{
                    key: value
                    for key, value in image_comparison.items()
                    if key not in {
                        "image_paths_a",
                        "image_paths_b",
                    }
                },
            }
        )

output_dir.mkdir(parents=True, exist_ok=True)
result_path = output_dir / "edge_compare_result.json"
result_path.write_text(
    json.dumps(
        {
            "complete_methods": complete_methods,
            "method_a": method_a,
            "method_b": method_b,
            "all_edges_identical": all_identical,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n================ RESULT ================", flush=True)
print(f"all_edges_identical={all_identical}", flush=True)
print(f"report={result_path}", flush=True)
print(f"montages={output_dir}", flush=True)
