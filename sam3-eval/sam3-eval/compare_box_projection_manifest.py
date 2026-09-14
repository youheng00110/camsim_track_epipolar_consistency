from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_PATH_FIELDS = (
    "image_path",
    "box_image_path",
    "rgb_path",
)


def resolve_image_path(
    raw_path: str,
    manifest_path: Path,
    box_root: Path,
    method_dir: Path,
    eval_root: Path,
) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path

    candidates = (
        manifest_path.parent / path,
        box_root / path,
        method_dir / path,
        eval_root / path,
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return candidates[0].resolve()


def canonical_key(
    image_path: Path,
    manifest_path: Path,
    box_root: Path,
) -> str:
    try:
        return image_path.relative_to(box_root).as_posix()
    except ValueError:
        pass

    rank_name = manifest_path.parent.name

    try:
        suffix = image_path.relative_to(manifest_path.parent)
    except ValueError:
        suffix = Path("images") / image_path.name

    return (Path(rank_name) / suffix).as_posix()


def extract_view_image_path(view: dict) -> str | None:
    for field in IMAGE_PATH_FIELDS:
        value = view.get(field)
        if value:
            return str(value)

    return None


def load_method_index(
    method_dir: Path,
    eval_root: Path,
) -> dict[str, Path]:
    box_root = method_dir / "box_projection"
    manifests = sorted(box_root.rglob("box_manifest.jsonl"))

    if not manifests:
        return {}

    print(
        f"[METHOD] {method_dir.name}: "
        f"{len(manifests)} box manifests",
        flush=True,
    )

    image_index: dict[str, Path] = {}
    video_count = 0

    for manifest_number, manifest_path in enumerate(manifests, start=1):
        print(
            f"  [MANIFEST] {manifest_number}/{len(manifests)} "
            f"{manifest_path.relative_to(method_dir)}",
            flush=True,
        )

        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    video_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON: {manifest_path}:{line_number}: {error}"
                    ) from error

                frames = video_record.get("frames", [])

                for frame in frames:
                    views = frame.get("views", [])

                    for view in views:
                        raw_path = extract_view_image_path(view)
                        if raw_path is None:
                            continue

                        image_path = resolve_image_path(
                            raw_path=raw_path,
                            manifest_path=manifest_path,
                            box_root=box_root,
                            method_dir=method_dir,
                            eval_root=eval_root,
                        )
                        key = canonical_key(
                            image_path=image_path,
                            manifest_path=manifest_path,
                            box_root=box_root,
                        )

                        if key in image_index:
                            previous = image_index[key]
                            if previous != image_path:
                                raise RuntimeError(
                                    f"Duplicate key in {method_dir.name}: {key}\n"
                                    f"  first:  {previous}\n"
                                    f"  second: {image_path}"
                                )
                            continue

                        image_index[key] = image_path

                video_count += 1

                if video_count % 200 == 0:
                    print(
                        f"    [PROGRESS] videos={video_count} "
                        f"images={len(image_index)}",
                        flush=True,
                    )

    print(
        f"  [DONE] videos={video_count} images={len(image_index)}",
        flush=True,
    )
    return image_index


def discover_methods(
    eval_root: Path,
    requested_methods: list[str] | None,
) -> dict[str, dict[str, Path]]:
    if requested_methods:
        method_dirs = [eval_root / name for name in requested_methods]
    else:
        method_dirs = sorted(
            path
            for path in eval_root.iterdir()
            if path.is_dir()
            and (path / "box_projection").is_dir()
        )

    if not method_dirs:
        raise RuntimeError(
            f"No method directory containing box_projection was found: "
            f"{eval_root}"
        )

    print(
        f"[DISCOVER] candidate methods={len(method_dirs)}",
        flush=True,
    )

    method_indices: dict[str, dict[str, Path]] = {}

    for method_dir in method_dirs:
        if not method_dir.is_dir():
            print(
                f"[SKIP] method directory not found: {method_dir}",
                flush=True,
            )
            continue

        index = load_method_index(
            method_dir=method_dir,
            eval_root=eval_root,
        )

        if index:
            method_indices[method_dir.name] = index

    if len(method_indices) < 2:
        raise RuntimeError(
            "At least two methods with readable box manifests are required. "
            f"Found: {list(method_indices)}"
        )

    return method_indices


def select_common_keys(
    method_indices: dict[str, dict[str, Path]],
    camera: str | None,
    frame_name: str | None,
    video_name: str | None,
) -> list[str]:
    key_sets = [
        set(image_index)
        for image_index in method_indices.values()
    ]
    common_keys = set.intersection(*key_sets)

    filtered_keys: list[str] = []

    for key in sorted(common_keys):
        parts = Path(key).parts
        stem = Path(key).stem

        if camera and stem != camera:
            continue
        if frame_name and frame_name not in parts:
            continue
        if video_name and video_name not in parts:
            continue

        filtered_keys.append(key)

    return filtered_keys


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
    method_indices: dict[str, dict[str, Path]],
    output_path: Path,
    columns: int,
    tile_width: int,
) -> dict:
    loaded: list[tuple[str, Path, Image.Image, str]] = []

    for method_name, image_index in method_indices.items():
        image_path = image_index[key]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Box image does not exist: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        digest = file_md5(image_path)
        loaded.append(
            (method_name, image_path, image, digest)
        )

    image_height = max(
        max(1, round(tile_width * image.height / image.width))
        for _, _, image, _ in loaded
    )
    label_height = 54
    title_height = 42
    columns = max(1, min(columns, len(loaded)))
    rows = math.ceil(len(loaded) / columns)
    canvas_width = columns * tile_width
    canvas_height = title_height + rows * (
        image_height + label_height
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (10, 12),
        key,
        fill=(255, 255, 255),
        font=font,
    )

    for index, item in enumerate(loaded):
        method_name, _, image, digest = item
        row = index // columns
        column = index % columns
        x0 = column * tile_width
        y0 = title_height + row * (
            image_height + label_height
        )

        fitted = ImageOps.contain(
            image,
            (tile_width, image_height),
            method=Image.Resampling.LANCZOS,
        )
        paste_x = x0 + (tile_width - fitted.width) // 2
        paste_y = y0 + (image_height - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

        draw.multiline_text(
            (x0 + 6, y0 + image_height + 5),
            f"{method_name}\nmd5={digest[:12]}",
            fill=(255, 255, 255),
            font=font,
            spacing=2,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(output_path, quality=95)

    digests = [item[3] for item in loaded]

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
            for method_name, image_path, image, digest in loaded
        },
    }


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--columns", type=int, default=4)
parser.add_argument("--tile-width", type=int, default=480)
parser.add_argument("--methods", nargs="*", default=None)
parser.add_argument("--camera", default=None)
parser.add_argument("--frame", default=None)
parser.add_argument("--video", default=None)
args = parser.parse_args()

eval_root = args.root.expanduser().resolve()

if not eval_root.is_dir():
    raise FileNotFoundError(
        f"Evaluation root does not exist: {eval_root}"
    )

output_dir = (
    args.output.expanduser().resolve()
    if args.output is not None
    else eval_root / "box_projection_compare"
)
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"[START] root={eval_root}",
    flush=True,
)
method_indices = discover_methods(
    eval_root=eval_root,
    requested_methods=args.methods,
)
common_keys = select_common_keys(
    method_indices=method_indices,
    camera=args.camera,
    frame_name=args.frame,
    video_name=args.video,
)

if not common_keys:
    counts = {
        name: len(index)
        for name, index in method_indices.items()
    }
    raise RuntimeError(
        "No common rank/video/frame/camera keys were found. "
        f"Method counts: {counts}"
    )

sample_count = min(
    args.samples,
    len(common_keys),
)
random_generator = random.Random(args.seed)
sampled_keys = random_generator.sample(
    common_keys,
    sample_count,
)

print(
    f"[COMMON] methods={len(method_indices)} "
    f"keys={len(common_keys)} samples={sample_count}",
    flush=True,
)
print(
    f"[OUTPUT] {output_dir}",
    flush=True,
)

records: list[dict] = []

for sample_index, key in enumerate(sampled_keys):
    safe_key = key.replace("/", "__")
    output_path = (
        output_dir
        / f"{sample_index:03d}__{safe_key}.jpg"
    )
    record = build_montage(
        key=key,
        method_indices=method_indices,
        output_path=output_path,
        columns=args.columns,
        tile_width=args.tile_width,
    )
    records.append(record)
    state = (
        "IDENTICAL"
        if record["all_identical"]
        else "DIFFERENT"
    )
    print(
        f"[{sample_index + 1:03d}/{sample_count:03d}] "
        f"{state}: {key}",
        flush=True,
    )

index_path = output_dir / "comparison_index.json"
index_path.write_text(
    json.dumps(
        {
            "root": str(eval_root),
            "seed": args.seed,
            "methods": list(method_indices),
            "common_key_count": len(common_keys),
            "samples": records,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    f"[DONE] index={index_path}",
    flush=True,
)
