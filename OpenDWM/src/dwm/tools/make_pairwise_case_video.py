import argparse
import json
import os
from pathlib import Path

import av
from PIL import Image, ImageDraw


METHOD_FOLDERS = {
    "Plucker-NoID": "0plucker_special_noid_preview_paired_200_merged200",
    "Implicit": "implicit_preview_paired_200_merged200",
    "No-Condition": "nocondition_preview_paired_200_merged200",
    "NoCond-18k": "nocondition18000_preview_paired_200_merged200",
    "PETR": "petr_preview_paired_200_merged200",
    "PV-only": "pvonly_preview_paired_200_merged200",
    "Token": "token_preview_paired_200_merged200",
}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=str, required=True)
    parser.add_argument("--video-index", type=int, required=True)
    parser.add_argument("--method-a", type=str, required=True)
    parser.add_argument("--method-b", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--view-indices", type=str, default="2,3,4")
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def load_manifest_item(root, video_index):
    manifest_path = os.path.join(root, "stflow_manifest.jsonl")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == video_index:
                line = line.strip()
                if len(line) == 0:
                    raise RuntimeError(f"Empty manifest line at index={video_index}")
                return json.loads(line)

    raise IndexError(f"video_index={video_index} not found in {manifest_path}")


def parse_view_indices(text):
    indices = []

    for item in text.split(","):
        item = item.strip()
        if len(item) == 0:
            continue
        indices.append(int(item))

    if len(indices) == 0:
        raise ValueError("view-indices is empty.")

    return indices


def resolve_path(root, relative_or_abs_path):
    if os.path.isabs(relative_or_abs_path):
        return relative_or_abs_path

    return os.path.join(root, relative_or_abs_path)


def load_resized_image(path, width):
    image = Image.open(path).convert("RGB")

    height = round(image.height * width / image.width)
    image = image.resize((width, height))

    return image


def draw_label(image, label):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, canvas.width, 26), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))

    return canvas


def concat_horizontal(images):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x_offset = 0

    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width

    return canvas


def concat_vertical(images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    y_offset = 0

    for image in images:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height

    return canvas


def build_row(root, manifest_item, row_name, time_index, view_indices, width, use_real):
    frame = manifest_item["frames"][time_index]
    row_images = []

    for view_index in view_indices:
        view = frame["views"][view_index]
        key = "real_image_path" if use_real else "image_path"

        image_path = resolve_path(root, view[key])
        image = load_resized_image(image_path, width)

        camera_name = view["camera"]
        label = f"{row_name} | t{time_index:03d} | {camera_name}"
        row_images.append(draw_label(image, label))

    return concat_horizontal(row_images)


def write_video(path, frames, fps):
    if len(frames) == 0:
        raise RuntimeError("No frames to write.")

    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", fps)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}

        for image in frames:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def save_contact_sheet(path, frames):
    if len(frames) == 0:
        return

    selected = []

    for index in [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]:
        selected.append(frames[index])

    sheet = concat_horizontal(selected)
    sheet.save(path)


def main():
    args = create_parser().parse_args()

    base_root = args.base_root
    view_indices = parse_view_indices(args.view_indices)

    if args.method_a not in METHOD_FOLDERS:
        raise KeyError(f"Unknown method-a: {args.method_a}")
    if args.method_b not in METHOD_FOLDERS:
        raise KeyError(f"Unknown method-b: {args.method_b}")

    root_a = os.path.join(base_root, METHOD_FOLDERS[args.method_a])
    root_b = os.path.join(base_root, METHOD_FOLDERS[args.method_b])

    item_a = load_manifest_item(root_a, args.video_index)
    item_b = load_manifest_item(root_b, args.video_index)

    frame_count = min(len(item_a["frames"]), len(item_b["frames"]))
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    frames = []

    for time_index in range(frame_count):
        rows = []

        rows.append(
            build_row(
                root_a,
                item_a,
                "GT",
                time_index,
                view_indices,
                args.resize_width,
                use_real=True,
            )
        )
        rows.append(
            build_row(
                root_a,
                item_a,
                args.method_a,
                time_index,
                view_indices,
                args.resize_width,
                use_real=False,
            )
        )
        rows.append(
            build_row(
                root_b,
                item_b,
                args.method_b,
                time_index,
                view_indices,
                args.resize_width,
                use_real=False,
            )
        )

        frames.append(concat_vertical(rows))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_a = args.method_a.replace("/", "_").replace(" ", "_")
    safe_b = args.method_b.replace("/", "_").replace(" ", "_")
    stem = f"case_{args.video_index:06d}_{safe_a}_vs_{safe_b}"

    video_path = output_dir / f"{stem}.mp4"
    sheet_path = output_dir / f"{stem}_sheet.jpg"

    write_video(str(video_path), frames, args.fps)
    save_contact_sheet(str(sheet_path), frames)

    print("[write]", video_path)
    print("[write]", sheet_path)


if __name__ == "__main__":
    main()
