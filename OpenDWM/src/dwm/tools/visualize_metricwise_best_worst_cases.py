import argparse
import json
import os
import subprocess
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


METRIC_DISPLAY = {
    "stflow_c": "ST-Flow-C",
    "temporal_l1": "Temporal-L1",
    "cross_raw_epi_px": "Cross-LoFTR Raw-Epi",
    "cross_gated_epi_px": "Cross-Gated Epi",
    "cycle_epi_px": "Cycle-Epi",
    "traj_epi_px": "Traj-Epi",
    "traj_inlier2": "Traj-Inlier@2",
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Visualize best/worst method pairs for metric-wise discriminative cases."
    )
    parser.add_argument("--base-root", type=str, required=True)
    parser.add_argument("--case-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gate", type=int, default=16)
    parser.add_argument(
        "--metrics",
        type=str,
        default="stflow_c,temporal_l1,cross_raw_epi_px,traj_epi_px,traj_inlier2",
    )
    parser.add_argument("--top-k-per-metric", type=int, default=1)
    parser.add_argument("--view-indices", type=str, default="2,3,4")
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--make-diagnostics", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--loftr-confidence", type=float, default=0.1)
    return parser


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_ints(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if len(values) == 0:
        raise ValueError("Empty index list.")
    return values


def parse_csv_strings(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(item)
    if len(values) == 0:
        raise ValueError("Empty metric list.")
    return values


def method_root(base_root, method_name):
    if method_name not in METHOD_FOLDERS:
        raise KeyError(f"Unknown method name: {method_name}")
    return os.path.join(base_root, METHOD_FOLDERS[method_name])


def load_manifest_item(root, video_index):
    manifest_path = os.path.join(root, "stflow_manifest.jsonl")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == video_index:
                return json.loads(line)
    raise IndexError(f"video_index={video_index} not found in {manifest_path}")


def resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_resized_image(path, width):
    image = Image.open(path).convert("RGB")
    height = round(image.height * width / image.width)
    return image.resize((width, height))


def draw_label(image, label):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 28), fill=(0, 0, 0))
    draw.text((6, 7), label, fill=(255, 255, 255))
    return canvas


def concat_horizontal(images):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))

    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width

    return canvas


def concat_vertical(images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))

    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height

    return canvas


def build_row(root, item, row_name, time_index, view_indices, resize_width, use_real):
    frame = item["frames"][time_index]
    row_images = []

    for view_index in view_indices:
        view = frame["views"][view_index]
        key = "real_image_path" if use_real else "image_path"
        image_path = resolve_path(root, view[key])
        image = load_resized_image(image_path, resize_width)
        label = f"{row_name} | t{time_index:03d} | {view['camera']}"
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

    indices = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    selected = [frames[index] for index in indices]
    sheet = concat_horizontal(selected)
    sheet.save(path)


def safe_filename(text):
    return (
        text.replace("/", "_")
        .replace(" ", "_")
        .replace("@", "at")
        .replace(":", "_")
    )


def make_pairwise_video(
    base_root,
    case,
    metric_name,
    metric_rank,
    output_dir,
    view_indices,
    resize_width,
    fps,
    max_frames,
):
    video_index = int(case["video_index"])
    best_method = case["best_method"]
    worst_method = case["worst_method"]

    root_best = method_root(base_root, best_method)
    root_worst = method_root(base_root, worst_method)

    item_best = load_manifest_item(root_best, video_index)
    item_worst = load_manifest_item(root_worst, video_index)

    frame_count = min(len(item_best["frames"]), len(item_worst["frames"]))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    frames = []
    metric_label = METRIC_DISPLAY.get(metric_name, metric_name)

    for time_index in range(frame_count):
        rows = []
        rows.append(
            build_row(
                root_best,
                item_best,
                "GT",
                time_index,
                view_indices,
                resize_width,
                True,
            )
        )
        rows.append(
            build_row(
                root_best,
                item_best,
                f"BEST {best_method}",
                time_index,
                view_indices,
                resize_width,
                False,
            )
        )
        rows.append(
            build_row(
                root_worst,
                item_worst,
                f"WORST {worst_method}",
                time_index,
                view_indices,
                resize_width,
                False,
            )
        )

        frame = concat_vertical(rows)
        frames.append(frame)

    stem = (
        f"{safe_filename(metric_name)}_rank{metric_rank:02d}_"
        f"idx{video_index:06d}_{safe_filename(best_method)}_vs_{safe_filename(worst_method)}"
    )
    video_path = os.path.join(output_dir, f"{stem}.mp4")
    sheet_path = os.path.join(output_dir, f"{stem}_sheet.jpg")
    summary_path = os.path.join(output_dir, f"{stem}_summary.json")

    write_video(video_path, frames, fps)
    save_contact_sheet(sheet_path, frames)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "metric_rank": metric_rank,
                "video_index": video_index,
                "video_id": case.get("video_id", ""),
                "best_method": best_method,
                "best_value": case.get("best_value"),
                "worst_method": worst_method,
                "worst_value": case.get("worst_value"),
                "gap": case.get("gap"),
                "alignment_ok": case.get("alignment_ok"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[write]", video_path)
    print("[write]", sheet_path)
    print("[write]", summary_path)

    return video_path


def load_method_video_result(base_root, method_name, gate, video_index):
    root = method_root(base_root, method_name)
    result_path = os.path.join(root, f"stflow_traj_result_gate{gate}.json")
    data = load_json(result_path)
    return data["videos"][video_index]


def worst_pair_for_metric(video_result, metric_name):
    pair_stats = video_result.get("pair_stats", {})
    if len(pair_stats) == 0:
        return None

    if metric_name == "cross_raw_epi_px":
        key = "cross_raw_epi_px"
    elif metric_name == "cross_gated_epi_px":
        key = "cross_epi_px"
    elif metric_name == "cycle_epi_px":
        key = "cycle_epi_px"
    else:
        key = "cross_raw_epi_px"

    ranked = []
    for pair_key, stats in pair_stats.items():
        value = stats.get(key, None)
        if value is None:
            continue
        ranked.append((float(value), pair_key))

    if len(ranked) == 0:
        return None

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def camera_name_for_view(root, video_index, view_index):
    item = load_manifest_item(root, video_index)
    view_count = len(item["frames"][0]["views"])
    view_index = min(max(view_index, 0), view_count - 1)
    return item["frames"][0]["views"][view_index]["camera"]


def run_command(command):
    print("[cmd]", " ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print("[warn] command failed:", " ".join(command), flush=True)


def make_diagnostics(
    base_root,
    case,
    metric_name,
    output_dir,
    gate,
    device,
    frame_stride,
    min_matches,
    max_matches,
    loftr_confidence,
):
    video_index = int(case["video_index"])
    best_method = case["best_method"]
    worst_method = case["worst_method"]

    if metric_name in ["cross_raw_epi_px", "cross_gated_epi_px", "cycle_epi_px"]:
        for method_name in [best_method, worst_method]:
            root = method_root(base_root, method_name)
            video_result = load_method_video_result(base_root, method_name, gate, video_index)
            pair_key = worst_pair_for_metric(video_result, metric_name)
            if pair_key is None or "__" not in pair_key:
                continue

            camera0, camera1 = pair_key.split("__", 1)
            diag_dir = os.path.join(
                output_dir,
                f"{safe_filename(metric_name)}_idx{video_index:06d}_{safe_filename(method_name)}_stflow_diag",
            )

            run_command([
                "python",
                "-m",
                "dwm.tools.visualize_stflow_debug",
                "--manifest",
                os.path.join(root, "stflow_manifest.jsonl"),
                "--output-dir",
                diag_dir,
                "--device",
                device,
                "--video-index",
                str(video_index),
                "--time-index",
                "3",
                "--frame-stride",
                str(frame_stride),
                "--camera0",
                camera0,
                "--camera1",
                camera1,
                "--camera-temporal",
                camera0,
                "--min-matches",
                str(min_matches),
                "--max-matches",
                str(max_matches),
                "--loftr-confidence",
                str(loftr_confidence),
            ])

    if metric_name in ["traj_epi_px", "traj_inlier2"]:
        for method_name in [best_method, worst_method]:
            root = method_root(base_root, method_name)
            camera = camera_name_for_view(root, video_index, 3)
            diag_dir = os.path.join(
                output_dir,
                f"{safe_filename(metric_name)}_idx{video_index:06d}_{safe_filename(method_name)}_traj_diag",
            )

            run_command([
                "python",
                "-m",
                "dwm.tools.visualize_traj_adherence",
                "--manifest",
                os.path.join(root, "stflow_manifest.jsonl"),
                "--output-dir",
                diag_dir,
                "--device",
                device,
                "--video-index",
                str(video_index),
                "--time-index",
                "3",
                "--frame-stride",
                str(frame_stride),
                "--camera",
                camera,
                "--min-matches",
                str(min_matches),
                "--max-matches",
                str(max_matches),
                "--loftr-confidence",
                str(loftr_confidence),
                "--error-vmax",
                "10.0",
            ])


def main():
    args = create_parser().parse_args()

    case_data = load_json(args.case_json)
    metrics = parse_csv_strings(args.metrics)
    view_indices = parse_csv_ints(args.view_indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metrics:
        category = case_data["categories"].get(metric_name, None)
        if category is None:
            print("[skip] metric not in case json:", metric_name)
            continue

        cases = category["cases"][: args.top_k_per_metric]
        metric_out = output_dir / safe_filename(metric_name)
        metric_out.mkdir(parents=True, exist_ok=True)

        for rank, case in enumerate(cases):
            make_pairwise_video(
                args.base_root,
                case,
                metric_name,
                rank,
                str(metric_out),
                view_indices,
                args.resize_width,
                args.fps,
                args.max_frames,
            )

            if args.make_diagnostics:
                make_diagnostics(
                    args.base_root,
                    case,
                    metric_name,
                    str(metric_out),
                    args.gate,
                    args.device,
                    args.frame_stride,
                    args.min_matches,
                    args.max_matches,
                    args.loftr_confidence,
                )

    print("[done] output:", output_dir)


if __name__ == "__main__":
    main()
