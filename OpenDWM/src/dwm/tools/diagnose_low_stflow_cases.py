import argparse
import json
import os
import subprocess


def create_parser():
    parser = argparse.ArgumentParser(
        description="Find low ST-Flow score cases and visualize likely failure reasons."
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--result-json", type=str, default="stflow_traj_result.json")
    parser.add_argument("--manifest-name", type=str, default="stflow_manifest.jsonl")
    parser.add_argument("--output-dir-name", type=str, default="diagnostics_low_score")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--loftr-confidence", type=float, default=0.1)
    parser.add_argument("--time-index", type=int, default=None)
    return parser


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def video_index_from_id(video_id):
    if "_video_" not in video_id:
        return None
    return int(video_id.split("_video_")[-1])


def normalized_components(video):
    temp = video.get("temporal_l1", float("nan"))
    cross = video.get("cross_epi_px", float("nan"))
    cycle = video.get("cycle_epi_px", float("nan"))

    temp_norm = min(temp / 0.15, 1.0)
    cross_norm = min(cross / 10.0, 1.0)
    cycle_norm = min(cycle / 10.0, 1.0)

    return {
        "temporal": temp_norm,
        "cross": cross_norm,
        "cycle": cycle_norm,
    }


def dominant_reason(video):
    comps = normalized_components(video)
    sorted_items = sorted(comps.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = sorted_items[0]

    if top_value < 0.7:
        return "mixed/moderate"

    if top_name == "temporal":
        return "temporal instability"
    if top_name == "cross":
        return "cross-view inconsistency"
    if top_name == "cycle":
        return "spatio-temporal cycle inconsistency"

    return "unknown"


def worst_pair(video):
    pair_stats = video.get("pair_stats", {})
    if len(pair_stats) == 0:
        return None, None

    ranked = []
    for pair_key, stats in pair_stats.items():
        cross = stats.get("cross_epi_px", float("nan"))
        cycle = stats.get("cycle_epi_px", float("nan"))
        score = max(cross, cycle)
        ranked.append((score, pair_key, stats))

    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    return ranked[0][1], ranked[0][2]


def split_pair(pair_key):
    if pair_key is None:
        return None, None
    if "__" not in pair_key:
        return None, None
    return pair_key.split("__", 1)


def safe_time_index(manifest_item, requested_time_index, frame_stride):
    frame_count = len(manifest_item["frames"])
    reference_count = int(manifest_item.get("reference_frame_count", 0))

    if requested_time_index is not None:
        candidate = requested_time_index
    else:
        candidate = reference_count

    if candidate + frame_stride >= frame_count:
        candidate = max(0, frame_count - frame_stride - 1)

    return candidate


def run_command(command):
    print("[cmd]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_case_summary(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = create_parser().parse_args()

    root = args.root
    result_path = os.path.join(root, args.result_json)
    manifest_path = os.path.join(root, args.manifest_name)
    output_root = os.path.join(root, args.output_dir_name)
    os.makedirs(output_root, exist_ok=True)

    result = load_json(result_path)
    manifest_items = load_manifest_items(manifest_path)

    videos = result["videos"]
    videos = sorted(videos, key=lambda x: x.get("stflow_score", 1e9))

    selected = videos[: args.top_k]

    print("==== low score cases ====")
    for rank, video in enumerate(selected):
        video_id = video.get("video_id", "")
        video_index = video_index_from_id(video_id)
        pair_key, pair_info = worst_pair(video)
        cam0, cam1 = split_pair(pair_key)
        reason = dominant_reason(video)

        print(
            f"[{rank}] video_id={video_id} index={video_index} "
            f"score={video.get('stflow_score')} "
            f"temp={video.get('temporal_l1')} "
            f"cross={video.get('cross_epi_px')} "
            f"cycle={video.get('cycle_epi_px')} "
            f"traj={video.get('traj_epi_px')} "
            f"reason={reason} "
            f"worst_pair={pair_key}"
        )

        if video_index is None or video_index >= len(manifest_items):
            print(f"[skip] cannot map video_id={video_id} to manifest index")
            continue

        manifest_item = manifest_items[video_index]
        time_index = safe_time_index(
            manifest_item,
            args.time_index,
            args.frame_stride,
        )

        case_dir = os.path.join(
            output_root,
            f"rank{rank:02d}_{video_id}_{reason.replace('/', '_').replace(' ', '_')}",
        )
        os.makedirs(case_dir, exist_ok=True)

        summary = {
            "rank": rank,
            "video_id": video_id,
            "video_index": video_index,
            "time_index": time_index,
            "reason": reason,
            "worst_pair": pair_key,
            "worst_pair_stats": pair_info,
            "metrics": video,
        }
        write_case_summary(os.path.join(case_dir, "case_summary.json"), summary)

        if cam0 is not None and cam1 is not None:
            stflow_dir = os.path.join(case_dir, "stflow_vis")
            run_command([
                "python",
                "-m",
                "dwm.tools.visualize_stflow_debug",
                "--manifest",
                manifest_path,
                "--output-dir",
                stflow_dir,
                "--device",
                args.device,
                "--video-index",
                str(video_index),
                "--time-index",
                str(time_index),
                "--frame-stride",
                str(args.frame_stride),
                "--camera0",
                cam0,
                "--camera1",
                cam1,
                "--camera-temporal",
                cam0,
                "--min-matches",
                str(args.min_matches),
                "--max-matches",
                str(args.max_matches),
                "--loftr-confidence",
                str(args.loftr_confidence),
            ])

            traj_dir = os.path.join(case_dir, "traj_vis")
            run_command([
                "python",
                "-m",
                "dwm.tools.visualize_traj_adherence",
                "--manifest",
                manifest_path,
                "--output-dir",
                traj_dir,
                "--device",
                args.device,
                "--video-index",
                str(video_index),
                "--time-index",
                str(time_index),
                "--frame-stride",
                str(args.frame_stride),
                "--camera",
                cam0,
                "--min-matches",
                str(args.min_matches),
                "--max-matches",
                str(args.max_matches),
                "--loftr-confidence",
                str(args.loftr_confidence),
                "--error-vmax",
                "10.0",
            ])

    print("[done] diagnostics saved to:", output_root)


if __name__ == "__main__":
    main()
