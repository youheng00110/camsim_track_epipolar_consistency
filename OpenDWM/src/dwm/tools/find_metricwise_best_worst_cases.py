import argparse
import hashlib
import json
import math
import os
from pathlib import Path


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
    parser.add_argument(
        "--base-root",
        type=str,
        default="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard",
    )
    parser.add_argument("--gate", type=int, default=16)
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--check-alignment", action="store_true")
    parser.add_argument("--probe-frame", type=int, default=3)
    parser.add_argument("--probe-view", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_div(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if math.isnan(a) or math.isnan(b) or b == 0:
        return float("nan")
    return a / b


def compute_stflow_c(video):
    stflow = safe_float(video.get("stflow_score"))
    edge_cov = safe_div(video.get("num_cross_edges"), video.get("num_cross_raw_edges"))
    inlier = safe_float(video.get("cross_inlier_ratio"))

    if any(math.isnan(x) for x in [stflow, edge_cov, inlier]):
        return float("nan")

    return stflow * edge_cov * inlier


def method_video_record(video):
    return {
        "video_id": video.get("video_id", ""),
        "stflow_c": compute_stflow_c(video),
        "temporal_l1": safe_float(video.get("temporal_l1")),
        "cross_raw_epi_px": safe_float(video.get("cross_raw_epi_px")),
        "cross_gated_epi_px": safe_float(video.get("cross_epi_px")),
        "cross_inlier_ratio": safe_float(video.get("cross_inlier_ratio")),
        "cycle_epi_px": safe_float(video.get("cycle_epi_px")),
        "traj_epi_px": safe_float(video.get("traj_epi_px")),
        "traj_inlier2": safe_float(video.get("traj_inlier2")),
        "traj_inlier4": safe_float(video.get("traj_inlier4")),
    }


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest_item(root, index):
    path = os.path.join(root, "stflow_manifest.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None


def real_signature(root, index, probe_frame, probe_view):
    item = load_manifest_item(root, index)
    if item is None:
        return None

    frames = item.get("frames", [])
    if not frames:
        return None

    t = min(max(probe_frame, 0), len(frames) - 1)
    views = frames[t].get("views", [])
    if not views:
        return None

    v = min(max(probe_view, 0), len(views) - 1)
    real_path = views[v].get("real_image_path", None)
    if real_path is None:
        return None

    if not os.path.isabs(real_path):
        real_path = os.path.join(root, real_path)

    if not os.path.exists(real_path):
        return None

    return file_md5(real_path)


def load_all_methods(base_root, gate, max_videos):
    methods = {}

    for method, folder in METHOD_FOLDERS.items():
        root = os.path.join(base_root, folder)
        result_path = os.path.join(root, f"stflow_traj_result_gate{gate}.json")

        if not os.path.exists(result_path):
            print(f"[skip] missing {method}: {result_path}")
            continue

        data = load_json(result_path)
        videos = data["videos"][:max_videos]

        methods[method] = {
            "root": root,
            "result_path": result_path,
            "records": [method_video_record(v) for v in videos],
        }

    return methods


def compare_metric_for_video(methods, video_index, metric_name, higher_better):
    values = []

    for method, data in methods.items():
        if video_index >= len(data["records"]):
            continue

        value = safe_float(data["records"][video_index].get(metric_name))
        if math.isnan(value):
            continue

        values.append((method, value))

    if len(values) < 2:
        return None

    if higher_better:
        best = max(values, key=lambda x: x[1])
        worst = min(values, key=lambda x: x[1])
    else:
        best = min(values, key=lambda x: x[1])
        worst = max(values, key=lambda x: x[1])

    gap = abs(best[1] - worst[1])

    return {
        "video_index": video_index,
        "metric": metric_name,
        "higher_better": higher_better,
        "best_method": best[0],
        "best_value": best[1],
        "worst_method": worst[0],
        "worst_value": worst[1],
        "gap": gap,
        "all_values": {method: value for method, value in values},
    }


def add_alignment(case, methods, probe_frame, probe_view):
    signatures = {}

    for method, data in methods.items():
        signatures[method] = real_signature(
            data["root"],
            case["video_index"],
            probe_frame,
            probe_view,
        )

    valid = [v for v in signatures.values() if v is not None]
    case["alignment_ok"] = len(set(valid)) == 1 if valid else None
    case["real_signatures"] = signatures

    return case


def collect_cases(methods, metric_name, higher_better, max_videos, top_k, check_alignment, probe_frame, probe_view):
    cases = []

    for index in range(max_videos):
        case = compare_metric_for_video(
            methods,
            index,
            metric_name,
            higher_better,
        )
        if case is None:
            continue

        if check_alignment:
            case = add_alignment(case, methods, probe_frame, probe_view)

        cases.append(case)

    cases = sorted(cases, key=lambda x: x["gap"], reverse=True)
    return cases[:top_k]


def print_cases(title, cases):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'rank':>4s} "
        f"{'idx':>4s} "
        f"{'metric':22s} "
        f"{'gap':>10s} "
        f"{'best':16s} "
        f"{'best_val':>10s} "
        f"{'worst':16s} "
        f"{'worst_val':>10s} "
        f"{'align':>8s}"
    )

    for rank, case in enumerate(cases):
        print(
            f"{rank:4d} "
            f"{case['video_index']:4d} "
            f"{case['metric']:22s} "
            f"{case['gap']:10.4f} "
            f"{case['best_method'][:16]:16s} "
            f"{case['best_value']:10.4f} "
            f"{case['worst_method'][:16]:16s} "
            f"{case['worst_value']:10.4f} "
            f"{str(case.get('alignment_ok', None)):>8s}"
        )


def main():
    args = create_parser().parse_args()

    methods = load_all_methods(args.base_root, args.gate, args.max_videos)
    if len(methods) < 2:
        raise RuntimeError("Need at least two methods.")

    tasks = [
        ("ST-Flow-C best/worst", "stflow_c", True),
        ("Temporal-L1 best/worst", "temporal_l1", False),
        ("Cross-LoFTR Raw-Epi best/worst", "cross_raw_epi_px", False),
        ("Cross-Gated Epi best/worst", "cross_gated_epi_px", False),
        ("Cycle-Epi best/worst", "cycle_epi_px", False),
        ("Traj-Epi best/worst", "traj_epi_px", False),
        ("Traj-Inlier@2 best/worst", "traj_inlier2", True),
    ]

    output = {
        "gate": args.gate,
        "base_root": args.base_root,
        "methods": list(methods.keys()),
        "categories": {},
    }

    for title, metric_name, higher_better in tasks:
        cases = collect_cases(
            methods,
            metric_name,
            higher_better,
            args.max_videos,
            args.top_k,
            args.check_alignment,
            args.probe_frame,
            args.probe_view,
        )
        output["categories"][metric_name] = {
            "title": title,
            "higher_better": higher_better,
            "cases": cases,
        }
        print_cases(title, cases)

    if args.output_json is None:
        output_json = os.path.join(
            args.base_root,
            f"metricwise_best_worst_cases_gate{args.gate}.json",
        )
    else:
        output_json = args.output_json

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n[write]", output_json)


if __name__ == "__main__":
    main()
