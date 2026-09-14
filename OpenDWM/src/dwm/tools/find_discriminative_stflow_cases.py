import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


DEFAULT_METHODS = {
    "Plucker-NoID": "0plucker_special_noid_preview_paired_200_merged200",
    "Implicit": "implicit_preview_paired_200_merged200",
    "No-Condition": "nocondition_preview_paired_200_merged200",
    "NoCond-18k": "nocondition18000_preview_paired_200_merged200",
    "PETR": "petr_preview_paired_200_merged200",
    "PV-only": "pvonly_preview_paired_200_merged200",
    "Token": "token_preview_paired_200_merged200",
}


QUALITY_KEYS = [
    "stflow_c",
    "temporal_quality",
    "cross_raw_quality",
    "traj_quality",
    "traj_inlier2",
]


def create_parser():
    parser = argparse.ArgumentParser(
        description="Find videos with largest per-method metric differences."
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard",
    )
    parser.add_argument(
        "--gate",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--check-alignment",
        action="store_true",
    )
    parser.add_argument(
        "--probe-frame",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--probe-view",
        type=int,
        default=3,
    )
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
    edge_cov = safe_div(
        video.get("num_cross_edges"),
        video.get("num_cross_raw_edges"),
    )
    inlier = safe_float(video.get("cross_inlier_ratio"))

    if any(math.isnan(x) for x in [stflow, edge_cov, inlier]):
        return float("nan")

    return stflow * edge_cov * inlier


def metric_record(video):
    stflow_c = compute_stflow_c(video)
    temporal_l1 = safe_float(video.get("temporal_l1"))
    cross_raw = safe_float(video.get("cross_raw_epi_px"))
    traj_epi = safe_float(video.get("traj_epi_px"))
    traj_inlier2 = safe_float(video.get("traj_inlier2"))

    return {
        "video_id": video.get("video_id", ""),
        "stflow_c": stflow_c,
        "temporal_l1": temporal_l1,
        "cross_raw_epi_px": cross_raw,
        "traj_epi_px": traj_epi,
        "traj_inlier2": traj_inlier2,
        "temporal_quality": -temporal_l1,
        "cross_raw_quality": -cross_raw,
        "traj_quality": -traj_epi,
    }


def robust_scale(values):
    values = np.array(
        [v for v in values if v is not None and not math.isnan(float(v))],
        dtype=np.float64,
    )

    if values.size == 0:
        return 1.0

    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25

    if iqr > 1e-8:
        return float(iqr)

    std = float(np.std(values))
    if std > 1e-8:
        return std

    return 1.0


def method_score(record, centers, scales):
    values = []

    for key in QUALITY_KEYS:
        value = safe_float(record.get(key))
        if math.isnan(value):
            continue

        z = (value - centers[key]) / scales[key]
        values.append(z)

    if len(values) == 0:
        return float("nan")

    return float(np.mean(values))


def file_md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def real_signature(root, manifest_item, probe_frame, probe_view):
    frames = manifest_item["frames"]
    t = min(max(probe_frame, 0), len(frames) - 1)

    views = frames[t]["views"]
    v = min(max(probe_view, 0), len(views) - 1)

    real_path = views[v].get("real_image_path", None)
    if real_path is None:
        return None

    if not os.path.isabs(real_path):
        real_path = os.path.join(root, real_path)

    if not os.path.exists(real_path):
        return None

    return file_md5(real_path)


def load_method_data(base_root, method_name, folder, gate, max_videos):
    root = os.path.join(base_root, folder)
    result_path = os.path.join(root, f"stflow_traj_result_gate{gate}.json")
    manifest_path = os.path.join(root, "stflow_manifest.jsonl")

    if not os.path.exists(result_path):
        raise FileNotFoundError(result_path)

    result = load_json(result_path)
    videos = result["videos"][:max_videos]

    manifests = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    manifests.append(json.loads(line))
                    if len(manifests) >= max_videos:
                        break

    records = [metric_record(video) for video in videos]

    return {
        "method": method_name,
        "folder": folder,
        "root": root,
        "result_path": result_path,
        "manifest_path": manifest_path,
        "videos": videos,
        "records": records,
        "manifests": manifests,
    }


def collect_centers_scales(method_data):
    centers = {}
    scales = {}

    for key in QUALITY_KEYS:
        values = []
        for data in method_data.values():
            for record in data["records"]:
                value = safe_float(record.get(key))
                if not math.isnan(value):
                    values.append(value)

        values_np = np.array(values, dtype=np.float64)
        centers[key] = float(np.median(values_np)) if values_np.size > 0 else 0.0
        scales[key] = robust_scale(values)

    return centers, scales


def per_video_case(index, method_data, centers, scales, check_alignment, probe_frame, probe_view):
    method_records = {}
    quality_scores = {}

    signatures = {}

    for method, data in method_data.items():
        if index >= len(data["records"]):
            continue

        record = data["records"][index]
        method_records[method] = record
        quality_scores[method] = method_score(record, centers, scales)

        if check_alignment and index < len(data["manifests"]):
            signatures[method] = real_signature(
                data["root"],
                data["manifests"][index],
                probe_frame,
                probe_view,
            )

    metric_gaps = {}

    for display_key, quality_key in [
        ("stflow_c_gap", "stflow_c"),
        ("temporal_l1_gap", "temporal_l1"),
        ("cross_raw_epi_gap", "cross_raw_epi_px"),
        ("traj_epi_gap", "traj_epi_px"),
        ("traj_inlier2_gap", "traj_inlier2"),
    ]:
        values = [
            safe_float(record.get(quality_key))
            for record in method_records.values()
        ]
        values = [v for v in values if not math.isnan(v)]
        metric_gaps[display_key] = float(max(values) - min(values)) if values else float("nan")

    normalized_spreads = []

    for key in QUALITY_KEYS:
        values = [
            safe_float(record.get(key))
            for record in method_records.values()
        ]
        values = [v for v in values if not math.isnan(v)]

        if len(values) < 2:
            continue

        spread = (max(values) - min(values)) / scales[key]
        normalized_spreads.append(spread)

    combined_gap = float(np.mean(normalized_spreads)) if normalized_spreads else float("nan")

    sorted_methods = sorted(
        quality_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_method = sorted_methods[0][0] if sorted_methods else ""
    worst_method = sorted_methods[-1][0] if sorted_methods else ""

    alignment_ok = None
    if check_alignment and signatures:
        valid_sigs = [sig for sig in signatures.values() if sig is not None]
        alignment_ok = len(set(valid_sigs)) == 1 if valid_sigs else None

    case = {
        "video_index": index,
        "video_id": next(iter(method_records.values())).get("video_id", ""),
        "combined_gap": combined_gap,
        "best_method": best_method,
        "worst_method": worst_method,
        "quality_scores": quality_scores,
        "metric_gaps": metric_gaps,
        "alignment_ok": alignment_ok,
        "real_signatures": signatures,
        "methods": method_records,
    }

    return case


def write_csv(path, cases):
    fieldnames = [
        "rank",
        "video_index",
        "video_id",
        "combined_gap",
        "best_method",
        "worst_method",
        "stflow_c_gap",
        "temporal_l1_gap",
        "cross_raw_epi_gap",
        "traj_epi_gap",
        "traj_inlier2_gap",
        "alignment_ok",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, case in enumerate(cases):
            row = {
                "rank": rank,
                "video_index": case["video_index"],
                "video_id": case["video_id"],
                "combined_gap": case["combined_gap"],
                "best_method": case["best_method"],
                "worst_method": case["worst_method"],
                "alignment_ok": case["alignment_ok"],
            }
            row.update(case["metric_gaps"])
            writer.writerow(row)


def print_cases(cases, top_k):
    print("\n================ DISCRIMINATIVE CASES ================\n")
    print(
        f"{'rank':>4s} "
        f"{'idx':>4s} "
        f"{'video_id':22s} "
        f"{'gap':>8s} "
        f"{'best':14s} "
        f"{'worst':14s} "
        f"{'STCgap':>8s} "
        f"{'TempGap':>8s} "
        f"{'RawGap':>8s} "
        f"{'TrajGap':>8s} "
        f"{'In2Gap':>8s} "
        f"{'align':>8s}"
    )

    for rank, case in enumerate(cases[:top_k]):
        gaps = case["metric_gaps"]
        print(
            f"{rank:4d} "
            f"{case['video_index']:4d} "
            f"{case['video_id']:22s} "
            f"{case['combined_gap']:8.3f} "
            f"{case['best_method'][:14]:14s} "
            f"{case['worst_method'][:14]:14s} "
            f"{gaps['stflow_c_gap']:8.3f} "
            f"{gaps['temporal_l1_gap']:8.4f} "
            f"{gaps['cross_raw_epi_gap']:8.3f} "
            f"{gaps['traj_epi_gap']:8.3f} "
            f"{gaps['traj_inlier2_gap']:8.3f} "
            f"{str(case['alignment_ok']):>8s}"
        )


def main():
    args = create_parser().parse_args()

    output_json = args.output_json
    if output_json is None:
        output_json = os.path.join(
            args.base_root,
            f"discriminative_cases_gate{args.gate}.json",
        )

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(
            args.base_root,
            f"discriminative_cases_gate{args.gate}.csv",
        )

    method_data = {}
    for method_name, folder in DEFAULT_METHODS.items():
        root = os.path.join(args.base_root, folder)
        result_path = os.path.join(root, f"stflow_traj_result_gate{args.gate}.json")
        if not os.path.exists(result_path):
            print(f"[skip] missing {method_name}: {result_path}")
            continue

        method_data[method_name] = load_method_data(
            args.base_root,
            method_name,
            folder,
            args.gate,
            args.max_videos,
        )

    if len(method_data) < 2:
        raise RuntimeError("Need at least two methods to compare.")

    centers, scales = collect_centers_scales(method_data)

    video_count = min(len(data["records"]) for data in method_data.values())
    video_count = min(video_count, args.max_videos)

    cases = []
    for index in range(video_count):
        case = per_video_case(
            index,
            method_data,
            centers,
            scales,
            args.check_alignment,
            args.probe_frame,
            args.probe_view,
        )
        cases.append(case)

    cases = sorted(cases, key=lambda x: x["combined_gap"], reverse=True)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gate": args.gate,
                "base_root": args.base_root,
                "methods": list(method_data.keys()),
                "centers": centers,
                "scales": scales,
                "cases": cases,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_csv(output_csv, cases)
    print_cases(cases, args.top_k)

    print("\n[write]", output_json)
    print("[write]", output_csv)


if __name__ == "__main__":
    main()
