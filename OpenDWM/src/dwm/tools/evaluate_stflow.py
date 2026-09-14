import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dwm.metrics.stflow import STFlowEvaluator


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate ST-Flow consistency from generated multi-view manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to stflow_manifest.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save result JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help=(
            "Start frame for ST-Flow/Traj evaluation. "
            "Default None means auto: skip pure GT-reference edges but keep transition edge."
        ),
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--loftr-confidence",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--camera-pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated camera-pair whitelist, e.g. "
            "CAM_02__CAM_05,CAM_01__CAM_06. If None, select by dataset_name."
        ),
    )
    parser.add_argument(
        "--pair-policy",
        type=str,
        default="dataset",
        choices=["dataset", "ring", "waymo"],
        help=(
            "Camera pair selection policy. dataset: waymo uses fixed overlapping "
            "pairs and nuplan uses ring pairs."
        ),
    )
    parser.add_argument(
        "--cross-gate-px",
        type=float,
        default=12.0,
        help="Epipolar gate threshold in pixels for cross-view LoFTR matches.",
    )
    return parser


def load_manifest_items(manifest_path, max_videos):
    items = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            items.append(json.loads(line))

            if max_videos is not None and len(items) >= max_videos:
                break

    return items


def add_coverage_score(result):
    raw_edges = float(result.get("num_cross_raw_edges", 0.0))
    gated_edges = float(result.get("num_cross_edges", 0.0))
    inlier_ratio = float(result.get("cross_inlier_ratio", 0.0))
    stflow_score = float(result.get("stflow_score", 0.0))

    if raw_edges <= 0:
        edge_coverage = 0.0
    else:
        edge_coverage = gated_edges / raw_edges

    result["edge_coverage"] = edge_coverage
    result["stflow_c_score"] = stflow_score * edge_coverage * inlier_ratio

    return result


def aggregate_video_results(video_results):
    scalar_keys = [
        "temporal_l1",
        "cross_raw_epi_px",
        "cross_epi_px",
        "cross_inlier_ratio",
        "cycle_epi_px",
        "stflow_error",
        "stflow_score",
        "stflow_d_error",
        "stflow_d_score",
        "stflow_c_score",
        "edge_coverage",
        "cycle_coverage",
        "traj_epi_px",
        "traj_inlier2",
        "traj_inlier4",
        "num_temporal_edges",
        "num_cross_raw_edges",
        "num_cross_edges",
        "num_cycle_edges",
        "num_traj_edges",
    ]
    output = {
        "num_videos": len(video_results),
        "videos": video_results,
        "mean": {},
    }

    for key in scalar_keys:
        values = []
        for result in video_results:
            value = result.get(key, float("nan"))
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            values.append(float(value))

        output["mean"][key] = float(np.mean(values)) if len(values) > 0 else float("nan")

    pair_values = {}
    for result in video_results:
        for pair_key, pair_result in result.get("pair_stats", {}).items():
            if pair_key not in pair_values:
                pair_values[pair_key] = {
                    "cross_raw_epi_px": [],
                    "cross_epi_px": [],
                    "cycle_epi_px": [],
                    "match_count": [],
                    "inlier_count": [],
                    "cross_inlier_ratio": [],
                }

            for metric_key in pair_values[pair_key]:
                metric_value = pair_result.get(metric_key, float("nan"))
                if metric_value is None:
                    continue
                if isinstance(metric_value, float) and np.isnan(metric_value):
                    continue
                pair_values[pair_key][metric_key].append(float(metric_value))

    output["camera_pair_mean"] = {}
    for pair_key, metric_values in pair_values.items():
        output["camera_pair_mean"][pair_key] = {}
        for metric_key, values in metric_values.items():
            output["camera_pair_mean"][pair_key][metric_key] = (
                float(np.mean(values)) if len(values) > 0 else float("nan")
            )

    mean_score = output["mean"].get("stflow_score", float("nan"))
    mean_coverage = output["mean"].get("edge_coverage", float("nan"))
    mean_inlier = output["mean"].get("cross_inlier_ratio", float("nan"))

    output["mean"]["stflow_c_score_per_video_mean"] = output["mean"].get(
        "stflow_c_score",
        float("nan"),
    )

    if (
        not np.isnan(mean_score)
        and not np.isnan(mean_coverage)
        and not np.isnan(mean_inlier)
    ):
        output["mean"]["stflow_c_score"] = float(
            mean_score * mean_coverage * mean_inlier
        )
    else:
        output["mean"]["stflow_c_score"] = float("nan")

    return output


def main():
    parser = create_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest_dir = str(manifest_path.parent)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = STFlowEvaluator(
        device=args.device,
        frame_stride=args.frame_stride,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        loftr_confidence=args.loftr_confidence,
        camera_pairs=args.camera_pairs,
        pair_policy=args.pair_policy,
        start_frame=args.start_frame,
        cross_gate_px=args.cross_gate_px,
    )

    manifest_items = load_manifest_items(args.manifest, args.max_videos)
    print(f"[stflow] Loaded {len(manifest_items)} videos from {args.manifest}")

    video_results = []
    for index, item in enumerate(manifest_items):
        with torch.no_grad():
            result = evaluator.evaluate_video(item, manifest_dir)
            result = add_coverage_score(result)

        video_results.append(result)
        print(
            "[stflow] {}/{} {} | dscore={:.3f} cscore={:.3f} score={:.3f} temp={:.5f} cross={:.3f} cycle={:.3f}".format(
                index + 1,
                len(manifest_items),
                result.get("video_id", ""),
                result.get("stflow_d_score", float("nan")),
                result.get("stflow_c_score", float("nan")),
                result["stflow_score"],
                result["temporal_l1"],
                result["cross_epi_px"],
                result["cycle_epi_px"],
            )
        )

    output = aggregate_video_results(video_results)
    output["eval_config"] = {
        "frame_stride": args.frame_stride,
        "min_matches": args.min_matches,
        "max_matches": args.max_matches,
        "loftr_confidence": args.loftr_confidence,
        "camera_pairs": args.camera_pairs,
        "pair_policy": args.pair_policy,
        "cross_gate_px": args.cross_gate_px,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[stflow] Saved result to {output_path}")
    print(json.dumps(output["mean"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()