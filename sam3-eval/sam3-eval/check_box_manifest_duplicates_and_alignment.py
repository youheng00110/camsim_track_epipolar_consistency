from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = [
    Path(
        "/inspire/qb-ilm/project/advanced-machine-learning/"
        "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000"
    ),
    Path(
        "/inspire/qb-ilm/project/advanced-machine-learning/"
        "yanjunchi-24040/camsim_lyh/output/eval/4camnuplanhard1000"
    ),
]

STABLE_VIDEO_FIELDS = (
    "sample_id",
    "dataset_index",
    "scene_token",
    "start_lidarpc_token",
    "end_lidarpc_token",
    "start_timestamp",
    "end_timestamp",
)

STABLE_FRAME_FIELDS = (
    "timestamp",
    "sample_token",
    "lidarpc_token",
    "frame_token",
)


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


def build_video_identity(video: dict[str, Any]) -> tuple[str | None, str]:
    identity: dict[str, Any] = {}

    for field in STABLE_VIDEO_FIELDS:
        if field in video and video[field] is not None:
            identity[field] = normalize_json_value(video[field])

    frames = video.get("frames", [])
    frame_identities: list[dict[str, Any]] = []
    pose_count = 0
    token_count = 0

    for frame in frames:
        frame_identity: dict[str, Any] = {
            "frame_index": frame.get("frame_index"),
        }

        for field in STABLE_FRAME_FIELDS:
            if field in frame and frame[field] is not None:
                frame_identity[field] = normalize_json_value(frame[field])
                token_count += 1

        if "T_ego_to_world" in frame and frame["T_ego_to_world"] is not None:
            frame_identity["T_ego_to_world"] = normalize_json_value(
                frame["T_ego_to_world"]
            )
            pose_count += 1

        frame_identities.append(frame_identity)

    identity["frames"] = frame_identities

    if identity.keys() == {"frames"} and pose_count == 0 and token_count == 0:
        return None, "insufficient_manifest_identity"

    if pose_count > 0:
        mode = f"ego_pose_sequence({pose_count})"
    elif token_count > 0:
        mode = f"frame_tokens({token_count})"
    else:
        mode = "stable_video_fields"

    payload = json.dumps(
        normalize_json_value(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest(), mode


def read_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    videos: list[dict[str, Any]] = []

    print(
        f"  [READ_START] {manifest_path}",
        flush=True,
    )

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                return videos, (
                    f"invalid JSON at line {line_number}: {error}"
                )

            if not isinstance(item, dict):
                return videos, (
                    f"line {line_number} is not a JSON object"
                )

            videos.append(item)

            if len(videos) % 50 == 0:
                print(
                    f"  [READ_PROGRESS] "
                    f"{manifest_path.name} "
                    f"videos={len(videos)}",
                    flush=True,
                )

    print(
        f"  [READ_DONE] "
        f"{manifest_path.name} "
        f"videos={len(videos)}",
        flush=True,
    )

    return videos, None


def signatures_from_videos(
    videos: list[dict[str, Any]],
) -> tuple[list[str | None], str]:
    signatures: list[str | None] = []
    first_mode = ""

    for index, video in enumerate(videos):
        signature, mode = build_video_identity(video)
        signatures.append(signature)

        if index == 0:
            first_mode = mode

    return signatures, first_mode


def prefix_match_length(
    signatures: list[str | None],
    restart_index: int,
) -> int:
    match_length = 0
    maximum = min(
        restart_index,
        len(signatures) - restart_index,
    )

    for offset in range(maximum):
        left = signatures[offset]
        right = signatures[restart_index + offset]

        if left is None or right is None or left != right:
            break

        match_length += 1

    return match_length


def expected_count_for_root(root: Path) -> int | None:
    name = root.name.lower()

    if name == "nuplanhard1000":
        return 258

    if name == "4camnuplanhard1000":
        return 263

    return None


def infer_method_rank_and_rgb(
    root: Path,
    box_manifest: Path,
) -> tuple[str, str, Path | None]:
    relative_parts = box_manifest.relative_to(root).parts

    if "box_projection" not in relative_parts:
        return "unknown", "", None

    box_index = relative_parts.index("box_projection")
    method = relative_parts[box_index - 1] if box_index > 0 else "unknown"

    rank = ""
    for part in relative_parts[box_index + 1:]:
        match = re.fullmatch(r"rank_(\d+)", part)

        if match:
            rank = f"rank_{int(match.group(1)):02d}"
            break

    if not rank:
        return method, rank, None

    method_root = root / method
    rgb_manifest = method_root / rank / "stflow_manifest.jsonl"
    return method, rank, rgb_manifest


def compare_signature_sequences(
    box_signatures: list[str | None],
    rgb_signatures: list[str | None],
) -> dict[str, Any]:
    compare_count = min(
        len(box_signatures),
        len(rgb_signatures),
    )
    matching_count = 0
    first_mismatch_line: int | None = None

    for index in range(compare_count):
        box_signature = box_signatures[index]
        rgb_signature = rgb_signatures[index]

        if (
            box_signature is not None
            and rgb_signature is not None
            and box_signature == rgb_signature
        ):
            matching_count += 1
            continue

        first_mismatch_line = index + 1
        break

    exact_match = (
        len(box_signatures) == len(rgb_signatures)
        and matching_count == len(box_signatures)
    )

    return {
        "rgb_count": len(rgb_signatures),
        "aligned_prefix": matching_count,
        "first_mismatch_line": first_mismatch_line,
        "exact_match": exact_match,
    }


def analyze_box_manifest(
    root: Path,
    box_manifest: Path,
) -> dict[str, Any]:
    method, rank, rgb_manifest = infer_method_rank_and_rgb(
        root,
        box_manifest,
    )
    expected_count = expected_count_for_root(root)
    box_videos, parse_error = read_manifest(box_manifest)

    result: dict[str, Any] = {
        "root": str(root),
        "method": method,
        "rank": rank,
        "box_manifest": str(box_manifest),
        "rgb_manifest": str(rgb_manifest) if rgb_manifest else "",
        "expected_count": expected_count,
        "box_count": len(box_videos),
        "rgb_count": None,
        "identity_mode": "",
        "first_repeat_lines": [],
        "prefix_match_lengths": [],
        "max_restart_prefix": 0,
        "duplicate_groups": 0,
        "duplicate_entries": 0,
        "aligned_prefix": 0,
        "first_rgb_mismatch_line": None,
        "box_rgb_exact_match": False,
        "status": "OK",
        "parse_error": parse_error or "",
    }

    if parse_error:
        result["status"] = "INVALID_BOX_JSON"
        return result

    if not box_videos:
        result["status"] = "EMPTY_BOX"
        return result

    box_signatures, identity_mode = signatures_from_videos(box_videos)
    result["identity_mode"] = identity_mode

    if box_signatures[0] is None:
        result["status"] = "INSUFFICIENT_BOX_IDENTITY"
        return result

    first_signature = box_signatures[0]
    repeat_indices = [
        index
        for index in range(1, len(box_signatures))
        if box_signatures[index] == first_signature
    ]
    restart_lengths = [
        prefix_match_length(box_signatures, index)
        for index in repeat_indices
    ]

    result["first_repeat_lines"] = [
        index + 1 for index in repeat_indices
    ]
    result["prefix_match_lengths"] = restart_lengths
    result["max_restart_prefix"] = max(
        restart_lengths,
        default=0,
    )

    valid_signatures = [
        signature
        for signature in box_signatures
        if signature is not None
    ]
    counts = Counter(valid_signatures)
    duplicate_counts = [
        count
        for count in counts.values()
        if count > 1
    ]
    result["duplicate_groups"] = len(duplicate_counts)
    result["duplicate_entries"] = sum(
        count - 1
        for count in duplicate_counts
    )

    box_duplicate_status = "OK"

    if repeat_indices and result["max_restart_prefix"] >= 2:
        box_duplicate_status = "BOX_RESTART_APPEND"
    elif repeat_indices:
        box_duplicate_status = "BOX_FIRST_VIDEO_REPEATED"
    elif result["duplicate_groups"] > 0:
        box_duplicate_status = "BOX_OTHER_DUPLICATES"

    count_status = "OK"

    if expected_count is not None:
        if len(box_videos) < expected_count:
            count_status = "BOX_INCOMPLETE"
        elif len(box_videos) > expected_count:
            count_status = "BOX_OVERSIZED"

    alignment_status = "OK"

    if rgb_manifest is None or not rgb_manifest.is_file():
        alignment_status = "RGB_MANIFEST_MISSING"
    else:
        rgb_videos, rgb_error = read_manifest(rgb_manifest)

        if rgb_error:
            alignment_status = "INVALID_RGB_JSON"
        else:
            rgb_signatures, _ = signatures_from_videos(rgb_videos)
            comparison = compare_signature_sequences(
                box_signatures,
                rgb_signatures,
            )
            result["rgb_count"] = comparison["rgb_count"]
            result["aligned_prefix"] = comparison["aligned_prefix"]
            result["first_rgb_mismatch_line"] = comparison[
                "first_mismatch_line"
            ]
            result["box_rgb_exact_match"] = comparison["exact_match"]

            if comparison["exact_match"]:
                alignment_status = "OK"
            elif comparison["aligned_prefix"] == 0:
                alignment_status = "BOX_RGB_MISMATCH_FROM_START"
            else:
                alignment_status = "BOX_RGB_PARTIAL_ALIGNMENT"

    statuses = [
        status
        for status in (
            box_duplicate_status,
            count_status,
            alignment_status,
        )
        if status != "OK"
    ]

    result["status"] = "+".join(statuses) if statuses else "OK"
    return result


def find_box_manifests(
    root: Path,
    include_merged: bool,
) -> list[Path]:
    manifests: list[Path] = []
    visited_directories = 0

    print(
        f"[SCAN_START] root={root}",
        flush=True,
    )

    prune_names = {
        "images",
        "paired_real",
        "visualizations",
        "masks",
        "__pycache__",
    }

    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name
            for name in directory_names
            if name not in prune_names
            and not name.startswith(".")
            and (
                include_merged
                or "merged" not in name.lower()
            )
        ]

        visited_directories += 1

        if visited_directories % 50 == 0:
            print(
                f"[SCAN_PROGRESS] "
                f"directories={visited_directories} "
                f"manifests={len(manifests)}",
                flush=True,
            )

        if "box_manifest.jsonl" not in file_names:
            continue

        manifest_path = (
            Path(current_root) / "box_manifest.jsonl"
        )
        relative_parts = manifest_path.relative_to(root).parts

        if (
            not include_merged
            and any(
                "merged" in part.lower()
                for part in relative_parts
            )
        ):
            continue

        manifests.append(manifest_path)

        print(
            f"[MANIFEST_FOUND] "
            f"{manifest_path.relative_to(root)}",
            flush=True,
        )

    manifests.sort()

    print(
        f"[SCAN_DONE] "
        f"directories={visited_directories} "
        f"box_manifests={len(manifests)}",
        flush=True,
    )

    return manifests


def save_reports(
    output_dir: Path,
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "box_manifest_check.json"
    csv_path = output_dir / "box_manifest_check.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "root",
        "method",
        "rank",
        "status",
        "expected_count",
        "box_count",
        "rgb_count",
        "duplicate_groups",
        "duplicate_entries",
        "first_repeat_lines",
        "prefix_match_lengths",
        "max_restart_prefix",
        "aligned_prefix",
        "first_rgb_mismatch_line",
        "box_rgb_exact_match",
        "identity_mode",
        "box_manifest",
        "rgb_manifest",
        "parse_error",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = dict(result)

            for field in (
                "first_repeat_lines",
                "prefix_match_lengths",
            ):
                row[field] = json.dumps(
                    row[field],
                    ensure_ascii=False,
                )

            writer.writerow(row)

    return json_path, csv_path


parser = argparse.ArgumentParser(
    description=(
        "Check box_manifest.jsonl for restart append, duplicate videos, "
        "expected count, and alignment with the corresponding RGB manifest."
    )
)
parser.add_argument(
    "--roots",
    nargs="*",
    type=Path,
    default=DEFAULT_ROOTS,
)
parser.add_argument(
    "--include-merged",
    action="store_true",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("./box_manifest_check"),
)
args = parser.parse_args()

all_results: list[dict[str, Any]] = []

for root_value in args.roots:
    root = root_value.expanduser().resolve()

    if not root.is_dir():
        print(f"[ROOT_MISSING] {root}", flush=True)
        continue

    manifests = find_box_manifests(
        root=root,
        include_merged=args.include_merged,
    )
    print(
        f"[ROOT] {root} box_manifests={len(manifests)}",
        flush=True,
    )

    for index, manifest in enumerate(manifests, start=1):
        result = analyze_box_manifest(
            root=root,
            box_manifest=manifest,
        )
        all_results.append(result)

        repeats = ",".join(
            str(value)
            for value in result["first_repeat_lines"]
        )
        if not repeats:
            repeats = "-"

        print(
            f"[{index:03d}/{len(manifests):03d}] "
            f"{result['status']:<60} "
            f"method={result['method']} "
            f"rank={result['rank'] or '-'} "
            f"box={result['box_count']} "
            f"rgb={result['rgb_count']} "
            f"repeat={repeats} "
            f"restart_prefix={result['max_restart_prefix']} "
            f"aligned_prefix={result['aligned_prefix']}",
            flush=True,
        )

if not all_results:
    raise RuntimeError("No box manifest was analyzed.")

json_path, csv_path = save_reports(
    output_dir=args.output_dir.expanduser().resolve(),
    results=all_results,
)

status_counts = Counter(
    result["status"]
    for result in all_results
)

print("\n================ SUMMARY ================", flush=True)

for status, count in sorted(status_counts.items()):
    print(f"{status}: {count}", flush=True)

print("\nProblematic box manifests:", flush=True)

problematic = [
    result
    for result in all_results
    if result["status"] != "OK"
]

if problematic:
    for result in problematic:
        print(
            f"- {result['status']}: "
            f"{result['method']} {result['rank']} "
            f"box={result['box_count']} "
            f"rgb={result['rgb_count']} "
            f"repeat={result['first_repeat_lines']} "
            f"aligned_prefix={result['aligned_prefix']} "
            f"path={result['box_manifest']}",
            flush=True,
        )
else:
    print("- none", flush=True)

print(f"\nJSON report: {json_path}", flush=True)
print(f"CSV report:  {csv_path}", flush=True)
