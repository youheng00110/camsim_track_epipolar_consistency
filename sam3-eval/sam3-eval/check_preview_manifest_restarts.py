from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

MANIFEST_NAMES = {
    "stflow_manifest.jsonl",
    "box_manifest.jsonl",
}

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
    """Make numeric JSON values stable enough for hashing."""
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
    """Build a path-independent identity from stable manifest metadata."""
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


def infer_method_and_rank(
    root: Path,
    manifest_path: Path,
) -> tuple[str, str]:
    relative_parts = manifest_path.relative_to(root).parts
    rank = ""

    for part in relative_parts:
        match = re.fullmatch(r"rank_(\d+)", part)
        if match:
            rank = f"rank_{int(match.group(1)):02d}"
            break

        match = re.search(r"__rank_(\d+)$", part)
        if match:
            rank = f"rank_{int(match.group(1)):02d}"
            break

    if "box_projection" in relative_parts:
        box_index = relative_parts.index("box_projection")
        method = relative_parts[box_index - 1] if box_index > 0 else "unknown"
    else:
        method = manifest_path.parent.name
        method = re.sub(r"__rank_\d+$", "", method)

    return method, rank


def is_merged_path(root: Path, manifest_path: Path) -> bool:
    relative_parts = manifest_path.relative_to(root).parts
    return any("merged" in part.lower() for part in relative_parts)


def find_manifests(
    root: Path,
    include_merged: bool,
) -> list[Path]:
    manifests: list[Path] = []

    for manifest_name in MANIFEST_NAMES:
        for path in root.rglob(manifest_name):
            if not path.is_file():
                continue
            if not include_merged and is_merged_path(root, path):
                continue
            manifests.append(path)

    return sorted(set(manifests))


def read_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    videos: list[dict[str, Any]] = []

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

    return videos, None


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


def analyze_manifest(
    root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    method, rank = infer_method_and_rank(root, manifest_path)
    videos, parse_error = read_manifest(manifest_path)

    result: dict[str, Any] = {
        "root": str(root),
        "method": method,
        "rank": rank,
        "manifest_name": manifest_path.name,
        "manifest_path": str(manifest_path),
        "video_count": len(videos),
        "first_video_id": "",
        "identity_mode": "",
        "first_repeat_lines": [],
        "first_repeat_video_ids": [],
        "prefix_match_lengths": [],
        "max_prefix_match_length": 0,
        "duplicate_groups": 0,
        "duplicate_video_entries": 0,
        "status": "OK",
        "parse_error": parse_error or "",
    }

    if parse_error:
        result["status"] = "INVALID_JSON"
        return result

    if not videos:
        result["status"] = "EMPTY"
        return result

    signatures: list[str | None] = []
    identity_modes: list[str] = []

    for video in videos:
        signature, mode = build_video_identity(video)
        signatures.append(signature)
        identity_modes.append(mode)

    result["first_video_id"] = str(
        videos[0].get("video_id", "")
    )
    result["identity_mode"] = identity_modes[0]

    if signatures[0] is None:
        result["status"] = "INSUFFICIENT_IDENTITY"
        return result

    first_signature = signatures[0]
    repeat_indices = [
        index
        for index in range(1, len(signatures))
        if signatures[index] == first_signature
    ]
    match_lengths = [
        prefix_match_length(signatures, index)
        for index in repeat_indices
    ]

    result["first_repeat_lines"] = [
        index + 1 for index in repeat_indices
    ]
    result["first_repeat_video_ids"] = [
        str(videos[index].get("video_id", ""))
        for index in repeat_indices
    ]
    result["prefix_match_lengths"] = match_lengths
    result["max_prefix_match_length"] = max(
        match_lengths,
        default=0,
    )

    valid_signatures = [
        signature
        for signature in signatures
        if signature is not None
    ]
    counts = Counter(valid_signatures)
    duplicate_counts = [
        count
        for count in counts.values()
        if count > 1
    ]
    result["duplicate_groups"] = len(duplicate_counts)
    result["duplicate_video_entries"] = sum(
        count - 1
        for count in duplicate_counts
    )

    if repeat_indices:
        if result["max_prefix_match_length"] >= 2:
            result["status"] = "CONFIRMED_RESTART_APPEND"
        else:
            result["status"] = "FIRST_VIDEO_REPEATED"
    elif result["duplicate_groups"] > 0:
        result["status"] = "OTHER_DUPLICATES"

    return result


def save_reports(
    output_dir: Path,
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "manifest_restart_report.json"
    csv_path = output_dir / "manifest_restart_report.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "root",
        "method",
        "rank",
        "manifest_name",
        "manifest_path",
        "video_count",
        "first_video_id",
        "identity_mode",
        "status",
        "first_repeat_lines",
        "first_repeat_video_ids",
        "prefix_match_lengths",
        "max_prefix_match_length",
        "duplicate_groups",
        "duplicate_video_entries",
        "parse_error",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = dict(result)
            for field in (
                "first_repeat_lines",
                "first_repeat_video_ids",
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
        "Detect preview manifests that were appended after a restart. "
        "The script checks whether the first video's stable pose/token "
        "sequence appears again later in the same JSONL manifest."
    )
)
parser.add_argument(
    "--roots",
    nargs="*",
    type=Path,
    default=DEFAULT_ROOTS,
    help="Evaluation roots. Defaults to nuplanhard1000 and 4camnuplanhard1000.",
)
parser.add_argument(
    "--include-merged",
    action="store_true",
    help="Also scan paths whose directory name contains 'merged'.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("./manifest_restart_check"),
    help="Directory for JSON and CSV reports.",
)
args = parser.parse_args()

all_results: list[dict[str, Any]] = []

for root_value in args.roots:
    root = root_value.expanduser().resolve()

    if not root.is_dir():
        print(f"[ROOT_MISSING] {root}", flush=True)
        continue

    manifests = find_manifests(
        root=root,
        include_merged=args.include_merged,
    )
    print(
        f"[ROOT] {root} manifests={len(manifests)}",
        flush=True,
    )

    for index, manifest_path in enumerate(manifests, start=1):
        result = analyze_manifest(
            root=root,
            manifest_path=manifest_path,
        )
        all_results.append(result)

        repeats = result["first_repeat_lines"]
        repeat_text = ",".join(str(value) for value in repeats)
        if not repeat_text:
            repeat_text = "-"

        print(
            f"[{index:03d}/{len(manifests):03d}] "
            f"{result['status']:<26} "
            f"method={result['method']} "
            f"rank={result['rank'] or '-'} "
            f"videos={result['video_count']} "
            f"first_repeat_lines={repeat_text} "
            f"prefix_match={result['max_prefix_match_length']}",
            flush=True,
        )

if not all_results:
    raise RuntimeError("No manifest was analyzed.")

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

print("\nSuspicious manifests:", flush=True)

suspicious = [
    result
    for result in all_results
    if result["status"] not in {"OK"}
]

if suspicious:
    for result in suspicious:
        print(
            f"- {result['status']}: "
            f"{result['method']} {result['rank']} "
            f"{result['manifest_path']} "
            f"repeat_lines={result['first_repeat_lines']} "
            f"prefix_match={result['max_prefix_match_length']}",
            flush=True,
        )
else:
    print("- none", flush=True)

print(f"\nJSON report: {json_path}", flush=True)
print(f"CSV report:  {csv_path}", flush=True)
