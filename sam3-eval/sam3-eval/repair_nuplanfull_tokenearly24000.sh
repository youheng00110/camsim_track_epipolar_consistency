#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BASE/_manifest_repair_backup_$STAMP"

mkdir -p "$BACKUP_DIR"

echo "BASE:       $BASE"
echo "BACKUP:     $BACKUP_DIR"
echo

python -u - "$BASE" "$BACKUP_DIR" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

base = Path(sys.argv[1]).resolve()
backup_root = Path(sys.argv[2]).resolve()

plans = [
    {
        "method": "nuplanfull_preview_paired_200",
        "expected_total": 327,
        "keep_start": 0,
        "keep_count": 258,
        "expected_repeat_index": 258,
    },
    {
        "method": "tokenearly24000_preview_paired_200",
        "expected_total": 347,
        "keep_start": 89,
        "keep_count": 258,
        "expected_repeat_index": 89,
    },
]


def video_signature(record: dict[str, Any]) -> str:
    frames = record.get("frames", [])
    payload = []

    for frame in frames:
        payload.append(
            {
                "frame_index": frame.get("frame_index"),
                "timestamp": frame.get("timestamp"),
                "T_ego_to_world": frame.get("T_ego_to_world"),
            }
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def resolve_existing_path(
    raw_path: str,
    rank_root: Path,
    method_root: Path,
    base_root: Path,
) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path

    candidates = [
        rank_root / path,
        method_root / path,
        base_root / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def replace_path_component(raw_path: str, old_id: str, new_id: str) -> str:
    path = Path(raw_path)
    parts = list(path.parts)
    replaced = False

    for index, part in enumerate(parts):
        if part == old_id:
            parts[index] = new_id
            replaced = True

    if not replaced:
        raise RuntimeError(
            f"Path does not contain video_id {old_id}: {raw_path}"
        )

    return str(Path(*parts))


def collect_video_dirs(
    record: dict[str, Any],
    rank_root: Path,
    method_root: Path,
    base_root: Path,
) -> set[Path]:
    video_id = str(record.get("video_id", ""))
    directories: set[Path] = set()

    if not video_id:
        raise RuntimeError("Manifest record has no video_id")

    for frame in record.get("frames", []):
        for view in frame.get("views", []):
            for field in ("image_path", "real_image_path"):
                raw_path = view.get(field)

                if not raw_path:
                    continue

                resolved = resolve_existing_path(
                    str(raw_path),
                    rank_root,
                    method_root,
                    base_root,
                )
                parts = list(resolved.parts)

                if video_id not in parts:
                    continue

                video_part_index = parts.index(video_id)
                video_dir = Path(*parts[: video_part_index + 1]).resolve()

                if not video_dir.is_relative_to(rank_root.resolve()):
                    raise RuntimeError(
                        f"Refuse to modify a video directory outside rank root: "
                        f"{video_dir} (rank root: {rank_root})"
                    )

                directories.add(video_dir)

    if not directories:
        for root_name in ("images", "paired_real"):
            candidate = rank_root / root_name / video_id

            if candidate.exists():
                directories.add(candidate)

    return directories


for plan in plans:
    method_name = str(plan["method"])
    method_root = base / method_name

    if not method_root.is_dir():
        raise FileNotFoundError(f"Method directory not found: {method_root}")

    print("=" * 80, flush=True)
    print(f"METHOD: {method_name}", flush=True)
    print("=" * 80, flush=True)

    for rank_index in range(4):
        rank_name = f"rank_{rank_index:02d}"
        rank_root = method_root / rank_name
        manifest_path = rank_root / "stflow_manifest.jsonl"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records = []

        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON: {manifest_path}:{line_number}: {error}"
                    ) from error

        expected_total = int(plan["expected_total"])
        keep_start = int(plan["keep_start"])
        keep_count = int(plan["keep_count"])
        repeat_index = int(plan["expected_repeat_index"])

        if len(records) != expected_total:
            raise RuntimeError(
                f"Unexpected line count for {manifest_path}: "
                f"{len(records)} != {expected_total}"
            )

        if video_signature(records[0]) != video_signature(records[repeat_index]):
            raise RuntimeError(
                f"Expected restart boundary was not verified: {manifest_path}, "
                f"line {repeat_index + 1} does not match line 1"
            )

        kept_records = records[keep_start : keep_start + keep_count]
        removed_records = records[:keep_start] + records[keep_start + keep_count :]

        if len(kept_records) != keep_count:
            raise RuntimeError(
                f"Kept record count mismatch for {manifest_path}: "
                f"{len(kept_records)} != {keep_count}"
            )

        kept_signatures = [video_signature(record) for record in kept_records]

        if len(set(kept_signatures)) != keep_count:
            raise RuntimeError(
                f"The selected 258 records still contain duplicate videos: "
                f"{manifest_path}"
            )

        backup_path = backup_root / method_name / rank_name / manifest_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, backup_path)

        print(
            f"[{rank_name}] backup manifest -> {backup_path}",
            flush=True,
        )

        removed_directories: set[Path] = set()

        for record in removed_records:
            removed_directories.update(
                collect_video_dirs(
                    record,
                    rank_root,
                    method_root,
                    base,
                )
            )

        for directory in sorted(removed_directories, key=lambda path: len(path.parts), reverse=True):
            if directory.exists():
                shutil.rmtree(directory)
                print(f"[{rank_name}] deleted {directory}", flush=True)

        normalized_records = []

        for new_index, record in enumerate(kept_records):
            old_video_id = str(record.get("video_id", ""))

            if "_video_" in old_video_id:
                dataset_prefix = old_video_id.rsplit("_video_", 1)[0]
            else:
                dataset_prefix = "nuplan"

            new_video_id = f"{dataset_prefix}_video_{new_index:06d}"
            old_directories = collect_video_dirs(
                record,
                rank_root,
                method_root,
                base,
            )

            if old_video_id != new_video_id:
                for old_directory in sorted(old_directories):
                    if not old_directory.exists():
                        raise FileNotFoundError(
                            f"Video directory missing before rename: {old_directory}"
                        )

                    if old_directory.name != old_video_id:
                        raise RuntimeError(
                            f"Unexpected video directory name: {old_directory}"
                        )

                    new_directory = old_directory.with_name(new_video_id)

                    if new_directory.exists():
                        raise FileExistsError(
                            f"Rename target already exists: {new_directory}"
                        )

                    os.rename(old_directory, new_directory)
                    print(
                        f"[{rank_name}] renamed {old_directory.name} -> "
                        f"{new_directory.name}",
                        flush=True,
                    )

                for frame in record.get("frames", []):
                    for view in frame.get("views", []):
                        for field in ("image_path", "real_image_path"):
                            raw_path = view.get(field)

                            if raw_path:
                                view[field] = replace_path_component(
                                    str(raw_path),
                                    old_video_id,
                                    new_video_id,
                                )

            record["video_id"] = new_video_id

            if "video_index" in record:
                record["video_index"] = new_index

            normalized_records.append(record)

        temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")

        with temporary_manifest.open("w", encoding="utf-8") as file:
            for record in normalized_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        os.replace(temporary_manifest, manifest_path)

        with manifest_path.open("r", encoding="utf-8") as file:
            repaired_count = sum(1 for line in file if line.strip())

        if repaired_count != keep_count:
            raise RuntimeError(
                f"Repaired manifest count mismatch: {manifest_path}: "
                f"{repaired_count} != {keep_count}"
            )

        print(
            f"[{rank_name}] repaired manifest: {repaired_count} videos",
            flush=True,
        )

print("All rank manifests repaired successfully.", flush=True)
PY

for METHOD in \
  nuplanfull_preview_paired_200 \
  tokenearly24000_preview_paired_200
do
  MERGED_DIR="$BASE/${METHOD}_merged1000"

  if [ -e "$MERGED_DIR" ]; then
    echo "Delete old merged directory: $MERGED_DIR"
    rm -rf -- "$MERGED_DIR"
  else
    echo "Merged directory does not exist, skip: $MERGED_DIR"
  fi
done

echo
echo "Repair completed."
echo "Manifest backups: $BACKUP_DIR"
echo
echo "Next: rerun restart check, then rerun merge."
