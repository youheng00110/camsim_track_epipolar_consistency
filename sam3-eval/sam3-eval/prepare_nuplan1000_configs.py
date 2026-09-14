from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Callable

import yaml

from shared_box_projection import video_pose_signature


BASE = Path(
    "/inspire/qb-ilm/project/advanced-machine-learning/"
    "yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000"
).resolve()

SHARED_BOX_ROOT = (
    BASE / "shared_box_preview_paired_200"
).resolve()

BASE_CONFIG = Path(
    "config_shared_box_nuplan_test.yaml"
).resolve()

CONFIG_ROOT = Path(
    "configs_nuplan1000_box3d"
).resolve()

OUTPUT_ROOT = (
    BASE / "sam3_nuplanhard1000_box3d"
).resolve()


ALIASES = [
    "plucker",
    "box",
    "implicit",
    "full",
    "petr",
    "pvonly",
    "nocondition",
    "token18000",
    "token24000",
    "tvself",
]


def matches_alias(alias: str, method_name: str) -> bool:
    name = method_name.lower()

    if "dwmpreview" in name:
        return False

    if alias == "plucker":
        return "plucker" in name

    if alias == "box":
        return (
            name.startswith("box")
            or "box30000" in name
        )

    if alias == "implicit":
        return "implicit" in name

    if alias == "full":
        return (
            "full" in name
            and "tvself" not in name
        )

    if alias == "petr":
        return "petr" in name

    if alias == "pvonly":
        return "pvonly" in name

    if alias == "nocondition":
        return "nocondition" in name

    if alias == "token18000":
        return (
            "token" in name
            and "18000" in name
            and "24000" not in name
        )

    if alias == "token24000":
        return (
            "token" in name
            and "24000" in name
        )

    if alias == "tvself":
        return "tvself" in name

    raise KeyError(alias)


def read_video_signatures(
    manifest_path: Path,
) -> tuple[int, set[str]]:
    line_count = 0
    signatures: set[str] = set()

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            line_count += 1
            signatures.add(
                video_pose_signature(record)
            )

    return line_count, signatures


if not BASE_CONFIG.is_file():
    raise FileNotFoundError(
        f"Base config does not exist: {BASE_CONFIG}"
    )

if not SHARED_BOX_ROOT.is_dir():
    raise FileNotFoundError(
        f"Shared Box root does not exist: "
        f"{SHARED_BOX_ROOT}"
    )

manifest_paths = sorted(
    path
    for path in BASE.rglob(
        "stflow_manifest.jsonl"
    )
    if path.parent.name.endswith(
        "_merged1000"
    )
)

method_manifests: dict[str, Path] = {}

for manifest_path in manifest_paths:
    relative_parent = (
        manifest_path.parent.relative_to(BASE)
    )
    method_name = str(relative_parent)
    method_name = method_name.replace(
        os.sep,
        "__",
    )
    method_manifests[method_name] = manifest_path

print("Detected merged1000 methods:")
for method_name in sorted(method_manifests):
    print(" ", method_name)

selected: dict[str, str] = {}

for alias in ALIASES:
    candidates = [
        method_name
        for method_name in method_manifests
        if matches_alias(alias, method_name)
    ]

    if len(candidates) != 1:
        print()
        print(
            f"ERROR: {alias!r} matched "
            f"{len(candidates)} methods:"
        )
        for candidate in candidates:
            print(" ", candidate)
        raise RuntimeError(
            f"Cannot uniquely resolve {alias}"
        )

    selected[alias] = candidates[0]

signature_sets: dict[str, set[str]] = {}

print()
print("Selected methods:")

for alias in ALIASES:
    method_name = selected[alias]
    manifest_path = method_manifests[
        method_name
    ]

    line_count, signatures = (
        read_video_signatures(manifest_path)
    )

    print(
        f"  {alias:12s} -> {method_name} "
        f"lines={line_count} "
        f"unique_videos={len(signatures)}"
    )

    if line_count != 1000:
        raise RuntimeError(
            f"{alias} manifest contains "
            f"{line_count} records, expected 1000: "
            f"{manifest_path}"
        )

    if len(signatures) != 1000:
        raise RuntimeError(
            f"{alias} contains "
            f"{len(signatures)} unique video "
            f"signatures, expected 1000"
        )

    signature_sets[alias] = signatures

reference_alias = "implicit"
reference_signatures = signature_sets[
    reference_alias
]

for alias in ALIASES:
    missing = (
        reference_signatures
        - signature_sets[alias]
    )
    extra = (
        signature_sets[alias]
        - reference_signatures
    )

    if missing or extra:
        raise RuntimeError(
            f"{alias} does not use the same "
            f"1000 videos as implicit: "
            f"missing={len(missing)}, "
            f"extra={len(extra)}"
        )

base_config = yaml.safe_load(
    BASE_CONFIG.read_text(encoding="utf-8")
)

CONFIG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

for alias in ALIASES:
    method_name = selected[alias]
    config = copy.deepcopy(base_config)

    config["paths"]["preview_root"] = str(
        BASE
    )
    config["paths"]["shared_box_root"] = str(
        SHARED_BOX_ROOT
    )
    config["paths"]["output_dir"] = str(
        OUTPUT_ROOT / alias
    )

    config["preview"][
        "manifest_glob"
    ] = "**/*_merged1000/stflow_manifest.jsonl"

    config["preview"][
        "include_methods"
    ] = [method_name]

    config["preview"][
        "exclude_methods"
    ] = ["*dwmpreview*"]

    config["preview"][
        "skip_reference_frames"
    ] = True

    config["shared_box"][
        "manifest_glob"
    ] = "**/box_manifest.jsonl"

    config["shared_box"][
        "strict_match"
    ] = True

    config["sources"] = {
        "generated": {
            "type": "preview_generated",
            "group_by_manifest": True,
        }
    }

    config["runtime"]["limit_frames"] = 0
    config["runtime"]["overwrite"] = True

    config["visualization"]["enabled"] = True
    config["visualization"][
        "max_frames_per_source"
    ] = 32
    config["visualization"][
        "draw_cuboid"
    ] = True
    config["visualization"][
        "draw_detection_box"
    ] = True

    config_path = (
        CONFIG_ROOT
        / f"generated_{alias}.yaml"
    )

    config_path.write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

# Paired-real 只从 implicit manifest 读取一次。
real_config = copy.deepcopy(base_config)

real_config["paths"]["preview_root"] = str(
    BASE
)
real_config["paths"]["shared_box_root"] = str(
    SHARED_BOX_ROOT
)
real_config["paths"]["output_dir"] = str(
    OUTPUT_ROOT / "pairedreal_implicit"
)

real_config["preview"][
    "manifest_glob"
] = "**/*_merged1000/stflow_manifest.jsonl"

real_config["preview"][
    "include_methods"
] = [selected["implicit"]]

real_config["preview"][
    "exclude_methods"
] = ["*dwmpreview*"]

real_config["preview"][
    "skip_reference_frames"
] = True

real_config["shared_box"][
    "manifest_glob"
] = "**/box_manifest.jsonl"

real_config["shared_box"][
    "strict_match"
] = True

real_config["sources"] = {
    "real": {
        "type": "preview_real",
        "group_by_manifest": False,
    }
}

real_config["runtime"]["limit_frames"] = 0
real_config["runtime"]["overwrite"] = True

real_config["visualization"]["enabled"] = True
real_config["visualization"][
    "max_frames_per_source"
] = 32
real_config["visualization"][
    "draw_cuboid"
] = True
real_config["visualization"][
    "draw_detection_box"
] = True

real_path = (
    CONFIG_ROOT
    / "pairedreal_implicit.yaml"
)

real_path.write_text(
    yaml.safe_dump(
        real_config,
        allow_unicode=True,
        sort_keys=False,
    ),
    encoding="utf-8",
)

mapping_path = (
    CONFIG_ROOT
    / "method_mapping.json"
)

mapping_path.write_text(
    json.dumps(
        selected,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("All 10 methods use the same 1000 videos.")
print("Configs saved to:", CONFIG_ROOT)
print("Outputs will be saved to:", OUTPUT_ROOT)
print("Method mapping:", mapping_path)
