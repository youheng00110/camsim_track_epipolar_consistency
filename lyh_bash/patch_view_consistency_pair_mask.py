#!/usr/bin/env python3
import argparse
import copy
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/camsim/OpenDWM"
)

OLD_UTIL_BLOCK = '''    crossview_mask = batch.get("crossview_mask")
    if crossview_mask is not None:
        crossview_mask = crossview_mask.bool().cpu()
        if crossview_mask.ndim == 4 and crossview_mask.shape[1] == 1:
            crossview_mask = crossview_mask.squeeze(1)
        if crossview_mask.ndim == 2:
            crossview_mask = crossview_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        if crossview_mask.shape != (
            batch_size,
            view_count,
            view_count,
        ):
            raise ValueError(
                "crossview_mask must be [B,V,V], got "
                f"{tuple(crossview_mask.shape)}"
            )
'''

NEW_UTIL_BLOCK = '''    pair_mask = batch.get("view_consistency_pair_mask")
    if enable_crossview and pair_mask is None:
        raise KeyError(
            "view_consistency_pair_mask is required when "
            "view_consistency_enable_crossview=True. "
            "Do not reuse crossview_mask here: crossview_mask describes "
            "cross-view attention groups, while view_consistency_pair_mask "
            "describes valid pair-wise epipolar neighbors."
        )
    if pair_mask is not None:
        pair_mask = pair_mask.bool().cpu()
        if pair_mask.ndim == 4 and pair_mask.shape[1] == 1:
            pair_mask = pair_mask.squeeze(1)
        if pair_mask.ndim == 2:
            pair_mask = pair_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        if pair_mask.shape != (
            batch_size,
            view_count,
            view_count,
        ):
            raise ValueError(
                "view_consistency_pair_mask must be [B,V,V], got "
                f"{tuple(pair_mask.shape)}"
            )
'''

OLD_MASK_ROW = '''        mask_row = (
            None
            if crossview_mask is None
            else crossview_mask[batch_index]
        )
'''

NEW_MASK_ROW = '''        mask_row = (
            None
            if pair_mask is None
            else pair_mask[batch_index]
        )
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Separate epipolar pair selection from cross-view attention mask, "
            "and generate canonical BEV-PV epipolar configs."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def backup_file(path: Path, stamp: str):
    if not path.exists():
        return None
    backup = path.with_name(path.name + f".bak_pairmask_{stamp}")
    shutil.copy2(path, backup)
    return backup


def camera_slots(sensor_channels):
    cameras = [
        item
        for item in sensor_channels
        if "lidar" not in str(item).lower()
    ]
    if len(cameras) != 8:
        raise ValueError(
            f"Expected exactly 8 camera slots after removing lidar, got "
            f"{len(cameras)}: {cameras}"
        )
    return cameras


def first_slot_by_name(cameras):
    result = {}
    for index, name in enumerate(cameras):
        result.setdefault(name, index)
    return result


def ring_pairs_from_canonical_order(cameras, canonical_order):
    slot = first_slot_by_name(cameras)
    present = [name for name in canonical_order if name in slot]
    unknown = sorted(set(cameras) - set(canonical_order))
    if unknown:
        raise ValueError(
            f"Unknown cameras for ring topology: {unknown}; cameras={cameras}"
        )
    if len(present) < 2:
        return []
    pairs = []
    for index, name_a in enumerate(present):
        name_b = present[(index + 1) % len(present)]
        a = slot[name_a]
        b = slot[name_b]
        if a != b:
            pair = tuple(sorted((a, b)))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def build_pairs(dataset_class, cameras):
    if ".waymo." in dataset_class:
        # User-specified non-ring physical chain:
        # SIDE_RIGHT -- FRONT_RIGHT -- FRONT -- FRONT_LEFT -- SIDE_LEFT
        chain_edges = [
            ("CAM_SIDE_RIGHT", "CAM_FRONT_RIGHT"),
            ("CAM_FRONT_RIGHT", "CAM_FRONT"),
            ("CAM_FRONT", "CAM_FRONT_LEFT"),
            ("CAM_FRONT_LEFT", "CAM_SIDE_LEFT"),
        ]
        slot = first_slot_by_name(cameras)
        expected = {
            "CAM_SIDE_RIGHT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_SIDE_LEFT",
        }
        unknown = sorted(set(cameras) - expected)
        if unknown:
            raise ValueError(
                f"Unknown Waymo cameras {unknown}; cameras={cameras}"
            )
        pairs = []
        for name_a, name_b in chain_edges:
            if name_a in slot and name_b in slot:
                pairs.append(tuple(sorted((slot[name_a], slot[name_b]))))
        return pairs

    if ".nuscenes." in dataset_class:
        canonical = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        return ring_pairs_from_canonical_order(cameras, canonical)

    if ".nuplan." in dataset_class:
        canonical = [
            "CAM_F0",
            "CAM_R0",
            "CAM_R1",
            "CAM_R2",
            "CAM_B0",
            "CAM_L2",
            "CAM_L1",
            "CAM_L0",
        ]
        return ring_pairs_from_canonical_order(cameras, canonical)

    if ".argoverse." in dataset_class:
        canonical = [
            "cameras/ring_front_center",
            "cameras/ring_front_right",
            "cameras/ring_side_right",
            "cameras/ring_rear_right",
            "cameras/ring_rear_left",
            "cameras/ring_side_left",
            "cameras/ring_front_left",
        ]
        return ring_pairs_from_canonical_order(cameras, canonical)

    raise ValueError(
        f"Unsupported dataset class for view-consistency topology: "
        f"{dataset_class}"
    )


def pairs_to_mask(pairs, view_count=8):
    mask = [[False for _ in range(view_count)] for _ in range(view_count)]
    for a, b in pairs:
        if a == b:
            continue
        mask[a][b] = True
        mask[b][a] = True
    return mask


def make_stub(mask):
    compact = json.dumps(mask, separators=(",", ":"))
    return [
        "content",
        {
            "_class_name": "torch.tensor",
            "data": {
                "_class_name": "json.loads",
                "s": compact,
            },
            "dtype": {
                "_class_name": "get_class",
                "class_name": "torch.bool",
            },
        },
    ]


def iter_aligned_dataset_nodes(value, path=""):
    if isinstance(value, dict):
        class_name = value.get("_class_name", "")
        if class_name == "dwm.datasets.lyh.bev_pv.AlignedBEVPVDataset":
            yield path or "<root>", value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from iter_aligned_dataset_nodes(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from iter_aligned_dataset_nodes(child, child_path)


def install_pair_masks(cfg):
    records = []
    aligned_nodes = list(iter_aligned_dataset_nodes(cfg))
    if not aligned_nodes:
        raise RuntimeError("No AlignedBEVPVDataset nodes found in config.")

    for node_path, aligned in aligned_nodes:
        bev_dataset = aligned["bev_dataset"]
        pv_dataset = aligned.get("pv_dataset")

        dataset_class = bev_dataset["_class_name"]
        cameras = camera_slots(bev_dataset["sensor_channels"])
        pairs = build_pairs(dataset_class, cameras)
        mask = pairs_to_mask(pairs, len(cameras))
        stub = make_stub(mask)

        bev_stub = bev_dataset.setdefault("stub_key_data_dict", {})
        bev_stub["view_consistency_pair_mask"] = copy.deepcopy(stub)

        # Keep PV config symmetric as well. AlignedBEVPVDataset currently
        # returns BEV keys as the authoritative batch keys, so BEV is the
        # required source; mirroring to PV avoids future ambiguity.
        if pv_dataset is not None:
            pv_stub = pv_dataset.setdefault("stub_key_data_dict", {})
            pv_stub["view_consistency_pair_mask"] = copy.deepcopy(stub)

        records.append(
            {
                "path": node_path,
                "dataset": dataset_class,
                "cameras": cameras,
                "pairs": pairs,
            }
        )
    return records


def patch_utils(utils_path: Path, stamp: str):
    text = utils_path.read_text(encoding="utf-8")
    already_patched = (
        'pair_mask = batch.get("view_consistency_pair_mask")' in text
        and "if enable_crossview and pair_mask is None:" in text
    )
    if already_patched:
        print("[utils] already patched:", utils_path)
        return

    if OLD_UTIL_BLOCK not in text:
        raise RuntimeError(
            "Could not find the expected crossview_mask block in "
            f"{utils_path}. Refusing to guess."
        )
    if OLD_MASK_ROW not in text:
        raise RuntimeError(
            "Could not find the expected mask_row block in "
            f"{utils_path}. Refusing to guess."
        )

    backup = backup_file(utils_path, stamp)
    text = text.replace(
        "Cross-view pairs use the existing crossview_mask at the same time.",
        "Cross-view pairs use view_consistency_pair_mask at the same time.",
    )
    text = text.replace(OLD_UTIL_BLOCK, NEW_UTIL_BLOCK, 1)
    text = text.replace(OLD_MASK_ROW, NEW_MASK_ROW, 1)
    utils_path.write_text(text, encoding="utf-8")
    print("[utils] patched:", utils_path)
    print("[utils] backup :", backup)


def patch_one_config(src: Path, canonical_dst: Path, stamp: str):
    if not src.exists():
        print("[config] skip missing:", src)
        return

    with src.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    records = install_pair_masks(cfg)

    src_backup = backup_file(src, stamp)
    if canonical_dst.exists() and canonical_dst.resolve() != src.resolve():
        canonical_backup = backup_file(canonical_dst, stamp)
    else:
        canonical_backup = None

    serialized = json.dumps(cfg, indent=4, ensure_ascii=False) + "\n"
    src.write_text(serialized, encoding="utf-8")
    canonical_dst.write_text(serialized, encoding="utf-8")

    print()
    print("[config] updated :", src)
    print("[config] backup  :", src_backup)
    print("[config] canonical:", canonical_dst)
    if canonical_backup is not None:
        print("[config] canonical backup:", canonical_backup)

    for record in records:
        print(
            "  -",
            record["dataset"],
            "at",
            record["path"],
        )
        print("    slots:", list(enumerate(record["cameras"])))
        print("    epipolar pairs:", record["pairs"])


def verify_config(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    missing = []
    for node_path, aligned in iter_aligned_dataset_nodes(cfg):
        bev_stub = aligned["bev_dataset"].get("stub_key_data_dict", {})
        if "view_consistency_pair_mask" not in bev_stub:
            missing.append(node_path)

    if missing:
        raise RuntimeError(
            f"{path} has aligned datasets missing view_consistency_pair_mask: "
            f"{missing}"
        )

    tc = cfg["pipeline"]["training_config"]
    print(
        f"[verify] {path.name}: "
        f"crossview={tc.get('view_consistency_enable_crossview')} "
        f"crossframe={tc.get('view_consistency_enable_crossframe')} "
        f"warmup={tc.get('view_consistency_loss_warmup_steps')} "
        f"weight={tc.get('view_consistency_loss_weight')}"
    )


def main():
    args = parse_args()
    root = args.root.resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    utils_path = root / "src/dwm/utils/view_consistency.py"
    if not utils_path.exists():
        raise FileNotFoundError(utils_path)

    patch_utils(utils_path, stamp)

    config_dir = root / "configs/lyh"

    # Keep the old filenames usable, but also write the requested canonical names.
    config_jobs = [
        (
            config_dir / "PV_track_train_epipolar.json",
            config_dir / "bev_pv_epipolar.json",
        ),
        (
            config_dir / "PV_track_train_epipolar_debug.json",
            config_dir / "bev_pv_epipolar_debug.json",
        ),
    ]

    for src, dst in config_jobs:
        patch_one_config(src, dst, stamp)

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(utils_path)],
        check=True,
    )

    verify_config(config_dir / "bev_pv_epipolar.json")
    verify_config(config_dir / "bev_pv_epipolar_debug.json")

    print()
    print("=== DONE ===")
    print("Original crossview_mask was NOT changed.")
    print(
        "Epipolar cross-view sampling now requires "
        "batch['view_consistency_pair_mask']."
    )
    print("Canonical configs:")
    print("  ", config_dir / "bev_pv_epipolar.json")
    print("  ", config_dir / "bev_pv_epipolar_debug.json")
    print("No training or evaluation was launched.")


if __name__ == "__main__":
    main()
