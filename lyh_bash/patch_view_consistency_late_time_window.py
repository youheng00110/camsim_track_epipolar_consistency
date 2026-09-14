#!/usr/bin/env python3
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/camsim/OpenDWM"
)

UTILS = ROOT / "src/dwm/utils/view_consistency.py"
CONFIG_DIR = ROOT / "configs/lyh"

CONFIGS = [
    CONFIG_DIR / "PV_track_train_epipolar.json",
    CONFIG_DIR / "PV_track_train_epipolar_debug.json",
    CONFIG_DIR / "bev_pv_epipolar.json",
    CONFIG_DIR / "bev_pv_epipolar_debug.json",
]

OLD_BLOCK = '''    all_selections = []
    max_pair_count = 0
    for batch_index in range(batch_size):
        valid_times = list(range(min(reference_count, frame_count), frame_count))
        if len(valid_times) == 0:
            valid_times = [frame_count - 1]
        sampled_times = sample_stratified_times(
            valid_times,
            sample_time_count,
            generator,
        )
'''

NEW_BLOCK = '''    all_selections = []
    max_pair_count = 0

    configured_time_start = training_config.get(
        "view_consistency_time_start",
        None,
    )
    configured_time_end = training_config.get(
        "view_consistency_time_end",
        None,
    )

    for batch_index in range(batch_size):
        if configured_time_start is None and configured_time_end is None:
            # Backward-compatible behavior for configs without an explicit
            # view-consistency time window.
            valid_times = list(
                range(
                    min(reference_count, frame_count),
                    frame_count,
                )
            )
            if len(valid_times) == 0:
                valid_times = [frame_count - 1]
        else:
            # start is inclusive, end is exclusive.
            time_start = (
                reference_count
                if configured_time_start is None
                else int(configured_time_start)
            )
            time_end = (
                frame_count
                if configured_time_end is None
                else int(configured_time_end)
            )

            time_start = max(
                min(reference_count, frame_count),
                0,
                time_start,
            )
            time_end = min(
                frame_count,
                time_end,
            )

            # Cross-view and cross-frame share the same sampled source time.
            # If cross-frame is enabled, reserve t+stride so every sampled
            # time can produce both spatial and temporal pairs.
            if enable_crossframe:
                time_end = min(
                    time_end,
                    frame_count - crossframe_stride,
                )

            valid_times = list(range(time_start, time_end))
            if len(valid_times) == 0:
                return None

        sampled_times = sample_stratified_times(
            valid_times,
            sample_time_count,
            generator,
        )
'''


def backup(path, stamp):
    dst = path.with_name(path.name + f".bak_timewindow_{stamp}")
    shutil.copy2(path, dst)
    return dst


def main():
    if not UTILS.exists():
        raise FileNotFoundError(UTILS)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    text = UTILS.read_text(encoding="utf-8")

    if "configured_time_start = training_config.get(" in text:
        print("[utils] time-window patch already present; source unchanged.")
    else:
        if OLD_BLOCK not in text:
            raise RuntimeError(
                "Expected time-sampling block was not found. "
                "Refusing to guess against your current code."
            )
        b = backup(UTILS, stamp)
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
        UTILS.write_text(text, encoding="utf-8")
        print("[utils] patched:", UTILS)
        print("[utils] backup :", b)

    # Only add the two time-window keys. Preserve depth and all other options.
    for path in CONFIGS:
        if not path.exists():
            print("[config] skip missing:", path)
            continue

        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        tc = cfg["pipeline"]["training_config"]
        old_start = tc.get("view_consistency_time_start")
        old_end = tc.get("view_consistency_time_end")

        tc["view_consistency_time_start"] = 10
        tc["view_consistency_time_end"] = 20

        b = backup(path, stamp)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
            f.write("\n")

        print("[config] updated:", path)
        print("[config] backup :", b)
        print(
            "         VC time window:",
            old_start,
            old_end,
            "-> 10 20",
        )
        print(
            "         depth preserved:",
            tc.get("view_consistency_min_depth"),
            tc.get("view_consistency_max_depth"),
        )

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(UTILS)],
        check=True,
    )

    print()
    print("=== VERIFY ===")
    for path in CONFIGS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        tc = cfg["pipeline"]["training_config"]
        print(path.name)
        print(
            "  time_start =",
            tc.get("view_consistency_time_start"),
        )
        print(
            "  time_end   =",
            tc.get("view_consistency_time_end"),
        )
        print(
            "  depth      =",
            tc.get("view_consistency_min_depth"),
            tc.get("view_consistency_max_depth"),
        )

    print()
    print("For a 20-frame clip with crossframe_stride=1 and both")
    print("crossview/crossframe enabled, source t is sampled from 10..18.")
    print("Cross-view:  t -> t")
    print("Cross-frame: t -> t+1, so target may reach frame 19.")
    print("No training was launched.")


if __name__ == "__main__":
    main()
