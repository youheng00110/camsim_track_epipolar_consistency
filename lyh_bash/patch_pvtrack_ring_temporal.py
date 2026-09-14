#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


WAYMO_ORDER = [
    "LIDAR_TOP",
    "CAM_SIDE_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_RIGHT",
    "CAM_SIDE_RIGHT",
]

# Camera slots 0..7 after the lidar entry:
#   0 SL, 1 FL, 2 FL(dummy), 3 F, 4 F(dummy), 5 FR, 6 FR(dummy), 7 SR
# Real-camera physical chain:
#   SIDE_LEFT -- FRONT_LEFT -- FRONT -- FRONT_RIGHT -- SIDE_RIGHT
# No SIDE_RIGHT <-> SIDE_LEFT wrap edge because Waymo has no rear cameras here.
WAYMO_MASK = [
    [1,1,0,0,0,0,0,0],
    [1,1,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,1,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,1,0,1],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,1],
]

ARGOVERSE_FULL_ORDER = [
    "lidar",
    "cameras/ring_side_left",
    "cameras/ring_front_left",
    "cameras/ring_front_center",
    "cameras/ring_front_center",
    "cameras/ring_front_right",
    "cameras/ring_side_right",
    "cameras/ring_rear_right",
    "cameras/ring_rear_left",
]

# Camera slots 0..7:
#   0 SL, 1 FL, 2 FC, 3 FC(dummy), 4 FR, 5 SR, 6 RR, 7 RL
# Full physical ring:
#   SL -- FL -- FC -- FR -- SR -- RR -- RL -- (back to SL)
ARGOVERSE_FULL_MASK = [
    [1,1,0,0,0,0,0,1],
    [1,1,1,0,0,0,0,0],
    [0,1,1,0,1,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,1,0,1,1,0,0],
    [0,0,0,0,1,1,1,0],
    [0,0,0,0,0,1,1,1],
    [1,0,0,0,0,0,1,1],
]


def mask_string(mask):
    return json.dumps(mask, separators=(",", ":"))


def assert_mask(mask, name):
    if len(mask) != 8 or any(len(row) != 8 for row in mask):
        raise ValueError(f"{name}: mask must be 8x8")
    for i in range(8):
        if mask[i][i] != 1:
            raise ValueError(f"{name}: self edge missing at {i}")
        for j in range(8):
            if mask[i][j] != mask[j][i]:
                raise ValueError(f"{name}: mask is not symmetric at ({i},{j})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--source",
        default="configs/lyh/PV_track_urope_tv_train.json",
    )
    parser.add_argument(
        "--target",
        default="configs/lyh/PV_track_urope_tv_temporal_ring_train.json",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / args.source
    target = root / args.target

    if not source.is_file():
        raise SystemExit(f"source config not found: {source}")
    if target.exists():
        raise SystemExit(
            f"target already exists; refusing to overwrite:\n  {target}"
        )

    cfg = json.loads(source.read_text())
    pipe = cfg["pipeline"]
    model = pipe["model"]
    common = pipe["common_config"]

    # Match the provided TV+temporal experiment semantics:
    # standalone temporal and joint TV are both enabled.
    model["enable_temporal"] = True
    model["temporal_attention_type"] = "rowwise"
    model["temporal_block_layers"] = [1, 5, 9, 13, 17, 21]
    model["temporal_gradient_checkpointing"] = True

    model["enable_tv"] = True
    model["tv_attention_type"] = "full"
    model["tv_block_layers"] = [1, 5, 9, 13, 17, 21]
    model["tv_gradient_checkpointing"] = True

    # Standalone cross-view remains disabled; cross-view interaction is in TV.
    model["enable_crossview"] = False
    model["crossview_block_layers"] = []
    model["crossview_gradient_checkpointing"] = False

    # Use the original PVTrack pipeline for this experiment.
    # It keeps disable_temporal independent from TV, matching the provided
    # TV+temporal reference config. The previously generated lyh pipeline
    # is preserved but not used by this new config.
    pipe["_class_name"] = "dwm.pipelines.camsim_track.CrossviewTemporalSD"
    common.pop("disable_tv", None)

    datasets = cfg["training_dataset"]["base_dataset"]["datasets"]

    waymo_count = 0
    av_full_count = 0

    for ds in datasets:
        cls = ds.get("_class_name", "")

        if cls == "dwm.datasets.track_pv.waymo.MotionDataset":
            waymo_count += 1
            ds["sensor_channels"] = WAYMO_ORDER
            ds["stub_key_data_dict"]["crossview_mask"][1]["data"]["s"] = (
                mask_string(WAYMO_MASK)
            )

        elif cls == "dwm.datasets.track_pv.argoverse.MotionDataset":
            channels = ds.get("sensor_channels", [])
            unique_cameras = set(channels[1:])

            # Only the 7-camera/full-ring Argoverse variant.
            required_full = {
                "cameras/ring_front_center",
                "cameras/ring_front_left",
                "cameras/ring_front_right",
                "cameras/ring_side_left",
                "cameras/ring_side_right",
                "cameras/ring_rear_left",
                "cameras/ring_rear_right",
            }
            if required_full.issubset(unique_cameras):
                av_full_count += 1
                ds["sensor_channels"] = ARGOVERSE_FULL_ORDER
                ds["stub_key_data_dict"]["crossview_mask"][1]["data"]["s"] = (
                    mask_string(ARGOVERSE_FULL_MASK)
                )

    if waymo_count != 1:
        raise SystemExit(f"expected exactly 1 Waymo dataset, found {waymo_count}")
    if av_full_count != 1:
        raise SystemExit(
            f"expected exactly 1 full-ring Argoverse dataset, found {av_full_count}"
        )

    assert_mask(WAYMO_MASK, "Waymo")
    assert_mask(ARGOVERSE_FULL_MASK, "Argoverse-full")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n")

    print("CREATED:", target)
    print()
    print("MODEL:")
    print("  enable_crossview =", model["enable_crossview"])
    print("  enable_temporal  =", model["enable_temporal"])
    print("  temporal_type    =", model["temporal_attention_type"])
    print("  temporal_layers  =", model["temporal_block_layers"])
    print("  enable_tv        =", model["enable_tv"])
    print("  tv_layers        =", model["tv_block_layers"])
    print()
    print("PIPELINE:")
    print(" ", pipe["_class_name"])
    print()
    print("WAYMO camera order:")
    print(" ", WAYMO_ORDER[1:])
    print("WAYMO mask:")
    for row in WAYMO_MASK:
        print(" ", row)
    print()
    print("ARGOVERSE full-ring camera order:")
    print(" ", ARGOVERSE_FULL_ORDER[1:])
    print("ARGOVERSE full-ring mask:")
    for row in ARGOVERSE_FULL_MASK:
        print(" ", row)
    print()
    print("NOTE: arbitrary camera permutation is NOT enabled.")
    print("Only physical-order-preserving cyclic shifts are safe with the current TV helper.")


if __name__ == "__main__":
    main()
