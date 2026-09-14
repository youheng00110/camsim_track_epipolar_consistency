#!/usr/bin/env python3
import argparse
import ast
import json
import py_compile
import shutil
from datetime import datetime
from pathlib import Path

OLD_BEV_AV2 = "dwm.datasets.bevs.argoverse.MotionDataset"
NEW_BEV_AV2 = "dwm.datasets.lyh.track_pv_bev_argoverse.MotionDataset"
TRACK_PV_AV2 = "dwm.datasets.track_pv.argoverse.MotionDataset"
ALIGNED = "dwm.datasets.lyh.bev_pv.AlignedBEVPVDataset"

DATASET_CODE = '"""AV2 BEV adapter built on the current track_pv Argoverse reader."""\n\nimport bisect\nimport numpy as np\nimport pyarrow.feather\nimport torch\n\nimport dwm.datasets.common\nfrom dwm.datasets.bevs.common import (\n    argoverse_class_id,\n    assign_stable_slot,\n    assert_unique_tracks,\n    require_track_id,\n)\nfrom dwm.datasets.track_pv.argoverse import MotionDataset as TrackPVMotionDataset\n\n\nclass MotionDataset(TrackPVMotionDataset):\n    """Track-PV AV2 reader plus BEV-specific outputs."""\n\n    def __init__(self, *args, layout_token_settings=None, max_boxes=64, **kwargs):\n        layout = {} if layout_token_settings is None else dict(layout_token_settings)\n        self.layout_max_boxes = int(layout.get("max_boxes", max_boxes))\n        self.max_annotation_time_error_ns = int(\n            layout.get("max_annotation_time_error_ns", 150_000_000)\n        )\n\n        hdmap_bev_settings = kwargs.get("hdmap_bev_settings")\n        if hdmap_bev_settings is not None:\n            hdmap_bev_settings = dict(hdmap_bev_settings)\n            # Keep static RGB map here. DatasetAdapter appends 10 bbox channels.\n            hdmap_bev_settings["append_dynamic_object_mask"] = False\n            kwargs["hdmap_bev_settings"] = hdmap_bev_settings\n\n        # Disable track_pv\'s per-frame bbox packing; rebuild stable slots below.\n        super().__init__(*args, layout_token_settings=None, **kwargs)\n\n    def get_bev_map_frame_reference(self, frame_items):\n        for item in frame_items:\n            if str(item.get("sensor", "")).lower() == "lidar":\n                return item\n        return super().get_bev_map_frame_reference(frame_items)\n\n    def _read_feather(self, item, filename):\n        path = f"{item[\'split\']}/{item[\'scene_id\']}/{filename}"\n        with self.fs.open(path) as f:\n            return pyarrow.feather.read_table(f).to_pydict()\n\n    def _nearest_annotation_range_safe(self, annotations, timestamp):\n        timestamps = annotations.get("timestamp_ns", [])\n        if len(timestamps) == 0:\n            return None, None, None\n\n        pos = bisect.bisect_left(timestamps, timestamp)\n        if pos == 0:\n            nearest = timestamps[0]\n        elif pos >= len(timestamps):\n            nearest = timestamps[-1]\n        else:\n            prev_ts = timestamps[pos - 1]\n            next_ts = timestamps[pos]\n            nearest = (\n                prev_ts\n                if abs(timestamp - prev_ts) <= abs(next_ts - timestamp)\n                else next_ts\n            )\n\n        nearest = int(nearest)\n        timestamp = int(timestamp)\n        if abs(nearest - timestamp) > self.max_annotation_time_error_ns:\n            # >150ms (default): empty bbox condition for this frame.\n            return None, None, None\n\n        start = bisect.bisect_left(timestamps, nearest)\n        stop = bisect.bisect_right(timestamps, nearest)\n        return nearest, start, stop\n\n    def _annotation_corners(\n        self, annotations, row, target_timestamp, annotation_timestamp, poses\n    ):\n        world_from_current = self.get_transform(\n            poses, "timestamp_ns", target_timestamp\n        ).astype(np.float32)\n        world_from_annotation = self.get_transform(\n            poses, "timestamp_ns", annotation_timestamp\n        ).astype(np.float32)\n        current_from_annotation = np.linalg.solve(\n            world_from_current, world_from_annotation\n        ).astype(np.float32)\n\n        scale = np.diag(\n            [annotations[k][row] for k in self.shape_keys] + [1.0]\n        ).astype(np.float32)\n        annotation_from_box = dwm.datasets.common.get_transform(\n            [annotations[k][row] for k in self.rotation_keys],\n            [annotations[k][row] for k in self.translation_keys],\n        ).astype(np.float32)\n        template = np.asarray(\n            self.default_3dbox_corner_template, dtype=np.float32\n        ).T\n        points = current_from_annotation @ annotation_from_box @ scale @ template\n        return torch.tensor(points[:3].T, dtype=torch.float32)\n\n    def _stable_layout_tokens(\n        self, item, annotations, poses, extrinsics, intrinsics\n    ):\n        if "track_uuid" not in annotations:\n            raise KeyError("Argoverse annotations.feather lacks official track_uuid.")\n\n        slot_by_track = {}\n        class_by_track = {}\n        frame_records = []\n\n        for frame_items in item["segment"]:\n            ref = self.get_bev_map_frame_reference(frame_items)\n            target_ts = int(ref["timestamp"])\n            anno_ts, start, stop = self._nearest_annotation_range_safe(\n                annotations, target_ts\n            )\n\n            records = []\n            frame_track_ids = []\n            if start is not None:\n                for row in range(start, stop):\n                    track_id = require_track_id(\n                        annotations["track_uuid"][row], "Argoverse"\n                    )\n                    class_id = argoverse_class_id(annotations["category"][row])\n                    slot = assign_stable_slot(\n                        track_id,\n                        class_id,\n                        slot_by_track,\n                        class_by_track,\n                        self.layout_max_boxes,\n                    )\n                    if slot is None:\n                        continue\n                    records.append((slot, class_id, row, target_ts, anno_ts))\n                    frame_track_ids.append(track_id)\n\n            assert_unique_tracks(frame_track_ids, "Argoverse")\n            frame_records.append(records)\n\n        camera_frames = [\n            [\n                sample\n                for sample in frame_items\n                if str(sample.get("sensor", "")).startswith("cameras/")\n            ]\n            for frame_items in item["segment"]\n        ]\n        time_count = len(camera_frames)\n        if time_count == 0:\n            raise ValueError("Argoverse BEV adapter received an empty segment.")\n        view_count = len(camera_frames[0])\n        if view_count == 0:\n            raise ValueError("Argoverse BEV adapter requires camera views.")\n        if any(len(frame) != view_count for frame in camera_frames):\n            raise ValueError("Argoverse camera view count changes inside sequence.")\n\n        corners = torch.zeros(\n            time_count, view_count, self.layout_max_boxes, 8, 3,\n            dtype=torch.float32,\n        )\n        classes = torch.zeros(\n            time_count, view_count, self.layout_max_boxes, dtype=torch.long\n        )\n        masks = torch.zeros(\n            time_count, view_count, self.layout_max_boxes, dtype=torch.float32\n        )\n\n        for t, records in enumerate(frame_records):\n            for slot, class_id, row, target_ts, anno_ts in records:\n                box = self._annotation_corners(\n                    annotations, row, target_ts, anno_ts, poses\n                )\n                box_np = box.numpy()\n                for v, sample in enumerate(camera_frames[t]):\n                    corners[t, v, slot] = box\n                    classes[t, v, slot] = class_id\n                    if self._layout_box_visible_in_av2_camera(\n                        box_np,\n                        target_ts,\n                        extrinsics,\n                        intrinsics,\n                        poses,\n                        sample,\n                    ):\n                        masks[t, v, slot] = 1.0\n\n        return corners, classes, masks\n\n    def __getitem__(self, index):\n        # images/camera geometry/3dbox/static BEV map all come from track_pv.\n        result = super().__getitem__(index)\n        item = self.items[index]\n\n        annotations = self._read_feather(item, "annotations.feather")\n        poses = self._read_feather(item, "city_SE3_egovehicle.feather")\n        extrinsics = self._read_feather(\n            item, "calibration/egovehicle_SE3_sensor.feather"\n        )\n        intrinsics = self._read_feather(\n            item, "calibration/intrinsics.feather"\n        )\n\n        result["reference_ego_transforms"] = torch.stack(\n            [\n                self.get_transform(\n                    poses,\n                    "timestamp_ns",\n                    int(self.get_bev_map_frame_reference(frame)["timestamp"]),\n                    "pt",\n                )\n                for frame in item["segment"]\n            ],\n            dim=0,\n        )\n\n        corners, classes, masks = self._stable_layout_tokens(\n            item, annotations, poses, extrinsics, intrinsics\n        )\n        result["bbox_token_corners"] = corners\n        result["bbox_token_classes"] = classes\n        result["bbox_token_masks"] = masks\n\n        if self.hdmap_bev_settings is not None and "hdmap_bev_images" not in result:\n            raise KeyError(\n                "Expected hdmap_bev_images from track_pv AV2 reader."\n            )\n        return result\n'


def backup(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(path.name + ".bak." + stamp)
    shutil.copy2(path, dst)
    return dst


def ensure_target(path, force):
    if not path.exists():
        return
    if not force:
        raise SystemExit(
            f"Target exists: {path}\n"
            "Use --force to back it up and regenerate."
        )
    print("backup:", backup(path))


def make_adapter_13ch(adapter):
    adapter["append_bbox_dynamic_bev_mask"] = True
    adapter["bbox_dynamic_bev_num_classes"] = 10
    adapter["bbox_dynamic_bev_map_key"] = "hdmap_bev_images"
    adapter["bbox_dynamic_bev_bbox_key"] = "bbox_token_corners"
    adapter["bbox_dynamic_bev_class_key"] = "bbox_token_classes"
    adapter["bbox_dynamic_bev_mask_key"] = "bbox_token_masks"
    adapter["bbox_dynamic_bev_x_min"] = -80.0
    adapter["bbox_dynamic_bev_x_max"] = 80.0
    adapter["bbox_dynamic_bev_y_min"] = -80.0
    adapter["bbox_dynamic_bev_y_max"] = 80.0


def convert_av2_bev(bev):
    max_boxes = int(bev.pop("max_boxes", 64))
    bev["_class_name"] = NEW_BEV_AV2
    bev["enable_camera_transforms"] = True
    bev["enable_ego_transforms"] = True
    bev["enable_synchronization_check"] = True
    bev["hide_lidar"] = True

    # AlignedBEVPVDataset keeps 3dbox_images from BEV side.
    bev["_3dbox_image_settings"] = {}

    # Avoid duplicate PV-only conditions on the BEV half.
    bev["hdmap_image_settings"] = None
    bev["instance_flow_image_settings"] = None
    bev["image_description_settings"] = None

    layout = dict(bev.get("layout_token_settings") or {})
    layout["max_boxes"] = max_boxes
    layout["max_annotation_time_error_ns"] = 150_000_000
    layout["bev_map_reference_channel"] = "lidar"
    bev["layout_token_settings"] = layout

    hdmap = dict(bev.get("hdmap_bev_settings") or {})
    if not hdmap:
        raise ValueError("AV2 BEV config has no hdmap_bev_settings.")
    hdmap["append_dynamic_object_mask"] = False
    hdmap["num_object_classes"] = 10
    bev["hdmap_bev_settings"] = hdmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--source-config",
        default="configs/lyh/BEV_PV_plucker_train.json",
    )
    parser.add_argument(
        "--target-config",
        default="configs/lyh/BEV_PV_plucker_train_av2pvbev.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    source_config = root / args.source_config
    target_config = root / args.target_config
    dataset_target = (
        root / "src/dwm/datasets/lyh/track_pv_bev_argoverse.py"
    )

    required = [
        source_config,
        root / "src/dwm/datasets/track_pv/argoverse.py",
        root / "src/dwm/datasets/bevs/common.py",
        root / "src/dwm/datasets/common.py",
        root / "src/dwm/datasets/lyh/bev_pv.py",
        root / "src/dwm/pipelines/lyh/bev_pv.py",
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Required source not found: {path}")

    common_text = (root / "src/dwm/datasets/common.py").read_text()
    for marker in (
        "append_bbox_dynamic_bev_mask",
        "bbox_dynamic_bev_map_key",
        "bbox_dynamic_bev_bbox_key",
        "bbox_dynamic_bev_class_key",
        "bbox_dynamic_bev_mask_key",
    ):
        if marker not in common_text:
            raise SystemExit(
                "Current DatasetAdapter lacks required BEV mask support: "
                + marker
            )

    ensure_target(dataset_target, args.force)
    ensure_target(target_config, args.force)

    config = json.loads(source_config.read_text())
    make_adapter_13ch(config["training_dataset"])
    if isinstance(config.get("validation_dataset"), dict):
        make_adapter_13ch(config["validation_dataset"])

    converted = 0
    for pair in config["training_dataset"]["base_dataset"]["datasets"]:
        if pair.get("_class_name") != ALIGNED:
            continue
        bev = pair.get("bev_dataset", {})
        pv = pair.get("pv_dataset", {})
        if bev.get("_class_name") != OLD_BEV_AV2:
            continue
        if pv.get("_class_name") != TRACK_PV_AV2:
            raise ValueError("AV2 PV side is not track_pv.argoverse.")
        convert_av2_bev(bev)
        converted += 1

    if converted == 0:
        raise ValueError("No AV2 BEV dataset found.")
    if int(config["pipeline"]["model"].get("bev_in_channels", -1)) != 13:
        raise ValueError("Expected model bev_in_channels=13.")

    ast.parse(DATASET_CODE)
    dataset_target.parent.mkdir(parents=True, exist_ok=True)
    dataset_target.write_text(DATASET_CODE)
    py_compile.compile(str(dataset_target), doraise=True)

    target_config.parent.mkdir(parents=True, exist_ok=True)
    target_config.write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n"
    )
    json.loads(target_config.read_text())

    print("=== CREATED ===")
    print(dataset_target)
    print(target_config)
    print("=== SEMANTICS ===")
    print("AV2 BEV base reader: track_pv.argoverse.MotionDataset")
    print("annotation safety threshold: 150 ms")
    print(">150 ms: empty bbox frame, no crash")
    print("bbox_token_*: stable track_uuid slots across time")
    print("map token: static hdmap_bev_images from track_pv")
    print("dynamic BEV: +10 bbox-derived channels in DatasetAdapter")
    print("final BEV map: 3 + 10 = 13 channels")
    print("reference_ego_transforms: LiDAR timeline")
    print("PV companions: unchanged")
    print("converted AV2 variants:", converted)


if __name__ == "__main__":
    main()
