import os
import torch
import json
import sys
import fsspec
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile
import debugpy
src_dir = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src"
sys.path.append(src_dir)
# ====== dataset ======
# 确保导入的是 ArgoDataset (MotionDataset)
from dwm.fs.dirfs import DirFileSystem
from dwm.datasets.argoverse import MotionDataset as ArgoDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# PATH 配置
# =========================




# =========================
# transform
# =========================

resize_to_tensor = T.Compose([
    T.Resize((256, 448)),
    T.ToTensor()
])


def _apply_nested_images(pil_nested):
    if pil_nested is None:
        return None
    out = []
    for row in pil_nested:
        row_t = []
        for im in row:
            if isinstance(im, Image.Image):
                row_t.append(resize_to_tensor(im))
            else:
                row_t.append(torch.zeros(3, 256, 448))
        out.append(row_t)
    return out


def _shape_nested(nested):
    if not nested:
        return "[]"
    Tn = len(nested)
    Vn = len(nested[0]) if Tn else 0
    x = nested[0][0] if (Tn and Vn) else None
    if torch.is_tensor(x):
        return f"[T={Tn}, V={Vn}, {tuple(x.shape)}]"
    else:
        return f"[T={Tn}, V={Vn}]"


# =========================
# Dataset 适配 Argo2
# =========================

def make_base_ds():
    fs = DirFileSystem(
        fs=DirFileSystem(
            path=ARGO2_ROOT
        ),
        enable_cached_info=True
    )

    ds = ArgoDataset(
                {
                    "_class_name": "dwm.datasets.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link"
                        },

                        "enable_cached_info": true
                    },
                    "split":"train",
                    "layout_token_settings": {
                    "enabled": true,
                    "max_boxes": 64,
                    "box_coordinate": "ego",
                    "box_format": "corners8",
                    "class_mapping": "driving_10cls",
                    "use_map_bev_token": true,
                    "map_token_key": "hdmap_bev_images"
                    },
                    "dataset_root":"/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link",
                    "index_json_path":"/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_json",
                    "balanced_json_path":"/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                         [6,2.5,0.95],[2,6,0.39]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_front_left",
                        "cameras/ring_front_center",
                        "cameras/ring_front_right",
                        "cameras/ring_side_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left",
                        "cameras/ring_side_left",
                        "cameras/ring_front_center"
                        
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "hdmap_bev_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/av2_sensor_caption_v2/av2_sensor_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/av2_sensor_caption_v2/av2_sensor_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,1,0,0,0,0,1,0],[1,1,1,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[1,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": { "_class_name": "get_class", "class_name": "torch.int64" }
                            }
                            ]
                    }
                },
    )

    print("\n========== DATASET DEBUG ==========")
    print("Dataset size:", len(ds))

    sample = ds[0]

    print("Sample keys:", sample.keys())

    if "images" in sample:
        print("Sequence length:", len(sample["images"]))
        print("Camera count:", len(sample["images"][0]))
        print("Example image size:", sample["images"][0][0].size)

    if "angle" in sample:
        print("Angle:", sample["angle"])

    if "dist" in sample:
        print("Dist:", sample["dist"])

    print("===================================\n")

    return ds


# =========================
# Adapter
# =========================

class SimpleDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = dict(self.base_ds[idx])
        if "images" in sample:
            sample["vae_images"] = _apply_nested_images(sample["images"])
        
        # 删除原始大对象以防 DataLoader 内存溢出
        for k in ("images", "lidar_points"):
            sample.pop(k, None)
        return sample


# =========================
# collate
# =========================

def collate_ignore_clip_text(batch):
    assert len(batch) == 1
    return batch[0]


# =========================
# debug print
# =========================

def pretty_print_sample(sample):
    print("keys:", list(sample.keys()))
    if "vae_images" in sample:
        print("vae_images:", _shape_nested(sample["vae_images"]))

    print("--- Metadata ---")
    # Argo2 常用 log_id
    for key in ["scene_name", "seq_id", "log_id", "angle", "dist", "start_timestamp", "end_timestamp"]:
        if key in sample:
            print(f"{key}: {sample[key]}")
    print("----------------")


# =========================
# main (保留远程调试逻辑)
# =========================

def main():
    # 保留 debugpy 配置
    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] waiting for attach...")
    debugpy.wait_for_client()
    print("[debugpy] attached")

    base_ds = make_base_ds()
    ds = SimpleDatasetAdapter(base_ds)

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        prefetch_factor=None,
        persistent_workers=False,
        collate_fn=collate_ignore_clip_text,
        pin_memory=False
    )

    print("Dataset size:", len(ds))

    hit_samples = 0
    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(f"\n==== sample {i} ====")
            pretty_print_sample(batch)

        if "angle" in batch and "dist" in batch:
            hit_samples += 1
            if i % 50 == 0:
                # 兼容性获取名称
                name = batch.get('scene_name') or batch.get('scene_id') or batch.get('Scene') or "Unknown"
                print(
                    f"[HIT] sample #{i}: name={name}, "
                    f"angle={batch['angle']:.4f}, dist={batch['dist']:.4f}"
                )

    print(f"\n[RESULT] metadata loaded for {hit_samples} / {len(ds)} samples")


if __name__ == "__main__":
    main()