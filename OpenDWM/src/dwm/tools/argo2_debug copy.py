import os
import torch
import json
import sys
import fsspec
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile
import debugpy
src_dir = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/src"
sys.path.append(src_dir)
# ====== dataset ======
# 确保导入的是 ArgoDataset (MotionDataset)
from dwm.datasets.argoverse import MotionDataset as ArgoDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# PATH 配置
# =========================

ARGO2_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_link"
INDEX_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_json"
BALANCED_JSON_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_balanced/balanced_windows.json"


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

    fs = fsspec.filesystem("file")

    # =========================
    # 扫描 train + val scenes
    # =========================

    scene_dirs = []

    for split in ["train", "val"]:
        split_dir = os.path.join(ARGO2_ROOT, split)

        if not os.path.exists(split_dir):
            continue

        for scene in os.listdir(split_dir):
            scene_path = os.path.join(split_dir, scene)

            if os.path.isdir(scene_path):
                scene_dirs.append(scene_path)

    print("Total scenes found:", len(scene_dirs))

    # =========================
    # 创建 dataset
    # =========================

    ds = ArgoDataset(
        fs=fs,
        dataset_root=ARGO2_ROOT,
        scene_dirs=scene_dirs,

        sequence_length=20,
        fps_stride_tuples=[(10, 2)],

        sensor_channels=[
            "cameras/ring_front_center", "cameras/ring_front_left",
            "cameras/ring_front_right", "cameras/ring_rear_left",
            "cameras/ring_rear_right", "cameras/ring_side_left",
            "cameras/ring_side_right"
        ],

        enable_camera_transforms=True,
        balanced_json_path=BALANCED_JSON_PATH,
        index_json_path=INDEX_PATH,
        split="train"
    )

    # =========================
    # Debug
    # =========================

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