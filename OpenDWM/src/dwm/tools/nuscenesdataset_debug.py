import os
import torch
import json
import sys
import fsspec
src_dir = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/src"
sys.path.append(src_dir)
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile
import debugpy

# ========= nuscenesDataset =========
# 建议把别名改回 NuscenesDataset，避免脑部混乱
from dwm.datasets.nuscenes import MotionDataset as NuscenesDataset 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========= 路径更新 =========
NUSC_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link"
BALANCED_JSON_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_train_balanced/balanced_windows_metadata.json"

# ========= transform（最小实现）=========
resize_to_tensor = T.Compose([
    T.Resize((256, 448)),  # (H, W)
    T.ToTensor()
])

def _apply_nested_images(pil_nested):
    if pil_nested is None:
        return None
    out = []
    for row in pil_nested:
        row_t =[]
        for im in row:
            if isinstance(im, Image.Image):
                row_t.append(resize_to_tensor(im))
            else:
                row_t.append(torch.zeros(3, 256, 448))  # None -> black
        out.append(row_t)
    return out

def _shape_nested(nested):
    if not nested:
        return "[]"
    Tn = len(nested)
    Vn = len(nested[0]) if Tn else 0
    x = nested[0][0] if (Tn and Vn) else None
    return f"[T={Tn}, V={Vn}, {tuple(x.shape)}]" if torch.is_tensor(x) else f"[T={Tn}, V={Vn}]"

# ========= base dataset =========
def make_base_ds():
    # 既然没有 pkl 了，我们直接把 json_path 传进去
    # 【注意】：稍后你在修改底层 nuscenes.py 时，__init__ 需要接收 `json_path` 这个参数来替代原先的 pkl_path
    fs = fsspec.filesystem("file")
    return NuscenesDataset(
        
        dataset_root=NUSC_ROOT,
        
        fs=fs,

        dataset_name="/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/nuscenes_link/interp_12Hz_trainval",

        split="interp_12Hz_trainval",

        balanced_json_path=BALANCED_JSON_PATH, 
        # 【修改点1】：sequence_length。
        # 你的 json 是以 window (片段) 为单位的，这里的 sequence_length 应该和你希望提取的帧数一致。
        # 如果长度不固定，底层需要处理 padding，或者在这里设为 None/最大值。
        sequence_length=20, 
        
        fps_stride_tuples=[(6,2,0.95)],
        
        # 【修改点2】：NuScenes 只有 6 个相机！(之前 NuPlan 是 8 个)
        sensor_channels=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'],
        
        enable_synchronization_check=False,
        
        # 【修改点3】：crossview_mask 的矩阵大小。
        # 因为现在是 6 个相机，mask 必须是 6x6 的（表示相机视野是否重叠）。我给你改成了 6x6。
        # 删除了底下重复且错误的 image_description_settings。
        stub_key_data_dict={
            "crossview_mask": [
                "content",
                torch.tensor(
                    [[1,1,0,0,0,1],
                     [1,1,1,0,0,0],[0,1,1,1,0,0],
                     [0,0,1,1,1,0],
                     [0,0,0,1,1,1],[1,0,0,0,1,1]], dtype=torch.bool)
            ]
        }
    )

# ========= adapter：只剥离沉重数据，保留元数据 =========
class SimpleDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = dict(self.base_ds[idx])

        if "images" in sample:
            sample["vae_images"] = _apply_nested_images(sample["images"])

        # 保留了 scene_name, angle, dist 等轻量级 JSON 字段，只 pop 掉没用的和占内存的
        for k in ("images", "lidar_points"): 
            sample.pop(k, None)

        return sample

def collate_ignore_clip_text(batch):
    assert len(batch) == 1, "建议 batch_size=1（与 cfg 一致）"
    return batch[0]

# ========= 打印 =========
def pretty_print_sample(sample):

    print("keys:", list(sample.keys()))

    if "vae_images" in sample:
        print("vae_images:", _shape_nested(sample["vae_images"]))

    print("--- Metadata ---")

    if "scene_name" in sample:
        print("scene_name:", sample["scene_name"])

    if "angle" in sample:
        print("angle:", sample["angle"].item())

    if "dist" in sample:
        print("dist:", sample["dist"].item())

    print("----------------")

# ========= main =========
def main():
    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] listening on 0.0.0.0:9876, waiting for VS Code to attach...")
    debugpy.wait_for_client()
    print("[debugpy] attached")

    base_ds = make_base_ds()
    ds = SimpleDatasetAdapter(base_ds)

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0, # Debug 阶段建议改为 0，能直接进底层断点！
        prefetch_factor=None, 
        shuffle=False,
        persistent_workers=False,
        collate_fn=collate_ignore_clip_text,
        pin_memory=False,
    )

    print(f"Dataset items: {len(ds)}")
    hit_samples = 0

    # 【修改点5】：循环里的测试逻辑，现在用来寻找你 JSON 里的 angle 和 dist
    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(f"\n==== sample {i} ====")
            pretty_print_sample(batch)

        # 检查你的底层修改是否成功把 angle 和 dist 传了上来
        if "angle" in batch and "dist" in batch:
            hit_samples += 1
            if i % 50 == 0:
                print(f"[HIT] sample #{i}: scene={batch.get('scene_name')}, angle={batch['angle']:.2f}, dist={batch['dist']:.2f}")

    print(f"\n[RESULT] Successfully loaded metadata for {hit_samples} / {len(ds)} samples.")

if __name__ == "__main__":
    main()