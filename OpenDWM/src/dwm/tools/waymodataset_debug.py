import os
import torch
import json
import sys
import fsspec
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile
import debugpy

##加入系统路径###

src_dir = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/src"

waymo_dir = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src"

sys.path.insert(0, src_dir)
sys.path.insert(0, waymo_dir)

# ====== dataset ======
from dwm.datasets.waymo import MotionDataset as WaymoDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# PATH
# =========================

WAYMO_ROOT = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_link"

BALANCED_JSON_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_balanced/balanced_windows_metadata.json"


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
# Dataset
# =========================

def make_base_ds():

    fs = fsspec.filesystem("file")

    INFO_DICT_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_balanced/training.info.json"

    BALANCED_JSON_PATH = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/waymo_balanced/balanced_windows_metadata.json"

    ds = WaymoDataset(
        fs=fs,

        # ⭐ frame offset index
        info_dict_path=INFO_DICT_PATH,

        dataset_root=WAYMO_ROOT,

        sequence_length=20,

        fps_stride_tuples=[(4, 2,0.8)],

        sensor_channels=[
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_SIDE_LEFT",
            "CAM_SIDE_RIGHT",
        ],

        # ======================
        # transforms
        # ======================
        enable_camera_transforms=True,
        enable_ego_transforms=True,

        # ======================
        # 3D box image
        # ======================
        _3dbox_image_settings={
            "image_size": [512, 512],
            "line_width": 2
        },

        # ======================
        # HD map image
        # ======================
        hdmap_image_settings={
            "image_size": [512, 512]
        },

        # ======================
        # BEV 3D box
        # ======================
        _3dbox_bev_settings={
            "image_size": [512, 512],
            "meters_per_pixel": 0.2
        },

        # ======================
        # BEV HD map
        # ======================
        hdmap_bev_settings={
            "image_size": [512, 512],
            "meters_per_pixel": 0.2
        },

        # ======================
        # image caption
        # ======================
        image_description_settings=None,

        # ======================
        # stub keys (optional)
        # ======================
        stub_key_data_dict=None,

        # ======================
        # balanced sampling
        # ======================
        balanced_json_path=BALANCED_JSON_PATH,
    )

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

        # 删除大数据
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

    if "scene_name" in sample:
        print("scene_name:", sample["scene_name"])

    if "seq_id" in sample:
        print("seq_id:", sample["seq_id"])

    if "angle" in sample:
        print("angle:", sample["angle"])

    if "dist" in sample:
        print("dist:", sample["dist"])

    print("----------------")


# =========================
# main
# =========================

def main():

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

                print(
                    f"[HIT] sample #{i}: scene={batch.get('scene_name')}, "
                    f"angle={batch['angle']}, dist={batch['dist']}"
                )

    print(
        f"\n[RESULT] metadata loaded for {hit_samples} / {len(ds)} samples"
    )


if __name__ == "__main__":
    main()