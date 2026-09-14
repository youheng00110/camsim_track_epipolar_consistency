import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageFile

PROJECT_ROOT = "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh"
SRC_DIR = os.path.join(PROJECT_ROOT, "OpenDWM/src")
NUPLAN_DEVKIT_DIR = os.path.join(PROJECT_ROOT, "nuplan-devkit-master")

sys.path.insert(0, NUPLAN_DEVKIT_DIR)
sys.path.insert(0, SRC_DIR)

from dwm.datasets.nuplan import MotionDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

HEIGHT = 288
WIDTH = 512
USE_TRAIN = False
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5"))
ENABLE_DEBUGPY = os.environ.get("ENABLE_DEBUGPY", "0") == "1"

SENSOR_ROOT = os.path.join(PROJECT_ROOT, "nuplan_prepo/mini_sensors")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "nuplan_cache")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "nuplan_link/plan_data/mini")
MAP_ROOT = os.path.join(PROJECT_ROOT, "nuplan_link/maps")

TRAIN_PKL = os.path.join(PROJECT_ROOT, "nuplan_prepo/mini_infos_train.pkl")
VAL_PKL = os.path.join(PROJECT_ROOT, "nuplan_prepo/mini_infos_val.pkl")

TRAIN_BALANCED_JSON = None
VAL_BALANCED_JSON = os.path.join(
    PROJECT_ROOT,
    "nuplan_balanced/val_nonidle_windows_metadata.json",
)

IMAGE_DESC_PATH = os.path.join(PROJECT_ROOT, "nuplan_prepo/nuplan_scene.json")

SENSOR_CHANNELS = [
    "CAM_L2",
    "CAM_L1",
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
]

CROSSVIEW_MASK = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1],
    ],
    dtype=torch.bool,
)

resize_to_tensor = T.Compose([
    T.Resize((HEIGHT, WIDTH)),
    T.ToTensor(),
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
                row_t.append(torch.zeros(3, HEIGHT, WIDTH))

        out.append(row_t)

    return out


def _shape_nested(nested):
    if not nested:
        return "[]"

    t_count = len(nested)
    v_count = len(nested[0]) if t_count else 0
    x = nested[0][0] if (t_count and v_count) else None

    if torch.is_tensor(x):
        return f"[T={t_count}, V={v_count}, {tuple(x.shape)}]"

    if isinstance(x, Image.Image):
        return f"[T={t_count}, V={v_count}, PIL={x.size}]"

    return f"[T={t_count}, V={v_count}, type={type(x).__name__}]"


def _as_tensor_stack(nested):
    if nested is None:
        return None

    rows = []

    for row in nested:
        rows.append(torch.stack(row, dim=0))

    return torch.stack(rows, dim=0)


def _first_existing_path(path_or_none):
    if path_or_none is None:
        return None

    if os.path.isfile(path_or_none):
        return path_or_none

    print("[WARN] file not found, skip:", path_or_none, flush=True)
    return None


def make_base_ds(train=True):
    data_pkl = TRAIN_PKL if train else VAL_PKL
    balanced_json = TRAIN_BALANCED_JSON if train else VAL_BALANCED_JSON
    balanced_json = _first_existing_path(balanced_json)

    if not os.path.isfile(data_pkl):
        raise FileNotFoundError(f"pkl not found: {data_pkl}")

    return MotionDataset(
        sensor_root=SENSOR_ROOT,
        pkl_path=data_pkl,
        balanced_json_path=balanced_json,
        cache_root=CACHE_ROOT,
        dataset_root=DATASET_ROOT,
        map_root=MAP_ROOT,
        sequence_length=35 if not train else 20,
        fps_stride_tuples=[(6,8,0.6)] if not train else [(6, 2, 0.5)],
        sensor_channels=SENSOR_CHANNELS,
        enable_camera_transforms=True,
        enable_ego_transforms=True,
        enable_synchronization_check=True,
        _3dbox_image_settings={},
        hdmap_image_settings={},
        stub_key_data_dict={
            "crossview_mask": ["content", CROSSVIEW_MASK],
            "dataset_tag": ["content", torch.tensor(0, dtype=torch.int64)],
        },
        image_description_settings={
            "path": IMAGE_DESC_PATH,
            "align_keys": ["time", "weather"],
            "reorder_keys": True,
        },
    )


class SimpleDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = dict(self.base_ds[idx])

        if "images" in sample:
            sample["vae_images"] = _apply_nested_images(sample["images"])

        if "3dbox_images" in sample:
            sample["3dbox_images"] = _apply_nested_images(sample["3dbox_images"])

        if "hdmap_images" in sample:
            sample["hdmap_images"] = _apply_nested_images(sample["hdmap_images"])

        if "image_description" in sample:
            sample["clip_text"] = sample["image_description"]

        for k in ("images", "lidar_points", "image_description"):
            sample.pop(k, None)

        return sample


def collate_single(batch):
    assert len(batch) == 1, "This debug script expects batch_size=1."
    return batch[0]


def pretty_print_sample(sample):
    print("keys:", list(sample.keys()))

    for key in ("vae_images", "3dbox_images", "hdmap_images"):
        if key in sample:
            print(key + ":", _shape_nested(sample[key]))

    for key in ("camera_intrinsics", "camera_transforms", "ego_transforms", "image_size", "crossview_mask"):
        if key in sample:
            value = sample[key]
            if torch.is_tensor(value):
                print(key + ":", tuple(value.shape), value.dtype)
            else:
                print(key + ":", type(value).__name__)

    if "fps" in sample:
        print("fps:", float(sample["fps"]))

    if "camera_names" in sample:
        print("camera_names:", sample["camera_names"])

    if "scene" in sample:
        print("scene:", sample["scene"])

    if "angle" in sample:
        print("angle:", sample["angle"])

    if "dist" in sample:
        print("dist:", sample["dist"])


def main():
    if ENABLE_DEBUGPY:
        import debugpy

        debugpy.listen(("0.0.0.0", 9876))
        print("[debugpy] listening on 0.0.0.0:9876, waiting for VS Code to attach...")
        debugpy.wait_for_client()
        print("[debugpy] attached")

    base_ds = make_base_ds(train=USE_TRAIN)
    ds = SimpleDatasetAdapter(base_ds)

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_single,
        pin_memory=False,
    )

    print("USE_TRAIN:", USE_TRAIN)
    print("Dataset items:", len(ds))

    for i, batch in enumerate(loader):
        print(f"\n==== sample {i} ====")
        pretty_print_sample(batch)

        if "vae_images" in batch:
            vae = _as_tensor_stack(batch["vae_images"])
            print("vae_images stacked:", tuple(vae.shape), vae.dtype, float(vae.min()), float(vae.max()))

        if "3dbox_images" in batch:
            box = _as_tensor_stack(batch["3dbox_images"])
            print("3dbox_images stacked:", tuple(box.shape), box.dtype, float(box.min()), float(box.max()))

        if "hdmap_images" in batch:
            hdmap = _as_tensor_stack(batch["hdmap_images"])
            print("hdmap_images stacked:", tuple(hdmap.shape), hdmap.dtype, float(hdmap.min()), float(hdmap.max()))

        if i + 1 >= MAX_SAMPLES:
            break

    print("[DONE]")


if __name__ == "__main__":
    main()
