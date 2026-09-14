import bisect
import math
import numpy as np
from PIL import ImageDraw
import torch
import transforms3d
import torch.nn.functional as F
import random
import os, time
import cv2

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode

INVALID_BIN = 0  # proj_depth 是 bins_u16 时，无效值是 0


def rs_bins_u16(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    x:
      - depth bins: [T,V,H,W] long/int
      - sem masks : [T,V,C,H,W] long/int/float (一般 0/1)
    return:
      - same ndim, resized to target H,W
    resize rule: nearest (不会引入新类别)
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[rs_bins_u16] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[rs_bins_u16] expect torch.Tensor, got {type(x)}")

    # case A: [T,V,H,W]
    if x.ndim == 4:
        T, V, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, 1, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        return y.reshape(T, V, height, width).long()

    # case B: [T,V,C,H,W]
    if x.ndim == 5:
        T, V, C, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, C, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        # sem 通常你希望保持 float(0/1)，也可以 .long()
        return y.reshape(T, V, C, height, width)

    raise ValueError(f"[rs_bins_u16] expect [T,V,H,W] or [T,V,C,H,W], got {tuple(x.shape)}")

def resize_clr_keep_invalid(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    x: torch.Tensor [T,V,3,H,W]  float in [0,1] (推荐) 或 uint8
    return: torch.FloatTensor [T,V,3,height,width] in [0,1]
    规则：
      - 下采样：黑色像素(0,0,0)不参与平均（weighted area）
      - 上采样：nearest
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[resize_clr_keep_invalid_tvchw] expect torch.Tensor, got {type(x)}")

    if x.ndim != 5:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] expect [T,V,3,H,W], got {tuple(x.shape)}")

    T, V, C, H0, W0 = x.shape
    if C != 3:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] channel must be 3, got C={C}")

    # to float [0,1]
    if x.dtype == torch.uint8:
        x_f = x.float() / 255.0
    else:
        x_f = x.float()

    x_f = x_f.reshape(T * V, 3, H0, W0)

    # upsample -> nearest
    if height >= H0 and width >= W0:
        y = F.interpolate(x_f, size=(height, width), mode="nearest")
        return y.reshape(T, V, 3, height, width)

    # downsample -> weighted area (ignore black)
    valid = (x_f.sum(dim=1, keepdim=True) > 0).float()  # [TV,1,H,W]

    num = F.interpolate(x_f * valid, size=(height, width), mode="area")
    den = F.interpolate(valid,       size=(height, width), mode="area")

    y = torch.zeros_like(num)
    m = den > 1e-6
    y[m.expand_as(y)] = (num / den.clamp_min(1e-6))[m.expand_as(y)]

    return y.reshape(T, V, 3, height, width)


class Copy():
    def __call__(self, a):
        return a


class FilterPoints():
    def __init__(self, min_distance: float = 0, max_distance: float = 1000.0):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, a):
        distances = a[:, :3].norm(dim=-1)
        mask = torch.logical_and(
            distances >= self.min_distance, distances < self.max_distance)

        return a[mask]


class TakePoints():
    def __init__(self, max_count: int = 32768):
        self.max_count = max_count

    def __call__(self, a):
        if a.shape[0] > self.max_count:
            indices = torch.randperm(a.shape[0])[:self.max_count]
            a = a[indices]

        return a
    
def flatten_nested_list(x):
    if isinstance(x, list):
        out = []
        for i in x:
            if isinstance(i, list):
                out.extend(flatten_nested_list(i))
            else:
                out.append(i)
        return out
    return [x]


def to_chw_tensor(img):
    if torch.is_tensor(img):
        x = img
        if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
            return x.float()
        if x.ndim == 3 and x.shape[-1] in [1, 3, 4]:
            return x.permute(2, 0, 1).float()
        if x.ndim == 2:
            return x.unsqueeze(0).float()
        raise ValueError(f"Unsupported tensor image shape: {tuple(x.shape)}")

    if isinstance(img, Image.Image):
        return ToTensor()(img)

    if isinstance(img, np.ndarray):
        x = torch.from_numpy(img)
        if x.ndim == 2:
            return x.unsqueeze(0).float()
        if x.ndim == 3 and x.shape[-1] in [1, 3, 4]:
            if x.dtype == torch.uint8:
                return x.permute(2, 0, 1).float() / 255.0
            return x.permute(2, 0, 1).float()
        if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
            return x.float()
        raise ValueError(f"Unsupported numpy image shape: {tuple(x.shape)}")

    raise TypeError(f"Unsupported image type: {type(img)}")


def resize_clr_keep_invalid(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[resize_clr_keep_invalid_tvchw] expect torch.Tensor, got {type(x)}")

    if x.ndim != 5:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] expect [T,V,3,H,W], got {tuple(x.shape)}")

    T, V, C, H0, W0 = x.shape
    if C != 3:
        raise ValueError(f"[resize_clr_keep_invalid_tvchw] channel must be 3, got C={C}")

    if x.dtype == torch.uint8:
        x_f = x.float() / 255.0
    else:
        x_f = x.float()

    x_f = x_f.reshape(T * V, 3, H0, W0)

    if height >= H0 and width >= W0:
        y = F.interpolate(x_f, size=(height, width), mode="nearest")
        return y.reshape(T, V, 3, height, width)

    valid = (x_f.sum(dim=1, keepdim=True) > 0).float()

    num = F.interpolate(x_f * valid, size=(height, width), mode="area")
    den = F.interpolate(valid, size=(height, width), mode="area")

    y = torch.zeros_like(num)
    m = den > 1e-6
    y[m.expand_as(y)] = (num / den.clamp_min(1e-6))[m.expand_as(y)]

    return y.reshape(T, V, 3, height, width)


def rs_bins_u16(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"[rs_bins_u16] invalid target size: H={height}, W={width}")

    if not torch.is_tensor(x):
        raise TypeError(f"[rs_bins_u16] expect torch.Tensor, got {type(x)}")

    if x.ndim == 4:
        T, V, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, 1, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        return y.reshape(T, V, height, width).long()

    if x.ndim == 5:
        T, V, C, H0, W0 = x.shape
        y = F.interpolate(
            x.reshape(T * V, C, H0, W0).float(),
            size=(height, width),
            mode="nearest"
        )
        return y.reshape(T, V, C, height, width)

    raise ValueError(f"[rs_bins_u16] expect [T,V,H,W] or [T,V,C,H,W], got {tuple(x.shape)}")


def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _extract_principal_point(k):
    if k is None:
        return None, None

    if not torch.is_tensor(k):
        k = torch.tensor(k)

    if k.shape[-2:] != (3, 3):
        raise ValueError(f"camera_intrinsics must end with (3, 3), got {tuple(k.shape)}")

    cx = float(k[0, 2].item())
    cy = float(k[1, 2].item())
    return cx, cy


def compute_crop_meta(
    h0: int,
    w0: int,
    target_h: int,
    target_w: int,
    k=None,
    crop_use_intrinsics_center: bool = True,
    crop_horizontal_anchor: float = 0.5,
    crop_vertical_anchor: float = 0.5,
):
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"invalid source size: H={h0}, W={w0}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"invalid target size: H={target_h}, W={target_w}")

    target_ratio = float(target_w) / float(target_h)
    src_ratio = float(w0) / float(h0)

    cx, cy = _extract_principal_point(k)

    if crop_use_intrinsics_center and cx is not None and cy is not None:
        center_x = cx
        center_y = cy
    else:
        center_x = (w0 - 1) * float(crop_horizontal_anchor)
        center_y = (h0 - 1) * float(crop_vertical_anchor)

    if abs(src_ratio - target_ratio) < 1e-8:
        crop_left = 0
        crop_top = 0
        crop_w = w0
        crop_h = h0

    elif src_ratio < target_ratio:
        # 图偏窄：优先裁上下，去掉天空和车头
        crop_w = w0
        crop_h = int(round(float(w0) / target_ratio))
        crop_h = min(crop_h, h0)
        crop_left = 0
        crop_top = int(round(center_y - crop_h * 0.5))
        crop_top = _clamp_int(crop_top, 0, h0 - crop_h)

    else:
        # 图偏宽：只能裁左右
        crop_h = h0
        crop_w = int(round(float(h0) * target_ratio))
        crop_w = min(crop_w, w0)
        crop_top = 0
        crop_left = int(round(center_x - crop_w * 0.5))
        crop_left = _clamp_int(crop_left, 0, w0 - crop_w)

    scale_x = float(target_w) / float(crop_w)
    scale_y = float(target_h) / float(crop_h)

    return {
        "orig_h": h0,
        "orig_w": w0,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "new_h": target_h,
        "new_w": target_w,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def apply_crop_resize_to_chw_tensor(x: torch.Tensor, meta: dict, mode: str = "bilinear") -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expect CHW tensor, got {tuple(x.shape)}")

    crop_top = meta["crop_top"]
    crop_left = meta["crop_left"]
    crop_h = meta["crop_h"]
    crop_w = meta["crop_w"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]

    x = x[:, crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]

    if mode in ["bilinear", "bicubic"]:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(new_h, new_w),
            mode=mode,
            align_corners=False
        ).squeeze(0)
    else:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(new_h, new_w),
            mode=mode
        ).squeeze(0)

    return x


def resize_and_crop_image(
    img,
    target_h: int,
    target_w: int,
    k=None,
    crop_use_intrinsics_center: bool = True,
    crop_horizontal_anchor: float = 0.5,
    crop_vertical_anchor: float = 0.5,
):
    x = to_chw_tensor(img)
    c, h0, w0 = x.shape

    meta = compute_crop_meta(
        h0=h0,
        w0=w0,
        target_h=target_h,
        target_w=target_w,
        k=k,
        crop_use_intrinsics_center=crop_use_intrinsics_center,
        crop_horizontal_anchor=crop_horizontal_anchor,
        crop_vertical_anchor=crop_vertical_anchor,
    )
    y = apply_crop_resize_to_chw_tensor(x, meta, mode="bilinear")
    return y, meta


def update_intrinsics_for_resize_crop(k, crop_left, crop_top, scale_x, scale_y):
    if not torch.is_tensor(k):
        k = torch.tensor(k)

    k = k.clone().float()
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"camera_intrinsics must end with (3, 3), got {tuple(k.shape)}")

    k[..., 0, 0] *= float(scale_x)
    k[..., 1, 1] *= float(scale_y)
    k[..., 0, 2] = (k[..., 0, 2] - float(crop_left)) * float(scale_x)
    k[..., 1, 2] = (k[..., 1, 2] - float(crop_top)) * float(scale_y)
    return k


def validate_resize_crop_update(k_before, k_after, meta, atol=1e-6):
    fx0 = float(k_before[0, 0].item())
    fy0 = float(k_before[1, 1].item())
    cx0 = float(k_before[0, 2].item())
    cy0 = float(k_before[1, 2].item())

    fx1 = float(k_after[0, 0].item())
    fy1 = float(k_after[1, 1].item())
    cx1 = float(k_after[0, 2].item())
    cy1 = float(k_after[1, 2].item())

    crop_left = float(meta["crop_left"])
    crop_top = float(meta["crop_top"])
    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])

    if abs(fx1 - scale_x * fx0) > atol:
        raise AssertionError(f"fx mismatch: expected {scale_x * fx0}, got {fx1}")
    if abs(fy1 - scale_y * fy0) > atol:
        raise AssertionError(f"fy mismatch: expected {scale_y * fy0}, got {fy1}")
    if abs(cx1 - ((cx0 - crop_left) * scale_x)) > atol:
        raise AssertionError(f"cx mismatch: expected {(cx0 - crop_left) * scale_x}, got {cx1}")
    if abs(cy1 - ((cy0 - crop_top) * scale_y)) > atol:
        raise AssertionError(f"cy mismatch: expected {(cy0 - crop_top) * scale_y}, got {cy1}")

    u0 = cx0
    v0 = cy0
    u1 = (u0 - crop_left) * scale_x
    v1 = (v0 - crop_top) * scale_y

    x0 = (u0 - cx0) / fx0
    y0 = (v0 - cy0) / fy0
    x1 = (u1 - cx1) / fx1
    y1 = (v1 - cy1) / fy1

    if abs(x0 - x1) > atol or abs(y0 - y1) > atol:
        raise AssertionError(
            f"principal ray mismatch: old=({x0}, {y0}), new=({x1}, {y1})"
        )


def crop_resize_tv_tensor_nearest(x: torch.Tensor, meta_nested):
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    if x.ndim == 4:
        t_count, v_count, _, _ = x.shape
        out_rows = []
        for t in range(t_count):
            out_row = []
            for v in range(v_count):
                cur = x[t, v].unsqueeze(0)
                cur = apply_crop_resize_to_chw_tensor(
                    cur,
                    meta_nested[t][v],
                    mode="nearest"
                ).squeeze(0)
                if x.dtype.is_floating_point:
                    out_row.append(cur.to(dtype=x.dtype))
                else:
                    out_row.append(cur.round().to(dtype=x.dtype))
            out_rows.append(torch.stack(out_row, dim=0))
        return torch.stack(out_rows, dim=0)

    if x.ndim == 5:
        t_count, v_count, _, _, _ = x.shape
        out_rows = []
        for t in range(t_count):
            out_row = []
            for v in range(v_count):
                cur = apply_crop_resize_to_chw_tensor(
                    x[t, v].float(),
                    meta_nested[t][v],
                    mode="nearest"
                )
                if x.dtype.is_floating_point:
                    out_row.append(cur.to(dtype=x.dtype))
                else:
                    out_row.append(cur.round().to(dtype=x.dtype))
            out_rows.append(torch.stack(out_row, dim=0))
        return torch.stack(out_rows, dim=0)

    raise ValueError(f"Unsupported tensor shape for crop_resize_tv_tensor_nearest: {tuple(x.shape)}")


def crop_resize_proj_clr_keep_invalid(x: torch.Tensor, meta_nested):
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    if x.ndim != 5:
        raise ValueError(f"proj_clr expect [T,V,3,H,W], got {tuple(x.shape)}")

    t_count, v_count, c, _, _ = x.shape
    out_rows = []
    for t in range(t_count):
        out_row = []
        for v in range(v_count):
            meta = meta_nested[t][v]
            crop_top = meta["crop_top"]
            crop_left = meta["crop_left"]
            crop_h = meta["crop_h"]
            crop_w = meta["crop_w"]
            new_h = meta["new_h"]
            new_w = meta["new_w"]

            cur = x[t, v:v + 1, :, crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
            cur = resize_clr_keep_invalid(cur.unsqueeze(0), new_h, new_w)[0, 0]
            out_row.append(cur)
        out_rows.append(torch.stack(out_row, dim=0))
    return torch.stack(out_rows, dim=0)
class RandomCameraOrderAugment:
    def __init__(
        self,
        enabled=True,
        prob=1.0,
        keep_first_n=0,
        record_perm_key="camera_order_perm",
        record_old_names_key="camera_names_before_permute",
    ):
        self.enabled = enabled
        self.prob = float(prob)
        self.keep_first_n = int(keep_first_n)
        self.record_perm_key = record_perm_key
        self.record_old_names_key = record_old_names_key

        self.tv_tensor_keys = [
            "images",
            "vae_images",
            "3dbox_images",
            "hdmap_images",
            "valid_mask",
            "padding_mask",
            "camera_intrinsics",
            "camera_transforms",
            "ego_transforms",
            "image_size",
            "distortion",
            "angle",
            "dist",
            "is_uncalibrated",
            "proj_depth",
            "proj_sem",
            "proj_clr",
        ]

        self.nested_list_keys = [
            "camera_names",
            "image_description",
            "clip_text",
        ]

    def make_perm(self, v_count):
        if not self.enabled:
            return torch.arange(v_count, dtype=torch.long)

        if random.random() >= self.prob:
            return torch.arange(v_count, dtype=torch.long)

        if self.keep_first_n <= 0:
            return torch.randperm(v_count)

        if self.keep_first_n >= v_count:
            return torch.arange(v_count, dtype=torch.long)

        fixed = torch.arange(self.keep_first_n, dtype=torch.long)
        tail = torch.randperm(v_count - self.keep_first_n) + self.keep_first_n
        return torch.cat([fixed, tail], dim=0)

    def permute_tensor(self, value, perm, v_count):
        if not torch.is_tensor(value):
            return value

        if v_count is None or v_count <= 1:
            return value

        perm = perm.to(value.device)

        # [V]
        if value.ndim == 1 and value.shape[0] == v_count:
            return value.index_select(0, perm)

        # [V, ...]
        if value.ndim >= 2 and value.shape[0] == v_count:
            return value.index_select(0, perm)

        # [T, V] or [T, V, ...]
        if value.ndim >= 2 and value.shape[1] == v_count:
            return value.index_select(1, perm)

        # reserve for possible [B, T, V, ...] style tensors
        if value.ndim >= 3 and value.shape[2] == v_count:
            return value.index_select(2, perm)

        return value

    def permute_nested_list(self, value, perm_list, v_count):
        if not isinstance(value, list):
            return value

        if len(value) == v_count:
            return [value[i] for i in perm_list]

        if len(value) > 0 and isinstance(value[0], list):
            out = []
            for row in value:
                if len(row) == v_count:
                    out.append([row[i] for i in perm_list])
                else:
                    out.append(row)
            return out

        return value

    def permute_crossview_mask(self, value, perm):
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        if value.ndim != 2:
            raise ValueError(
                "RandomCameraOrderAugment expects per-sample crossview_mask "
                f"with shape [V, V], but got {tuple(value.shape)}."
            )

        perm = perm.to(value.device)
        value = value.index_select(0, perm)
        value = value.index_select(1, perm)
        return value

    def __call__(self, item, v_count):
        perm = self.make_perm(v_count)
        perm_list = perm.tolist()

        if self.record_perm_key is not None:
            item[self.record_perm_key] = perm.clone()

        if (
            self.record_old_names_key is not None
            and "camera_names" in item
        ):
            item[self.record_old_names_key] = item["camera_names"]

        for key in self.tv_tensor_keys:
            if key in item:
                item[key] = self.permute_tensor(item[key], perm, v_count)

        for key in self.nested_list_keys:
            if key in item:
                item[key] = self.permute_nested_list(
                    item[key],
                    perm_list,
                    v_count,
                )

        if "crossview_mask" in item:
            item["crossview_mask"] = self.permute_crossview_mask(
                item["crossview_mask"],
                perm,
            )

        return item    


def _mdtoken_to_numpy_or_none(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _mdtoken_order_polygon_np(pts):
    pts = np.asarray(pts, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[0] < 3:
        return []

    center = pts.mean(axis=0)
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )
    order = np.argsort(angles)

    return [
        (float(pts[i, 0]), float(pts[i, 1]))
        for i in order
    ]


def _mdtoken_xy_to_uv_np(x, y, w, h, x_min, x_max, y_min, y_max):
    u = (float(x) - float(x_min)) / max(float(x_max - x_min), 1e-6)
    v = (float(y_max) - float(y)) / max(float(y_max - y_min), 1e-6)

    u = u * float(w)
    v = v * float(h)

    return u, v


def _mdtoken_select_time_slice(arr, t):
    if arr is None:
        return None

    arr = np.asarray(arr)

    if arr.ndim >= 1 and arr.shape[0] > t:
        return arr[t]

    return arr


def _mdtoken_select_bev_boxes(arr, t):
    cur = _mdtoken_select_time_slice(arr, t)

    if cur is None:
        return None

    cur = np.asarray(cur)

    if cur.ndim == 4 and cur.shape[-2:] == (8, 3):
        return cur[0]

    if cur.ndim == 3 and cur.shape[-2:] == (8, 3):
        return cur

    return cur


def _mdtoken_select_bev_classes(arr, t):
    cur = _mdtoken_select_time_slice(arr, t)

    if cur is None:
        return None

    cur = np.asarray(cur)

    if cur.ndim == 2:
        return cur[0]

    return cur


def _mdtoken_select_bev_masks(arr, t):
    cur = _mdtoken_select_time_slice(arr, t)

    if cur is None:
        return None

    cur = np.asarray(cur)

    if cur.ndim == 2:
        return (cur > 0).any(axis=0).astype(np.float32)

    return cur


def _mdtoken_rasterize_bev_boxes(
    boxes,
    classes,
    masks,
    h,
    w,
    num_classes,
    x_min,
    x_max,
    y_min,
    y_max,
):
    dyn = np.zeros((num_classes, h, w), dtype=np.float32)

    if boxes is None or masks is None:
        return dyn

    boxes = np.asarray(boxes, dtype=np.float32)
    masks = np.asarray(masks)

    if boxes.ndim != 3 or boxes.shape[-2:] != (8, 3):
        return dyn

    if classes is None:
        classes = np.zeros((boxes.shape[0],), dtype=np.int64)
    else:
        classes = np.asarray(classes)

    for i in range(boxes.shape[0]):
        if i >= masks.shape[0]:
            continue

        if float(masks[i]) <= 0:
            continue

        cls_id = int(classes[i]) if i < classes.shape[0] else 0

        if cls_id < 0 or cls_id >= num_classes:
            continue

        box = boxes[i]

        if not np.isfinite(box).all():
            continue

        bottom_idx = np.argsort(box[:, 2])[:4]
        bottom_xy = box[bottom_idx, :2]

        pts = np.asarray(
            [
                _mdtoken_xy_to_uv_np(
                    x,
                    y,
                    w,
                    h,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )
                for x, y in bottom_xy
            ],
            dtype=np.float32,
        )

        if pts[:, 0].max() < 0:
            continue

        if pts[:, 0].min() >= w:
            continue

        if pts[:, 1].max() < 0:
            continue

        if pts[:, 1].min() >= h:
            continue

        poly = _mdtoken_order_polygon_np(pts)

        if len(poly) < 3:
            continue

        channel = Image.fromarray(
            (dyn[cls_id] * 255.0).astype(np.uint8),
            mode="L",
        )
        draw = ImageDraw.Draw(channel)
        draw.polygon(poly, fill=255)
        dyn[cls_id] = np.asarray(channel, dtype=np.float32) / 255.0

    return dyn


def _mdtoken_append_dynamic_to_tensor_maps(
    maps_tensor,
    bboxes_np,
    classes_np,
    masks_np,
    num_classes,
    x_min,
    x_max,
    y_min,
    y_max,
):
    device = maps_tensor.device
    dtype = maps_tensor.dtype

    maps_np = maps_tensor.detach().float().cpu().numpy()

    if maps_np.ndim == 4:
        t_count, _, h, w = maps_np.shape
        out = np.zeros((t_count, 3 + num_classes, h, w), dtype=np.float32)

        for t in range(t_count):
            boxes_t = _mdtoken_select_bev_boxes(bboxes_np, t)
            classes_t = _mdtoken_select_bev_classes(classes_np, t)
            masks_t = _mdtoken_select_bev_masks(masks_np, t)

            dyn = _mdtoken_rasterize_bev_boxes(
                boxes_t,
                classes_t,
                masks_t,
                h,
                w,
                num_classes,
                x_min,
                x_max,
                y_min,
                y_max,
            )

            out[t, :3] = maps_np[t, :3]
            out[t, 3:] = dyn

        return torch.from_numpy(out).to(device=device, dtype=dtype)

    if maps_np.ndim == 5:
        t_count, view_count, _, h, w = maps_np.shape
        out = np.zeros((t_count, view_count, 3 + num_classes, h, w), dtype=np.float32)

        for t in range(t_count):
            boxes_t = _mdtoken_select_bev_boxes(bboxes_np, t)
            classes_t = _mdtoken_select_bev_classes(classes_np, t)
            masks_t = _mdtoken_select_bev_masks(masks_np, t)

            dyn = _mdtoken_rasterize_bev_boxes(
                boxes_t,
                classes_t,
                masks_t,
                h,
                w,
                num_classes,
                x_min,
                x_max,
                y_min,
                y_max,
            )

            for v in range(view_count):
                out[t, v, :3] = maps_np[t, v, :3]
                out[t, v, 3:] = dyn

        return torch.from_numpy(out).to(device=device, dtype=dtype)

    return maps_tensor


def _mdtoken_append_dynamic_to_hwc_array(
    arr,
    t,
    bboxes_np,
    classes_np,
    masks_np,
    num_classes,
    x_min,
    x_max,
    y_min,
    y_max,
):
    arr = np.asarray(arr)

    if arr.ndim != 3:
        return arr

    h, w = arr.shape[:2]
    static = arr[:, :, :3].astype(np.uint8)

    boxes_t = _mdtoken_select_bev_boxes(bboxes_np, t)
    classes_t = _mdtoken_select_bev_classes(classes_np, t)
    masks_t = _mdtoken_select_bev_masks(masks_np, t)

    dyn_chw = _mdtoken_rasterize_bev_boxes(
        boxes_t,
        classes_t,
        masks_t,
        h,
        w,
        num_classes,
        x_min,
        x_max,
        y_min,
        y_max,
    )
    dyn_hwc = np.transpose(
        (dyn_chw * 255.0).astype(np.uint8),
        (1, 2, 0),
    )

    return np.concatenate([static, dyn_hwc], axis=-1)


def _append_dynamic_bev_masks_from_bbox_tokens(
    result,
    map_key="hdmap_bev_images",
    bbox_key="bbox_token_corners",
    class_key="bbox_token_classes",
    mask_key="bbox_token_masks",
    num_classes=10,
    x_min=-80.0,
    x_max=80.0,
    y_min=-80.0,
    y_max=80.0,
):
    if os.environ.get("DWM_DISABLE_BBOX_DYNAMIC_BEV", "0") == "1":
        return result

    if map_key not in result:
        return result

    if bbox_key not in result:
        return result

    if mask_key not in result:
        return result

    maps = result[map_key]
    bboxes_np = _mdtoken_to_numpy_or_none(result.get(bbox_key, None))
    classes_np = _mdtoken_to_numpy_or_none(result.get(class_key, None))
    masks_np = _mdtoken_to_numpy_or_none(result.get(mask_key, None))

    if bboxes_np is None or masks_np is None:
        return result

    if torch.is_tensor(maps):
        result[map_key] = _mdtoken_append_dynamic_to_tensor_maps(
            maps,
            bboxes_np,
            classes_np,
            masks_np,
            int(num_classes),
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
        )
        return result

    if isinstance(maps, (list, tuple)):
        new_maps = []

        for t, map_item in enumerate(maps):
            new_maps.append(
                _mdtoken_append_dynamic_to_hwc_array(
                    map_item,
                    t,
                    bboxes_np,
                    classes_np,
                    masks_np,
                    int(num_classes),
                    float(x_min),
                    float(x_max),
                    float(y_min),
                    float(y_max),
                )
            )

        result[map_key] = new_maps
        return result

    maps_np = np.asarray(maps)

    if maps_np.ndim == 4:
        new_maps = []

        for t in range(maps_np.shape[0]):
            new_maps.append(
                _mdtoken_append_dynamic_to_hwc_array(
                    maps_np[t],
                    t,
                    bboxes_np,
                    classes_np,
                    masks_np,
                    int(num_classes),
                    float(x_min),
                    float(x_max),
                    float(y_min),
                    float(y_max),
                )
            )

        result[map_key] = np.stack(new_maps, axis=0)
        return result

    if maps_np.ndim == 3:
        result[map_key] = _mdtoken_append_dynamic_to_hwc_array(
            maps_np,
            0,
            bboxes_np,
            classes_np,
            masks_np,
            int(num_classes),
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
        )
        return result

    return result

class DatasetAdapter(torch.utils.data.Dataset):
    @staticmethod
    def apply_transform(transform, a, stack: bool = True):
        if isinstance(a, list):
            result = [
                DatasetAdapter.apply_transform(transform, i, stack) for i in a
            ]
            if stack:
                result = torch.stack(result)
            return result
        return transform(a)

    @staticmethod
    def infer_tv_from_item(item):
        if "camera_intrinsics" in item:
            k = item["camera_intrinsics"]
            if not torch.is_tensor(k):
                k = torch.tensor(k)
            if k.ndim == 4:
                return int(k.shape[0]), int(k.shape[1])
            if k.ndim == 3:
                return int(k.shape[0]), None

        if "images" in item:
            imgs = item["images"]
            if isinstance(imgs, list) and len(imgs) > 0:
                if isinstance(imgs[0], list):
                    return len(imgs), len(imgs[0])
                return len(imgs), None

        raise ValueError("Cannot infer T/V from item")

    @staticmethod
    def ensure_nested_list_tv(x, t_count=None, v_count=None, key="images"):
        if not isinstance(x, list) or len(x) == 0:
            raise ValueError(f"item['{key}'] must be a non-empty list")

        if isinstance(x[0], list):
            out = x
            if t_count is not None and len(out) != t_count:
                raise ValueError(
                    f"{key} time count mismatch: len={len(out)} vs expected T={t_count}"
                )
            if v_count is not None:
                for t in range(len(out)):
                    if len(out[t]) != v_count:
                        raise ValueError(
                            f"{key} view count mismatch at t={t}: "
                            f"len={len(out[t])} vs expected V={v_count}"
                        )
            return out

        if t_count is None:
            raise ValueError(f"{key} is flat list but T is unknown")
        if v_count is None:
            if len(x) != t_count:
                raise ValueError(
                    f"{key} flat list length mismatch: len={len(x)} vs expected T={t_count}"
                )
            return [[x[t]] for t in range(t_count)]

        if len(x) != t_count * v_count:
            raise ValueError(
                f"{key} flat list length mismatch: len={len(x)} vs expected T*V={t_count * v_count}"
            )

        out = []
        flat_idx = 0
        for t in range(t_count):
            row = []
            for v in range(v_count):
                row.append(x[flat_idx])
                flat_idx += 1
            out.append(row)
        return out

    @staticmethod
    def stack_nested_list_tv(x, key="images"):
        if not isinstance(x, list) or len(x) == 0:
            raise ValueError(f"{key} must be a non-empty nested list")
        if not isinstance(x[0], list):
            raise ValueError(f"{key} must be nested as [T][V], got flat list")
        rows = []
        for t in range(len(x)):
            rows.append(torch.stack(x[t], dim=0))
        return torch.stack(rows, dim=0)

    @staticmethod
    def crop_resize_nested_images_keep_tv(
        nested_images,
        height,
        width,
        intrinsics_src=None,
        crop_use_intrinsics_center: bool = True,
        crop_horizontal_anchor: float = 0.5,
        crop_vertical_anchor: float = 0.5,
    ):
        t_count = len(nested_images)
        v_count = len(nested_images[0])

        if intrinsics_src is not None and not torch.is_tensor(intrinsics_src):
            intrinsics_src = torch.tensor(intrinsics_src)

        resized_nested = []
        meta_nested = []

        for t in range(t_count):
            resized_row = []
            meta_row = []

            if len(nested_images[t]) != v_count:
                raise ValueError(
                    f"images view count mismatch at t={t}: "
                    f"{len(nested_images[t])} vs expected {v_count}"
                )

            for v in range(v_count):
                k_this = None
                if intrinsics_src is not None:
                    if intrinsics_src.ndim == 4:
                        k_this = intrinsics_src[t, v]
                    elif intrinsics_src.ndim == 3:
                        k_this = intrinsics_src[t]
                    else:
                        raise ValueError(
                            f"Unsupported camera_intrinsics shape: {tuple(intrinsics_src.shape)}"
                        )

                resized_img, meta = resize_and_crop_image(
                    img=nested_images[t][v],
                    target_h=height,
                    target_w=width,
                    k=k_this,
                    crop_use_intrinsics_center=crop_use_intrinsics_center,
                    crop_horizontal_anchor=crop_horizontal_anchor,
                    crop_vertical_anchor=crop_vertical_anchor,
                )
                resized_row.append(resized_img)
                meta_row.append(meta)

            resized_nested.append(resized_row)
            meta_nested.append(meta_row)

        return resized_nested, meta_nested

    @staticmethod
    def build_size_tensors_from_meta(meta_nested):
        t_count = len(meta_nested)
        v_count = len(meta_nested[0])

        image_size_before_resize_crop = torch.zeros(
            t_count, v_count, 2, dtype=torch.long
        )
        image_size_after_crop_before_resize = torch.zeros(
            t_count, v_count, 2, dtype=torch.long
        )
        image_size_tensor = torch.zeros(
            t_count, v_count, 2, dtype=torch.long
        )

        for t in range(t_count):
            for v in range(v_count):
                meta = meta_nested[t][v]
                image_size_before_resize_crop[t, v] = torch.tensor(
                    [meta["orig_w"], meta["orig_h"]], dtype=torch.long
                )
                image_size_after_crop_before_resize[t, v] = torch.tensor(
                    [meta["crop_w"], meta["crop_h"]], dtype=torch.long
                )
                image_size_tensor[t, v] = torch.tensor(
                    [meta["new_w"], meta["new_h"]], dtype=torch.long
                )

        return (
            image_size_before_resize_crop,
            image_size_after_crop_before_resize,
            image_size_tensor,
        )

    @staticmethod
    def update_intrinsics_nested(k_src, meta_nested):
        if not torch.is_tensor(k_src):
            k_src = torch.tensor(k_src)

        k_new = k_src.clone().float()

        if k_new.ndim == 4:
            t_count = k_new.shape[0]
            v_count = k_new.shape[1]

            if len(meta_nested) != t_count:
                raise ValueError(
                    f"camera_intrinsics T mismatch: {len(meta_nested)} vs {t_count}"
                )

            for t in range(t_count):
                if len(meta_nested[t]) != v_count:
                    raise ValueError(
                        f"camera_intrinsics V mismatch at t={t}: "
                        f"{len(meta_nested[t])} vs {v_count}"
                    )
                for v in range(v_count):
                    meta = meta_nested[t][v]
                    k_new[t, v] = update_intrinsics_for_resize_crop(
                        k_new[t, v],
                        meta["crop_left"],
                        meta["crop_top"],
                        meta["scale_x"],
                        meta["scale_y"],
                    )
            return k_src, k_new

        if k_new.ndim == 3:
            t_count = k_new.shape[0]
            if len(meta_nested) != t_count:
                raise ValueError(
                    f"camera_intrinsics T mismatch: {len(meta_nested)} vs {t_count}"
                )
            for t in range(t_count):
                meta = meta_nested[t][0]
                k_new[t] = update_intrinsics_for_resize_crop(
                    k_new[t],
                    meta["crop_left"],
                    meta["crop_top"],
                    meta["scale_x"],
                    meta["scale_y"],
                )
            return k_src, k_new

        raise ValueError(f"Unsupported camera_intrinsics shape: {tuple(k_new.shape)}")
    
    def __init__(
        self,
        base_dataset,
        transform_list,
        pop_list=None,
        enable_geometry_check=False,
        crop_use_intrinsics_center=True,
        crop_horizontal_anchor=0.5,
        crop_vertical_anchor=0.5,
        default_height=288,
        default_width=512,
        random_camera_order_augment=None,        append_bbox_dynamic_bev_mask=True,
        bbox_dynamic_bev_num_classes=10,
        bbox_dynamic_bev_map_key="hdmap_bev_images",
        bbox_dynamic_bev_bbox_key="bbox_token_corners",
        bbox_dynamic_bev_class_key="bbox_token_classes",
        bbox_dynamic_bev_mask_key="bbox_token_masks",
        bbox_dynamic_bev_x_min=-80.0,
        bbox_dynamic_bev_x_max=80.0,
        bbox_dynamic_bev_y_min=-80.0,
        bbox_dynamic_bev_y_max=80.0,

    ):
        self.base_dataset = base_dataset
        self.transform_list = transform_list
        self.pop_list = pop_list
        self.enable_geometry_check = enable_geometry_check
        self._geometry_check_done = False
        self.crop_use_intrinsics_center = crop_use_intrinsics_center
        self.crop_horizontal_anchor = crop_horizontal_anchor
        self.crop_vertical_anchor = crop_vertical_anchor
        self.default_height = default_height
        self.default_width = default_width

        self.append_bbox_dynamic_bev_mask = bool(append_bbox_dynamic_bev_mask)
        self.bbox_dynamic_bev_num_classes = int(bbox_dynamic_bev_num_classes)
        self.bbox_dynamic_bev_map_key = bbox_dynamic_bev_map_key
        self.bbox_dynamic_bev_bbox_key = bbox_dynamic_bev_bbox_key
        self.bbox_dynamic_bev_class_key = bbox_dynamic_bev_class_key
        self.bbox_dynamic_bev_mask_key = bbox_dynamic_bev_mask_key
        self.bbox_dynamic_bev_x_min = float(bbox_dynamic_bev_x_min)
        self.bbox_dynamic_bev_x_max = float(bbox_dynamic_bev_x_max)
        self.bbox_dynamic_bev_y_min = float(bbox_dynamic_bev_y_min)
        self.bbox_dynamic_bev_y_max = float(bbox_dynamic_bev_y_max)


        if random_camera_order_augment is None:
            self.random_camera_order_augment = None
        else:
            self.random_camera_order_augment = RandomCameraOrderAugment(
                **random_camera_order_augment
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        num_frame = None

        if isinstance(index, int):
            idx = index
            height = self.default_height
            width = self.default_width
            item = self.base_dataset[idx]


        elif isinstance(index, str):
            idx, num_frame, height, width = [int(val) for val in index.split("-")]

            item = self.base_dataset[idx]
            start_f = random.randint(0, len(item["images"]) - num_frame)

            new_item = {}

            for k, v in item.items():
                if k == "fps" or k == "crossview_mask":
                    new_item[k] = v
                    continue

                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    new_item[k] = v
                    continue

                if isinstance(v, np.ndarray) and v.ndim == 0:
                    new_item[k] = v
                    continue

                if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or isinstance(v, str):
                    new_item[k] = v
                    continue

                new_item[k] = v[start_f:start_f + num_frame]

            item = new_item

        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        t_count, v_count = DatasetAdapter.infer_tv_from_item(item)
        image_resize_crop_meta = None

        if "images" in item:
            nested_images = DatasetAdapter.ensure_nested_list_tv(
                item["images"], t_count=t_count, v_count=v_count, key="images"
            )

            intrinsics_src = item["camera_intrinsics"] if "camera_intrinsics" in item else None

            resized_nested_images, meta_nested = DatasetAdapter.crop_resize_nested_images_keep_tv(
                nested_images,
                height,
                width,
                intrinsics_src=intrinsics_src,
                crop_use_intrinsics_center=self.crop_use_intrinsics_center,
                crop_horizontal_anchor=self.crop_horizontal_anchor,
                crop_vertical_anchor=self.crop_vertical_anchor,
            )

            item["images"] = resized_nested_images
            image_resize_crop_meta = meta_nested

            (
                image_size_before_resize_crop,
                image_size_after_crop_before_resize,
                image_size_tensor,
            ) = DatasetAdapter.build_size_tensors_from_meta(meta_nested)

            item["image_size"] = image_size_tensor

            if "camera_intrinsics" in item:
                k_src, k_new = DatasetAdapter.update_intrinsics_nested(
                    item["camera_intrinsics"], meta_nested
                )
                #item["camera_intrinsics_before_resize_crop"] = k_src
                item["camera_intrinsics"] = k_new

                if self.enable_geometry_check and not self._geometry_check_done:
                    if k_src.ndim == 4:
                        validate_resize_crop_update(
                            k_src[0, 0],
                            k_new[0, 0],
                            meta_nested[0][0]
                        )
                    else:
                        validate_resize_crop_update(
                            k_src[0],
                            k_new[0],
                            meta_nested[0][0]
                        )

                    self._geometry_check_done = True
                    print(
                        "[DatasetAdapter] geometry check passed: "
                        "crop+resize updates fx/fy/cx/cy correctly."
                    )

        for i in self.transform_list:
            old_key = i["old_key"]
            new_key = i["new_key"]
            stack = i.get("stack", True)

            if old_key == "images":
                src = item[old_key]
                src = DatasetAdapter.ensure_nested_list_tv(
                    src, t_count=t_count, v_count=v_count, key=old_key
                )
                item[new_key] = DatasetAdapter.stack_nested_list_tv(
                    src, key=new_key
                ) if stack else src
                continue

            if old_key in ["3dbox_images", "hdmap_images"]:
                if image_resize_crop_meta is None:
                    raise ValueError(
                        f"{old_key} exists but images crop meta is missing"
                    )

                src = DatasetAdapter.ensure_nested_list_tv(
                    item[old_key], t_count=t_count, v_count=v_count, key=old_key
                )

                out_nested = []
                for t in range(t_count):
                    out_row = []
                    for v in range(v_count):
                        resized_img = apply_crop_resize_to_chw_tensor(
                            to_chw_tensor(src[t][v]),
                            image_resize_crop_meta[t][v],
                            mode="bilinear"
                        )
                        out_row.append(resized_img)
                    out_nested.append(out_row)

                item[new_key] = DatasetAdapter.stack_nested_list_tv(
                    out_nested, key=new_key
                ) if stack else out_nested
                continue

            if old_key == "proj_depth":
                if image_resize_crop_meta is None:
                    raise ValueError("proj_depth exists but images crop meta is missing")

                src = item["proj_depth"]
                if isinstance(src, list):
                    out = []
                    for x in src:
                        out.append(crop_resize_tv_tensor_nearest(x, image_resize_crop_meta))
                    item[new_key] = torch.stack(out) if stack else out
                else:
                    item[new_key] = crop_resize_tv_tensor_nearest(src, image_resize_crop_meta)
                continue

            if old_key == "proj_sem":
                if image_resize_crop_meta is None:
                    raise ValueError("proj_sem exists but images crop meta is missing")

                src = item["proj_sem"]
                if isinstance(src, list):
                    out = []
                    for x in src:
                        out.append(crop_resize_tv_tensor_nearest(x, image_resize_crop_meta))
                    item[new_key] = torch.stack(out) if stack else out
                else:
                    item[new_key] = crop_resize_tv_tensor_nearest(src, image_resize_crop_meta)
                continue

            if old_key == "proj_clr":
                if image_resize_crop_meta is None:
                    raise ValueError("proj_clr exists but images crop meta is missing")

                src = item["proj_clr"]
                if isinstance(src, list):
                    out = []
                    for c in src:
                        out.append(crop_resize_proj_clr_keep_invalid(c, image_resize_crop_meta))
                    item[new_key] = torch.stack(out) if stack else out
                else:
                    item[new_key] = crop_resize_proj_clr_keep_invalid(src, image_resize_crop_meta)
                continue

            if getattr(i["transform"], "is_temporal_transform", False):
                item[new_key] = DatasetAdapter.apply_temporal_transform(
                    i["transform"], item[old_key]
                )
            else:
                item[new_key] = DatasetAdapter.apply_transform(
                    i["transform"], item[old_key], stack
                )

        if self.random_camera_order_augment is not None:
            item = self.random_camera_order_augment(item, v_count)

        if self.pop_list is not None:
            for i in self.pop_list:
                if i in item:
                    item.pop(i)

        if self.append_bbox_dynamic_bev_mask:
            item = _append_dynamic_bev_masks_from_bbox_tokens(
                item,
                map_key=self.bbox_dynamic_bev_map_key,
                bbox_key=self.bbox_dynamic_bev_bbox_key,
                class_key=self.bbox_dynamic_bev_class_key,
                mask_key=self.bbox_dynamic_bev_mask_key,
                num_classes=self.bbox_dynamic_bev_num_classes,
                x_min=self.bbox_dynamic_bev_x_min,
                x_max=self.bbox_dynamic_bev_x_max,
                y_min=self.bbox_dynamic_bev_y_min,
                y_max=self.bbox_dynamic_bev_y_max,
            )

        return item


class ConcatMotionDataset(torch.utils.data.Dataset):
    """Concatenate multiple datasets with given ratio. It is implemented for
    the training recipe in Vista(https://arxiv.org/abs/2405.17398).

    Args:
        datasets: a list of datasets.
        ratios: a list of ratios for each dataset.
    """

    def __init__(self, datasets: list, ratios: list):
        self.datasets = datasets
        self.full_size = math.ceil(
            max([
                len(dataset) / ratio
                for dataset, ratio in zip(datasets, ratios)
            ]))
        self.ranges = torch.cumsum(
            torch.tensor([int(ratio * self.full_size) for ratio in ratios]),
            dim=0)

    def __len__(self):
        return self.full_size

    def __getitem__(self, index):
        for i, range in enumerate(self.ranges):
            if index < range:
                return self.datasets[i][index % len(self.datasets[i])]

        raise Exception(f"invalid index {index}")


class CollateFnIgnoring():
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, item_list: list):
        ignored = [
            (key, [item.pop(key) for item in item_list])
            for key in self.keys
        ]
        result = {}
        keys = item_list[0].keys()

        for k in keys:
            vlist = [item[k] for item in item_list]

            try:
                result[k] = torch.utils.data.default_collate(vlist)
            except Exception as e:
                print("\n========== COLLATE ERROR ==========")
                print("key:", k)
                for i, v in enumerate(vlist):
                    if isinstance(v, torch.Tensor):
                        print(f"[{i}] shape={tuple(v.shape)}, dtype={v.dtype}, contig={v.is_contiguous()}")
                    else:
                        print(f"[{i}] type={type(v)}")
                raise 
        for key, value in ignored:
            result[key] = value

        return result


def find_nearest(list: list, value, return_item=False):
    i = bisect.bisect_left(list, value)
    if i == 0:
        pass
    elif i >= len(list):
        i = len(list) - 1
    else:
        diff_0 = value - list[i - 1]
        diff_1 = list[i] - value
        if i > 0 and diff_0 <= diff_1:
            i -= 1

    return list[i] if return_item else i



def find_nearest_2hz(timestamps, sdl, value, *, window=2, max_extra_us=0):
    i = bisect.bisect_left(timestamps, value)
    cand = [j for j in range(i - window, i + window + 1) if 0 <= j < len(timestamps)]

    best = min(
        cand,
        key=lambda j: (abs(timestamps[j] - value), 0 if timestamps[j] <= value else 1)
    )
    best_diff = abs(timestamps[best] - value)

    pref = [j for j in cand if len(sdl[j].get("token", "")) == 32]
    if pref:
        best_pref = min(
            pref,
            key=lambda j: (abs(timestamps[j] - value), 0 if timestamps[j] <= value else 1)
        )
        if abs(timestamps[best_pref] - value) <= best_diff + max_extra_us:
            return best_pref

    return best


def get_transform(rotation: list, translation: list, output_type: str = "np"):
    result = np.eye(4)
    result[:3, :3] = transforms3d.quaternions.quat2mat(rotation)
    result[:3, 3] = np.array(translation)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the get_transform()")


def make_intrinsic_matrix(fx_fy: list, cx_cy: list, output_type: str = "np"):
    result = np.diag(fx_fy + [1])
    result[:2, 2] = np.array(cx_cy)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the make_intrinsic_matrix()")


def project_line(
    a: np.array, b: np.array, near_z: float = 0.05, far_z: float = 512.0
):
    if (a[2] < near_z and b[2] < near_z) or (a[2] > far_z and b[2] > far_z):
        return None

    ca = a
    cb = b
    if a[2] >= near_z and b[2] < near_z:
        r = (near_z - b[2]) / (a[2] - b[2])
        cb = a * r + b * (1 - r)
    elif a[2] < near_z and b[2] >= near_z:
        r = (b[2] - near_z) / (b[2] - a[2])
        ca = a * r + b * (1 - r)

    if a[2] > far_z and b[2] <= far_z:
        r = (far_z - b[2]) / (a[2] - b[2])
        ca = a * r + b * (1 - r)
    elif a[2] <= far_z and b[2] > far_z:
        r = (b[2] - far_z) / (b[2] - a[2])
        cb = a * r + b * (1 - r)

    pa = ca[:2] / ca[2]
    pb = cb[:2] / cb[2]
    return (pa[0], pa[1], pb[0], pb[1])


def draw_edges_to_image(
    draw: ImageDraw.ImageDraw, points: np.array, edge_indices: list,
    pen_color: tuple, pen_width: int
):
    for a, b in edge_indices:
        xy = project_line(points[:, a], points[:, b])
        if xy is not None:
            draw.line(xy, fill=pen_color, width=pen_width)


def draw_3dbox_image(
    draw: ImageDraw.ImageDraw, view_transform: np.array,
    list_annotation_func, get_world_transform_func, get_annotation_label,
    pen_width: int, color_table: dict, corner_templates: list,
    edge_indices: list
):
    corner_templates_np = np.array(corner_templates).transpose()
    for sa in list_annotation_func():
        sa_label = get_annotation_label(sa)
        if sa_label in color_table:
            pen_color = tuple(color_table[sa_label])
            world_transform = get_world_transform_func(sa)
            p = view_transform @ world_transform @ corner_templates_np
            draw_edges_to_image(draw, p, edge_indices, pen_color, pen_width)


def align_image_description_crossview(caption_list: list, settings: dict):
    if "align_keys" in settings:
        for k in settings["align_keys"]:
            value_count = {}
            for i in caption_list:
                if i[k] not in value_count:
                    value_count[i[k]] = 0

                value_count[i[k]] += 1

            dominated_value = max(value_count, key=value_count.get)
            for i in caption_list:
                i[k] = dominated_value

    return caption_list


def make_image_description_string(
    caption_dict: dict, settings: dict, random_state: np.random.RandomState
):
    """Make the image description string from the caption dict with given
    settings.

    Args:
        caption_dict (dict): The caption dict contains textual descriptions of
            various categories such as time, environment, and more.
        settings (dict): The dict of settings to decide how to compose the
            final image descrption string.
            * selected_keys (list if exist): The value in the caption dict is
                used when its key in the list of selected keys.
            * reorder_keys (bool if exist): If set to True, the elements used
                to compose text descriptions in caption_dict will be shuffled.
            * drop_rates (dict if exist): The entries in the dict are the
                probabilities of the corresponding key elements in the
                caption_dict being dropped.
        random_state (np.random.RandomState): The random state for reproducible
            randomness.
    """
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]
    selected_keys = settings.get(
        "selected_keys", default_image_description_keys)

    if "reorder_keys" in settings and settings["reorder_keys"]:
        new_order = random_state.permutation(len(selected_keys))
        selected_keys = [selected_keys[i] for i in new_order]

    if "drop_rates" in settings:
        drop = {
            k: random_state.rand() <= v
            for k, v in settings["drop_rates"].items()
        }
        selected_keys = [
            i for i in selected_keys
            if i not in drop or not drop[i]
        ]

    result = ". ".join([caption_dict[j] for j in selected_keys])
    # result = ". ".join([(caption_dict[j].split(",", 1)[0] if j == "weather" else caption_dict[j])
    #                     for j in selected_keys])
    return result


def add_stub_key_data(stub_key_data_dict, result: dict):
    """Add the stub key and data into the result dict.

    Args:
        stub_key_data_dict (dict or None): If set, the items are used to create
            stub item for the result dict. The value of this dict should be
            tuple. If the first item of the value tuple is "tensor", a tensor
            filled with the 3rd item in the shape of 2nd item is created as
            the stub data. Otherwise the 2nd item of the value tuple is
            deserialized as the stub data.
        result (dict): The result dict to insert created stub items.
    """

    if stub_key_data_dict is None:
        return

    for key, data in stub_key_data_dict.items():
        if key not in result.keys():
            if data[0] == "tensor":
                shape, value = data[1:]
                result[key] = value * torch.ones(shape)
            else:
                result[key] = data[1]



# -------------------------------- proj ----------------------------------





# ---------- png io ----------

def _safe_save_png(pil_img, p: str):
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pil_img.save(tmp, format="PNG", compress_level=1, optimize=False)
    os.replace(tmp, p)

def _try_open_png(p: str):
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


# ---------- lock ----------

def _acquire_lock(lock: str, timeout=30, stale=120, sleep=0.02):
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale:
                    os.remove(lock)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Lock timeout: {lock}")
            time.sleep(sleep)

def _release_lock(lock: str):
    try:
        os.remove(lock)
    except FileNotFoundError:
        pass


# ---------- u16 png io ----------

def _safe_save_u16_png(u16_img: np.ndarray, p: str):
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(u16_img.astype(np.uint16), mode="I;16").save(
        tmp, format="PNG", compress_level=1
    )
    os.replace(tmp, p)

def _try_open_u16_png(p: str):
    try:
        with Image.open(p) as im:
            im.load()
            if im.mode != "I;16":
                im = im.convert("I;16")
            return np.array(im, dtype=np.uint16)
    except Exception:
        return None


# ---------- cache subdir (for method wrapper) ----------

def ensure_cache_subdir(cache_root: str, subdir: str):
    root = os.path.abspath(cache_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"cache_root not found (won't create parent): {root}")

    sub = os.path.abspath(os.path.join(root, subdir))
    if os.path.dirname(sub) != root:
        raise ValueError(f"refuse to create nested cache dir: {sub}")

    os.makedirs(sub, exist_ok=True)
    return sub


# ---------- depth binning + vis ----------

def depth_to_logbins_u16(depth: np.ndarray, *, invalid=-300.0, n_bins=256, far_m=25.0, gamma=1.0) -> np.ndarray:
    assert 1 <= n_bins <= 65535
    out = np.zeros(depth.shape, np.uint16)

    m = depth != invalid
    if not np.any(m):
        return out

    far_m = float(far_m)
    near_m = far_m * 0.6
    near_frac = 0.85

    nb1 = max(1, int(round(n_bins * near_frac)))
    nb1 = min(nb1, n_bins - 1)
    nb2 = n_bins - nb1

    idx = np.flatnonzero(m)
    d = np.clip(depth[m].astype(np.float32), 0.0, far_m)

    m1 = d <= near_m
    if np.any(m1):
        x1 = np.log1p(d[m1]) / (np.log1p(near_m) + 1e-6)
        x1 = np.clip(x1, 0.0, 1.0)
        if gamma != 1.0:
            x1 = x1 ** float(gamma)
        out.reshape(-1)[idx[m1]] = (x1 * (nb1 - 1)).astype(np.uint16) + 1

    if nb2 > 0:
        m2 = ~m1
        if np.any(m2):
            dd = np.clip(d[m2] - near_m, 0.0, far_m - near_m)
            x2 = np.log1p(dd) / (np.log1p(far_m - near_m) + 1e-6)
            x2 = np.clip(x2, 0.0, 1.0)
            out.reshape(-1)[idx[m2]] = (x2 * (nb2 - 1)).astype(np.uint16) + 1 + nb1

    return out

def depth_to_linbins_u16(depth: np.ndarray, *, invalid=-300.0, n_bins=256, far_m=25.0) -> np.ndarray:
    assert 1 <= n_bins <= 65535
    out = np.zeros(depth.shape, np.uint16)

    m = depth != invalid
    if not np.any(m):
        return out

    d = np.clip(depth[m].astype(np.float32), 0.0, float(far_m))
    x = d / (float(far_m) + 1e-6)
    x = np.clip(x, 0.0, 1.0)

    out[m] = (x * (n_bins - 1)).astype(np.int32).astype(np.uint16) + 1
    return out

def visualize_bins_u16(bins_u16: np.ndarray, *, n_bins=256, invalid_bin=0, colormap=cv2.COLORMAP_TURBO):
    b = bins_u16.astype(np.int32)
    valid = b != int(invalid_bin)
    gray = np.zeros(b.shape, np.uint8)
    if np.any(valid):
        x = (b - 1) / max(1, (int(n_bins) - 1))
        x = np.clip(x, 0.0, 1.0)
        gray[valid] = (x[valid] * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(gray, colormap)
    vis[~valid] = (0, 0, 0)
    return vis


# ---------- downsample ----------

def downsample_depth_blockwise(depth_img, target_size, invalid=-300.0):
    m = (depth_img != invalid)
    d = cv2.resize(depth_img.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    m2 = cv2.resize(m.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.full(d.shape, invalid, np.float32)
    out[m2] = d[m2]
    return out

def downsample_clr_blockwise(img, target_size, blur_ksize=1):
    out = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    if blur_ksize and blur_ksize > 1:
        out = cv2.blur(out, (blur_ksize, blur_ksize))
    return out.astype(np.uint8)

# def downsample_clr_blockwise(img_u8, target_size, ks=5, beta=0.15):
#     """
#     img_u8: (H,W,3) uint8
#     target_size: (H2,W2)
#     ks: 全局平滑核大小(奇数)，越大越平滑
#     beta: 稀疏区稳定项，越大越不跳/越暗
#     """
#     H2, W2 = int(target_size[0]), int(target_size[1])

#     img = img_u8.astype(np.float32)
#     valid = (img_u8.sum(axis=2) > 0).astype(np.float32)  # (H,W)

#     num = cv2.resize(img * valid[..., None], (W2, H2), interpolation=cv2.INTER_AREA)
#     den = cv2.resize(valid, (W2, H2), interpolation=cv2.INTER_AREA)

#     if ks and ks > 1:
#         num = cv2.boxFilter(num, ddepth=-1, ksize=(ks, ks), normalize=True)
#         den = cv2.boxFilter(den, ddepth=-1, ksize=(ks, ks), normalize=True)
        
#     out = num / (den[..., None] + float(beta))

#     return np.clip(out, 0, 255).astype(np.uint8)