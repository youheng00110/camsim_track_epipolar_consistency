import os, json, time, pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm
import dwm.common
import dwm.datasets.common



from PIL import Image, ImageDraw, ImageFile
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString, Polygon

from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_sensor_token_map_name_from_db
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.common.actor_state.state_representation import Point2D

from dwm.datasets.common import (
    _safe_save_png, _try_open_png,
    _safe_save_u16_png, _try_open_u16_png,
    _acquire_lock, _release_lock,
    ensure_cache_subdir,
    depth_to_logbins_u16, depth_to_linbins_u16,
    visualize_bins_u16,
    downsample_depth_blockwise, downsample_clr_blockwise,
)

cv2.setNumThreads(0)

def vis_from_bins_cache(bins_u16_cache: np.ndarray, n_bins: int) -> Image.Image:
    """
    bins_u16_cache: uint16, value in [0..n_bins], invalid_bin=0
    """
    bins_u16_cache = np.asarray(bins_u16_cache, np.uint16)
    vis_bgr = visualize_bins_u16(
        bins_u16_cache,
        n_bins=int(n_bins),
        invalid_bin=0,
        colormap=cv2.COLORMAP_TURBO,
    )
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(vis_rgb, "RGB")

# -------- splat expand --------
def _expand_uv(u, v, z, r, H, W):
    if r <= 0:
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u[m], v[m], z[m]

    d = np.arange(-r, r + 1, dtype=np.int32)
    dx, dy = np.meshgrid(d, d)
    dx = dx.reshape(-1); dy = dy.reshape(-1)

    uu = (u[:, None] + dx[None, :]).reshape(-1)
    vv = (v[:, None] + dy[None, :]).reshape(-1)
    zz = np.repeat(z, dx.size)

    m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    return uu[m], vv[m], zz[m]

# -------- core: project depth + sem(one-hot) + clr(rgb) --------

def project_depth_only(
    pts_xyz: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, invalid_depth=-300.0, splat=None
):
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    if pts_xyz is None or pts_xyz.shape[0] == 0:
        return np.full((H, W), float(invalid_depth), np.float32)

    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]

    u = (p[:, 0] / z).astype(np.int32)
    v = (p[:, 1] / z).astype(np.int32)

    depth = np.full((H, W), np.inf, np.float32)

    z2, u2, v2 = z.copy(), u.copy(), v.copy()
    for zmax, r in splat:
        mm = z2 <= zmax
        if not np.any(mm):
            continue
        uu, vv, zz = _expand_uv(u2[mm], v2[mm], z2[mm], r, H, W)
        np.minimum.at(depth.reshape(-1), vv * W + uu, zz)
        z2, u2, v2 = z2[~mm], u2[~mm], v2[~mm]
        if z2.size == 0:
            break

    depth[~np.isfinite(depth)] = float(invalid_depth)
    return depth


def project_sem_only(
    pts_xyz: np.ndarray,
    pts_sem: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, splat=None, n_actor_classes=3
):
    """
    returns sem: (H,W,K) uint8 {0,1}
    """
    H, W = int(ori_hw[0]), int(ori_hw[1])
    K = int(n_actor_classes)
    if splat is None:
        splat = [(1e9, 0)]

    sem = np.zeros((H, W, K), np.uint8)
    if pts_xyz is None or pts_xyz.shape[0] == 0 or pts_sem is None or pts_sem.shape[0] == 0:
        return sem

    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]
    sid = pts_sem[m].astype(np.int32)

    # keep only 1..K
    good = (sid >= 1) & (sid <= K)
    if not np.any(good):
        return sem

    u = (p[:, 0] / z).astype(np.int32)[good]
    v = (p[:, 1] / z).astype(np.int32)[good]
    zc = z[good].astype(np.float32)
    sid = sid[good].astype(np.int32)

    flat, zz, sid2 = _collect_splat(u, v, zc, sid, H, W, splat)
    if flat.size == 0:
        return sem

    uf, _, sf = _zbuffer_first(flat, zz, sid2.astype(np.int32))
    sf = sf - 1
    ok = (sf >= 0) & (sf < K)
    sem.reshape(-1, K)[uf[ok], sf[ok]] = 1
    return sem


def project_clr_only(
    clr_xyz: np.ndarray,
    clr_rgb: np.ndarray,
    image_from_lidar: np.ndarray,
    ori_hw,
    *, splat=None
):
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    clr = np.zeros((H, W, 3), np.uint8)
    if clr_xyz is None or clr_xyz.shape[0] == 0:
        return clr

    xyz1c = np.concatenate([clr_xyz.astype(np.float32), np.ones((clr_xyz.shape[0], 1), np.float32)], 1)
    pc = xyz1c @ image_from_lidar.T
    zc = pc[:, 2]
    mc = zc > 1e-5
    pc, zc, rgb = pc[mc], zc[mc], clr_rgb[mc].astype(np.uint8)

    uc = (pc[:, 0] / zc).astype(np.int32)
    vc = (pc[:, 1] / zc).astype(np.int32)

    flat, zz, rr = _collect_splat(uc, vc, zc.astype(np.float32), rgb, H, W, splat)
    if flat.size == 0:
        return clr

    uf, _, rgbf = _zbuffer_first(flat, zz, rr)
    clr.reshape(-1, 3)[uf] = rgbf
    return clr



def project_depth_sem_clr(
    pts_xyz: np.ndarray, pts_sem: np.ndarray,
    clr_xyz: np.ndarray, clr_rgb: np.ndarray,
    image_from_lidar: np.ndarray, ori_hw,
    *, invalid_depth=-300.0, splat=None, n_actor_classes=3
):
    """
    pts_xyz: (N,3) lidar
    pts_sem: (N,) int, 0=bg/ignore, 1..K=classes
    clr_xyz: (M,3) lidar
    clr_rgb: (M,3) uint8
    return:
      depth: (H,W) float32
      sem  : (H,W,K) uint8(0/1)
      clr  : (H,W,3) uint8
    """
    H, W = int(ori_hw[0]), int(ori_hw[1])
    if splat is None:
        splat = [(1e9, 0)]

    # ----- depth -----
    xyz1 = np.concatenate([pts_xyz.astype(np.float32), np.ones((pts_xyz.shape[0], 1), np.float32)], 1)
    p = xyz1 @ image_from_lidar.T
    z = p[:, 2]
    m = z > 1e-5
    p, z = p[m], z[m]
    sem_id = pts_sem[m].astype(np.int32) if pts_sem is not None else None

    u = (p[:, 0] / z).astype(np.int32)
    v = (p[:, 1] / z).astype(np.int32)

    depth = np.full((H, W), np.inf, np.float32)
    # depth 用 min-at（快）
    z2, u2, v2 = z.copy(), u.copy(), v.copy()
    for zmax, r in splat:
        mm = z2 <= zmax
        if not np.any(mm):
            continue
        uu, vv, zz = _expand_uv(u2[mm], v2[mm], z2[mm], r, H, W)
        np.minimum.at(depth.reshape(-1), vv * W + uu, zz)
        z2, u2, v2 = z2[~mm], u2[~mm], v2[~mm]
        if z2.size == 0:
            break
    depth[~np.isfinite(depth)] = float(invalid_depth)

    # ----- sem (z-buffer) -----
    
    sem = np.zeros((H, W, int(n_actor_classes)), np.uint8)
    if sem_id is not None and sem_id.size > 0:
        good = (sem_id >= 1) & (sem_id <= n_actor_classes)
        zc, uc, vc, sid = z[good], u[good], v[good], sem_id[good].astype(np.int32)
        flat, zz, sid2 = _collect_splat(uc, vc, zc, sid, H, W, splat)
        if flat.size > 0:
            uf, _, sf = _zbuffer_first(flat, zz, sid2.astype(np.int32))
            sf = sf - 1
            ok = (sf >= 0) & (sf < n_actor_classes)
            sem.reshape(-1, n_actor_classes)[uf[ok], sf[ok]] = 1
        
    # ----- clr (z-buffer) -----
    clr = np.zeros((H, W, 3), np.uint8)
    if clr_xyz is not None and clr_xyz.shape[0] > 0:
        xyz1c = np.concatenate([clr_xyz.astype(np.float32), np.ones((clr_xyz.shape[0], 1), np.float32)], 1)
        pc = xyz1c @ image_from_lidar.T
        zc = pc[:, 2]
        mc = zc > 1e-5
        pc, zc, rgb = pc[mc], zc[mc], clr_rgb[mc].astype(np.uint8)
        uc = (pc[:, 0] / zc).astype(np.int32)
        vc = (pc[:, 1] / zc).astype(np.int32)

        flat, zz, rr = _collect_splat(uc, vc, zc, rgb, H, W, splat)
        if flat.size > 0:
            uf, _, rgbf = _zbuffer_first(flat, zz, rr)
            clr.reshape(-1, 3)[uf] = rgbf

    return depth, sem, clr

############ tools ##################


def _zbuffer_first(flat: np.ndarray, z: np.ndarray, payload: np.ndarray | None):
    """
    flat: (M,) int32  像素扁平索引 v*W+u
    z:    (M,) float32 深度
    payload: (M, C) 或 (M,) 可选，随最近点一起取
    return: uniq_flat, uniq_z, uniq_payload
    """
    order = np.lexsort((z, flat))          # flat 升序；同 flat 内 z 升序
    flat2 = flat[order]
    z2 = z[order]
    first = np.r_[True, flat2[1:] != flat2[:-1]]
    uf = flat2[first]
    uz = z2[first]
    if payload is None:
        return uf, uz, None
    p2 = payload[order]
    return uf, uz, p2[first]


def _collect_splat(u, v, z, payload, H, W, splat):
    """
    splat: list[(zmax, r)]
    输出所有扩张后的 flat/z/payload（不会做 zbuffer）
    """
    flat_all, z_all, p_all = [], [], []
    u = u.astype(np.int32); v = v.astype(np.int32); z = z.astype(np.float32)

    for zmax, r in splat:
        mm = z <= zmax
        if not np.any(mm):
            continue

        uu = u[mm]; vv = v[mm]; zz = z[mm]
        pp = payload[mm] if payload is not None else None

        if r > 0:
            d = np.arange(-r, r + 1, dtype=np.int32)
            dx, dy = np.meshgrid(d, d)
            dx = dx.reshape(-1); dy = dy.reshape(-1)
            k2 = dx.size

            uu = (uu[:, None] + dx[None, :]).reshape(-1)
            vv = (vv[:, None] + dy[None, :]).reshape(-1)
            zz = np.repeat(zz, k2)
            if pp is not None:
                pp = np.repeat(pp, k2, axis=0) if pp.ndim == 2 else np.repeat(pp, k2)

        m_in = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu = uu[m_in]; vv = vv[m_in]; zz = zz[m_in]
        flat_all.append(vv * W + uu)
        z_all.append(zz)
        if pp is not None:
            p_all.append(pp[m_in])

        # 处理剩余远处点
        keep = ~mm
        u = u[keep]; v = v[keep]; z = z[keep]
        if payload is not None:
            payload = payload[keep]
        if z.size == 0:
            break

    flat = np.concatenate(flat_all, 0) if flat_all else np.zeros((0,), np.int32)
    z = np.concatenate(z_all, 0) if z_all else np.zeros((0,), np.float32)
    p = np.concatenate(p_all, 0) if (payload is not None and p_all) else None
    return flat, z, p

def layer_to_key(layer_name):
    if layer_name in (SemanticMapLayer.CARPARK_AREA, SemanticMapLayer.WALKWAYS, SemanticMapLayer.INTERSECTION):
        return "drivable_area"
    if layer_name == SemanticMapLayer.LANE:
        return "lane"
    if layer_name == SemanticMapLayer.CROSSWALK:
        return "ped_crossing"
    if layer_name == SemanticMapLayer.STOP_LINE:
        return "lane"  
    return None

def _find_nearest(sorted_arr, x):
    i = int(np.searchsorted(sorted_arr, x))
    if i <= 0:
        return 0
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    return i - 1 if abs(sorted_arr[i - 1] - x) <= abs(sorted_arr[i] - x) else i


def make_image_description_string(caption_dict: dict, settings: dict, random_state=None):
    settings = settings or {}
    rng = random_state if random_state is not None else np.random.default_rng()

    keys = ["time", "weather"]
    if settings.get("reorder_keys", False):
        keys = [keys[i] for i in rng.permutation(len(keys))]

    dr = settings.get("drop_rates")
    if dr:
        keys = [k for k in keys if not (k in dr and rng.random() <= dr[k])]

    return ". ".join(str(caption_dict[k]) for k in keys if k in caption_dict)

_BOX_EDGES = [
    # bottom face (0-1-2-3)
    (0, 1), (1, 2), (2, 3), (3, 0),
    # verticals
    (0, 4), (1, 5), (2, 6), (3, 7),
    # top face (4-5-6-7)
    (4, 5), (5, 6), (6, 7), (7, 4),
    # extra "direction" lines (keep your style)
    (0, 5), (0, 7)
]

def _yaw_to_Rz(yaw):
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _box_corners_lidar_xyz(box7, z_is_center=False):
    # box7: [x,y,z, dx,dy,dz, yaw] in LiDAR frame
    x, y, z, dx, dy, dz, yaw = box7[:7].astype(np.float32)
    xs = np.array([ dx/2,  dx/2, -dx/2, -dx/2,  dx/2,  dx/2, -dx/2, -dx/2], np.float32)
    ys = np.array([ dy/2, -dy/2, -dy/2,  dy/2,  dy/2, -dy/2, -dy/2,  dy/2], np.float32)
    if z_is_center:
        zs = np.array([ -dz/2, -dz/2, -dz/2, -dz/2,  dz/2,  dz/2,  dz/2,  dz/2], np.float32)
    else:
        zs = np.array([ 0, 0, 0, 0, dz, dz, dz, dz], np.float32)

    R = _yaw_to_Rz(yaw)
    pts = np.stack([xs, ys, zs], axis=0)    # (3,8)
    pts = (R @ pts).T                       # (8,3)
    pts += np.array([x, y, z], np.float32)
    return pts


def _project_pts(pts_xyz, T_4x4):
    pts_h = np.concatenate([pts_xyz.astype(np.float32),
                            np.ones((pts_xyz.shape[0], 1), np.float32)], axis=1)
    q = pts_h @ T_4x4.T
    z = np.clip(q[:, 2], 1e-5, 1e9)
    u = q[:, 0] / z
    v = q[:, 1] / z
    return np.stack([u, v, q[:, 2]], axis=1)


def _lidar2image_from_caminfo(cam_info, K3):
    # cam_info: contains sensor2lidar_rotation/translation
    R_c2l = np.asarray(cam_info["sensor2lidar_rotation"], np.float32).reshape(3, 3)
    t_c2l = np.asarray(cam_info["sensor2lidar_translation"], np.float32).reshape(3)

    # invert: lidar -> cam
    R_l2c = R_c2l.T
    t_l2c = -R_l2c @ t_c2l

    Rt = np.concatenate([R_l2c, t_l2c[:, None]], axis=1)      # 3x4
    P  = np.asarray(K3, np.float32).reshape(3, 3) @ Rt        # 3x4

    l2i = np.eye(4, dtype=np.float32)
    l2i[:3, :4] = P
    return l2i


def _infer_TV(result: dict):
    v = result.get("vae_images", None)
    if torch.is_tensor(v) and v.ndim >= 2:
        return int(v.shape[0]), int(v.shape[1])
    if isinstance(v, (list, tuple)):
        T = len(v)
        V = max((len(x) for x in v), default=0)
        return T, V

    c = result.get("camera_intrinsics", None)
    if torch.is_tensor(c) and c.ndim >= 2:
        return int(c.shape[0]), int(c.shape[1])

    p = result.get("pts", None)
    if torch.is_tensor(p) and p.ndim >= 1:
        return int(p.shape[0]), 0

    return 1, 0

def _se3_from_qt(q, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(q).rotation_matrix.astype(np.float32)
    T[:3, 3] = np.asarray(t, np.float32).reshape(3)
    return T

def _se3_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

############ tools ##################

class NuPlanDataset(torch.utils.data.Dataset):
    """
    - 3dbox_images：用 gt_line (7DoF boxes) 投影画线框（颜色严格对齐 MotionDataset）
    - hdmap_images：用 nuplan map 投影（drivable_area/lane/ped_crossing），并按 MotionDataset 颜色表画“轮廓线”
    """

    default_3dbox_color_table = {
        "ped": (255, 0, 0),
        "bike": (128, 255, 0),
        "car": (0, 0, 255),
    }
    default_hdmap_color_table = {
        "drivable_area": (0, 0, 255),
        "lane": (0, 255, 0),
        "ped_crossing": (255, 0, 0)
    }


    def __init__(
        self,
        pkl_path,# 必须传入数据集名称
        sensor_root,
        cache_root,
        dataset_root,
        map_root,
        map_version='nuplan-maps-v1.0',
        sequence_length=1,
        fps_stride_tuples=(),
        sensor_channels=('CAM_L1','CAM_L0','CAM_F0','CAM_R0','CAM_R1','CAM_R2','CAM_B0','CAM_L2'),
        scene_key="db_name",
        timestamp_key="timestamp",
        token_key="token",
        enable_synchronization_check=True,
        max_time_error_ratio=0.5,
        max_time_error_us=None,
        enable_sample_data=False,
        _3dbox_pen_width=8,
        hdmap_pen_width=8,
        hdmap_patch_radius=100.0,
        near_plane=1e-8,
        min_polygon_area=2000,
        cam_key="cam",
        ego2global_key="ego2global",
        stub_key_data_dict=None,
        image_description_settings=None,
        projected_pc_settings=None, 
        balanced_json_path=None,
    ):
        self.sensor_root = sensor_root
        self.cache_root = cache_root
        self.dataset_root = dataset_root
        self.map_factory = NuPlanMapFactory(get_maps_db(map_root=map_root, map_version=map_version))

        self.sequence_length = int(sequence_length)
        self.fps_stride_tuples = list(fps_stride_tuples)
        self.sensor_channels = list(sensor_channels)
        self.scene_key = scene_key
        self.timestamp_key = timestamp_key
        self.token_key = token_key
        self.enable_synchronization_check = bool(enable_synchronization_check)
        self.max_time_error_ratio = float(max_time_error_ratio)
        self.max_time_error_us = None if max_time_error_us is None else int(max_time_error_us)
        self.enable_sample_data = bool(enable_sample_data)

        self._3dbox_pen_width = int(_3dbox_pen_width)
        self.hdmap_pen_width = int(hdmap_pen_width)
        self.hdmap_patch_radius = float(hdmap_patch_radius)
        self.near_plane = float(near_plane)
        self.min_polygon_area = float(min_polygon_area)

        
        self.cam_key = cam_key
        self.ego2global_key = ego2global_key
        self.stub_key_data_dict = stub_key_data_dict
        
        # --- [MOD] 注释掉 Projection 相关初始化 ---
        # self.projected_pc_settings = projected_pc_settings
        # self.do_cache = bool(projected_pc_settings and projected_pc_settings.get("do_cache", True))
        # self._actor_root = ... 

        # nuplan layers
        self.polygon_layer_names = [
            SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS, SemanticMapLayer.CARPARK_AREA,
        ]
        self.line_layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]

        
        self._map_cache = {}
        self._token2map = {}
        self._hdgeom_cache = {}
        self._hdgeom_cache_max = 4096
        
        if image_description_settings:
            with open(image_description_settings.get('path'), "r", encoding="utf-8") as f:
                self.text_anno = json.load(f)
            self.text_settings = image_description_settings

        
        # ###
        # 读pickle生成原始window
        # #####
        
        # #####
        # 读 pickle 生成原始 window
        # #####
        infos = self._load_infos(pkl_path)
        
        # 1. 建立全局索引映射 (非常重要！)
        # 假设你的 nuplan_info.py 脚本在制作时没存 sample_idx，
        # 我们根据 step=2 手动推算。如果存了，直接取 info['sample_idx']
        self.scenes = {}
        self.scene_ts = {}
        self.scene_global_idxs = {} # 新增：存储每一帧在原始 DB 里的全局索引

        for info in infos:
            scene_id = info.get("db_name") or info.get("scene_name")
            if scene_id:
                if scene_id not in self.scenes:
                    self.scenes[scene_id] = []
                    self.scene_ts[scene_id] = []
                    self.scene_global_idxs[scene_id] = []
                
                self.scenes[scene_id].append(info)
                self.scene_ts[scene_id].append(info["timestamp"])
                
                # 获取/推算全局索引
                # 方案 A: 如果 pkl 已经带了 idx (推荐在制作 pkl 脚本里加上)
                g_idx = info.get("sample_idx") 
                # 方案 B: 如果没有，根据当前场景已有的帧数 * step 推算 (前提是顺序没断)
                if g_idx is None:
                    g_idx = (len(self.scenes[scene_id]) - 1) * 2 
                
                self.scene_global_idxs[scene_id].append(g_idx)

        # 转换为 numpy 方便切片
        for s in self.scene_ts:
            self.scene_ts[s] = np.array(self.scene_ts[s])
            self.scene_global_idxs[s] = np.array(self.scene_global_idxs[s])

        raw_items = self._build_items()
        print("DEBUG total raw windows:", len(raw_items))
        if len(raw_items) > 0:
            print("DEBUG example window:", raw_items[0])
        else:
            print("DEBUG: raw_items is EMPTY")
        
        #####
        ###对已有window进行匹配
        ####
        
        if balanced_json_path:
            with open(balanced_json_path) as f:
                raw_intervals = json.load(f)

            interval_map = {}
            for e in raw_intervals:
                interval_map.setdefault(e["seq_id"], []).append(e)

            # 注意：如果 Window 是 4s (10fps)，Interval 也是 4s (20fps)
            # 那么 Window 的全局长度是 80 (因为跨越了 80 个原始索引)
            # 而 Interval 的全局长度可能只有 40。
            # 所以 OVERLAP_RATIO 设为 0.5 是合理的，或者降到 0.4
            OVERLAP_RATIO = 0.5 
            matched_items = []

            for item in tqdm(raw_items, desc="Interval Matching"):
                scene_id = item["scene"]
                if scene_id not in interval_map:
                    continue

                local_idxs = item["indices"]
                global_idxs = self.scene_global_idxs[scene_id]

                # --- 核心：获取该 Window 的全局起止 Index ---
                win_g_start = global_idxs[local_idxs[0]]
                win_g_end   = global_idxs[local_idxs[-1]]
                # 物理跨度（在原始 20Hz 下的帧数）
                win_g_len   = win_g_end - win_g_start + 1

                for interval in interval_map[scene_id]:
                    int_g_start = interval["start_idx"]
                    int_g_end   = interval["end_idx"]

                    # 计算全局索引层面的重合
                    overlap = min(win_g_end, int_g_end) - max(win_g_start, int_g_start) + 1
                    
                    if overlap > 0:
                        ratio = overlap / win_g_len
                        
                        if ratio >= OVERLAP_RATIO:
                            item["angle"] = interval["angle"]
                            item["dist"]  = interval["dist"]
                            # 注入全局索引信息方便 Debug
                            item["g_idx_range"] = (win_g_start, win_g_end)
                            
                            matched_items.append(item)
                            
                            # Debug 输出第一个匹配成功的
                            if len(matched_items) == 1:
                                print(f"\n[FIRST MATCH DEBUG]")
                                print(f"Scene: {scene_id}")
                                print(f"Window Global: {win_g_start} -> {win_g_end} (Len: {win_g_len})")
                                print(f"Interval Global: {int_g_start} -> {int_g_end}")
                                print(f"Overlap: {overlap} | Ratio: {ratio:.4f}")
                            break

            self.items = matched_items
            print(f"\n[NuPlan Matching] Summary:")
            print(f"Total Raw Windows: {len(raw_items)}")
            print(f"Matched Windows: {len(self.items)}")
            print("DEBUG intervals:", len(raw_intervals))
            print("DEBUG interval example:", raw_intervals[0])
            print(
                f"[NuPlan Matching] Total windows: {len(raw_items)}, "
                f"Matched: {len(self.items)}"
            )
            print("window scene:", raw_items[0]["scene"])
            print("interval seq:", raw_intervals[0]["seq_id"])

        else:
            self.items = raw_items

        self._token2map_max = 20000
        ensure_cache_subdir(self.cache_root, "3dbox_images")
        ensure_cache_subdir(self.cache_root, "hdmap_images")
        """
        # pts_proj
        
        self._bg_scene = {}
        if projected_pc_settings and "color_scene_by_location" in projected_pc_settings:
            self._bg_scene = {k: np.load(v, allow_pickle=False)
                for k, v in projected_pc_settings["color_scene_by_location"].items()}
        #balanceroot########
        self.balanced_json_path= balanced_json_path
        
        
        
        # actor root / templates 
        self._actor_root = projected_pc_settings.get("actor_root", None) if projected_pc_settings else None
        self._actor_tpl = {}
        tpl_root = projected_pc_settings.get("actor_template_root", None) if projected_pc_settings else None
        if tpl_root and not self._actor_tpl:
            for fn in os.listdir(tpl_root):
                if fn.endswith(".pkl"):
                    with open(os.path.join(tpl_root, fn), "rb") as f:
                        self._actor_tpl[fn[:-4]] = pickle.load(f)
        """

        
    def _png_path(self, subdir, token):
        return os.path.join(self.cache_root, subdir, f"{token}.png")

    @staticmethod
    def _load_infos(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
        if not isinstance(infos, list):
            raise TypeError("PKL must be list[dict] or {'infos': list[dict]}")
        return infos
    # ---------------------------
    # clip indices
    # ---------------------------
    def _build_scenes(self, infos):
        scenes = {}
        for info in infos:
            scene = info.get(self.scene_key)
            t = info.get(self.timestamp_key)
            if scene is None or t is None:
                continue
            scenes.setdefault(scene, []).append(info)

        scene_ts = {}
        for s, lst in scenes.items():
            lst.sort(key=lambda x: x[self.timestamp_key])
            scene_ts[s] = np.asarray([x[self.timestamp_key] for x in lst], dtype=np.int64)

        scenes = {s: lst for s, lst in scenes.items() if len(lst) >= self.sequence_length}
        scene_ts = {s: ts for s, ts in scene_ts.items() if len(ts) >= self.sequence_length}
        return scenes, scene_ts

    def _build_items(self):
        items = []
        for scene, ts in self.scene_ts.items():
            for fps, stride in self.fps_stride_tuples:
                items.extend(self._enumerate_segments(scene, ts, fps, stride))
        return items

    def _enumerate_segments(self, scene, ts, fps, stride):
        T, N = self.sequence_length, len(ts)
        items = []

        if float(fps) == 0.0:
            s = max(1, int(stride))
            for start in range(0, N - T + 1, s):
                items.append({"scene": scene, "fps": 0.0, "indices": list(range(start, start + T))})
            return items

        fps = float(fps)
        dt_us = int(round(1e6 / fps))
        seq_dur_us = (T - 1) * dt_us

        t_begin = int(ts[0])
        t_last_begin = int(ts[-1] - seq_dur_us)
        if t_last_begin < t_begin:
            return items

        stride_sec = float(stride)
        stride_us = dt_us if stride_sec <= 0 else int(round(stride_sec * 1e6))

        max_err_us = self.max_time_error_us
        if max_err_us is None:
            max_err_us = int(self.max_time_error_ratio * dt_us)

        t = t_begin
        while t <= t_last_begin:
            wanted = np.array([t + i * dt_us for i in range(T)], dtype=np.int64)
            idxs = [_find_nearest(ts, w) for w in wanted]
            if len(set(idxs)) != T:
                t += stride_us
                continue

            if self.enable_synchronization_check:
                picked = ts[np.asarray(idxs, dtype=np.int64)]
                if int(np.max(np.abs(picked - wanted))) > max_err_us:
                    t += stride_us
                    continue

            items.append({"scene": scene, "fps": fps, "indices": idxs})
            t += stride_us

        return items

    # ---------------------------
    # paths + calib
    # ---------------------------
    def _get_sensor_paths(self, info):
        img = info.get('img_filename') or []
        m = {p.split("/")[-2]: p for p in img}
        return [os.path.join(self.sensor_root, m.get(ch, "")) for ch in self.sensor_channels]

    def _get_cam_info(self, info, cam_ch):
        cam = info.get(self.cam_key) or {}
        return cam.get(cam_ch)

    def _img_token(self, info, cam_idx):
        tok = info.get(self.timestamp_key, "t")
        return f"{tok}_{cam_idx}"

    # ---------------------------
    # hdmap projection (nuplan)
    # ---------------------------
    @staticmethod
    def _clip_points_behind_camera(points, near_plane: float, is_polygon: bool):
        pts = []
        assert points.shape[0] == 3
        n = points.shape[1]
        m = n if is_polygon else n - 1
        for i1 in range(m):
            i2 = (i1 + 1) % n
            p1, p2 = points[:, i1], points[:, i2]
            z1, z2 = p1[2], p2[2]
            if z1 >= near_plane and z2 >= near_plane:
                if len(pts) == 0 or np.any(pts[-1] != p1):
                    pts.append(p1)
                pts.append(p2)
            elif z1 < near_plane and z2 < near_plane:
                continue
            else:
                if z1 <= z2:
                    pa, pb = p1, p2
                else:
                    pa, pb = p2, p1
                za, zb = pa[2], pb[2]
                d = pb - pa
                alpha = (near_plane - zb) / (za - zb)
                clipped = pa + (1 - alpha) * d
                if z1 >= near_plane and (len(pts) == 0 or np.any(pts[-1] != p1)):
                    pts.append(p1)
                pts.append(clipped)
                if z2 >= near_plane:
                    pts.append(p2)
        return np.array(pts).transpose() if len(pts) else np.zeros((3, 0), np.float32)

    def _perspective_coords(self, points, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, is_polygon):
        # keep your current "flatten z" behavior exactly
        ego2global_t = np.concatenate([ego2global_t[:2], np.zeros_like(ego2global_t)[:1]], axis=-1)
        points = points - np.array(ego2global_t).reshape((-1, 1))
        points = np.dot(ego2global_r.T, points)

        points = points - np.array(sensor2ego_t).reshape((-1, 1))
        points = np.dot(Quaternion(sensor2ego_r).rotation_matrix.T, points)

        depths = points[2, :]
        if np.all(depths < self.near_plane):
            return None

        points = self._clip_points_behind_camera(points, self.near_plane, is_polygon)
        if is_polygon and (points.size == 0 or points.shape[1] < 3):
            return None

        points = view_points(points, cam_intrinsic, normalize=True)

        inside = np.ones(points.shape[1], dtype=bool)
        inside &= points[0, :] > 1
        inside &= points[0, :] < im_size[0] - 1
        inside &= points[1, :] > 1
        inside &= points[1, :] < im_size[1] - 1
        if np.all(~inside):
            return None
        return points

    @staticmethod
    def _draw_linestring_rgb(canvas_rgb, geom, color_bgr, thickness):
        if geom is None or geom.is_empty:
            return

        segs = []

        gtype = geom.geom_type
        if gtype == "LineString":
            segs.append(np.asarray(geom.coords).round().astype(np.int32).reshape(-1, 1, 2))

        elif gtype == "MultiLineString":
            for ls in geom.geoms:
                if ls.is_empty:
                    continue
                segs.append(np.asarray(ls.coords).round().astype(np.int32).reshape(-1, 1, 2))

        elif gtype == "GeometryCollection":
            for g in geom.geoms:
                if g.is_empty:
                    continue
                if g.geom_type == "LineString":
                    segs.append(np.asarray(g.coords).round().astype(np.int32).reshape(-1, 1, 2))
                elif g.geom_type == "MultiLineString":
                    for ls in g.geoms:
                        if ls.is_empty:
                            continue
                        segs.append(np.asarray(ls.coords).round().astype(np.int32).reshape(-1, 1, 2))

        else:
            return

        if segs:
            cv2.polylines(canvas_rgb, segs, isClosed=False, color=color_bgr, thickness=int(thickness))

    def _get_numap(self, log_db_path, token):
        k = (log_db_path, token)
        if k not in self._token2map:
            self._token2map[k] = get_sensor_token_map_name_from_db(
                log_db_path, get_lidarpc_sensor_data(), token
            )
            if len(self._token2map) >= self._token2map_max:
                self._token2map.pop(next(iter(self._token2map)))
                
        name = self._token2map[k]
        if name not in self._map_cache:
            self._map_cache[name] = self.map_factory.build_map_from_name(name)
        return self._map_cache[name]

    def _get_hdmap_world_geom(self, info):
        token = info["lidarpc_token"]
        db_name = info[self.scene_key]
        db_path = db_name if str(db_name).endswith(".db") else str(db_name) + ".db"
        log_db_path = os.path.join(self.dataset_root, db_path)
        key = (log_db_path, token)

        hit = self._hdgeom_cache.get(key, None)
        if hit is not None:
            return hit

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        center = Point2D(float(ego2global[0, 3]), float(ego2global[1, 3]))

        numap = self._get_numap(log_db_path, token)

        # ---- polygons
        nearest_poly = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.polygon_layer_names)
        if SemanticMapLayer.STOP_LINE in nearest_poly:
            nearest_poly[SemanticMapLayer.STOP_LINE] = [
                sp for sp in nearest_poly[SemanticMapLayer.STOP_LINE]
                if sp.stop_line_type != StopLineType.TURN_STOP
            ]

        # 1) drivable union exteriors (world xy)
        drivable_layers = {
            SemanticMapLayer.LANE,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        }
        drivable_polys = []
        for ln, objs in nearest_poly.items():
            if ln not in drivable_layers:
                continue
            for o in objs:
                poly = getattr(o, "polygon", None)
                if poly is not None and (not poly.is_empty):
                    drivable_polys.append(poly)

        drivable_exteriors = []
        if drivable_polys:
            uni = unary_union(drivable_polys)
            geoms = uni.geoms if uni.geom_type == "MultiPolygon" else [uni]
            for g in geoms:
                ext = np.asarray(g.exterior.coords, np.float32)
                if ext.shape[0] >= 2:
                    drivable_exteriors.append(ext)

        # 2) divider lines (world xy) via seg count
        nearest_line = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.line_layer_names)

        seg_cnt = {}
        for _, objs in nearest_line.items():
            for lane in objs:
                for path in (lane.left_boundary.discrete_path, lane.right_boundary.discrete_path):
                    pts = [(float(p.x), float(p.y)) for p in path]
                    for i in range(len(pts) - 1):
                        k = self._seg_key(pts[i], pts[i + 1])
                        seg_cnt[k] = seg_cnt.get(k, 0) + 1

        divider_lines = []
        for _, objs in nearest_line.items():
            for lane in objs:
                for path in (lane.left_boundary.discrete_path, lane.right_boundary.discrete_path):
                    pts = [(float(p.x), float(p.y)) for p in path]
                    if len(pts) < 2:
                        continue
                    flags = [seg_cnt.get(self._seg_key(pts[i], pts[i + 1]), 0) >= 2 for i in range(len(pts) - 1)]
                    if (sum(flags) / max(1, len(flags))) < 0.6:
                        continue
                    divider_lines.append(np.asarray(pts, np.float32))

        # 3) crosswalk exteriors (world xy)
        crosswalk_exteriors = []
        for obj in nearest_poly.get(SemanticMapLayer.CROSSWALK, []):
            poly = getattr(obj, "polygon", None)
            if poly is None or poly.is_empty:
                continue
            ext = np.asarray(poly.exterior.coords, np.float32)
            if ext.shape[0] >= 3:
                crosswalk_exteriors.append(ext)

        out = {
            "drivable_exteriors": drivable_exteriors,
            "divider_lines": divider_lines,
            "crosswalk_exteriors": crosswalk_exteriors,
        }

        if len(self._hdgeom_cache) >= self._hdgeom_cache_max:
            self._hdgeom_cache.pop(next(iter(self._hdgeom_cache)))
        self._hdgeom_cache[key] = out
        return out

    # ---------- _get_hdmap_image ----------
    def _get_hdmap_image(self, info, cam_ch, im_size, cam_intrinsic_3x3):
        W, H = map(int, im_size)
        canvas = np.zeros((H, W, 3), np.uint8)

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        ego2global_t, ego2global_r = ego2global[:3, 3], ego2global[:3, :3]

        cam_info = self._get_cam_info(info, cam_ch)
        if cam_info is None:
            return Image.fromarray(canvas[..., ::-1])

        sensor2ego_t = np.asarray(cam_info["sensor2ego_translation"], np.float32)
        sensor2ego_r = cam_info["sensor2ego_rotation"]

        # cached heavy geometry (per token)
        geom = self._get_hdmap_world_geom(info)

        # inner clip polygon (same as your code)
        margin = 2.0
        scene_poly = Polygon([(margin, margin), (margin, H - margin), (W - margin, H - margin), (W - margin, margin)])

        # colors (same as your code)
        blue_rgb = (0, 0, 255)
        blue_bgr = (blue_rgb[2], blue_rgb[1], blue_rgb[0])

        green_rgb = (0, 255, 0)
        green_bgr = (green_rgb[2], green_rgb[1], green_rgb[0])

        red_rgb = (255, 0, 0)
        red_bgr = (red_rgb[2], red_rgb[1], red_rgb[0])

        # --- 1) blue: drivable union exteriors
        for ext_xy in geom["drivable_exteriors"]:
            pts3 = np.vstack([ext_xy.T, np.zeros((1, ext_xy.shape[0]), np.float32)])  # 3xN
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=True
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue
            if line.geom_type == "LineString":
                line = MultiLineString([line])
            self._draw_linestring_rgb(canvas, line, blue_bgr, thickness=int(self.hdmap_pen_width))

        # --- 2) green: divider lines (already filtered in world)
        for pts in geom["divider_lines"]:
            pts2 = pts.T.astype(np.float32)  # 2xN
            pts3 = np.vstack([pts2, np.zeros((1, pts2.shape[1]), np.float32)])
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=False
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue
            if line.geom_type == "LineString":
                line = MultiLineString([line])
            self._draw_linestring_rgb(canvas, line, green_bgr, thickness=max(1, int(self.hdmap_pen_width // 2)))

        # --- 3) red: crosswalk exteriors
        for ext_xy in geom["crosswalk_exteriors"]:
            pts3 = np.vstack([ext_xy.T, np.zeros((1, ext_xy.shape[0]), np.float32)])  # 3xN
            q = self._perspective_coords(
                pts3, (W, H),
                ego2global_t, ego2global_r,
                sensor2ego_t, sensor2ego_r,
                cam_intrinsic_3x3,
                is_polygon=True
            )
            if q is None:
                continue
            xy_img = np.stack([q[0], q[1]], axis=1).astype(np.float32)
            line = LineString([(float(x), float(y)) for x, y in xy_img]).intersection(scene_poly)
            if line.is_empty:
                continue

            if line.geom_type == "LineString":
                line = MultiLineString([line])
            elif line.geom_type == "GeometryCollection":
                parts = [g for g in line.geoms if g.geom_type in ("LineString", "MultiLineString")]
                if not parts:
                    continue
                line = unary_union(parts)
                if line.geom_type == "LineString":
                    line = MultiLineString([line])
                elif line.geom_type != "MultiLineString":
                    continue
            elif line.geom_type != "MultiLineString":
                continue

            self._draw_linestring_rgb(canvas, line, red_bgr, thickness=int(self.hdmap_pen_width))

        return Image.fromarray(canvas[..., ::-1])

    # ---------------------------
    # 3dbox from gt_line
    # ---------------------------
    def _get_3dbox_image_from_gtline(self, info, im_size, lidar2image_4x4):
        W, H = int(im_size[0]), int(im_size[1])
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        boxes = info.get("gt_boxes")
        names = info.get('gt_names')
        names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else None

        for i, box in enumerate(boxes):
            color = (0, 0, 0)
            if names is not None and i < len(names):
                color = self.default_3dbox_color_table.get(str(names[i]), color)

            corners = _box_corners_lidar_xyz(box[:7], z_is_center=True)
            uvz = _project_pts(corners, lidar2image_4x4)
            if np.any(uvz[:, 2] <= 1e-5):
                continue

            pts2 = uvz[:, :2]
            if (pts2[:, 0].max() < 0) or (pts2[:, 0].min() > W) or (pts2[:, 1].max() < 0) or (pts2[:, 1].min() > H):
                continue

            for a, b in _BOX_EDGES:
                xa, ya = pts2[a]
                xb, yb = pts2[b]
                draw.line((float(xa), float(ya), float(xb), float(yb)),
                          fill=tuple(color), width=self._3dbox_pen_width)

        return img

    # ---------------------------
    # pts_proj
    # ---------------------------
    
    @staticmethod
    def _as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]

    @staticmethod
    def _global_to_ego(xyz_g: np.ndarray, R_ego2glb: np.ndarray, t_ego_glb: np.ndarray):
        # xyz_g: (N,3), R: (3,3), t: (3,)
        return (xyz_g - t_ego_glb[None, :]) @ R_ego2glb

    def _pick_car_template(self, dz: float):
        dz = float(dz)
        if dz < 1.8:  return self._actor_tpl.get("sedan")
        if dz < 2.05: return self._actor_tpl.get("suv")
        if dz < 2.7:  return self._actor_tpl.get("pickup")
        return None

    @staticmethod
    def _seg_key(p, q, quant=0.2):
        p = tuple(np.round(np.asarray(p) / quant).astype(np.int32))
        q = tuple(np.round(np.asarray(q) / quant).astype(np.int32))
        return (p, q) if p <= q else (q, p)
        
    def _compose_points_for_frame(self, info):
        s = self.projected_pc_settings or {}
        radius = float(s.get("radius", 150.0))
        min_actor_pts = int(s.get("min_actor_points", 30000))

        ego2global = np.asarray(info[self.ego2global_key], np.float32)
        R = ego2global[:3, :3]   # ego->global 
        t = ego2global[:3, 3]

        # ---------- 1) background ----------
        bg = self._bg_scene.get(info.get(self.scene_key), None)
        if bg is None:
            bg_xyz_l = np.zeros((0, 3), np.float32)
            bg_rgb   = np.zeros((0, 3), np.uint8)
        else:
            bg = np.asarray(bg)
            xyz_g = bg[:, :3].astype(np.float32)
            rgb   = np.clip(bg[:, 3:6], 0, 255).astype(np.uint8)

            m = (xyz_g[:, 0] > t[0] - radius) & (xyz_g[:, 0] < t[0] + radius) & \
                (xyz_g[:, 1] > t[1] - radius) & (xyz_g[:, 1] < t[1] + radius)
            xyz_g, rgb = xyz_g[m], rgb[m]

            bg_xyz_l = self._global_to_ego(xyz_g, R, t).astype(np.float32)
            bg_rgb   = rgb

        bg_sem = np.zeros((bg_xyz_l.shape[0],), np.int32)

        # ---------- 2) actors ----------
        boxes = info.get("gt_boxes", None)
        if boxes is None:
            boxes = []
        names  = self._as_list(info.get("gt_names", None))
        tracks = info.get("track_token", None)
        if tracks is None:
            tracks = info.get("gt_track_token", None)
        tracks = self._as_list(tracks)
        if len(tracks) == 0:
            tracks = [None] * len(boxes)

        name2id = {"car": 1, "ped": 2, "bike": 3}

        act_xyz_list, act_sem_list = [], []
        act_clr_xyz_list, act_clr_rgb_list = [], []

        for i, box in enumerate(boxes):
            # --- box pose（你原来漏了这几个，后面 Rz/x/y/z/dz/yaw 都靠它）---
            x, y, z, dx, dy, dz, yaw = box[:7].astype(np.float32)

            cls = str(names[i]) if i < len(names) else ""
            sid = name2id.get(cls, 0)
            if sid == 0:
                continue

            tok = tracks[i] if i < len(tracks) else None

            actor_track = None
            if tok is not None and self._actor_root is not None:
                p = os.path.join(self._actor_root, f"{tok}.npy")
                if os.path.isfile(p):
                    actor_track = np.asarray(np.load(p))

                    # 空 track 直接当不存在
                    if actor_track.shape[0] == 0:
                        actor_track = None

            # --- sem/depth 用：优先 track 点多，否则模板 ---
            if cls == "car":
                actor_sem = self._pick_car_template(dz)
            elif cls == "ped":
                actor_sem = self._actor_tpl.get("ped")
            elif cls == "bike":
                actor_sem = self._actor_tpl.get("bike")
            if actor_sem is None:
                continue

            # ====== sem 点 ======
            actor_sem = np.asarray(actor_sem)
            xyz_sem = actor_sem[:, :3].astype(np.float32)
            Rz = _yaw_to_Rz(yaw)
            xyz_sem = (Rz @ xyz_sem.T).T + np.array([x, y, z], np.float32)[None, :]
            act_xyz_list.append(xyz_sem)
            act_sem_list.append(np.full((xyz_sem.shape[0],), sid, np.int32))

            # ====== clr 点：只用“有RGB的track”，没有就不投影 ======
            if actor_track is None or actor_track.shape[1] < 6:
                continue  # 关键：没RGB就跳过这个actor的clr

            xyz_clr = actor_track[:, :3].astype(np.float32)
            xyz_clr = (Rz @ xyz_clr.T).T + np.array([x, y, z], np.float32)[None, :]

            a_rgb = actor_track[:, 3:6]
            if a_rgb.size == 0:
                continue  # 防 nanmax 空数组

            mx = np.nanmax(a_rgb)
            a_rgb_u8 = (np.clip(a_rgb * 255.0, 0, 255).astype(np.uint8)
                        if mx <= 1.5 else np.clip(a_rgb, 0, 255).astype(np.uint8))

            act_clr_xyz_list.append(xyz_clr)
            act_clr_rgb_list.append(a_rgb_u8)

        act_xyz = np.concatenate(act_xyz_list, 0) if act_xyz_list else np.zeros((0, 3), np.float32)
        act_sem = np.concatenate(act_sem_list, 0) if act_sem_list else np.zeros((0,), np.int32)
        act_clr_xyz = np.concatenate(act_clr_xyz_list, 0) if act_clr_xyz_list else np.zeros((0, 3), np.float32)
        act_clr_rgb = np.concatenate(act_clr_rgb_list, 0) if act_clr_rgb_list else np.zeros((0, 3), np.uint8)

        pts_xyz = np.concatenate([bg_xyz_l, act_xyz], 0).astype(np.float32)
        pts_sem = np.concatenate([bg_sem,   act_sem], 0).astype(np.int32)
        clr_xyz = np.concatenate([bg_xyz_l, act_clr_xyz], 0).astype(np.float32)
        clr_rgb = np.concatenate([bg_rgb,   act_clr_rgb], 0).astype(np.uint8)

        return pts_xyz, pts_sem, clr_xyz, clr_rgb
    
    # ---------------------------
    # main api
    # ---------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        scene, idxs, fps = item["scene"], item["indices"], item["fps"]

        seq = [self.scenes[scene][i] for i in idxs]
        location = seq[0]['location']
        t0 = seq[0][self.timestamp_key]
        pts = torch.tensor([(x[self.timestamp_key] - t0 + 500) // 1000 for x in seq], dtype=torch.float32)

        # 把匹配到的 angle 和 dist 放入 result
        result = {
            "fps": torch.tensor(float(fps), dtype=torch.float32), 
            "pts": pts,
            "angle": item.get("angle", 0.0),
            "dist": item.get("dist", 0.0),
            "scene_id": scene,
            "name": scene
        }

        if self.enable_sample_data:
            result["sample_data"] = seq
            result["scene"] = {"token": scene}

        # ---- [加载图像及标定逻辑：保持不变] ----
        images, cam_Ks, img_sizes, dists = [], [], [], []
        cam_infos = []
        for x in seq:
            paths = self._get_sensor_paths(x)
            row_imgs, row_K, row_sz, row_dist = [], [], [], []
            row_infos = []  
            for cam_i, (cam_ch, p) in enumerate(zip(self.sensor_channels, paths)):
                im = _try_open_png(p)
                cam_info = self._get_cam_info(x, cam_ch)
                row_infos.append(cam_info)
                if im is None or cam_info is None:
                    W, H = 1920, 1080
                    row_imgs.append(Image.new("RGB", (W, H)))
                    row_K.append(np.eye(4, dtype=np.float32))
                    row_sz.append((W, H))
                    row_dist.append(np.zeros((0,), np.float32))
                else:
                    row_imgs.append(im)
                    K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
                    K_pad = np.eye(4, dtype=np.float32); K_pad[:3, :3] = K3
                    row_K.append(K_pad)
                    row_sz.append(im.size)
                    row_dist.append(np.asarray(cam_info.get("distortion", []), np.float32).reshape(-1))
            cam_infos.append(row_infos); images.append(row_imgs); cam_Ks.append(row_K)
            img_sizes.append(row_sz); dists.append(row_dist)

        result["images"] = images
        result["camera_intrinsics"] = torch.tensor(np.asarray(cam_Ks), dtype=torch.float32)
        result["image_size"] = torch.tensor(np.asarray([[[w,h] for (w,h) in row] for row in img_sizes]), dtype=torch.long)
        result["distortion"] = dists

        # ---- 3dbox_images (gt_line) cached
        cached = []
        all_hit = True
        for t, x in enumerate(seq):
            row = []
            for v, _ in enumerate(self.sensor_channels):
                tok = self._img_token(x, v)
                p = self._png_path("3dbox_images", tok)
                img = _try_open_png(p) if os.path.isfile(p) else None
                if img is None:
                    all_hit = False
                row.append(img)
            cached.append(row)

        if all_hit:
            result["3dbox_images"] = cached
        else:
            imgs = []
            for t, x in enumerate(seq):
                row = []
                for v, cam_ch in enumerate(self.sensor_channels):
                    W, H = img_sizes[t][v]

                    cam_info = self._get_cam_info(x, cam_ch)
                    if cam_info is None:
                        row.append(Image.new("RGB", (int(W), int(H))))
                        continue

                    K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)

                    l2i = _lidar2image_from_caminfo(cam_info, K3)
                    row.append(self._get_3dbox_image_from_gtline(x, (W, H), l2i))
                imgs.append(row)

            result["3dbox_images"] = imgs
            for t, x in enumerate(seq):
                for v, _ in enumerate(self.sensor_channels):
                    tok = self._img_token(x, v)
                    p = self._png_path("3dbox_images", tok)
                    lock = p + ".lock"
                    if os.path.isfile(p) and _try_open_png(p) is not None:
                        continue
                    _acquire_lock(lock, timeout=30, stale=120, sleep=0.02)
                    try:
                        if os.path.isfile(p) and _try_open_png(p) is not None:
                            continue
                        _safe_save_png(imgs[t][v], p)
                    finally:
                        _release_lock(lock)

        # ---- hdmap_images (nuplan) cached
        cached = []
        all_hit = True
        for t, x in enumerate(seq):
            row = []
            for v, _ in enumerate(self.sensor_channels):
                tok = self._img_token(x, v)
                p = self._png_path("hdmap_images", tok)
                img = _try_open_png(p) if os.path.isfile(p) else None
                if img is None:
                    all_hit = False
                row.append(img)
            cached.append(row)

        if all_hit:
            result["hdmap_images"] = cached
        else:
            imgs = []
            for t, x in enumerate(seq):
                row = []
                for v, cam_ch in enumerate(self.sensor_channels):
                    W, H = img_sizes[t][v]
                    cam_intr = cam_Ks[t][v][:3, :3]
                    row.append(self._get_hdmap_image(x, cam_ch, (W, H), cam_intr))
                imgs.append(row)

            result["hdmap_images"] = imgs
            for t, x in enumerate(seq):
                for v, _ in enumerate(self.sensor_channels):
                    tok = self._img_token(x, v)
                    p = self._png_path("hdmap_images", tok)
                    lock = p + ".lock"
                    if os.path.isfile(p) and _try_open_png(p) is not None:
                        continue
                    _acquire_lock(lock, timeout=30, stale=120, sleep=0.02)
                    try:
                        if os.path.isfile(p) and _try_open_png(p) is not None:
                            continue
                        _safe_save_png(imgs[t][v], p)
                    finally:
                        _release_lock(lock)
        ''' 
        # ---- pts proj (nuplan) cached -> proj_depth/proj_sem/proj_clr (tensors)
        if self.projected_pc_settings:
            s = self.projected_pc_settings

            final_hw = s.get("final_hw", None)                 # (H,W) or None
            invalid  = float(s.get("invalid_depth", -300.0))
            splat    = s.get("splat", [(1e9, 0)])

            n_bins = int(s.get("depth_bins", 256))
            gamma  = float(s.get("log_gamma", 1.0))
            far_m  = float(s.get("radius", 50.0))
            mode   = str(s.get("depth_bin_mode", "log")).lower()
            n_cls  = int(s.get("n_actor_classes", 3))

            if "data_type" not in s:
                raise KeyError("projected_pc_settings['data_type'] must be set: one of {'depth','clr','all'}")
            dt = str(s["data_type"]).lower().strip()
            if dt not in ("depth", "clr", "all"):
                raise ValueError(f"Invalid projected_pc_settings['data_type']={dt!r}, expected 'depth'/'clr'/'all'")
            want_sem_out = False
            # depth/clr 按 data_type 控制
            if dt == "depth":
                want_depth_out, want_clr_out = True, False
            elif dt == "clr":
                want_depth_out, want_clr_out = False, True
            elif dt == "all":
                want_depth_out, want_clr_out = True, True
            else:
                want_depth_out, want_clr_out = False, False
                
            read_vis_depth = bool(s.get("read_vis_depth", False))
            if read_vis_depth and (not want_depth_out):
                raise ValueError("read_vis_depth=True requires depth input. Set data_type='depth' or 'all'.")
            
            depth_dir = s.get("depth_dir", f"proj_depth_g{gamma}_b{n_bins}")
            sem_dir   = s.get("sem_dir",   "proj_sem")
            clr_dir   = s.get("clr_dir",   "proj_clr")
            vis_dir   = s.get("vis_dir",   f"proj_depth_vis_g{gamma}_b{n_bins}")

            # 只为“需要的”创建子目录（需要啥存啥）
            if want_depth_out:
                ensure_cache_subdir(self.cache_root, depth_dir)
            if want_sem_out:
                ensure_cache_subdir(self.cache_root, sem_dir)
            if want_clr_out:
                ensure_cache_subdir(self.cache_root, clr_dir)
            if read_vis_depth:
                ensure_cache_subdir(self.cache_root, vis_dir)

            # output holders
            proj_depth = [] if want_depth_out else None
            proj_sem   = [] if want_sem_out else None
            proj_clr   = [] if want_clr_out else None
            proj_vis   = [] if read_vis_depth else None

            for t, info in enumerate(seq):

                V = len(self.sensor_channels)

                toks = [None] * V
                p_d_list = [None] * V
                p_s_list = [None] * V
                p_c_list = [None] * V
                p_v_list = [None] * V

                ex_d = [False] * V
                ex_s = [False] * V
                ex_c = [False] * V
                ex_v = [False] * V

                for v, cam_ch in enumerate(self.sensor_channels):
                    tok = self._img_token(info, v)
                    toks[v] = tok

                    if want_depth_out:
                        p = os.path.join(self.cache_root, depth_dir, f"{tok}.png")
                        p_d_list[v] = p
                        ex_d[v] = os.path.isfile(p)

                    if want_sem_out:
                        p = os.path.join(self.cache_root, sem_dir, f"{tok}.png")
                        p_s_list[v] = p
                        ex_s[v] = os.path.isfile(p)

                    if want_clr_out:
                        p = os.path.join(self.cache_root, clr_dir, f"{tok}.png")
                        p_c_list[v] = p
                        ex_c[v] = os.path.isfile(p)

                    if read_vis_depth:
                        p = os.path.join(self.cache_root, vis_dir, f"{tok}.png")
                        p_v_list[v] = p
                        ex_v[v] = os.path.isfile(p)

                need_any_compute = False
                if want_depth_out and (not all(ex_d)):
                    need_any_compute = True
                if want_sem_out and (not all(ex_s)):
                    need_any_compute = True
                if want_clr_out and (not all(ex_c)):
                    need_any_compute = True

                if need_any_compute:
                    pts_xyz, pts_sem, clr_xyz, clr_rgb = self._compose_points_for_frame(info)
                else:
                    pts_xyz = pts_sem = clr_xyz = clr_rgb = None

                row_d = [] if want_depth_out else None
                row_s = [] if want_sem_out else None
                row_c = [] if want_clr_out else None
                row_v = [] if read_vis_depth else None

                # ============================================================
                # [MOD] 进入 view loop：直接复用 tok/path/exist，减少 IO
                # ============================================================
                for v, cam_ch in enumerate(self.sensor_channels):
                    tok = toks[v]

                    p_d = p_d_list[v] if want_depth_out else None
                    p_s = p_s_list[v] if want_sem_out else None
                    p_c = p_c_list[v] if want_clr_out else None
                    p_v = p_v_list[v] if read_vis_depth else None

                    # fast path read (no lock) —— 复用 ex_*，不再重复 isfile
                    bins_u16 = _try_open_u16_png(p_d) if (want_depth_out and ex_d[v]) else None
                    sem_img  = _try_open_png(p_s)     if (want_sem_out   and ex_s[v]) else None
                    clr_img  = _try_open_png(p_c)     if (want_clr_out   and ex_c[v]) else None
                    vis_img  = _try_open_png(p_v)     if (read_vis_depth and ex_v[v]) else None

                    # decide missing
                    need_depth = want_depth_out and (bins_u16 is None)
                    need_sem   = want_sem_out   and (sem_img  is None)
                    need_clr   = want_clr_out   and (clr_img  is None)

                    cam_info = self._get_cam_info(info, cam_ch)
                    if cam_info is None:
                        # fill zeros with correct size
                        W, H = img_sizes[t][v]
                        H0, W0 = int(H), int(W)
                        if final_hw is not None:
                            H0, W0 = int(final_hw[0]), int(final_hw[1])

                        if want_depth_out:
                            row_d.append(torch.zeros((H0, W0), dtype=torch.long))
                        if want_sem_out:
                            row_s.append(torch.zeros((n_cls, H0, W0), dtype=torch.float32))
                        if want_clr_out:
                            row_c.append(torch.zeros((3, H0, W0), dtype=torch.float32))
                        if read_vis_depth:
                            row_v.append(torch.zeros((3, H0, W0), dtype=torch.float32))
                        continue

                    # -------- B) depth/sem/clr missing: acquire only relevant locks, then compute only what’s missing --------
                    if need_depth or need_sem or need_clr:
                        locks = []
                        if need_depth and p_d: locks.append(p_d + ".lock")
                        if need_sem   and p_s: locks.append(p_s + ".lock")
                        if need_clr   and p_c: locks.append(p_c + ".lock")
                        # (vis lock not bundled here; keep vis independent)
                        locks.sort()

                        for lk in locks:
                            _acquire_lock(lk, timeout=30, stale=120, sleep=0.02)

                        try:
                            # re-check inside lock (someone may have written)
                            if need_depth and p_d and self.do_cache and os.path.isfile(p_d):
                                temp = _try_open_u16_png(p_d)
                                bins_u16 = temp if temp is not None else bins_u16
                            if need_sem and p_s and self.do_cache and os.path.isfile(p_s):
                                temp = _try_open_png(p_s)
                                sem_img = temp if temp is not None else sem_img
                            if need_clr and p_c and self.do_cache and os.path.isfile(p_c):
                                temp = _try_open_png(p_c)
                                clr_img = temp if temp is not None else clr_img

                            need_depth = want_depth_out and (bins_u16 is None)
                            need_sem   = want_sem_out   and (sem_img  is None)
                            need_clr   = want_clr_out   and (clr_img  is None)

                            if need_depth or need_sem or need_clr:
                                W, H = img_sizes[t][v]
                                ori_hw = (int(H), int(W))

                                K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
                                l2i = _lidar2image_from_caminfo(cam_info, K3)

                                # compute only the missing branches
                                depth = None
                                sem   = None
                                clr   = None
                                bins  = None

                                if need_depth:
                                    depth = project_depth_only(
                                        pts_xyz, l2i, ori_hw,
                                        invalid_depth=invalid, splat=splat
                                    )

                                if need_sem:
                                    sem = project_sem_only(
                                        pts_xyz, pts_sem, l2i, ori_hw,
                                        splat=splat, n_actor_classes=n_cls
                                    )

                                if need_clr:
                                    clr = project_clr_only(
                                        clr_xyz, clr_rgb, l2i, ori_hw,
                                        splat=splat
                                    )

                                # downsample to final_hw if requested
                                if final_hw is not None:
                                    if depth is not None:
                                        depth = downsample_depth_blockwise(depth, final_hw, invalid=invalid)
                                    if sem is not None:
                                        # sem: (H,W,K) -> resize with nearest
                                        sem_u8 = (sem * 255).astype(np.uint8)
                                        sem_u8 = cv2.resize(
                                            sem_u8, (final_hw[1], final_hw[0]),
                                            interpolation=cv2.INTER_NEAREST
                                        )
                                        sem = (sem_u8 > 127).astype(np.uint8)
                                    if clr is not None:
                                        clr = downsample_clr_blockwise(clr, final_hw)

                                # save/cache per-branch
                                if need_depth:
                                    if mode in ("linear", "lin", "abs"):
                                        bins = depth_to_linbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m)
                                    else:
                                        bins = depth_to_logbins_u16(depth, invalid=invalid, n_bins=n_bins, far_m=far_m, gamma=gamma)

                                    bins_u16 = np.asarray(bins, np.uint16)  # 直接 bins(0..n_bins), invalid=0

                                    if self.do_cache and p_d:
                                        _safe_save_u16_png(bins_u16, p_d)

                                if need_sem:
                                    if n_cls != 3:
                                        raise ValueError(f"proj_sem cache uses RGB png, require n_actor_classes=3, got {n_cls}")
                                    sem_rgb = (sem * 255).astype(np.uint8)  # (H,W,K) with K=3
                                    sem_img = Image.fromarray(sem_rgb, "RGB")
                                    if self.do_cache and p_s:
                                        _safe_save_png(sem_img, p_s)

                                if need_clr:
                                    clr_u8 = clr.astype(np.uint8)
                                    clr_img = Image.fromarray(clr_u8, "RGB")
                                    if self.do_cache and p_c:
                                        _safe_save_png(clr_img, p_c)

                        finally:
                            for lk in reversed(locks):
                                _release_lock(lk)
                    
                    # ===== ensure vis_depth cached/read when requested =====
                    if read_vis_depth:
                        # depth 必须存在（你前面已经校验过 data_type）
                        if bins_u16 is None:
                            raise RuntimeError(f"read_vis_depth=True but depth bins missing: {p_d}")

                        # 先尝试读 cache（无锁快读）
                        if vis_img is None and p_v and os.path.isfile(p_v):
                            vis_img = _try_open_png(p_v)

                        # cache 没有就生成并存（按你规则：read_vis_depth 存在就一定存/读）
                        if vis_img is None:
                            vis_img = vis_img = vis_from_bins_cache(bins_u16, n_bins)
                            if self.do_cache and p_v:
                                lk = p_v + ".lock"
                                _acquire_lock(lk, timeout=30, stale=120, sleep=0.02)
                                try:
                                    # double check under lock
                                    if (not os.path.isfile(p_v)) or (_try_open_png(p_v) is None):
                                        _safe_save_png(vis_img, p_v)
                                finally:
                                    _release_lock(lk)
                    # ======================================================

                    # -------- to tensors (only append what we output) --------
                    W, H = img_sizes[t][v]
                    H0, W0 = int(H), int(W)
                    if final_hw is not None:
                        H0, W0 = int(final_hw[0]), int(final_hw[1])

                    if want_depth_out:
                        if bins_u16 is None:
                            raise RuntimeError(f"proj depth cache still None: {p_d}")
                        row_d.append(torch.from_numpy(np.asarray(bins_u16, np.int64)))

                    if want_sem_out:
                        if sem_img is None:
                            raise RuntimeError(f"proj sem cache still None: {p_s}")
                        sem_np = (np.asarray(sem_img, np.uint8) > 127).astype(np.float32)
                        row_s.append(torch.from_numpy(sem_np).permute(2, 0, 1))

                    if want_clr_out:
                        if clr_img is None:
                            raise RuntimeError(f"proj clr cache still None: {p_c}")
                        clr_np = np.asarray(clr_img, np.uint8).copy()
                        row_c.append(torch.from_numpy(clr_np).permute(2, 0, 1).float() / 255.)

                    if read_vis_depth:
                        if vis_img is None:
                            raise RuntimeError(f"vis depth still None (need depth bins): {p_v}")
                        vis_np = np.asarray(vis_img, np.uint8).copy()
                        row_v.append(torch.from_numpy(vis_np).permute(2, 0, 1).float() / 255.)

                # stack rows per frame
                if want_depth_out:
                    proj_depth.append(torch.stack(row_d, 0))
                if want_sem_out:
                    proj_sem.append(torch.stack(row_s, 0))
                if want_clr_out:
                    proj_clr.append(torch.stack(row_c, 0))
                if read_vis_depth:
                    proj_vis.append(torch.stack(row_v, 0))

            # write to result (only what requested)
            if want_depth_out:
                result["proj_depth"] = torch.stack(proj_depth, 0)
            if want_sem_out:
                result["proj_sem"]   = torch.stack(proj_sem,   0)
            if want_clr_out:
                result["proj_clr"]   = torch.stack(proj_clr,   0)
            if read_vis_depth:
                result["vis_depth"]  = torch.stack(proj_vis,   0)
               '''
        # camera_transforms: ego_from_camera
        if "camera_transforms" not in result:
            cam_T = []
            for row_infos in cam_infos:  # [V]
                row_T = []
                for ci in row_infos:
                    if ci is None:
                        row_T.append(np.eye(4, dtype=np.float32))
                    else:
                        row_T.append(_se3_from_qt(ci["sensor2ego_rotation"], ci["sensor2ego_translation"]))
                cam_T.append(row_T)
            result["camera_transforms"] = torch.tensor(np.asarray(cam_T), dtype=torch.float32)  # [T,V,4,4]

        # ego_transforms: world_from_ego (same per view in a frame)
        if "ego_transforms" not in result:
            ego_T = []
            V = len(self.sensor_channels)
            for x in seq:
                world_from_ego = np.asarray(x[self.ego2global_key], np.float32)
                ego_T.append([world_from_ego] * V)
            result["ego_transforms"] = torch.tensor(np.asarray(ego_T), dtype=torch.float32)  # [T,V,4,4]

        if "lidar_transforms" not in result:
            T = len(seq)
            I4 = torch.eye(4, dtype=torch.float32)
            result["lidar_transforms"] = I4.view(1, 1, 4, 4).repeat(T, 1, 1, 1)  # [T,1,4,4]
            
        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)
        
        if "image_description" not in result:
            T = len(seq)
            V = len(self.sensor_channels)

            db_name = seq[0].get("db_name", None)

            cap = self.text_anno.get(db_name, {}) or {}
            text_main = make_image_description_string(cap, self.text_settings)

            tail = f"This is a nuplan video clip from {location}" if location else "This is a nuplan video clip"
            text = f"{text_main}. {tail}" if text_main else tail

            result["image_description"] = [[text] * V for _ in range(T)]
                
        return result
