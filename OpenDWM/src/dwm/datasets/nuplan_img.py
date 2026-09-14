import os, time, pickle as pkl
import numpy as np
import torch
import cv2
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

ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(3)


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


def _safe_save_png(pil_img, p):
    tmp = p + ".tmp"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pil_img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, p)


def _try_open_png(p):
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


def _try_open_image(p):
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except Exception:
        return None


def _acquire_lock(lock, timeout=30, stale=120, sleep=0.02):
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


def _find_nearest(sorted_arr, x):
    i = int(np.searchsorted(sorted_arr, x))
    if i <= 0:
        return 0
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    return i - 1 if abs(sorted_arr[i - 1] - x) <= abs(sorted_arr[i] - x) else i

_BOX_EDGES = [
    # bottom face (0-1-2-3)
    (0, 1), (1, 2), (2, 3), (3, 0),
    # verticals
    (0, 4), (1, 5), (2, 6), (3, 7),
    # top face (4-5-6-7)
    (4, 5), (5, 6), (6, 7), (7, 4),
    # extra "direction" lines (keep your style)
    (6, 3), (6, 5),
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

    def _png_path(self, subdir, token):
        return os.path.join(self.cache_root, subdir, f"{token}.png")

    @staticmethod
    def _load_infos(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pkl.load(f)
        infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
        if not isinstance(infos, list):
            raise TypeError("PKL must be list[dict] or {'infos': list[dict]}")
        return infos

    def __init__(
        self,
        sensor_root,
        pkl_path,
        cache_root,
        dataset_root,                 # dataset_root/{db_name}.db
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
        # render
        _3dbox_pen_width=8,
        hdmap_pen_width=8,
        hdmap_patch_radius=100.0,
        near_plane=1e-8,
        min_polygon_area=2000,
        # cam calib
        cam_key="cam",
        # ego pose
        ego2global_key="ego2global",         # 4x4
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

        # nuplan layers
        self.polygon_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]
        self.line_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]

        infos = self._load_infos(pkl_path)
        self.scenes, self.scene_ts = self._build_scenes(infos)
        self.items = self._build_items()
        
        self._map_cache = {}        # map_name -> numap
        self._token2map = {}        # (log_db_path, lidar_token) -> map_name
        self._hdgeom_cache = {}     # (log_db_path, lidar_token) -> precomputed world-geometry
        self._hdgeom_cache_max = 4096

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
        tok = info.get(self.token_key, None)
        if tok is None:
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

    def _get_perspective_lines(self, nearest_vector_map, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic):
        ret = {}
        for layer_name in self.line_layer_names:
            objs = nearest_vector_map.get(layer_name, [])
            for map_obj in objs:
                for path in [map_obj.left_boundary.discrete_path, map_obj.right_boundary.discrete_path]:
                    pts2 = np.array([[p.x, p.y] for p in path], dtype=np.float32).T
                    pts3 = np.vstack((pts2, np.zeros((1, pts2.shape[1]), np.float32)))
                    q = self._perspective_coords(pts3, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, False)
                    if q is None:
                        continue
                    xy = [(float(a), float(b)) for a, b in zip(q[0], q[1])]
                    ret.setdefault(layer_name, []).append(LineString(xy))
        return ret

    def _get_perspective_polygons(self, nearest_vector_map, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic):
        ret = {}
        for layer_name in self.polygon_layer_names:
            objs = nearest_vector_map.get(layer_name, [])
            for obj in objs:
                poly: Polygon = obj.polygon
                pts2 = np.array(poly.exterior.xy, dtype=np.float32)
                pts3 = np.vstack((pts2, np.zeros((1, pts2.shape[1]), np.float32)))
                q = self._perspective_coords(pts3, im_size, ego2global_t, ego2global_r, sensor2ego_t, sensor2ego_r, cam_intrinsic, True)
                if q is None:
                    continue
                xy = np.stack([q[0], q[1]], axis=1).astype(np.float32)  # (N,2)
                poly_proj = Polygon([(float(a), float(b)) for a, b in xy])
                if poly_proj.area < self.min_polygon_area:
                    continue
                ret.setdefault(layer_name, []).append(xy)
        return ret

    @staticmethod
    def _draw_polyline_rgb(canvas_rgb, xy, color_bgr, thickness):
        if xy is None or len(xy) < 2:
            return
        pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas_rgb, [pts], isClosed=True, color=color_bgr, thickness=int(thickness))

    @staticmethod
    def _draw_linestring_rgb(canvas_rgb, line: MultiLineString, color_bgr, thickness):
        def ic(x): return np.array(x).round().astype(np.int32)
        segs = [ic(ls.coords).reshape(-1, 1, 2) for ls in line.geoms]
        if len(segs):
            cv2.polylines(canvas_rgb, segs, isClosed=False, color=color_bgr, thickness=int(thickness))


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

        token = info["lidarpc_token"]  # 保证和 ego2global 同步
        db_name = info[self.scene_key]
        db_path = db_name if str(db_name).endswith(".db") else str(db_name) + ".db"
        log_file_full_path = os.path.join(self.dataset_root, db_path)

        map_name = get_sensor_token_map_name_from_db(log_file_full_path, get_lidarpc_sensor_data(), token)
        numap = self.map_factory.build_map_from_name(map_name)

        center = Point2D(float(ego2global[0, 3]), float(ego2global[1, 3]))

        # 用 polygon_layer_names 取对象（包含 lane/intersection/walkways等）
        nearest_poly = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.polygon_layer_names)
        if SemanticMapLayer.STOP_LINE in nearest_poly:
            nearest_poly[SemanticMapLayer.STOP_LINE] = [
                sp for sp in nearest_poly[SemanticMapLayer.STOP_LINE]
                if sp.stop_line_type != StopLineType.TURN_STOP
            ]

        # --- 1) 蓝色：drivable area union 外轮廓
        # 只用真正“路面”的层做 union，避免 crosswalk/stop_line 把边界搞花
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
                if hasattr(o, "polygon") and o.polygon is not None and (not o.polygon.is_empty):
                    drivable_polys.append(o.polygon)

        # 内缩裁剪框，去掉图像边缘的框线
        margin = 2.0
        scene_poly = Polygon([(margin, margin), (margin, H - margin), (W - margin, H - margin), (W - margin, margin)])

        blue_rgb = (0, 0, 255)
        blue_bgr = (blue_rgb[2], blue_rgb[1], blue_rgb[0])

        if drivable_polys:
            uni = unary_union(drivable_polys)
            # 可能是 Polygon 或 MultiPolygon：都取 exterior 画
            geoms = uni.geoms if uni.geom_type == "MultiPolygon" else [uni]
            for g in geoms:
                ext_xy = np.asarray(g.exterior.coords, np.float32)  # (N,2) in global
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

        # --- 2) 绿色：lane divider（共享边界）只画“出现>=2次”的边界
        nearest_line = numap.get_proximal_map_objects(center, self.hdmap_patch_radius, self.line_layer_names)

        # 统计 map 坐标下的边界段出现次数
        def _seg_key(p, q, quant=0.2):
            p = tuple(np.round(np.asarray(p) / quant).astype(np.int32))
            q = tuple(np.round(np.asarray(q) / quant).astype(np.int32))
            return (p, q) if p <= q else (q, p)

        seg_cnt = {}
        for ln, objs in nearest_line.items():
            for lane in objs:
                for path in [lane.left_boundary.discrete_path, lane.right_boundary.discrete_path]:
                    pts = [(float(p.x), float(p.y)) for p in path]
                    for i in range(len(pts) - 1):
                        k = _seg_key(pts[i], pts[i + 1])
                        seg_cnt[k] = seg_cnt.get(k, 0) + 1

        green_rgb = (0, 255, 0)
        green_bgr = (green_rgb[2], green_rgb[1], green_rgb[0])

        for ln, objs in nearest_line.items():
            for lane in objs:
                for path in [lane.left_boundary.discrete_path, lane.right_boundary.discrete_path]:
                    pts = [(float(p.x), float(p.y)) for p in path]
                    if len(pts) < 2:
                        continue

                    # 判定这条边界是否为 divider：多数段出现次数>=2
                    flags = []
                    for i in range(len(pts) - 1):
                        flags.append(seg_cnt.get(_seg_key(pts[i], pts[i + 1]), 0) >= 2)
                    if (sum(flags) / max(1, len(flags))) < 0.6:
                        continue  # 更像 road edge，丢掉（蓝色负责外边缘）

                    pts2 = np.asarray(pts, np.float32).T  # 2xN
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
        
        # --- 3) 红色：CROSSWALK polygon exterior
        red_rgb = (255, 0, 0)
        red_bgr = (red_rgb[2], red_rgb[1], red_rgb[0])

        for obj in nearest_poly.get(SemanticMapLayer.CROSSWALK, []):
            poly = getattr(obj, "polygon", None)
            if poly is None or poly.is_empty:
                continue

            ext_xy = np.asarray(poly.exterior.coords, np.float32)  # (N,2) global
            if ext_xy.shape[0] < 3:
                continue

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

            # intersection 结果可能是 LineString / MultiLineString / GeometryCollection
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

        boxes = info.get("gt_line")
        names = info.get('gt_names')
        names = list(names) if isinstance(names, (list, tuple, np.ndarray)) else None

        for i, box in enumerate(boxes):
            color = (0, 0, 0)
            if names is not None and i < len(names):
                color = self.default_3dbox_color_table.get(str(names[i]), color)

            corners = _box_corners_lidar_xyz(box[:7], z_is_center=False)
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
    # main api
    # ---------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        scene, idxs, fps = item["scene"], item["indices"], item["fps"]

        seq = [self.scenes[scene][i] for i in idxs]
        t0 = seq[0][self.timestamp_key]
        pts = torch.tensor([(x[self.timestamp_key] - t0 + 500) // 1000 for x in seq], dtype=torch.float32)

        result = {"fps": torch.tensor(float(fps), dtype=torch.float32), "pts": pts}

        if self.enable_sample_data:
            result["sample_data"] = seq
            result["scene"] = {"token": scene}

        # ---- images: undistort + intrinsics
        images, cam_Ks, img_sizes, dists = [], [], [], []
        for x in seq:
            paths = self._get_sensor_paths(x)
            row_imgs, row_K, row_sz, row_dist = [], [], [], []

            for cam_i, (cam_ch, p) in enumerate(zip(self.sensor_channels, paths)):
                im = _try_open_image(p)
                cam_info = self._get_cam_info(x, cam_ch)

                if im is None or cam_info is None:
                    W, H = 1920, 1080
                    row_imgs.append(Image.new("RGB", (W, H)))
                    K_pad = np.eye(4, dtype=np.float32)
                    row_K.append(K_pad)
                    row_sz.append((W, H))
                    row_dist.append(np.zeros((0,), np.float32))
                    continue

                row_imgs.append(im)

                # 原K（3x3 -> pad成4x4）
                K3 = np.asarray(cam_info["camera_intrinsics"], np.float32).reshape(3, 3)
                K_pad = np.eye(4, dtype=np.float32)
                K_pad[:3, :3] = K3
                row_K.append(K_pad)

                # 原尺寸
                row_sz.append(im.size)  # (W,H)

                # 原畸变系数（保留）
                dist = np.asarray(cam_info.get("distortion", []), np.float32).reshape(-1)
                row_dist.append(dist)

            images.append(row_imgs)
            cam_Ks.append(row_K)
            img_sizes.append(row_sz)
            dists.append(row_dist)

        result["images"] = images
        result["camera_intrinsics"] = torch.tensor(np.asarray(cam_Ks), dtype=torch.float32)
        result["image_size"] = torch.tensor(
            np.asarray([[[w, h] for (w, h) in row] for row in img_sizes], dtype=np.int64),
            dtype=torch.long
        )
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
                        try:
                            os.remove(lock)
                        except FileNotFoundError:
                            pass

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
                        try:
                            os.remove(lock)
                        except FileNotFoundError:
                            pass

        return result
