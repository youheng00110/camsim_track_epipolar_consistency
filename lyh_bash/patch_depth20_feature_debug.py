#!/usr/bin/env python3
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM')
SRC = ROOT / 'src/dwm/tools/debug_view_consistency_features.py'
DST = ROOT / 'src/dwm/tools/debug_view_consistency_features_depth20.py'

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 20.0

if not SRC.exists():
    raise FileNotFoundError(SRC)

text = SRC.read_text(encoding='utf-8')
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if DST.exists():
    backup = DST.with_name(DST.name + f'.bak_{stamp}')
    shutil.copy2(DST, backup)
    print('[backup]', backup)

marker = '\ndef main():\n'
if marker not in text:
    raise RuntimeError('Could not find def main(); refusing to guess.')

helpers = r'''

DEPTH20_MIN_M = 0.1
DEPTH20_MAX_M = 20.0


def _depth20_visible_interval(ray_b_direction, translation_b):
    """Intersect source-camera Z depth [0.1m,20m] with z_B > 0."""
    lo = float(DEPTH20_MIN_M)
    hi = float(DEPTH20_MAX_M)
    z_eps = 1e-4

    a = float(ray_b_direction[2])
    b = float(translation_b[2])
    if abs(a) < 1e-12:
        return None if b <= z_eps else (lo, hi)

    boundary = (z_eps - b) / a
    margin = max(1e-5, 1e-5 * (hi - lo))
    if a > 0.0:
        lo = max(lo, boundary + margin)
    else:
        hi = min(hi, boundary - margin)

    return None if hi <= lo else (lo, hi)


def build_depth20_epipolar_segment_mask(
    query_index,
    feature_height,
    feature_width,
    camera_intrinsics_a_norm,
    camera_intrinsics_b_norm,
    camera2referego_a,
    camera2referego_b,
    lower_half_flat,
    band_width,
):
    """Finite epipolar tube for camera-A Z depth in [0.1m,20m]."""
    ka = vc.scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_a_norm.detach().float().cpu(),
        feature_height,
        feature_width,
    )
    kb = vc.scale_normalized_intrinsics_to_feature_grid(
        camera_intrinsics_b_norm.detach().float().cpu(),
        feature_height,
        feature_width,
    )
    ta = camera2referego_a.detach().float().cpu()
    tb = camera2referego_b.detach().float().cpu()

    camera_b_from_camera_a = torch.linalg.inv(tb) @ ta
    rotation = camera_b_from_camera_a[:3, :3]
    translation = camera_b_from_camera_a[:3, 3]

    query_coord = vc.build_patch_homogeneous_coordinates(
        torch.tensor([int(query_index)], dtype=torch.long),
        feature_width,
    )[0]
    ray_a = torch.linalg.solve(ka, query_coord)
    ray_b_direction = rotation @ ray_a

    depth_interval = _depth20_visible_interval(
        ray_b_direction,
        translation,
    )
    if depth_interval is None:
        empty = torch.zeros(feature_height * feature_width, dtype=torch.bool)
        return empty, None, None

    depth_near, depth_far = depth_interval

    def project(depth):
        point_a = ray_a * float(depth)
        point_b = rotation @ point_a + translation
        projected = kb @ point_b
        if float(projected[2]) <= 1e-6:
            return None
        return projected[:2] / projected[2]

    point_near = project(depth_near)
    point_far = project(depth_far)
    if point_near is None or point_far is None:
        empty = torch.zeros(feature_height * feature_width, dtype=torch.bool)
        return empty, None, depth_interval

    indices = torch.arange(feature_height * feature_width, dtype=torch.long)
    points = vc.build_patch_homogeneous_coordinates(
        indices,
        feature_width,
    )[:, :2]

    segment = point_far - point_near
    segment_norm2 = float(torch.dot(segment, segment))
    if segment_norm2 < 1e-10:
        distances = torch.linalg.vector_norm(points - point_near[None], dim=1)
    else:
        alpha = ((points - point_near[None]) @ segment) / segment_norm2
        alpha = alpha.clamp(0.0, 1.0)
        closest = point_near[None] + alpha[:, None] * segment[None]
        distances = torch.linalg.vector_norm(points - closest, dim=1)

    positive_mask = (
        lower_half_flat.bool().cpu()
        & distances.le(float(band_width))
    )
    return positive_mask, (point_near, point_far), depth_interval


def _clip_segment_to_feature_rect(p0, p1, width, height):
    """Liang-Barsky clipping in feature-grid coordinates."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0, float(width) - x0, y0, float(height) - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
            continue
        r = qi / pi
        if pi < 0.0:
            u1 = max(u1, r)
        else:
            u2 = min(u2, r)
        if u1 > u2:
            return None
    return (
        (x0 + u1 * dx, y0 + u1 * dy),
        (x0 + u2 * dx, y0 + u2 * dy),
    )


def draw_depth20_topk(
    image,
    segment_points,
    positive_indices,
    topk_indices,
    feature_height,
    feature_width,
):
    out = image.convert('RGBA')
    draw = ImageDraw.Draw(out, 'RGBA')
    width, height = out.size
    scale_x = width / float(feature_width)
    scale_y = height / float(feature_height)

    for index in positive_indices:
        box = patch_box(index, feature_width, scale_x, scale_y)
        draw.rectangle(box, fill=(255, 215, 0, 55))
        draw.rectangle(box, outline=(255, 165, 0, 100), width=1)

    if segment_points is not None:
        clipped = _clip_segment_to_feature_rect(
            segment_points[0],
            segment_points[1],
            feature_width,
            feature_height,
        )
        if clipped is not None:
            (x0, y0), (x1, y1) = clipped
            draw.line(
                [
                    (x0 * scale_x, y0 * scale_y),
                    (x1 * scale_x, y1 * scale_y),
                ],
                fill=(0, 255, 255, 255),
                width=3,
            )

        far = segment_points[1]
        fx = float(far[0]) * scale_x
        fy = float(far[1]) * scale_y
        if 0 <= fx < width and 0 <= fy < height:
            r = 4
            draw.ellipse((fx-r, fy-r, fx+r, fy+r), fill=(0, 255, 255, 255))
            draw.text(
                (fx + 5, fy - 7),
                '20m',
                fill=(0, 0, 0, 255),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 255),
            )

    for rank, index in enumerate(topk_indices, start=1):
        box = patch_box(index, feature_width, scale_x, scale_y)
        draw.rectangle(box, outline=(0, 255, 0, 255), width=4)
        draw.text(
            (box[0] + 2, box[1] + 1),
            str(rank),
            fill=(0, 0, 0, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    return out.convert('RGB')
'''

text = text.replace(marker, helpers + marker, 1)

old_geometry = '''        fundamental = vc.build_fundamental_matrix_on_feature_grid(
            camera_intrinsics_a_norm=camera_intrinsics_norm[
                0, time_a, view_a
            ],
            camera_intrinsics_b_norm=camera_intrinsics_norm[
                0, time_b, view_b
            ],
            camera2referego_a=camera2referego[
                0, time_a, view_a
            ],
            camera2referego_b=camera2referego[
                0, time_b, view_b
            ],
            feature_height=feature_height,
            feature_width=feature_width,
        ).detach().cpu()

        query_coord = vc.build_patch_homogeneous_coordinates(
            torch.tensor([query_index], dtype=torch.long),
            feature_width,
        )[0]

        line = fundamental @ query_coord
        denominator = torch.sqrt(
            line[0].square() + line[1].square()
        ).clamp(min=1e-6)
        distances = torch.abs(
            all_coordinates @ line
        ) / denominator

        positive_mask = (
            lower_half_flat
            & distances.le(band_width)
        )
        positive_indices_tensor = positive_mask.nonzero(
            as_tuple=False
        ).flatten()
'''
new_geometry = '''        (
            positive_mask,
            segment_points,
            valid_depth_range,
        ) = build_depth20_epipolar_segment_mask(
            query_index=query_index,
            feature_height=feature_height,
            feature_width=feature_width,
            camera_intrinsics_a_norm=camera_intrinsics_norm[
                0, time_a, view_a
            ],
            camera_intrinsics_b_norm=camera_intrinsics_norm[
                0, time_b, view_b
            ],
            camera2referego_a=camera2referego[
                0, time_a, view_a
            ],
            camera2referego_b=camera2referego[
                0, time_b, view_b
            ],
            lower_half_flat=lower_half_flat,
            band_width=band_width,
        )
        positive_indices_tensor = positive_mask.nonzero(
            as_tuple=False
        ).flatten()
'''
if old_geometry not in text:
    raise RuntimeError('Could not find old infinite-line geometry block; refusing to guess.')
text = text.replace(old_geometry, new_geometry, 1)

old_draw = '''        panel_epi = draw_epipolar_topk(
            image_b,
            line,
            positive_indices,
            top_indices,
            feature_height,
            feature_width,
        )
'''
new_draw = '''        panel_epi = draw_depth20_topk(
            image_b,
            segment_points,
            positive_indices,
            top_indices,
            feature_height,
            feature_width,
        )
'''
if old_draw not in text:
    raise RuntimeError('Could not find old epipolar drawing call; refusing to guess.')
text = text.replace(old_draw, new_draw, 1)

old_label = '''                f"B epipolar band + raw-feature Top-{len(top_indices)}"
'''
new_label = '''                f"B depth 0.1-20m segment + raw-feature Top-{len(top_indices)}"
'''
if old_label not in text:
    raise RuntimeError('Could not find old right-panel label; refusing to guess.')
text = text.replace(old_label, new_label, 1)

old_record = '''            "positive_patch_count": len(positive_indices),
            "topk_patch_indices": top_indices,
'''
new_record = '''            "positive_patch_count": len(positive_indices),
            "depth_range_m": [DEPTH20_MIN_M, DEPTH20_MAX_M],
            "valid_depth_range_m": (
                None
                if valid_depth_range is None
                else [float(valid_depth_range[0]), float(valid_depth_range[1])]
            ),
            "segment_near_xy_feature_grid": (
                None
                if segment_points is None
                else [float(segment_points[0][0]), float(segment_points[0][1])]
            ),
            "segment_far_xy_feature_grid": (
                None
                if segment_points is None
                else [float(segment_points[1][0]), float(segment_points[1][1])]
            ),
            "topk_patch_indices": top_indices,
'''
if old_record not in text:
    raise RuntimeError('Could not find metadata record insertion point; refusing to guess.')
text = text.replace(old_record, new_record, 1)

text = text.replace('/ "raw_feature_debug"', '/ "raw_feature_depth20_debug"', 1)
text = text.replace(
    'f"t{time_b:02d}_v{view_b}_raw1536.png"',
    'f"t{time_b:02d}_v{view_b}_depth20m_raw1536.png"',
    1,
)

old_meta = '''        "requested_sigma": args.sigma,
        "actual_sigma": sigma_value,
'''
new_meta = '''        "requested_sigma": args.sigma,
        "actual_sigma": sigma_value,
        "depth_range_m": [DEPTH20_MIN_M, DEPTH20_MAX_M],
'''
if old_meta not in text:
    raise RuntimeError('Could not find global metadata insertion point; refusing to guess.')
text = text.replace(old_meta, new_meta, 1)

old_console = '''    print("=== RAW FEATURE CAPTURED ===", flush=True)
'''
new_console = '''    print("depth_range_m  =", (DEPTH20_MIN_M, DEPTH20_MAX_M), flush=True)
    print("=== RAW FEATURE CAPTURED ===", flush=True)
'''
if old_console not in text:
    raise RuntimeError('Could not find console insertion point; refusing to guess.')
text = text.replace(old_console, new_console, 1)

DST.write_text(text, encoding='utf-8')
subprocess.run([sys.executable, '-m', 'py_compile', str(DST)], check=True)

print('[created]', DST)
print('[source ]', SRC)
print('[depth  ] camera-A Z depth: 0.1m <= d <= 20m')
print('[note   ] formal model/loss/config were NOT modified')
