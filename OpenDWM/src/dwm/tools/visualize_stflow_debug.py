import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

try:
    from torchvision.utils import flow_to_image
except Exception:
    flow_to_image = None

from dwm.metrics.stflow import STFlowEvaluator


def create_parser():
    parser = argparse.ArgumentParser(
        description="Visualize ST-Flow temporal flow, cross-view matches, cycle matches, and cross-view flow."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--camera0", type=str, default=None)
    parser.add_argument("--camera1", type=str, default=None)
    parser.add_argument("--view0", type=int, default=None)
    parser.add_argument("--view1", type=int, default=None)

    parser.add_argument("--camera-temporal", type=str, default=None)
    parser.add_argument("--view-temporal", type=int, default=None)

    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--loftr-confidence", type=float, default=0.1)
    parser.add_argument("--pair-policy", type=str, default="dataset", choices=["dataset", "ring", "waymo"])
    parser.add_argument("--max-draw-matches", type=int, default=80)
    return parser


def load_manifest_items(manifest_path):
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def tensor_image_to_pil(image_tensor):
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.detach().cpu().clamp(0, 1)
    return to_pil_image(image_tensor)


def save_tensor_image(image_tensor, path):
    tensor_image_to_pil(image_tensor).save(path)


def flow_to_pil_image(flow_tensor):
    if flow_to_image is None:
        raise RuntimeError("torchvision.utils.flow_to_image is not available in this torchvision version.")

    flow_img = flow_to_image(flow_tensor.detach().cpu())
    if flow_img.ndim == 4:
        flow_img = flow_img[0]
    return to_pil_image(flow_img)


def error_map_to_pil(error_map):
    error = error_map.detach().cpu().numpy()
    error = error - error.min()
    error = error / (error.max() + 1e-8)
    error = (error * 255).astype(np.uint8)
    return Image.fromarray(error, mode="L")


def resolve_view_index(camera_names, camera_name=None, view_index=None):
    if camera_name is not None:
        if camera_name not in camera_names:
            raise KeyError(f"camera_name={camera_name} not in camera_names={camera_names}")
        return camera_names.index(camera_name)
    if view_index is not None:
        return view_index
    return None


def score_to_color(value, vmax):
    if vmax <= 0:
        return (0, 255, 0)
    x = float(np.clip(value / vmax, 0.0, 1.0))
    r = int(255 * x)
    g = int(255 * (1.0 - x))
    return (r, g, 0)


def compute_fundamental(evaluator, K0, T0, K1, T1):
    if hasattr(evaluator, "fundamental_from_transforms"):
        return evaluator.fundamental_from_transforms(K0, T0, K1, T1)

    if hasattr(evaluator, "fundamental_from_camera_to_ego"):
        return evaluator.fundamental_from_camera_to_ego(K0, T0, K1, T1)

    raise AttributeError("STFlowEvaluator has no fundamental matrix function.")


def select_default_pair(evaluator, manifest_item, data):
    camera_names = data["camera_names"]
    transforms_t0 = data["transforms"][0]

    if hasattr(evaluator, "select_camera_pairs"):
        pairs = evaluator.select_camera_pairs(manifest_item, transforms_t0, camera_names)
    else:
        pairs = evaluator.camera_ring_pairs(transforms_t0, camera_names)

    if len(pairs) == 0:
        raise RuntimeError("No camera pairs available for visualization.")

    return pairs[0]


def valid_next_points(evaluator, points0, points1, points0_next, points1_next, mask0_next, mask1_next):
    valid0 = evaluator.points_inside_mask(points0_next, mask0_next)
    valid1 = evaluator.points_inside_mask(points1_next, mask1_next)
    valid = valid0 & valid1

    return points0[valid], points1[valid], points0_next[valid], points1_next[valid], valid


def draw_matches(image0, image1, points0, points1, values=None, max_draw=80, title_text=None):
    img0 = tensor_image_to_pil(image0)
    img1 = tensor_image_to_pil(image1)

    w0, h0 = img0.size
    canvas = Image.new("RGB", (w0 + img1.size[0], max(h0, img1.size[1]) + 30), (0, 0, 0))
    canvas.paste(img0, (0, 30))
    canvas.paste(img1, (w0, 30))

    draw = ImageDraw.Draw(canvas)

    if title_text is not None:
        draw.text((10, 5), title_text, fill=(255, 255, 255))

    if points0.shape[0] == 0:
        draw.text((10, h0 + 5), "No matches", fill=(255, 0, 0))
        return canvas

    pts0 = points0.detach().cpu().numpy()
    pts1 = points1.detach().cpu().numpy()

    if values is not None and values.numel() > 0:
        vals = values.detach().cpu().numpy()
        vmax = max(float(np.percentile(vals, 90)), 1e-6)
    else:
        vals = None
        vmax = 1.0

    if pts0.shape[0] > max_draw:
        indices = np.linspace(0, pts0.shape[0] - 1, max_draw).astype(int)
    else:
        indices = np.arange(pts0.shape[0])

    for idx in indices:
        x0, y0 = pts0[idx]
        x1, y1 = pts1[idx]

        color = score_to_color(vals[idx], vmax) if vals is not None else (0, 255, 0)

        draw.line((x0, y0 + 30, x1 + w0, y1 + 30), fill=color, width=2)
        r = 3
        draw.ellipse((x0 - r, y0 + 30 - r, x0 + r, y0 + 30 + r), outline=color, width=2)
        draw.ellipse((x1 + w0 - r, y1 + 30 - r, x1 + w0 + r, y1 + 30 + r), outline=color, width=2)

    return canvas


def draw_arrow(draw, p0, p1, color, width=2, head_len=8):
    x0, y0 = p0
    x1, y1 = p1

    draw.line((x0, y0, x1, y1), fill=color, width=width)

    dx = x1 - x0
    dy = y1 - y0
    norm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-6
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux

    hx = x1 - head_len * ux
    hy = y1 - head_len * uy
    left = (hx + 0.5 * head_len * px, hy + 0.5 * head_len * py)
    right = (hx - 0.5 * head_len * px, hy - 0.5 * head_len * py)

    draw.polygon([(x1, y1), left, right], fill=color)


def draw_cross_view_flow(
    image0,
    image1,
    points0,
    points1,
    delta0,
    delta1,
    values=None,
    max_draw=80,
    title_text=None,
):
    img0 = tensor_image_to_pil(image0)
    img1 = tensor_image_to_pil(image1)

    w0, h0 = img0.size
    gap = 20
    title_h = 36
    canvas = Image.new("RGB", (w0 + img1.size[0] + gap, max(h0, img1.size[1]) + title_h), (0, 0, 0))
    canvas.paste(img0, (0, title_h))
    canvas.paste(img1, (w0 + gap, title_h))

    draw = ImageDraw.Draw(canvas)

    if title_text is not None:
        draw.text((10, 8), title_text, fill=(255, 255, 255))

    draw.text((8, title_h + 8), "cam0 @ t", fill=(255, 255, 255))
    draw.text((w0 + gap + 8, title_h + 8), "cam1 @ t", fill=(255, 255, 255))

    if points0.shape[0] == 0:
        draw.text((10, title_h + 32), "No valid cross-view flow tracks", fill=(255, 0, 0))
        return canvas

    pts0 = points0.detach().cpu().numpy()
    pts1 = points1.detach().cpu().numpy()
    d0 = delta0.detach().cpu().numpy()
    d1 = delta1.detach().cpu().numpy()

    if values is not None and values.numel() > 0:
        vals = values.detach().cpu().numpy()
        vmax = max(float(np.percentile(vals, 90)), 1e-6)
    else:
        vals = None
        vmax = 1.0

    if pts0.shape[0] > max_draw:
        indices = np.linspace(0, pts0.shape[0] - 1, max_draw).astype(int)
    else:
        indices = np.arange(pts0.shape[0])

    for idx in indices:
        p0 = (float(pts0[idx, 0]), float(pts0[idx, 1]) + title_h)
        q0 = (float(pts0[idx, 0] + d0[idx, 0]), float(pts0[idx, 1] + d0[idx, 1]) + title_h)

        p1 = (float(pts1[idx, 0]) + w0 + gap, float(pts1[idx, 1]) + title_h)
        q1 = (float(pts1[idx, 0] + d1[idx, 0]) + w0 + gap, float(pts1[idx, 1] + d1[idx, 1]) + title_h)

        color = score_to_color(vals[idx], vmax) if vals is not None else (0, 255, 0)

        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=1)
        draw_arrow(draw, p0, q0, color=color, width=2, head_len=8)
        draw_arrow(draw, p1, q1, color=color, width=2, head_len=8)

        r = 3
        draw.ellipse((p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r), outline=color, width=2)
        draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), outline=color, width=2)

    return canvas


def draw_stflow_graph(
    image0_t,
    image1_t,
    image0_next,
    image1_next,
    points0_t,
    points1_t,
    points0_next,
    points1_next,
    values=None,
    max_draw=80,
    title_text=None,
):
    img00 = tensor_image_to_pil(image0_t)
    img01 = tensor_image_to_pil(image1_t)
    img10 = tensor_image_to_pil(image0_next)
    img11 = tensor_image_to_pil(image1_next)

    w, h = img00.size
    title_h = 38
    gap = 18

    canvas = Image.new("RGB", (w * 2 + gap, h * 2 + gap + title_h), (0, 0, 0))

    x00, y00 = 0, title_h
    x01, y01 = w + gap, title_h
    x10, y10 = 0, title_h + h + gap
    x11, y11 = w + gap, title_h + h + gap

    canvas.paste(img00, (x00, y00))
    canvas.paste(img01, (x01, y01))
    canvas.paste(img10, (x10, y10))
    canvas.paste(img11, (x11, y11))

    draw = ImageDraw.Draw(canvas)

    if title_text is not None:
        draw.text((10, 8), title_text, fill=(255, 255, 255))

    draw.text((x00 + 8, y00 + 8), "cam0 @ t", fill=(255, 255, 255))
    draw.text((x01 + 8, y01 + 8), "cam1 @ t", fill=(255, 255, 255))
    draw.text((x10 + 8, y10 + 8), "cam0 @ t+dt", fill=(255, 255, 255))
    draw.text((x11 + 8, y11 + 8), "cam1 @ t+dt", fill=(255, 255, 255))

    if points0_t.shape[0] == 0:
        draw.text((10, h * 2 + gap + title_h - 24), "No valid ST-flow tracks", fill=(255, 0, 0))
        return canvas

    pts0 = points0_t.detach().cpu().numpy()
    pts1 = points1_t.detach().cpu().numpy()
    pts0n = points0_next.detach().cpu().numpy()
    pts1n = points1_next.detach().cpu().numpy()

    if values is not None and values.numel() > 0:
        vals = values.detach().cpu().numpy()
        vmax = max(float(np.percentile(vals, 90)), 1e-6)
    else:
        vals = None
        vmax = 1.0

    if pts0.shape[0] > max_draw:
        indices = np.linspace(0, pts0.shape[0] - 1, max_draw).astype(int)
    else:
        indices = np.arange(pts0.shape[0])

    for idx in indices:
        p00 = (float(pts0[idx, 0]) + x00, float(pts0[idx, 1]) + y00)
        p01 = (float(pts1[idx, 0]) + x01, float(pts1[idx, 1]) + y01)
        p10 = (float(pts0n[idx, 0]) + x10, float(pts0n[idx, 1]) + y10)
        p11 = (float(pts1n[idx, 0]) + x11, float(pts1n[idx, 1]) + y11)

        color = score_to_color(vals[idx], vmax) if vals is not None else (0, 255, 0)

        draw.line((p00[0], p00[1], p01[0], p01[1]), fill=color, width=1)
        draw.line((p00[0], p00[1], p10[0], p10[1]), fill=color, width=2)
        draw.line((p01[0], p01[1], p11[0], p11[1]), fill=color, width=2)
        draw.line((p10[0], p10[1], p11[0], p11[1]), fill=color, width=1)

        r = 3
        for px, py in [p00, p01, p10, p11]:
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)

    return canvas


def main():
    args = create_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_items = load_manifest_items(args.manifest)
    manifest_item = manifest_items[args.video_index]
    manifest_dir = os.path.dirname(args.manifest)

    evaluator = STFlowEvaluator(
        device=args.device,
        frame_stride=args.frame_stride,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        loftr_confidence=args.loftr_confidence,
        pair_policy=args.pair_policy,
    )

    data = evaluator.load_frame_data(manifest_item, manifest_dir)
    camera_names = data["camera_names"]

    t0 = args.time_index
    t1 = t0 + args.frame_stride
    if t1 >= len(data["images"]):
        raise ValueError(
            f"time_index={t0} with frame_stride={args.frame_stride} exceeds video length={len(data['images'])}"
        )

    view0 = resolve_view_index(camera_names, args.camera0, args.view0)
    view1 = resolve_view_index(camera_names, args.camera1, args.view1)

    if view0 is None or view1 is None:
        default_pair = select_default_pair(evaluator, manifest_item, data)
        if view0 is None:
            view0 = default_pair[0]
        if view1 is None:
            view1 = default_pair[1]

    cam0 = camera_names[view0]
    cam1 = camera_names[view1]

    view_temporal = resolve_view_index(camera_names, args.camera_temporal, args.view_temporal)
    if view_temporal is None:
        view_temporal = view0
    cam_temporal = camera_names[view_temporal]

    image_t = data["images"][t0][view_temporal]
    image_t1 = data["images"][t1][view_temporal]
    mask_t = data["masks"][t0][view_temporal]
    mask_t1 = data["masks"][t1][view_temporal]

    flow = evaluator.run_raft(image_t, image_t1)
    warped, inside = evaluator.warp_image(image_t, flow)
    valid = inside & mask_t & mask_t1
    error_map = torch.abs(warped - image_t1).mean(dim=1)[0]
    temporal_l1 = error_map[valid].mean().item() if valid.sum().item() > 0 else float("nan")

    save_tensor_image(image_t, os.path.join(args.output_dir, f"temporal_{cam_temporal}_t{t0:03d}.png"))
    save_tensor_image(image_t1, os.path.join(args.output_dir, f"temporal_{cam_temporal}_t{t1:03d}.png"))
    flow_to_pil_image(flow).save(os.path.join(args.output_dir, f"temporal_{cam_temporal}_flow_t{t0:03d}_to_t{t1:03d}.png"))
    save_tensor_image(warped, os.path.join(args.output_dir, f"temporal_{cam_temporal}_warp_t{t0:03d}_to_t{t1:03d}.png"))
    error_map_to_pil(error_map).save(os.path.join(args.output_dir, f"temporal_{cam_temporal}_warp_error_t{t0:03d}_to_t{t1:03d}.png"))

    image0 = data["images"][t0][view0]
    image1 = data["images"][t0][view1]
    mask0 = data["masks"][t0][view0]
    mask1 = data["masks"][t0][view1]

    points0, points1, _ = evaluator.run_loftr(image0, image1)
    points0, points1 = evaluator.filter_matched_points(points0, points1, mask0, mask1)

    F_cross = compute_fundamental(
        evaluator,
        data["intrinsics"][t0][view0],
        data["transforms"][t0][view0],
        data["intrinsics"][t0][view1],
        data["transforms"][t0][view1],
    )
    cross_epi = evaluator.sampson_error_px(points0, points1, F_cross)
    cross_median = torch.median(cross_epi).item() if points0.shape[0] > 0 else float("nan")

    cross_vis = draw_matches(
        image0,
        image1,
        points0,
        points1,
        values=cross_epi,
        max_draw=args.max_draw_matches,
        title_text=f"Cross-view {cam0} vs {cam1} @ t={t0}, median epi={cross_median:.3f}",
    )
    cross_vis.save(os.path.join(args.output_dir, f"cross_{cam0}__{cam1}_t{t0:03d}.png"))

    image0_next = data["images"][t1][view0]
    image1_next = data["images"][t1][view1]
    mask0_next = data["masks"][t1][view0]
    mask1_next = data["masks"][t1][view1]

    flow0 = evaluator.run_raft(data["images"][t0][view0], image0_next)
    flow1 = evaluator.run_raft(data["images"][t0][view1], image1_next)

    delta0 = evaluator.sample_flow_at_points(flow0, points0)
    delta1 = evaluator.sample_flow_at_points(flow1, points1)

    points0_next_all = points0 + delta0
    points1_next_all = points1 + delta1
    points0_track, points1_track, points0_next, points1_next, valid_cycle = valid_next_points(
        evaluator,
        points0,
        points1,
        points0_next_all,
        points1_next_all,
        mask0_next,
        mask1_next,
    )
    delta0_track = delta0[valid_cycle]
    delta1_track = delta1[valid_cycle]

    F_cycle = compute_fundamental(
        evaluator,
        data["intrinsics"][t1][view0],
        data["transforms"][t1][view0],
        data["intrinsics"][t1][view1],
        data["transforms"][t1][view1],
    )
    cycle_epi = evaluator.sampson_error_px(points0_next, points1_next, F_cycle)
    cycle_median = torch.median(cycle_epi).item() if points0_next.shape[0] > 0 else float("nan")

    cycle_vis = draw_matches(
        image0_next,
        image1_next,
        points0_next,
        points1_next,
        values=cycle_epi,
        max_draw=args.max_draw_matches,
        title_text=f"Cycle {cam0} vs {cam1} @ t={t1}, median epi={cycle_median:.3f}",
    )
    cycle_vis.save(os.path.join(args.output_dir, f"cycle_{cam0}__{cam1}_t{t1:03d}.png"))

    cross_flow_vis = draw_cross_view_flow(
        image0,
        image1,
        points0_track,
        points1_track,
        delta0_track,
        delta1_track,
        values=cycle_epi,
        max_draw=args.max_draw_matches,
        title_text=(
            f"Cross-view flow {cam0}<->{cam1}, t={t0}->t={t1}, "
            f"cycle median={cycle_median:.3f}"
        ),
    )
    cross_flow_vis.save(
        os.path.join(
            args.output_dir,
            f"cross_flow_{cam0}__{cam1}_t{t0:03d}_to_t{t1:03d}.png",
        )
    )

    st_graph = draw_stflow_graph(
        image0,
        image1,
        image0_next,
        image1_next,
        points0_track,
        points1_track,
        points0_next,
        points1_next,
        values=cycle_epi,
        max_draw=args.max_draw_matches,
        title_text=(
            f"ST-Flow graph {cam0}<->{cam1}, t={t0}->t={t1}, "
            f"cycle median={cycle_median:.3f}"
        ),
    )
    st_graph.save(
        os.path.join(
            args.output_dir,
            f"st_graph_{cam0}__{cam1}_t{t0:03d}_to_t{t1:03d}.png",
        )
    )

    summary = {
        "video_id": manifest_item.get("video_id", ""),
        "dataset_name": manifest_item.get("dataset_name", ""),
        "time_index": t0,
        "frame_stride": args.frame_stride,
        "temporal_camera": cam_temporal,
        "cross_camera_pair": [cam0, cam1],
        "temporal_l1": float(temporal_l1),
        "cross_epi_px_median": float(cross_median),
        "cycle_epi_px_median": float(cycle_median),
        "num_cross_matches": int(points0.shape[0]),
        "num_cycle_matches": int(points0_next.shape[0]),
        "num_st_tracks": int(points0_track.shape[0]),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[vis] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
