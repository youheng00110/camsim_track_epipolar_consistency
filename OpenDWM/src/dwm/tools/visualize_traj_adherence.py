import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from dwm.metrics.stflow import STFlowEvaluator


def create_parser():
    parser = argparse.ArgumentParser(
        description="Visualize Temporal-Epipolar Trajectory Adherence."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--view", type=int, default=0)

    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--max-matches", type=int, default=256)
    parser.add_argument("--loftr-confidence", type=float, default=0.1)
    parser.add_argument("--max-draw-matches", type=int, default=120)
    parser.add_argument("--error-vmax", type=float, default=10.0)
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


def score_to_color(value, vmax):
    x = float(np.clip(value / max(vmax, 1e-6), 0.0, 1.0))
    r = int(255 * x)
    g = int(255 * (1.0 - x))
    return (r, g, 0)


def resolve_view_index(camera_names, camera_name, view_index):
    if camera_name is not None:
        if camera_name not in camera_names:
            raise KeyError(f"camera={camera_name} not in camera_names={camera_names}")
        return camera_names.index(camera_name)
    return view_index


def get_ego_xy(ego_transform):
    if ego_transform is None:
        return None
    value = ego_transform.detach().cpu().numpy()
    return np.array([value[0, 3], value[1, 3]], dtype=np.float32)


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


def draw_temporal_traj_matches(
    image0,
    image1,
    points0,
    points1,
    errors,
    vmax,
    max_draw,
    title_text,
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
    draw.text((10, 8), title_text, fill=(255, 255, 255))
    draw.text((8, title_h + 8), "camera @ t", fill=(255, 255, 255))
    draw.text((w0 + gap + 8, title_h + 8), "camera @ t+dt", fill=(255, 255, 255))

    if points0.shape[0] == 0:
        draw.text((10, title_h + 32), "No temporal matches", fill=(255, 0, 0))
        return canvas

    pts0 = points0.detach().cpu().numpy()
    pts1 = points1.detach().cpu().numpy()
    vals = errors.detach().cpu().numpy()

    if pts0.shape[0] > max_draw:
        indices = np.linspace(0, pts0.shape[0] - 1, max_draw).astype(int)
    else:
        indices = np.arange(pts0.shape[0])

    for idx in indices:
        p0 = (float(pts0[idx, 0]), float(pts0[idx, 1]) + title_h)
        p1 = (float(pts1[idx, 0]) + w0 + gap, float(pts1[idx, 1]) + title_h)
        color = score_to_color(vals[idx], vmax)

        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=2)
        r = 3
        draw.ellipse((p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r), outline=color, width=2)
        draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), outline=color, width=2)

    return canvas


def draw_ego_path(positions, segment_errors, output_path, vmax):
    valid_positions = [p for p in positions if p is not None]
    if len(valid_positions) < 2:
        canvas = Image.new("RGB", (900, 700), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), "No valid T_ego_to_world trajectory", fill=(255, 0, 0))
        canvas.save(output_path)
        return

    pts = np.stack(valid_positions, axis=0)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    scale_xy = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1e-6)

    width = 900
    height = 700
    margin = 80
    scale = (min(width, height) - 2 * margin) / scale_xy

    canvas = Image.new("RGB", (width, height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    def project(xy):
        x = (xy[0] - center[0]) * scale + width * 0.5
        y = height * 0.5 - (xy[1] - center[1]) * scale
        return (float(x), float(y))

    draw.text((20, 20), "Target ego trajectory; segment color = Traj-Epi error", fill=(255, 255, 255))
    draw.text((20, 45), "green=low error, red=high error", fill=(200, 200, 200))

    for index in range(len(positions) - 1):
        if positions[index] is None or positions[index + 1] is None:
            continue

        p0 = project(positions[index])
        p1 = project(positions[index + 1])
        error_value = segment_errors.get(index, None)

        if error_value is None or np.isnan(error_value):
            color = (160, 160, 160)
        else:
            color = score_to_color(error_value, vmax)

        draw_arrow(draw, p0, p1, color=color, width=4, head_len=10)

    start = project(valid_positions[0])
    end = project(valid_positions[-1])
    draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill=(0, 255, 0))
    draw.ellipse((end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7), fill=(255, 0, 0))
    draw.text((start[0] + 8, start[1] + 8), "start", fill=(0, 255, 0))
    draw.text((end[0] + 8, end[1] + 8), "end", fill=(255, 0, 0))

    canvas.save(output_path)


def compute_traj_errors_for_video(evaluator, data, view_index, frame_stride, min_matches):
    images = data["images"]
    masks = data["masks"]
    intrinsics = data["intrinsics"]
    transforms = data["transforms"]
    ego_transforms = data.get("ego_transforms", [])

    frame_count = len(images)
    segment_errors = {}
    segment_inlier2 = {}
    segment_inlier4 = {}
    segment_match_count = {}

    for time0 in range(0, frame_count - frame_stride, frame_stride):
        time1 = time0 + frame_stride

        if len(ego_transforms) <= time1:
            continue
        if ego_transforms[time0] is None or ego_transforms[time1] is None:
            continue

        image0 = images[time0][view_index]
        image1 = images[time1][view_index]

        points0, points1, _ = evaluator.run_loftr(image0, image1)
        points0, points1 = evaluator.filter_matched_points(
            points0,
            points1,
            masks[time0][view_index],
            masks[time1][view_index],
        )

        if points0.shape[0] < min_matches:
            continue

        T_cam_to_world0 = ego_transforms[time0] @ transforms[time0][view_index]
        T_cam_to_world1 = ego_transforms[time1] @ transforms[time1][view_index]

        F_traj = evaluator.fundamental_from_camera_to_world(
            intrinsics[time0][view_index],
            T_cam_to_world0,
            intrinsics[time1][view_index],
            T_cam_to_world1,
        )

        errors = evaluator.sampson_error_px(points0, points1, F_traj)
        segment_errors[time0] = float(torch.median(errors).item())
        segment_inlier2[time0] = float((errors < 2.0).float().mean().item())
        segment_inlier4[time0] = float((errors < 4.0).float().mean().item())
        segment_match_count[time0] = int(points0.shape[0])

    return segment_errors, segment_inlier2, segment_inlier4, segment_match_count


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
    )

    data = evaluator.load_frame_data(manifest_item, manifest_dir)
    camera_names = data["camera_names"]
    view_index = resolve_view_index(camera_names, args.camera, args.view)
    camera_name = camera_names[view_index]

    ego_transforms = data.get("ego_transforms", [])
    positions = [get_ego_xy(ego_transform) for ego_transform in ego_transforms]

    segment_errors, segment_inlier2, segment_inlier4, segment_match_count = compute_traj_errors_for_video(
        evaluator,
        data,
        view_index,
        args.frame_stride,
        args.min_matches,
    )

    draw_ego_path(
        positions,
        segment_errors,
        os.path.join(args.output_dir, "target_ego_path_colored_by_traj_epi.png"),
        args.error_vmax,
    )

    t0 = args.time_index
    t1 = t0 + args.frame_stride
    if t1 >= len(data["images"]):
        raise ValueError(
            f"time_index={t0} with frame_stride={args.frame_stride} exceeds video length={len(data['images'])}"
        )

    image0 = data["images"][t0][view_index]
    image1 = data["images"][t1][view_index]

    points0, points1, _ = evaluator.run_loftr(image0, image1)
    points0, points1 = evaluator.filter_matched_points(
        points0,
        points1,
        data["masks"][t0][view_index],
        data["masks"][t1][view_index],
    )

    if (
        len(ego_transforms) > t1
        and ego_transforms[t0] is not None
        and ego_transforms[t1] is not None
        and points0.shape[0] >= args.min_matches
    ):
        T_cam_to_world0 = ego_transforms[t0] @ data["transforms"][t0][view_index]
        T_cam_to_world1 = ego_transforms[t1] @ data["transforms"][t1][view_index]

        F_traj = evaluator.fundamental_from_camera_to_world(
            data["intrinsics"][t0][view_index],
            T_cam_to_world0,
            data["intrinsics"][t1][view_index],
            T_cam_to_world1,
        )
        errors = evaluator.sampson_error_px(points0, points1, F_traj)
        traj_median = float(torch.median(errors).item())
        traj_inlier2 = float((errors < 2.0).float().mean().item())
        traj_inlier4 = float((errors < 4.0).float().mean().item())
    else:
        errors = torch.empty((0,), dtype=torch.float32, device=evaluator.device)
        traj_median = float("nan")
        traj_inlier2 = float("nan")
        traj_inlier4 = float("nan")

    temporal_vis = draw_temporal_traj_matches(
        image0,
        image1,
        points0,
        points1,
        errors,
        args.error_vmax,
        args.max_draw_matches,
        (
            f"Traj-Epi temporal matches {camera_name}, t={t0}->t={t1}, "
            f"median={traj_median:.3f}, inlier@4={traj_inlier4:.3f}"
        ),
    )
    temporal_vis.save(
        os.path.join(
            args.output_dir,
            f"traj_temporal_{camera_name}_t{t0:03d}_to_t{t1:03d}.png",
        )
    )

    summary = {
        "video_id": manifest_item.get("video_id", ""),
        "dataset_name": manifest_item.get("dataset_name", ""),
        "camera": camera_name,
        "time_index": t0,
        "frame_stride": args.frame_stride,
        "traj_epi_px_median": traj_median,
        "traj_inlier2": traj_inlier2,
        "traj_inlier4": traj_inlier4,
        "num_matches": int(points0.shape[0]),
        "path_segments": {
            str(k): {
                "traj_epi_px": segment_errors[k],
                "traj_inlier2": segment_inlier2.get(k, float("nan")),
                "traj_inlier4": segment_inlier4.get(k, float("nan")),
                "match_count": segment_match_count.get(k, 0),
            }
            for k in sorted(segment_errors.keys())
        },
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[traj-vis] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
