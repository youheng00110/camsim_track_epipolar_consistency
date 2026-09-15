#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import dwm.common
from dwm.models.urope.urope import _invert_se3, _invert_intrinsics
from dwm.models.crossview_temporal_dit_PLUCKER_TVROW import (
    build_tv_view_index_from_crossview_mask,
)

SLOT_NAMES = ("left", "self", "right")


def pct(num, den):
    return float("nan") if den == 0 else 100.0 * num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-scale", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--min-depth", type=float, default=2.0)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--depth-count", type=int, default=6)
    args = p.parse_args()

    device = torch.device(args.device)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("generator_seed", 0))
    torch.manual_seed(seed)

    # Dataset-only diagnostic:
    # initialize filesystem/global states needed by datasets, but skip
    # distributed/FSDP states such as device_mesh.
    for key, value in config.get("global_state", {}).items():
        class_name = value.get("_class_name", "") if isinstance(value, dict) else ""

        if key == "device_mesh" or class_name.startswith("torch.distributed."):
            print(
                f"[urope-z] skip distributed global_state: "
                f"{key} ({class_name})",
                flush=True,
            )
            continue

        print(
            f"[urope-z] init global_state: {key} ({class_name})",
            flush=True,
        )
        dwm.common.global_state[key] = (
            dwm.common.create_instance_from_config(value)
        )

    dataset = dwm.common.create_instance_from_config(config["training_dataset"])
    loader_kwargs = dwm.common.instantiate_config(config["training_dataloader"])
    loader_kwargs["batch_size"] = 1
    loader_kwargs["num_workers"] = 0
    loader_kwargs.pop("prefetch_factor", None)
    loader_kwargs.pop("persistent_workers", None)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        **loader_kwargs,
        shuffle=True,
        generator=g,
    )

    depth_step = (args.max_depth - args.min_depth) / args.depth_count
    depths = args.min_depth + depth_step * torch.arange(
        args.depth_count, device=device, dtype=torch.float32
    )

    total = torch.zeros(3, args.depth_count, dtype=torch.long)
    neg = torch.zeros_like(total)
    pos = torch.zeros_like(total)
    pos_in = torch.zeros_like(total)
    neg_abs_in = torch.zeros_like(total)

    self_err_sum = 0.0
    self_err_count = 0
    self_err_max = 0.0
    left_self = 0
    right_self = 0
    view_target_count = 0
    processed = 0
    shape_line = None
    fps_seen = []

    with torch.no_grad():
        for batch in loader:
            if processed >= args.batches:
                break
            processed += 1
            print(f"[urope-z] batch {processed}/{args.batches}", flush=True)

            required = (
                "camera_intrinsics", "camera_transforms",
                "image_size", "crossview_mask",
            )
            missing = [k for k in required if k not in batch]
            if missing:
                raise KeyError(f"missing batch keys: {missing}")

            cam = batch["camera_transforms"].to(device, torch.float32)
            K0 = batch["camera_intrinsics"].to(device, torch.float32)
            image_size = batch["image_size"].to(device, torch.float32)
            B, T, V = cam.shape[:3]

            if "ego_transforms" in batch:
                ego = batch["ego_transforms"].to(device, torch.float32)
                ego = ego[:, :, -V:, ...]
            else:
                ego = torch.eye(4, device=device).view(1,1,1,4,4).expand(B,T,V,4,4)

            # Exactly the pipeline semantics:
            # camera -> ego -> world -> reference ego.
            camera2world = ego @ cam
            camera2ref = (
                torch.linalg.inv(ego[:, 0, 0])[:, None, None]
                @ camera2world
            )

            K_norm = K0.clone()
            K_norm[..., 0, 0] /= image_size[..., 0]
            K_norm[..., 1, 1] /= image_size[..., 1]
            K_norm[..., 0, 2] /= image_size[..., 0]
            K_norm[..., 1, 2] /= image_size[..., 1]

            if "vae_images" not in batch:
                raise KeyError("vae_images missing; cannot infer actual resized image size")
            image_h = int(batch["vae_images"].shape[-2])
            image_w = int(batch["vae_images"].shape[-1])
            divisor = args.vae_scale * args.patch_size
            if image_h % divisor or image_w % divisor:
                raise ValueError(
                    f"image={image_h}x{image_w} not divisible by {divisor}"
                )
            H = image_h // divisor
            W = image_w // divisor
            shape_line = f"image={image_h}x{image_w}, URoPE patch={H}x{W}"

            K = K_norm.clone()
            K[..., 0, 0] *= W
            K[..., 1, 1] *= H
            K[..., 0, 2] *= W
            K[..., 1, 2] *= H

            viewmats = torch.linalg.inv(camera2ref)
            view_index, _ = build_tv_view_index_from_crossview_mask(
                batch["crossview_mask"], B, V, device
            )

            target_v_bv = torch.arange(V, device=device)[None].expand(B, V)
            left_self += int((view_index[..., 0] == target_v_bv).sum())
            right_self += int((view_index[..., 2] == target_v_bv).sum())
            view_target_count += B * V

            flat_b = torch.arange(B, device=device)[:,None,None].expand(B,T,V).reshape(-1)
            flat_t = torch.arange(T, device=device)[None,:,None].expand(B,T,V).reshape(-1)
            flat_v = torch.arange(V, device=device)[None,None,:].expand(B,T,V).reshape(-1)
            N = flat_b.numel()
            src_v = view_index[flat_b, flat_v]  # [N,3] = left/self/right

            q_view = viewmats[flat_b, flat_t, flat_v]
            q_K = K[flat_b, flat_t, flat_v]
            s_view = viewmats[flat_b[:,None], flat_t[:,None], src_v]
            s_K = K[flat_b[:,None], flat_t[:,None], src_v]

            s_cam2world = _invert_se3(s_view)
            s_K_inv = _invert_intrinsics(s_K)

            gx, gy = torch.meshgrid(
                torch.arange(W, device=device, dtype=torch.float32),
                torch.arange(H, device=device, dtype=torch.float32),
                indexing="xy",
            )
            pix = torch.stack(
                (gx + 0.5, gy + 0.5, torch.ones_like(gx)), dim=-1
            ).reshape(1,1,H*W,3).expand(N,3,-1,-1)

            rays_cam = torch.einsum("nsij,nspj->nspi", s_K_inv, pix)
            rays_o_world = s_cam2world[..., :3, 3]
            rays_d_world = torch.einsum(
                "nsij,nspj->nspi", s_cam2world[..., :3, :3], rays_cam
            )

            Rq = q_view[..., :3, :3]
            tq = q_view[..., :3, 3]
            rays_o_q = torch.einsum("nij,nsj->nsi", Rq, rays_o_world) + tq[:,None]
            rays_d_q = torch.einsum("nij,nspj->nspi", Rq, rays_d_world)

            points_q = rays_o_q[:,None,:,None,:] + (
                rays_d_q[:,None] * depths[None,:,None,None,None]
            )
            projected = torch.einsum("nij,ndspj->ndspi", q_K, points_q)

            z = projected[..., 2]
            px = projected[..., 0]
            py = projected[..., 1]
            eps = 1e-5
            u = px / (z + eps)
            v = py / (z + eps)
            ua = px / (z.abs() + eps)
            va = py / (z.abs() + eps)

            for s in range(3):
                for d in range(args.depth_count):
                    zz = z[:,d,s]
                    is_neg = zz <= 0
                    is_pos = zz > 0
                    in_pos = (
                        is_pos
                        & (u[:,d,s] >= 0) & (u[:,d,s] < W)
                        & (v[:,d,s] >= 0) & (v[:,d,s] < H)
                    )
                    in_neg_abs = (
                        is_neg
                        & (ua[:,d,s] >= 0) & (ua[:,d,s] < W)
                        & (va[:,d,s] >= 0) & (va[:,d,s] < H)
                    )
                    total[s,d] += zz.numel()
                    neg[s,d] += int(is_neg.sum())
                    pos[s,d] += int(is_pos.sum())
                    pos_in[s,d] += int(in_pos.sum())
                    neg_abs_in[s,d] += int(in_neg_abs.sum())

            # self slot sanity: same camera should reproject to the same patch center.
            exp_u = (gx + 0.5).reshape(1, -1)
            exp_v = (gy + 0.5).reshape(1, -1)
            for d in range(args.depth_count):
                valid = z[:,d,1] > 0
                err = torch.maximum(
                    (u[:,d,1] - exp_u).abs(),
                    (v[:,d,1] - exp_v).abs(),
                )
                if valid.any():
                    e = err[valid]
                    self_err_sum += float(e.sum())
                    self_err_count += e.numel()
                    self_err_max = max(self_err_max, float(e.max()))

            if "fps" in batch:
                x = batch["fps"]
                fps_seen.append(
                    x.detach().cpu().reshape(-1).tolist()
                    if torch.is_tensor(x) else str(x)
                )

    lines = []
    lines.append("URoPE current-time z diagnostic")
    lines.append("=" * 86)
    lines.append(f"batches={processed}")
    lines.append(shape_line or "shape=unknown")
    lines.append("depths=" + ", ".join(f"{float(x):.1f}m" for x in depths.cpu()))
    lines.append(
        f"left slot replaced by self={pct(left_self, view_target_count):.3f}%"
    )
    lines.append(
        f"right slot replaced by self={pct(right_self, view_target_count):.3f}%"
    )
    if fps_seen:
        lines.append(f"fps seen={fps_seen}")
    lines.append("")
    lines.append(
        f"{'slot':>7} {'depth':>7} {'neg_z%':>10} "
        f"{'posFOV/all%':>12} {'posFOV|z+%':>12} {'negAbsFOV%':>12}"
    )
    lines.append("-" * 86)

    for s, name in enumerate(SLOT_NAMES):
        for d in range(args.depth_count):
            tt = int(total[s,d])
            nn = int(neg[s,d])
            pp = int(pos[s,d])
            pi = int(pos_in[s,d])
            nai = int(neg_abs_in[s,d])
            lines.append(
                f"{name:>7} {float(depths[d]):7.1f} {pct(nn,tt):10.3f} "
                f"{pct(pi,tt):12.3f} {pct(pi,pp):12.3f} {pct(nai,nn):12.3f}"
            )
        lines.append("")

    mean_self_err = self_err_sum / self_err_count if self_err_count else float("nan")
    lines.append("Self-camera sanity")
    lines.append(f"mean max(|du|,|dv|)={mean_self_err:.8f} patch pixels")
    lines.append(f"max  max(|du|,|dv|)={self_err_max:.8f} patch pixels")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- self neg_z should be ~0%, self reprojection error should be ~0.")
    lines.append("- left/right neg_z only a few %: abs(z) probably minor.")
    lines.append("- left/right neg_z tens of %: behind-camera geometry is substantial.")
    lines.append("- high negAbsFOV% is strongest warning: abs(z) folds behind-camera points into plausible in-FOV positions.")

    report = "\n".join(lines) + "\n"
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
