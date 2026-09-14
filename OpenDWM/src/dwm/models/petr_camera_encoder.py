from typing import Optional, Sequence
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PETRCameraEncoder(nn.Module):
    """
    DriveArena-OPE style camera position encoder for OpenDWM.

    Input:
        camera_intrinsics: [N, 3, 3], already scaled to latent/token grid.
        camera2referego:   [N, 4, 4]
        height, width:     token grid size.

    Output:
        [N, H, W, C]
    """

    def __init__(
        self,
        in_channels: int,
        depth_num: int = 64,
        depth_start: float = 1.0,
        depth_max: Optional[float] = None,
        position_range: Optional[Sequence[float]] = None,
        LID: bool = False,
        block_out_channels: Sequence[int] = (256, 256),
        use_valid_mask: bool = True,
        eps: float = 1e-5,
        output_scale: float = 1.0,
        debug: bool = False,
        debug_interval: int = 200,
    ):
        super().__init__()
        self.debug = bool(debug)
        self.debug_interval = int(debug_interval)
        self.register_buffer(
            "_debug_step",
            torch.zeros([], dtype=torch.long),
            persistent=False,
        )
        if position_range is None:
            position_range = (-80.0, -80.0, -5.0, 80.0, 80.0, 5.0)

        if len(position_range) != 6:
            raise ValueError(
                "position_range should be "
                "[x_min, y_min, z_min, x_max, y_max, z_max]."
            )

        if depth_num <= 0:
            raise ValueError("depth_num should be positive.")

        if len(block_out_channels) == 0:
            raise ValueError("block_out_channels should not be empty.")

        self.depth_num = int(depth_num)
        self.depth_start = float(depth_start)
        self.depth_max = float(position_range[3] if depth_max is None else depth_max)
        self.position_dim = 3 * self.depth_num
        self.LID = bool(LID)
        self.use_valid_mask = bool(use_valid_mask)
        self.eps = float(eps)
        self.output_scale = float(output_scale)

        if self.depth_max <= self.depth_start:
            raise ValueError("depth_max should be larger than depth_start.")

        position_min = torch.tensor(position_range[:3], dtype=torch.float32)
        position_max = torch.tensor(position_range[3:], dtype=torch.float32)

        self.register_buffer("position_min", position_min, persistent=False)
        self.register_buffer("position_max", position_max, persistent=False)

        self.position_encoder = nn.Sequential(
            nn.Conv2d(
                self.position_dim,
                self.depth_num * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.depth_num * 4,
                self.depth_num * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.conv_in = nn.Conv2d(
            self.depth_num * 2,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList()
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                nn.Conv2d(
                    channel_in,
                    channel_in,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in,
                    channel_out,
                    kernel_size=3,
                    padding=1,
                )
            )

        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            in_channels,
            kernel_size=1,
        )

        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        camera_intrinsics: torch.Tensor,
        camera2referego: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if camera_intrinsics.ndim != 3:
            raise ValueError(
                f"camera_intrinsics should be [N, 3, 3], "
                f"got {camera_intrinsics.shape}."
            )

        if camera2referego.ndim != 3:
            raise ValueError(
                f"camera2referego should be [N, 4, 4], "
                f"got {camera2referego.shape}."
            )

        if camera_intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                f"camera_intrinsics should be [N, 3, 3], "
                f"got {camera_intrinsics.shape}."
            )

        if camera2referego.shape[-2:] != (4, 4):
            raise ValueError(
                f"camera2referego should be [N, 4, 4], "
                f"got {camera2referego.shape}."
            )

        input_dtype = camera_intrinsics.dtype
        device = camera_intrinsics.device
        n = camera_intrinsics.shape[0]

        k = camera_intrinsics.to(dtype=torch.float32)
        cam2ego = camera2referego.to(dtype=torch.float32)

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )

        ones = torch.ones_like(xs)
        pixel_grid = torch.stack([xs, ys, ones], dim=-1)
        pixel_grid = pixel_grid.reshape(height * width, 3)

        if self.LID:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=device,
                dtype=torch.float32,
            )
            index_1 = index + 1.0
            bin_size = (
                (self.depth_max - self.depth_start)
                / (self.depth_num * (1.0 + self.depth_num))
            )
            depth_bins = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=device,
                dtype=torch.float32,
            )
            bin_size = (self.depth_max - self.depth_start) / self.depth_num
            depth_bins = self.depth_start + bin_size * index

        image_points = pixel_grid[:, None, :] * depth_bins[None, :, None]

        k_inv = torch.linalg.inv(k)
        points_cam = torch.einsum("nij,mdj->nmdi", k_inv, image_points)
        points_cam = points_cam.reshape(
            n,
            height,
            width,
            self.depth_num,
            3,
        )

        points_ego = torch.einsum(
            "nij,nhwdj->nhwdi",
            cam2ego[:, :3, :3],
            points_cam,
        )
        points_ego = points_ego + cam2ego[:, None, None, None, :3, 3]

        position_min = self.position_min.to(device=device, dtype=torch.float32)
        position_max = self.position_max.to(device=device, dtype=torch.float32)
        position_span = torch.clamp(position_max - position_min, min=self.eps)

        norm_coords = (
            points_ego - position_min.view(1, 1, 1, 1, 3)
        ) / position_span.view(1, 1, 1, 1, 3)

        valid_mask = (norm_coords > 0.0).all(dim=-1) & (norm_coords < 1.0).all(dim=-1)

        norm_coords = norm_coords.clamp(self.eps, 1.0 - self.eps)
        norm_coords = torch.log(norm_coords / (1.0 - norm_coords))

        if self.use_valid_mask:
            norm_coords = norm_coords * valid_mask[..., None].to(norm_coords.dtype)

        coords_feat = norm_coords.permute(0, 3, 4, 1, 2).contiguous()
        coords_feat = coords_feat.reshape(
            n,
            self.depth_num * 3,
            height,
            width,
        )
        if self.debug:
            self._debug_step += 1
            rank = int(os.environ.get("RANK", "0"))
            debug_step = int(self._debug_step.item())

            if rank == 0 and debug_step % self.debug_interval == 0:
                with torch.no_grad():
                    k0 = k.reshape(-1, 3, 3)[0]
                    cam0 = cam2ego.reshape(-1, 4, 4)[0]
                    valid_ratio = valid_mask.float().mean()

                    print("\n[PETR-Encoder][geometry]")
                    print("step:", debug_step)
                    print("N,H,W,D:", n, height, width, self.depth_num)
                    print("input dtype:", input_dtype)
                    print("K shape:", tuple(k.shape))
                    print("cam2ego shape:", tuple(cam2ego.shape))
                    print("K[0]:")
                    print(k0.detach().cpu())
                    print(
                        "fx fy cx cy:",
                        float(k0[0, 0]),
                        float(k0[1, 1]),
                        float(k0[0, 2]),
                        float(k0[1, 2]),
                    )
                    print("cam2ego[0] translation:", cam0[:3, 3].detach().cpu().tolist())
                    print(
                        "depth range:",
                        float(depth_bins.min()),
                        float(depth_bins.max()),
                    )
                    print(
                        "points_ego min/max/mean:",
                        float(points_ego.min()),
                        float(points_ego.max()),
                        float(points_ego.mean()),
                    )
                    print(
                        "norm_coords before sigmoid min/max:",
                        float(((points_ego - position_min.view(1, 1, 1, 1, 3)) / position_span.view(1, 1, 1, 1, 3)).min()),
                        float(((points_ego - position_min.view(1, 1, 1, 1, 3)) / position_span.view(1, 1, 1, 1, 3)).max()),
                    )
                    print("valid_ratio:", float(valid_ratio))
        conv_dtype = self.conv_in.weight.dtype
        coords_feat = coords_feat.to(dtype=conv_dtype)

        embedding = self.position_encoder(coords_feat)
        embedding = self.conv_in(embedding)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        if self.debug:
            rank = int(os.environ.get("RANK", "0"))
            debug_step = int(self._debug_step.item())

            if rank == 0 and debug_step % self.debug_interval == 0:
                with torch.no_grad():
                    print("[PETR-Encoder][feature before conv_out]")
                    print("coords_feat:", tuple(coords_feat.shape), coords_feat.dtype)
                    print(
                        "coords_feat mean/std/min/max:",
                        float(coords_feat.float().mean()),
                        float(coords_feat.float().std()),
                        float(coords_feat.float().min()),
                        float(coords_feat.float().max()),
                    )
                    print("pre_out embedding:", tuple(embedding.shape), embedding.dtype)
                    print(
                        "pre_out mean/std/min/max:",
                        float(embedding.float().mean()),
                        float(embedding.float().std()),
                        float(embedding.float().min()),
                        float(embedding.float().max()),
                    )
        embedding = self.conv_out(embedding)
        embedding = embedding * self.output_scale
        if self.debug:
            rank = int(os.environ.get("RANK", "0"))
            debug_step = int(self._debug_step.item())

            if rank == 0 and debug_step % self.debug_interval == 0:
                with torch.no_grad():
                    print("[PETR-Encoder][output]")
                    print("output:", tuple(embedding.shape), embedding.dtype)
                    print(
                        "output mean/std/min/max:",
                        float(embedding.float().mean()),
                        float(embedding.float().std()),
                        float(embedding.float().min()),
                        float(embedding.float().max()),
                    )
                    print(
                        "nan/inf:",
                        bool(torch.isnan(embedding).any()),
                        bool(torch.isinf(embedding).any()),
                    )
                    if torch.cuda.is_available():
                        print(
                            "cuda mem allocated MB:",
                            round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
                        )
                        print(
                            "cuda max mem allocated MB:",
                            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
                        )
        embedding = embedding.permute(0, 2, 3, 1).contiguous()
        return embedding.to(dtype=input_dtype)