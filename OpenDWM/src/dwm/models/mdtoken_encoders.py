import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_freqs: int = 4,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0,
                float(num_freqs - 1),
                steps=num_freqs,
            )
        else:
            freq_bands = torch.linspace(
                1.0,
                2.0 ** float(num_freqs - 1),
                steps=num_freqs,
            )

        self.register_buffer("freq_bands", freq_bands, persistent=False)

        out_dim = input_dim * num_freqs * 2
        if include_input:
            out_dim += input_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"FourierFeatures expects last dim={self.input_dim}, "
                f"but got {tuple(x.shape)}."
            )

        freq_shape = [1] * x.ndim
        freq_shape[-1] = self.num_freqs

        freq_bands = self.freq_bands.to(device=x.device, dtype=x.dtype)
        scaled = x.unsqueeze(-1) * freq_bands.view(freq_shape)
        scaled = scaled * math.pi

        sin_feat = torch.sin(scaled).flatten(-2)
        cos_feat = torch.cos(scaled).flatten(-2)

        if self.include_input:
            return torch.cat([x, sin_feat, cos_feat], dim=-1)

        return torch.cat([sin_feat, cos_feat], dim=-1)


class CameraParamTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        num: int = 7,
        num_freqs: int = 4,
        out_dim: int = 1536,
        include_input: bool = True,
        log_sampling: bool = True,
        zero_after_proj: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num = num

        self.embedder = FourierFeatures(
            input_dim=input_dim,
            num_freqs=num_freqs,
            include_input=include_input,
            log_sampling=log_sampling,
        )

        self.emb2token = nn.Linear(
            self.embedder.out_dim * num,
            out_dim,
            bias=True,
        )

        self.uncond_cam = nn.Parameter(torch.randn(input_dim, num))
        self.after_proj = nn.Linear(out_dim, out_dim, bias=True)

        if zero_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        param: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            param: [N, 3, 7] or [N, 4, 7]
            mask:  [N], 1 means keep camera, 0 means use uncond camera

        Returns:
            token: [N, C]
        """
        if param.ndim != 3:
            raise ValueError(
                f"CameraParamTokenEncoder expects [N, 3, 7], "
                f"but got {tuple(param.shape)}."
            )

        if param.shape[1] == 4:
            param = param[:, :3]

        if param.shape[1:] != (self.input_dim, self.num):
            raise ValueError(
                f"CameraParamTokenEncoder expects [N, {self.input_dim}, {self.num}], "
                f"but got {tuple(param.shape)}."
            )

        if mask is not None:
            mask = mask.to(device=param.device)
            param = torch.where(
                mask[:, None, None].bool(),
                param,
                self.uncond_cam[None].to(device=param.device, dtype=param.dtype),
            )

        n = param.shape[0]
        param = param.transpose(1, 2).contiguous()
        emb = self.embedder(param.reshape(n * self.num, self.input_dim))
        emb = emb.reshape(n, self.num, -1).flatten(1)

        token = self.emb2token(emb)
        token = self.after_proj(token)
        return token


class BBoxTokenEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 32,
        points_per_box: int = 8,
        input_dim: int = 3,
        class_token_dim: int = 768,
        num_freqs: int = 4,
        out_dim: int = 1536,
        hidden_dim: int = 768,
        position_min=(-80.0, -80.0, -5.0),
        position_range=(160.0, 160.0, 10.0),
        zero_after_proj: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.points_per_box = points_per_box
        self.input_dim = input_dim

        self.register_buffer(
            "position_min",
            torch.tensor(position_min, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "position_range",
            torch.tensor(position_range, dtype=torch.float32),
            persistent=False,
        )

        self.embedder = FourierFeatures(
            input_dim=input_dim,
            num_freqs=num_freqs,
            include_input=True,
            log_sampling=True,
        )

        pos_dim = self.embedder.out_dim * points_per_box
        self.class_embedding = nn.Embedding(num_classes, class_token_dim)

        self.bbox_proj = nn.Linear(pos_dim, hidden_dim)
        self.second_linear = nn.Sequential(
            nn.Linear(hidden_dim + class_token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.null_pos_feature = nn.Parameter(torch.zeros(pos_dim))
        self.mask_pos_feature = nn.Parameter(torch.zeros(pos_dim))
        self.null_class_feature = nn.Parameter(torch.zeros(class_token_dim))
        self.mask_class_feature = nn.Parameter(torch.zeros(class_token_dim))

        self.after_proj = nn.Linear(out_dim, out_dim, bias=True)

        if zero_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        bboxes: torch.Tensor,
        classes: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            bboxes:  [N, S, 8, 3]
            classes: [N, S]
            masks:   [N, S]
                     1  = keep real box
                     0  = null / padded box
                    -1  = masked box

        Returns:
            tokens: [N, S, C]
        """
        if bboxes.ndim != 4:
            raise ValueError(
                f"BBoxTokenEncoder expects bboxes [N, S, P, 3], "
                f"but got {tuple(bboxes.shape)}."
            )

        if bboxes.shape[-2:] != (self.points_per_box, self.input_dim):
            raise ValueError(
                f"BBoxTokenEncoder expects last dims "
                f"({self.points_per_box}, {self.input_dim}), "
                f"but got {tuple(bboxes.shape)}."
            )

        if classes.shape != bboxes.shape[:2]:
            raise ValueError(
                f"classes shape should be {tuple(bboxes.shape[:2])}, "
                f"but got {tuple(classes.shape)}."
            )

        if masks is None:
            masks = torch.ones_like(classes, dtype=bboxes.dtype)
        else:
            masks = masks.to(device=bboxes.device)

        classes = classes.to(device=bboxes.device).long()
        classes = torch.clamp(classes, min=0, max=self.num_classes - 1)

        position_min = self.position_min.to(device=bboxes.device, dtype=bboxes.dtype)
        position_range = self.position_range.to(device=bboxes.device, dtype=bboxes.dtype)
        bboxes_norm = (bboxes - position_min.view(1, 1, 1, 3)) / \
            position_range.view(1, 1, 1, 3)
        bboxes_norm = torch.clamp(bboxes_norm, 0.0, 1.0)

        pos_emb = self.embedder(bboxes_norm)
        pos_emb = pos_emb.flatten(2)

        null_keep = (masks != 0).to(dtype=bboxes.dtype).unsqueeze(-1)
        visible_keep = (masks > 0).to(dtype=bboxes.dtype).unsqueeze(-1)

        pos_emb = pos_emb * null_keep + \
            self.null_pos_feature[None, None].to(pos_emb) * (1.0 - null_keep)
        pos_emb = pos_emb * visible_keep + \
            self.mask_pos_feature[None, None].to(pos_emb) * (1.0 - visible_keep)

        cls_emb = self.class_embedding(classes)
        cls_emb = cls_emb * null_keep + \
            self.null_class_feature[None, None].to(cls_emb) * (1.0 - null_keep)
        cls_emb = cls_emb * visible_keep + \
            self.mask_class_feature[None, None].to(cls_emb) * (1.0 - visible_keep)

        pos_token = F.silu(self.bbox_proj(pos_emb))
        token = self.second_linear(torch.cat([pos_token, cls_emb], dim=-1))
        token = self.after_proj(token)
        return token


class BEVMapTokenEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 1536,
        hidden_dim: int = 768,
        token_grid_size=(8, 8),
        zero_after_proj: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.token_grid_size = token_grid_size

        mid_dim = max(hidden_dim // 2, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(token_grid_size)
        self.proj = nn.Linear(hidden_dim, out_dim, bias=True)

        self.after_proj = nn.Linear(out_dim, out_dim, bias=True)
        if zero_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            maps: [N, C, H, W]

        Returns:
            tokens: [N, token_grid_h * token_grid_w, C_out]
        """
        if maps.ndim != 4:
            raise ValueError(
                f"BEVMapTokenEncoder expects maps [N, C, H, W], "
                f"but got {tuple(maps.shape)}."
            )

        if maps.shape[1] != self.in_channels:
            raise ValueError(
                f"BEVMapTokenEncoder expects {self.in_channels} channels, "
                f"but got {maps.shape[1]}."
            )

        x = self.conv(maps)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.after_proj(x)
        return x

class BEVMapResidualEncoder(nn.Module):
    """
    MagicDrive-style BEV map encoder.

    The BEV map is encoded as a spatial feature and added to DiT patch tokens.
    BBox conditions can still stay as cross-attention/context tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 1536,
        hidden_dim: int = 256,
        zero_out: bool = True,
        zero_after_proj=None,
        token_grid_size=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        if zero_after_proj is not None:
            zero_out = bool(zero_after_proj)

        mid_dim_1 = max(hidden_dim // 4, 64)
        mid_dim_2 = max(hidden_dim // 2, 128)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim_1, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_dim_1, mid_dim_2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_dim_2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.proj = nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1)

        if zero_out:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, maps: torch.Tensor, target_hw) -> torch.Tensor:
        """
        Args:
            maps:      [N, C, H, W]
            target_hw: (latent_patch_h, latent_patch_w)

        Returns:
            residual: [N, out_dim, latent_patch_h, latent_patch_w]
        """
        if maps.ndim != 4:
            raise ValueError(
                f"BEVMapResidualEncoder expects maps [N, C, H, W], "
                f"but got {tuple(maps.shape)}."
            )

        if maps.shape[1] != self.in_channels:
            raise ValueError(
                f"BEVMapResidualEncoder expects C={self.in_channels}, "
                f"but got C={maps.shape[1]}."
            )

        x = self.conv(maps)
        x = self.proj(x)
        x = F.interpolate(
            x,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return x

