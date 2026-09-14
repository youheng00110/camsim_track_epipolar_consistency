from typing import Optional, Union
import einops
import diffusers
import torch
import math
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import dwm.models.adapters
from dwm.models.crossview_temporal import VTSelfAttentionBlock, AlphaBlender, Mixer
from dwm.models.mdtoken_encoders import (
    BBoxTokenEncoder,
    BEVMapResidualEncoder,
    BEVMapTokenEncoder,
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(
            self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class PluckerEncoder(torch.nn.Module):
    def __init__(
        self,
        dir_octaves=4,
        dir_start_octave=0,
        moment_octaves=8,
        moment_start_octave=0,
        cond_proj_dim=72,
        in_channels=1536,
    ):
        super().__init__()
        self.dir_encoding = PositionalEncoding(
            num_octaves=dir_octaves,
            start_octave=dir_start_octave,
        )
        self.moment_encoding = PositionalEncoding(
            num_octaves=moment_octaves,
            start_octave=moment_start_octave,
        )
        self.proj = torch.nn.Linear(
            cond_proj_dim,
            in_channels,
            bias=False,
        )

    def forward(self, rays_d, rays_m):
        batch_size, height, width, _ = rays_d.shape

        rays_d = rays_d.flatten(1, 2)
        rays_m = rays_m.flatten(1, 2)

        dir_enc = self.dir_encoding(rays_d)
        moment_enc = self.moment_encoding(rays_m)

        x = torch.cat([dir_enc, moment_enc], dim=-1)
        x = x.view(batch_size, height, width, x.shape[-1])

        return self.proj(x)

def get_rays(
        camera_intrinsics: torch.tensor,
        camera_transforms: torch.tensor,
        target_size: Union[int, tuple[int]],):
    ''' get rays
        Args:
            camera_transforms: (B*T*V, 4, 4), cam2world
            intrinsics: (B*T*V, 3, 3)
        Returns:
            rays_o: [B*T*V, 3]
            rays_d: [B*T*V, H, W, 3]
    '''
    ###########检查是否更新内参#####
    '''
    if not hasattr(get_rays, "_debug_printed"):
        get_rays._debug_printed = False

    if not get_rays._debug_printed:
        print("=== get_rays check ===")
        print("target_size:", target_size)
        print("camera_intrinsics shape:", camera_intrinsics.shape)
        print("camera_transforms shape:", camera_transforms.shape)#
        k0 = camera_intrinsics.reshape(-1, 3, 3)[0]
        print("K sample:")
        print(k0)
        print("fx fy cx cy:",
              float(k0[0, 0]),
             float(k0[1, 1]),
              float(k0[0, 2]),
              float(k0[1, 2]))
        get_rays._debug_printed = False
        '''
    device = camera_transforms.device
    dtype = camera_transforms.dtype
    camera_transforms = camera_transforms.to(dtype=torch.float32)
    camera_intrinsics = camera_intrinsics.to(dtype=torch.float32)

    H, W = (target_size, target_size) if isinstance(
        target_size, int) else target_size
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs])  # (H*W, 3)
    directions = torch.inverse(
        camera_intrinsics) @ points_coord.unsqueeze(0)  # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions  # (batch_size, 3, H*W)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.transpose(1, 2).view(-1, H, W, 3).to(dtype=dtype)

    rays_o = camera_transforms[:, :3, 3].to(dtype=dtype)  # (batch_size, 3)

    return rays_o, rays_d


class DiTCrossviewTemporalConditionModel(diffusers.SD3Transformer2DModel):

    def _mdtoken_map_to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """
        x: [C, H, W]
        return: [H, W, 3] uint8
        """
        x = x.detach().float().cpu()

        if x.ndim != 3:
            raise ValueError(f"expect [C,H,W], got {tuple(x.shape)}")

        gray = x.abs().amax(dim=0)

        g_min = gray.min()
        g_max = gray.max()

        if (g_max - g_min) > 1e-6:
            gray = (gray - g_min) / (g_max - g_min)
        else:
            gray = torch.zeros_like(gray)

        img = (gray.numpy() * 255.0).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)
        return img


    def _bev_xy_to_pixel(self, x, y, w, h):
        x_min, x_max = -80.0, 80.0
        y_min, y_max = -80.0, 80.0

        u = (x - x_min) / max(x_max - x_min, 1e-6) * (w - 1)
        v = (y_max - y) / max(y_max - y_min, 1e-6) * (h - 1)

        return float(u), float(v)


    def _order_bev_polygon(self, pts: np.ndarray):
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        return [tuple(pts[i]) for i in order]


    def _draw_bbox_on_bev(self, bev_img, bboxes, classes=None, masks=None):
        """
        bev_img: [H, W, 3]
        bboxes: [S, 8, 3]
        masks: [S]
        """
        canvas = Image.fromarray(bev_img.copy())
        draw = ImageDraw.Draw(canvas)

        if bboxes is None:
            return np.asarray(canvas)

        bboxes = bboxes.detach().float().cpu()
        if masks is not None:
            masks = masks.detach().cpu()

        h, w = bev_img.shape[:2]

        for i in range(bboxes.shape[0]):
            if masks is not None and float(masks[i]) <= 0:
                continue

            box = bboxes[i]   # [8,3]

            # 只取 z 最小的 4 个点，当作底面
            bottom_idx = torch.argsort(box[:, 2])[:4]
            bottom_xy = box[bottom_idx, :2].numpy()

            pix = np.array(
                [self._bev_xy_to_pixel(p[0], p[1], w, h) for p in bottom_xy],
                dtype=np.float32
            )

            poly = self._order_bev_polygon(pix)
            draw.line(poly + [poly[0]], fill=(255, 60, 60), width=2)

            cx = float(pix[:, 0].mean())
            cy = float(pix[:, 1].mean())
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(0, 255, 0))

        return np.asarray(canvas)


    def _put_title(self, img: np.ndarray, text: str) -> np.ndarray:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        h, w = img.shape[:2]
        draw.rectangle((0, 0, w, 20), fill=(0, 0, 0))
        draw.text((4, 3), text, fill=(255, 255, 255))
        return np.asarray(pil)


    def _concat_h(self, imgs):
        max_h = max(im.shape[0] for im in imgs)
        padded = []
        for im in imgs:
            h, w = im.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
                im = np.concatenate([im, pad], axis=0)
            padded.append(im)
        return np.concatenate(padded, axis=1)


    def _concat_v(self, imgs):
        max_w = max(im.shape[1] for im in imgs)
        padded = []
        for im in imgs:
            h, w = im.shape[:2]
            if w < max_w:
                pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
                im = np.concatenate([im, pad], axis=1)
            padded.append(im)
        return np.concatenate(padded, axis=0)


    def _write_mp4_av(self, frames, out_path, fps=4):
        if len(frames) == 0:
            return

        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]

        with av.open(out_path, mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            for fr in frames:
                video_frame = av.VideoFrame.from_ndarray(fr, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)


    @torch.no_grad()
    def _maybe_dump_mdtoken_video(
        self,
        bbox_token_input: torch.Tensor = None,
        bbox_class_input: torch.Tensor = None,
        bbox_mask_input: torch.Tensor = None,
        map_token_input: torch.Tensor = None,
    ):
        # 用环境变量开关，避免平时训练一直写
        if os.environ.get("DWM_DEBUG_MDTOKEN_VIDEO", "0") != "1":
            return

        # 多卡只让 rank0 写
        if int(os.environ.get("RANK", "0")) != 0:
            return

        if map_token_input is None:
            return


        # debug 目标：
        #   不按 denoise forward 保存；
        #   只在 map / box 条件内容发生变化时保存。
        # 这样一个 preview item 通常只保存一次，
        # 下一个 preview item 条件变了才会保存新视频。
        if not hasattr(self, "_mdtoken_debug_seen_signatures"):
            self._mdtoken_debug_seen_signatures = set()

        if not hasattr(self, "_mdtoken_debug_dump_count"):
            self._mdtoken_debug_dump_count = 0

        if not hasattr(self, "_mdtoken_debug_call_count"):
            self._mdtoken_debug_call_count = 0

        self._mdtoken_debug_call_count += 1

        max_dump_count = int(os.environ.get("DWM_DEBUG_MDTOKEN_VIDEO_MAX", "16"))

        if self._mdtoken_debug_dump_count >= max_dump_count:
            return

        with torch.no_grad():
            nonzero_per_b = (
                map_token_input.detach().float().abs() > 1e-6
            ).flatten(1).sum(dim=1)

            sig_b = int(torch.argmax(nonzero_per_b).item())

            map_sig_tensor = map_token_input.detach().float()[sig_b]
            map_sig = (
                tuple(map_sig_tensor.shape),
                round(float(map_sig_tensor.mean().item()), 6),
                round(float(map_sig_tensor.std().item()), 6),
                round(float(map_sig_tensor.abs().sum().item()), 3),
            )

            if bbox_token_input is not None:
                bbox_sig_tensor = bbox_token_input.detach().float()[sig_b]
                bbox_sig = (
                    tuple(bbox_sig_tensor.shape),
                    round(float(bbox_sig_tensor.mean().item()), 6),
                    round(float(bbox_sig_tensor.std().item()), 6),
                    round(float(bbox_sig_tensor.abs().sum().item()), 3),
                )
            else:
                bbox_sig = None

            if bbox_mask_input is not None:
                mask_sig_tensor = bbox_mask_input.detach().float()[sig_b]
                mask_sig = (
                    tuple(mask_sig_tensor.shape),
                    round(float(mask_sig_tensor.sum().item()), 3),
                )
            else:
                mask_sig = None

            debug_signature = (map_sig, bbox_sig, mask_sig)

        if debug_signature in self._mdtoken_debug_seen_signatures:
            return

        self._mdtoken_debug_seen_signatures.add(debug_signature)

        print(
            "[mdtoken debug] new condition detected: "
            f"dump_id={self._mdtoken_debug_dump_count}, "
            f"call_id={self._mdtoken_debug_call_count}, "
            f"sig_b={sig_b}, "
            f"map_sig={map_sig}, "
            f"bbox_sig={bbox_sig}, "
            f"mask_sig={mask_sig}",
            flush=True,
        )

        # 现在这里期望已经是 [B,T,V,...] 了
        B, T, V = map_token_input.shape[:3]
        print("[mdtoken debug] selected frame stats:")
        for bi in range(B):
            for ti in range(min(T, 8)):
                for vi in range(V):
                    x = map_token_input[bi, ti, vi]
                    print(
                        f"  b={bi} t={ti} v={vi} "
                        f"min={float(x.min()):.4f} "
                        f"max={float(x.max()):.4f} "
                        f"mean={float(x.mean()):.6f} "
                        f"nonzero={int((x.abs() > 1e-6).sum())}"
                    )
        if not hasattr(self, "_mdtoken_debug_stat_printed"):
            self._mdtoken_debug_stat_printed = False

            if not self._mdtoken_debug_stat_printed:
                print("[mdtoken debug] map shape:", tuple(map_token_input.shape))
                print("[mdtoken debug] map min/max/mean/nonzero:",
                    float(map_token_input.min()),
                    float(map_token_input.max()),
                    float(map_token_input.mean()),
                    int((map_token_input.abs() > 1e-6).sum()))

                if bbox_token_input is not None:
                    print("[mdtoken debug] bbox shape:", tuple(bbox_token_input.shape))
                    print("[mdtoken debug] bbox min/max/mean:",
                        float(bbox_token_input.min()),
                        float(bbox_token_input.max()),
                        float(bbox_token_input.mean()))

                if bbox_mask_input is not None:
                    print("[mdtoken debug] bbox mask unique:",
                        torch.unique(bbox_mask_input.detach().cpu()))

                self._mdtoken_debug_stat_printed = True
        b = 0
        best_nonzero = -1

        for bi in range(B):
            cur_nonzero = int((map_token_input[bi].abs() > 1e-6).sum())
            if cur_nonzero > best_nonzero:
                best_nonzero = cur_nonzero
                b = bi

        print(f"[mdtoken debug] choose batch index b={b}, nonzero={best_nonzero}")
        max_t = T
        max_v = min(V, 8)

        frames = []

        for t in range(max_t):
            row_panels = []

            for v in range(max_v):
                bev = self._mdtoken_map_to_uint8(map_token_input[b, t, v])

                bev_plain = self._put_title(bev, f"map  t={t} v={v}")

                bev_box = bev
                if bbox_token_input is not None:
                    bev_box = self._draw_bbox_on_bev(
                        bev_box,
                        bbox_token_input[b, t, v],
                        None if bbox_class_input is None else bbox_class_input[b, t, v],
                        None if bbox_mask_input is None else bbox_mask_input[b, t, v],
                    )
                bev_box = self._put_title(bev_box, f"map + bbox  t={t} v={v}")

                panel = self._concat_h([bev_plain, bev_box])
                row_panels.append(panel)

            frame = self._concat_v(row_panels)
            frames.append(frame)

        out_path = os.environ.get(
            "DWM_DEBUG_MDTOKEN_VIDEO_PATH",
            "./outputs/debug_mdtoken/mdtoken_debug.mp4",
        )

        stem, ext = os.path.splitext(str(out_path))
        if ext == "":
            ext = ".mp4"

        dump_id = self._mdtoken_debug_dump_count
        call_id = getattr(self, "_mdtoken_debug_call_count", 0)

        indexed_out_path = f"{stem}_dump{dump_id:03d}_call{call_id:04d}{ext}"
        png_path = f"{stem}_dump{dump_id:03d}_call{call_id:04d}_first_frame.png"

        Path(png_path).parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(frames[0]).save(png_path)
        print(f"[mdtoken debug] saved first frame png to: {png_path}")

        self._write_mp4_av(frames, indexed_out_path, fps=4)
        print(f"[mdtoken debug] saved video to: {indexed_out_path}")

        self._mdtoken_debug_dump_count += 1
    @diffusers.configuration_utils.register_to_config        
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        projection_class_embeddings_input_dim: int = None,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = False,
        enable_temporal: bool = False,
        crossview_attention_type: str = None,
        temporal_attention_type: str = None,
        merge_factor: float = 2, merge_strategy="learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "",
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mdtoken_bbox_config: Optional[dict] = None,
        mdtoken_map_config: Optional[dict] = None,
        mdtoken_map_cross_attn: bool = True,
        disable_view_index_embedding: bool = False,
        mask_module=None,
        mdtoken_add_plucker_to_hidden: bool = False,
        mdtoken_plucker_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs
        )
        self.crossview_gradient_checkpointing = crossview_gradient_checkpointing
        self.temporal_gradient_checkpointing = temporal_gradient_checkpointing
        self.disable_view_emb_on_temporal_module = disable_view_emb_on_temporal_module
        self.disable_view_index_embedding = bool(disable_view_index_embedding)
        self.mdtoken_add_plucker_to_hidden = bool(mdtoken_add_plucker_to_hidden)
        self.mdtoken_plucker_scale = float(mdtoken_plucker_scale)
        self.mdtoken_map_cross_attn = bool(mdtoken_map_cross_attn)
        # image condition adapter
        if condition_image_adapter_config is not None:
            self.condition_image_adapter = \
                dwm.models.adapters.ImageAdapter(
                    **condition_image_adapter_config
                )
        else:
            self.condition_image_adapter = None

        # views and frames index embedding
        inner_dim = attention_head_dim * num_attention_heads
        self.index_proj = diffusers.models.embeddings.Timesteps(
            inner_dim, True, 0)

        self.perspective_modeling_type = perspective_modeling_type

        if perspective_modeling_type in ["explicit", "mdtoken_nopv"]:
            self.rayencoder = PluckerEncoder(
                dir_octaves=4,
                moment_octaves=8,
                cond_proj_dim=72,
                in_channels=self.inner_dim,
            )

        elif perspective_modeling_type == "implicit":
            self.view_cam_proj = diffusers.models.embeddings.Timesteps(
                num_channels=256,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
            )
            self.view_embedding = diffusers.models.embeddings.TimestepEmbedding(
                in_channels=projection_class_embeddings_input_dim,
                time_embed_dim=self.inner_dim,
            )

        if perspective_modeling_type == "mdtoken_nopv":
            self._mdtoken_debug_dumped = False
            if mdtoken_bbox_config is None:
                mdtoken_bbox_config = {}
            if mdtoken_map_config is None:
                mdtoken_map_config = {}

            self.bbox_token_encoder = BBoxTokenEncoder(
                out_dim=self.inner_dim,
                **mdtoken_bbox_config,
            )
            self.map_residual_encoder = BEVMapResidualEncoder(
                out_dim=self.inner_dim,
                **mdtoken_map_config,
            )
            if self.mdtoken_map_cross_attn:
                self.map_token_encoder = BEVMapTokenEncoder(
                    out_dim=self.inner_dim,
                    **mdtoken_map_config,
                )
            else:
                self.map_token_encoder = None
        self._mdtoken_debug_dumped = False
        self.enable_crossview = enable_crossview
        self.crossview_attention_type = crossview_attention_type
        self.crossview_block_layers = crossview_block_layers
        if enable_crossview:
            self.view_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

            self.crossview_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(crossview_block_layers))
            ])

            self.view_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

        self.enable_temporal = enable_temporal
        self.temporal_attention_type = temporal_attention_type
        self.temporal_block_layers = temporal_block_layers
        if enable_temporal:
            self.time_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

            self.temporal_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(temporal_block_layers))
            ])

            self.time_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

        # Depth Net 
        self.depth_net = None   # TODO: support joint training of image and lidar

        # Mask Reconstruction
        self.mask_module = mask_module

    def forward_crossview_block_and_mix_result(
        self, crossview_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        view_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, height: int, disable_crossview: torch.BoolTensor,
        crossview_attention_mask, crossview_attention_index
    ):
        crossview_hidden_states = hidden_states + \
            view_emb      # [b*T*V, h*w, c]
        if self.crossview_attention_type == "fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) hw c -> bt v hw c", v=view_count)
            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).flatten(0, 1)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, crossview_attention_index)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t) (v n) hw c -> (b v) (t n hw) c",
                v=view_count, n=3, t=sequence_length)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b v) (t n hw) c -> (b t v) n hw c",
                v=view_count, t=sequence_length, n=3)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, :, :]

        elif self.crossview_attention_type == "adj_fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) hw c -> b t v hw c",
                v=view_count, t=sequence_length)

            adj_frame_index = torch.cat(
                (torch.arange(view_count).unsqueeze(0),
                 torch.arange(view_count).unsqueeze(0)))
            adj_frame_index[0, 1:] -= 1
            adj_frame_index = adj_frame_index.t().flatten()
            adj_frame_index = adj_frame_index.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1).expand(batch_size, -1, view_count,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).to(
                                             crossview_attention_index)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, adj_frame_index)

            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length*2, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1])
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 2, crossview_attention_index)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "b (t l) (v n) hw c -> (b t v) (l n hw) c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) (l n hw) c -> (b t v) l n hw c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, 1, :, :]

        elif self.crossview_attention_type == "full":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> bt (h v w) c",
                v=view_count, w=width)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "bt (h v w) c -> (bt v) (h w) c",
                v=view_count, w=width)

        elif self.crossview_attention_type == "rowwise":
            if crossview_attention_mask is not None:
                crossview_attention_mask = crossview_attention_mask\
                    .repeat_interleave(width, 2)\
                    .repeat_interleave(width, 1)\
                    .repeat_interleave(sequence_length*height, 0)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> (bt h) (v w) c",
                w=width, v=view_count)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt h) (v w) c -> (bt v) (h w) c",
                bt=batch_size*sequence_length, v=view_count)

        else:
            raise f"Not support {self.crossview_attention_type}"

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            crossview_hidden_states.view(
                batch_size, sequence_length * view_count,
                *crossview_hidden_states.shape[1:]),
            image_only_indicator=disable_crossview).flatten(0, 1)

    def forward_temporal_block_and_mix_result(
        self, temporal_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        sequence_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, disable_temporal: torch.BoolTensor
    ):
        temporal_hidden_states = hidden_states + sequence_emb
        if self.temporal_attention_type == "full":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v) (t hw) c",
                b=batch_size, t=sequence_length)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v) (t hw) c -> (b t v) hw c",
                b=batch_size, t=sequence_length)
        elif self.temporal_attention_type == "rowwise":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) (h w) c -> (b v h) (t w) c",
                b=batch_size, v=view_count, w=width)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v h) (t w) c -> (b t v) (h w) c",
                b=batch_size, v=view_count, w=width)
        else:
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v hw) t c",
                b=batch_size, t=sequence_length)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v hw) t c -> (b t v) hw c",
                b=batch_size, v=view_count, t=sequence_length)

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            temporal_hidden_states.view(
                batch_size, sequence_length * view_count,
                *temporal_hidden_states.shape[1:]),
            image_only_indicator=disable_temporal).flatten(0, 1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        frustum_bev_residuals: torch.Tensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_image_tensor: torch.Tensor = None,
        disable_crossview: torch.BoolTensor = None,
        disable_temporal: torch.BoolTensor = None,
        crossview_attention_mask: torch.Tensor = None,
        crossview_attention_index: torch.Tensor = None,
        camera_intrinsics: torch.Tensor = None,
        camera_transforms: torch.Tensor = None,   
        camera_intrinsics_norm: torch.Tensor = None,
        camera2referego: torch.Tensor = None,
        added_time_ids: torch.Tensor = None,
        noise: torch.Tensor = None,
        return_dict: bool = False,
        camera_param_token: torch.Tensor = None,
        camera_token_mask: torch.Tensor = None,
        bbox_token_input: torch.Tensor = None,
        bbox_class_input: torch.Tensor = None,
        bbox_mask_input: torch.Tensor = None,
        map_token_input: torch.Tensor = None,
    ):
        should_add_dim = len(sample.shape) < 6
        if should_add_dim:
            sample = sample.unsqueeze(2)
            timestep = timestep.unsqueeze(2)
            if condition_image_tensor is not None:
                condition_image_tensor = condition_image_tensor.unsqueeze(2)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(2)
            if disable_temporal is not None:
                disable_temporal = disable_temporal.unsqueeze(2)
            if pooled_projections is not None:
                pooled_projections = pooled_projections.unsqueeze(2)

        hidden_states = sample
        batch_size, sequence_length, view_count, _, height, width = \
            hidden_states.shape

        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        self.view_count = view_count
        self.width = width

        hidden_states = hidden_states.flatten(0, 2)     # [b, 16, 32, 56]
        pooled_projections = pooled_projections.flatten(0, 2)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 2)


        hidden_states = self.pos_embed(hidden_states)   # [b, 448, 1536]    
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states)    # [b, 154, 1536]
        extra_condition_tokens = []
        timestep = timestep.flatten(0, 2).to(hidden_states.device)

        temb = self.time_text_embed(
            timestep,
            pooled_projections,
        )
        if self.perspective_modeling_type == "mdtoken_nopv":

            if bbox_token_input is not None:
                if bbox_class_input is None:
                    raise ValueError(
                        "bbox_class_input is required when bbox_token_input is provided."
                    )

                if bbox_token_input.ndim == 5:
                    bbox_token_input = bbox_token_input[:, :, None].expand(
                        -1, -1, view_count, -1, -1, -1
                    )

                if bbox_class_input.ndim == 3:
                    bbox_class_input = bbox_class_input[:, :, None].expand(
                        -1, -1, view_count, -1
                    )

                if bbox_mask_input is None:
                    bbox_mask_input = torch.ones_like(
                        bbox_class_input, dtype=hidden_states.dtype
                    )
                elif bbox_mask_input.ndim == 3:
                    bbox_mask_input = bbox_mask_input[:, :, None].expand(
                        -1, -1, view_count, -1
                    )

            if map_token_input is not None:
                if map_token_input.ndim == 5:
                    map_token_input = map_token_input[:, :, None].expand(
                        -1, -1, view_count, -1, -1, -1
                    )

                if map_token_input.ndim != 6:
                    raise ValueError(
                        "map_token_input should be [B, T, C, H, W] "
                        "or [B, T, V, C, H, W], "
                        f"but got {tuple(map_token_input.shape)}."
                    )

            # ===== 在这里导出可视化 =====
            self._maybe_dump_mdtoken_video(
                bbox_token_input=bbox_token_input,
                bbox_class_input=bbox_class_input,
                bbox_mask_input=bbox_mask_input,
                map_token_input=map_token_input,
            )

            # ===== 然后再正常编码 =====
            if bbox_token_input is not None:
                bbox_flat = bbox_token_input.flatten(0, 2).to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                bbox_class_flat = bbox_class_input.flatten(0, 2).to(
                    device=hidden_states.device,
                )
                bbox_mask_flat = bbox_mask_input.flatten(0, 2).to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

                bbox_tokens = self.bbox_token_encoder(
                    bbox_flat,
                    bbox_class_flat,
                    bbox_mask_flat,
                )
                extra_condition_tokens.append(bbox_tokens)

            if map_token_input is not None:
                map_flat = map_token_input.flatten(0, 2).to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

                map_residual = self.map_residual_encoder(
                    map_flat,
                    target_hw=(height, width),
                )
                map_residual = map_residual.flatten(2).transpose(1, 2)

                if map_residual.shape != hidden_states.shape:
                    raise ValueError(
                        "BEV map residual shape mismatch: "
                        f"map_residual={tuple(map_residual.shape)}, "
                        f"hidden_states={tuple(hidden_states.shape)}."
                    )

                hidden_states = hidden_states + map_residual

                if getattr(self, "map_token_encoder", None) is not None:
                    map_tokens = self.map_token_encoder(map_flat)
                    extra_condition_tokens.append(map_tokens)

            if len(extra_condition_tokens) > 0:
                encoder_hidden_states = torch.cat(
                    extra_condition_tokens + [encoder_hidden_states],
                    dim=1,
                )


            # 第一版建议先不要加 view_cam_emb，保持纯 context token 注入
            view_cam_emb = hidden_states.new_zeros(
                batch_size * sequence_length * view_count,
                1,
                hidden_states.shape[-1],
            )

        if self.perspective_modeling_type == "implicit":
            view_emb = self.view_cam_proj(added_time_ids.flatten()) \
                .to(dtype=hidden_states.dtype)
            view_cam_emb = self.view_embedding(
                view_emb.view(batch_size * sequence_length * view_count, -1)
            ).unsqueeze(1)

        elif self.perspective_modeling_type in ["explicit", "mdtoken_nopv"]:
            if camera_intrinsics_norm is None:
                raise ValueError(
                    "camera_intrinsics_norm is required for Plücker camera encoding."
                )

            if self.perspective_modeling_type == "mdtoken_nopv":
                # Layout tokens and BEV map are represented in the current ego frame.
                # Use camera-to-current-ego for Plücker rays to keep the camera frame
                # consistent with bbox/map conditions, especially after turns.
                camera_for_ray = camera_transforms
            else:
                camera_for_ray = camera2referego
                if camera_for_ray is None:
                    camera_for_ray = camera_transforms

            if camera_for_ray is None:
                raise ValueError(
                    "camera_transforms is required for mdtoken_nopv Plücker camera "
                    "encoding, or camera2referego / camera_transforms is required "
                    "for explicit camera encoding."
                )
            if not hasattr(self, "_mdtoken_plucker_frame_debug_printed"):
                self._mdtoken_plucker_frame_debug_printed = False

            if not self._mdtoken_plucker_frame_debug_printed:
                print(
                    "[mdtoken plucker frame] "
                    f"type={self.perspective_modeling_type}, "
                    f"use_camera_transforms={camera_for_ray is camera_transforms}, "
                    f"has_camera2referego={camera2referego is not None}, "
                    f"has_camera_transforms={camera_transforms is not None}",
                    flush=True,
                )
                self._mdtoken_plucker_frame_debug_printed = True
            camera_intrinsics_norm = camera_intrinsics_norm.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).clone()

            camera_for_ray = camera_for_ray.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            camera_intrinsics_norm[..., 0, 0] = \
                camera_intrinsics_norm[..., 0, 0] * width
            camera_intrinsics_norm[..., 1, 1] = \
                camera_intrinsics_norm[..., 1, 1] * height
            camera_intrinsics_norm[..., 0, 2] = \
                camera_intrinsics_norm[..., 0, 2] * width
            camera_intrinsics_norm[..., 1, 2] = \
                camera_intrinsics_norm[..., 1, 2] * height

            rays_o, rays_d = get_rays(
                camera_intrinsics_norm.flatten(0, 2),
                camera_for_ray.flatten(0, 2),
                (height, width),
            )

            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            rays_o_map = rays_o[:, None, None, :].expand_as(rays_d)
            rays_m = torch.cross(rays_o_map, rays_d, dim=-1)

            raymap = self.rayencoder(rays_d, rays_m)
            view_cam_emb = raymap.flatten(1, 2)
            # 新增：让 Plücker 直接进入 image latent token
            if (
                self.perspective_modeling_type == "mdtoken_nopv"
                and self.mdtoken_add_plucker_to_hidden
            ):
                hidden_states = hidden_states + self.mdtoken_plucker_scale * view_cam_emb

        condition_residuals = None if \
            self.condition_image_adapter is None or \
            condition_image_tensor is None else \
            self.condition_image_adapter(condition_image_tensor)

        if self.mask_module is not None and noise is not None:
            hidden_states = einops.rearrange(hidden_states, "(b t v) hw c -> (b v) t hw c", 
                t=sequence_length, v=view_count)
            noise = einops.rearrange(noise, "b t v c h w -> (b v) c t h w")
            hidden_states, mask_metas, condition_residuals = self.mask_module.random_masking(
                hidden_states, noise, height, width, timestep, condition_residuals=condition_residuals)
            hidden_states = einops.rearrange(hidden_states, "(b v) t hw c -> (b t v) hw c", 
                t=sequence_length, v=view_count)
            ori_width = width
            width = int(width*(1-self.mask_module.mask_ratio))

        for i, block in enumerate(self.transformer_blocks):
            if self.mask_module is not None and noise is not None and\
                self.mask_module.is_first_decoder_layer(
                    i, len(self.transformer_blocks)):
                hidden_states = einops.rearrange(hidden_states, 
                    "(b t v) hw c -> (b v t) hw c", t=sequence_length, v=view_count)
                temb_v_first = einops.rearrange(temb, 
                    "(b t v) c -> (b v t) c", t=sequence_length, v=view_count)
                hidden_states = self.mask_module.mask_reconstruction(
                    hidden_states, mask_metas, ori_shape=(batch_size*view_count, 
                    sequence_length, height, ori_width), y_t=y_t, y_lens=y_lens, 
                    temb=temb_v_first)
                hidden_states = einops.rearrange(hidden_states, 
                    "(b v t) hw c -> (b t v) hw c", t=sequence_length, v=view_count)
                width = int(width/(1-self.mask_module.mask_ratio))

            if condition_residuals is not None and len(condition_residuals) > 0:
                hidden_states = hidden_states + \
                    condition_residuals.pop(0).flatten(0, 2)\
                    .flatten(2).permute(0, 2, 1)    

            # text-spatio
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                encoder_hidden_states, hidden_states = \
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        use_reentrant=False
                    )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb
                )

            if encoder_hidden_states is not None:
                self.encoder_hidden_states_var = torch.var(encoder_hidden_states).item()

            # temporal
            if self.enable_temporal and i in self.temporal_block_layers:
                sequence_emb = torch\
                    .arange(sequence_length, device=hidden_states.device)\
                    .unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, view_count)

                sequence_emb = self.index_proj(sequence_emb.flatten())\
                    .to(dtype=hidden_states.dtype)

                sequence_emb = self.time_pos_embeds[
                    self.temporal_block_layers.index(i)](sequence_emb).unsqueeze(1)

                # 原来这里错误依赖 self.enable_crossview
                # 单视角 / enable_crossview=False 时也应该让 temporal 看到相机几何
                if not self.disable_view_emb_on_temporal_module:
                    sequence_emb = sequence_emb + view_cam_emb

                if self.training and self.temporal_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_temporal_block_and_mix_result,
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states,
                        sequence_emb,
                        batch_size,
                        sequence_length,
                        view_count,
                        width,
                        disable_temporal,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_temporal_block_and_mix_result(
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states,
                        sequence_emb,
                        batch_size,
                        sequence_length,
                        view_count,
                        width,
                        disable_temporal)

            # cross-view
            if self.enable_crossview and i in self.crossview_block_layers:
                if (
                    self.perspective_modeling_type in ["camtoken_nopv", "mdtoken_nopv"]
                    and self.disable_view_index_embedding
                ):
                    view_emb = hidden_states.new_zeros(
                        batch_size * sequence_length * view_count,
                        1,
                        hidden_states.shape[-1],
                    )
                else:
                    view_emb = torch\
                        .arange(view_count, device=hidden_states.device)\
                        .unsqueeze(0).unsqueeze(0)\
                        .repeat(batch_size, sequence_length, 1)
                    view_emb = self.index_proj(view_emb.flatten())\
                        .to(dtype=hidden_states.dtype)
                    view_emb = self.view_pos_embeds[
                        self.crossview_block_layers.index(i)](view_emb).unsqueeze(1)

                view_emb = view_emb + view_cam_emb

                if self.training and self.crossview_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_crossview_block_and_mix_result,
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb, batch_size,
                        sequence_length, view_count, width, height, disable_crossview,
                        crossview_attention_mask,
                        crossview_attention_index,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_crossview_block_and_mix_result(
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(
                            i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb,
                        batch_size, sequence_length, view_count, width, height,
                        disable_crossview, crossview_attention_mask,
                        crossview_attention_index)

        # debug code
        self.hidden_states_var = hidden_states.var().item()
        if self.enable_temporal:
            self.temporal_embedding_var = sequence_emb.var().item()

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                batch_size, sequence_length, view_count,
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )
        result = [output]

        if should_add_dim:
            output = output.squeeze(2)
            result = [output]

        if return_dict:
            return {
                "noise_pred": output,
            }

        return result, encoder_hidden_states, pooled_projections