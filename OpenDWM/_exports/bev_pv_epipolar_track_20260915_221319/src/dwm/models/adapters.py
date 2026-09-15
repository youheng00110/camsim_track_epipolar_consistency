import diffusers.models.adapter
import torch
import einops
from typing import Optional


class ImageAdapter(torch.nn.Module):
    def __init__(
        self, in_channels: int = 3,
        channels: list = [320, 320, 640, 1280, 1280],
        is_downblocks: list = [False, True, True, True, False],
        num_res_blocks: int = 2, downscale_factor: int = 8,
        use_zero_convs: bool = False, zero_gate_coef: Optional[float] = None,
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor ** 2
        self.unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        self.body = torch.nn.ModuleList([
            diffusers.models.adapter.AdapterBlock(
                in_channels if i == 0 else channels[i - 1], channels[i],
                num_res_blocks, down=is_downblocks[i])
            for i in range(len(channels))
        ])
        self.gradient_checkpointing = gradient_checkpointing

        self.zero_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(channel, channel, 1)
            for channel in channels
        ]) if use_zero_convs else [None for _ in channels]
        for i in self.zero_convs:
            if i is not None:
                torch.nn.init.zeros_(i.weight)
                torch.nn.init.zeros_(i.bias)

        self.zero_gate_coef = zero_gate_coef
        self.zero_gates = torch.nn.Parameter(torch.zeros(len(channels))) \
            if zero_gate_coef else None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        base_shape = x.shape[:-3]
        x = self.unshuffle(x.flatten(0, -4))
        features = []
        for i, (block, zero_conv) in enumerate(zip(self.body, self.zero_convs)):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)

            x_out = x
            if zero_conv is not None:
                x_out = zero_conv(x_out)

            if self.zero_gates is not None:
                x_out = x_out * torch.tanh(
                    self.zero_gate_coef * self.zero_gates[i])

            features.append(x_out.view(*base_shape, *x_out.shape[1:]))
        return features if not return_features else features[-1]

def zero_module(module: torch.nn.Module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
def get_temporal_kernel_size(temporal_downsample_factor: int) -> int:
    if temporal_downsample_factor % 2 == 0:
        return temporal_downsample_factor + 1
    return temporal_downsample_factor
class TemporalConditionImageAdapter(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        hidden_channels: Optional[int] = None,
        temporal_downsample_factor: int = 4,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.depth = int(depth)
        self.hidden_channels = int(hidden_channels or out_channels)
        self.temporal_downsample_factor = int(temporal_downsample_factor)

        kernel_t = get_temporal_kernel_size(self.temporal_downsample_factor)

        self.stem = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=int(in_channels),
                out_channels=self.hidden_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
            ),
            torch.nn.SiLU(),
            torch.nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
            ),
            torch.nn.SiLU(),
        )

        self.temporal = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=(kernel_t, 1, 1),
                stride=(self.temporal_downsample_factor, 1, 1),
                padding=(kernel_t // 2, 0, 0),
            ),
            torch.nn.SiLU(),
        )

        self.projs = torch.nn.ModuleList(
            [
                zero_module(
                    torch.nn.Conv3d(
                        in_channels=self.hidden_channels,
                        out_channels=self.out_channels,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                    )
                )
                for _ in range(self.depth)
            ]
        )
    def forward(
        self,
        condition_image_tensor: torch.Tensor,
        target_sequence_length: int,
        target_patch_size,
    ):
        if condition_image_tensor.ndim != 5:
            raise ValueError(
                f"condition_image_tensor must be 5D [(B*V), C, T, H, W], "
                f"but got shape {tuple(condition_image_tensor.shape)}"
            )

        if len(target_patch_size) != 2:
            raise ValueError(
                f"target_patch_size must be (patch_height, patch_width), "
                f"but got {target_patch_size}"
            )

        batch_size_total, _, dense_t, _, _ = condition_image_tensor.shape
        patch_height, patch_width = target_patch_size

        x = torch.nn.functional.adaptive_avg_pool3d(
            condition_image_tensor,
            output_size=(dense_t, patch_height, patch_width),
        )

        x = self.stem(x)
        x = self.temporal(x)

        if x.shape[2] != target_sequence_length:
            raise ValueError(
                f"TemporalConditionImageAdapter got T={x.shape[2]}, "
                f"expected {target_sequence_length}. "
                f"dense_t={dense_t}, temporal_downsample_factor={self.temporal_downsample_factor}"
            )

        outputs = []
        for proj in self.projs:
            y = proj(x)
            y = einops.rearrange(
                y,
                "b c t h w -> b (t h w) c",
                t=target_sequence_length,
                h=patch_height,
                w=patch_width,
            )
            outputs.append(y)

        return outputs