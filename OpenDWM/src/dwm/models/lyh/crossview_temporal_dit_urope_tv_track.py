"""
URoPE-inside-TV + PVTrack model entry.

PVTrack remains an image-condition branch:
    PV box / HD map / instance flow
        -> condition_image_tensor
        -> ImageAdapter
        -> condition residuals
        -> DiT hidden states

URoPE-TV remains the joint attention branch:
    TV local 3x3 temporal-view topology
        + URoPE camera geometry inside the same attention.

This wrapper intentionally adds no new trainable module. It only exposes a
separate experiment class and absorbs legacy token kwargs that camsim_track
may optionally produce in other configurations.
"""

from dwm.models.lyh.crossview_temporal_dit_urope_tv import (
    DiTCrossviewTemporalConditionModel as URoPETVModel,
)


class DiTCrossviewTemporalConditionModel(URoPETVModel):
    """URoPE-TV with the existing PVTrack ImageAdapter conditioning path."""

    def forward(
        self,
        *args,
        camera_param_token=None,
        camera_token_mask=None,
        bbox_token_input=None,
        bbox_class_input=None,
        bbox_mask_input=None,
        map_token_input=None,
        **kwargs,
    ):
        del camera_param_token
        del camera_token_mask
        del bbox_token_input
        del bbox_class_input
        del bbox_mask_input
        del map_token_input

        return super().forward(
            *args,
            **kwargs,
        )
