"""
PVTrack wrapper for current-time-only URoPE-TV.

PVTrack remains unchanged:
box/map/instance-flow -> 9-channel ImageAdapter -> DiT residuals.
"""

from dwm.models.lyh.crossview_temporal_dit_urope_tv_tself import (
    DiTCrossviewTemporalConditionModel as URoPETVCurrentTimeModel,
)


class DiTCrossviewTemporalConditionModel(URoPETVCurrentTimeModel):
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
        return super().forward(*args, **kwargs)
