"""Aligned BEV + PV dataset composition for the joint conditioning experiment."""

from typing import Any

import torch


class AlignedBEVPVDataset(torch.utils.data.Dataset):
    """
    Keep the BEV sample as the authoritative target/geometry sample and attach
    the missing PV image-condition fields from the matching track_pv sample.

    This preserves the BEV stable-slot annotations and BEV map implementation,
    while reusing track_pv for PV HD-map rendering and instance flow.
    """

    PV_REQUIRED_KEYS = (
        "3dbox_images",
        "hdmap_images",
        "instance_flow_images",
    )
    PV_COPY_KEYS = (
        "hdmap_images",
        "instance_flow_images",
    )

    def __init__(
        self,
        bev_dataset,
        pv_dataset,
        verify_alignment: bool = True,
    ):
        self.bev_dataset = bev_dataset
        self.pv_dataset = pv_dataset
        self.verify_alignment = bool(verify_alignment)

        bev_length = len(self.bev_dataset)
        pv_length = len(self.pv_dataset)
        if bev_length != pv_length:
            raise ValueError(
                "BEV/PV dataset lengths differ: "
                f"bev={bev_length}, pv={pv_length}. "
                "They must use the same split and balanced windows."
            )

    def __len__(self):
        return len(self.bev_dataset)

    @staticmethod
    def _shape(value: Any):
        return tuple(value.shape) if torch.is_tensor(value) else None

    def _verify(self, index: int, bev: dict, pv: dict):
        for key in ("camera_intrinsics", "camera_transforms"):
            if key not in bev or key not in pv:
                raise KeyError(
                    f"Aligned BEV/PV dataset requires {key!r} on both sides."
                )
            if self._shape(bev[key]) != self._shape(pv[key]):
                raise ValueError(
                    f"BEV/PV {key} shape mismatch at index {index}: "
                    f"bev={self._shape(bev[key])}, pv={self._shape(pv[key])}."
                )

        if "crossview_mask" in bev and "crossview_mask" in pv:
            bev_mask = bev["crossview_mask"]
            pv_mask = pv["crossview_mask"]
            if torch.is_tensor(bev_mask) and torch.is_tensor(pv_mask):
                if bev_mask.shape != pv_mask.shape or not torch.equal(
                    bev_mask.bool(),
                    pv_mask.bool(),
                ):
                    raise ValueError(
                        f"BEV/PV crossview_mask mismatch at index {index}."
                    )

        if "3dbox_images" not in bev:
            raise KeyError(
                f"BEV dataset did not produce '3dbox_images' at index {index}."
            )

        for key in self.PV_REQUIRED_KEYS:
            if key not in pv:
                raise KeyError(
                    f"PV dataset did not produce {key!r} at index {index}."
                )

    def __getitem__(self, index):
        bev = self.bev_dataset[index]
        pv = self.pv_dataset[index]

        if self.verify_alignment:
            self._verify(index, bev, pv)

        result = dict(bev)

        # Keep BEV's own 3dbox_images so the original BEV box-weighted loss
        # remains exactly on its original data path. Only attach PV fields
        # that BEV base does not produce.
        for key in self.PV_COPY_KEYS:
            result[key] = pv[key]

        return result
