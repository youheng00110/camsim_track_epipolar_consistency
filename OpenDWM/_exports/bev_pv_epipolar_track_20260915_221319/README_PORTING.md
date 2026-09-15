# BEV + PV + Epipolar + Track-ID package

This package contains the current OpenDWM implementation for BEV conditioning, PV raster conditioning, independent PV condition dropout, epipolar view-consistency loss, and Track-ID instance consistency loss.

The Track loss uses spatial pairs (same time, adjacent cameras) and temporal pairs (same camera, adjacent frames). Positives are matching Track IDs; negatives are vehicle instances with other Track IDs in the target frame/view. Hard negatives require the same class, a different Track ID, and similar 3D size.

The total objective is `L = L_SD + lambda_epi * L_epi + lambda_track * L_track`.

When porting, keep camera slot order identical, ensure the dataset annotation fields exist (`bbox_token_*` and stable Track IDs), and reconfigure SD3/base checkpoint, dataset roots, and output paths. NuPlan Track IDs intentionally require `info["track_token"]`; there is no `gt_track_token` fallback.
