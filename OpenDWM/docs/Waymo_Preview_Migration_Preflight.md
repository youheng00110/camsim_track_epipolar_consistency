# Waymo preview migration preflight

## Files

- Config: `configs/camsim/waymo/waymopluckerpreview_quantum_resume.json`
- Preflight: `tests/waymo_preview_preflight.py`
- Shared resume helper: `src/dwm/utils/preview.py`

The Waymo pipeline remains `dwm.pipelines.camsim.CrossviewTemporalSD`. The
BEVPVEpi pipeline was not substituted.

## Resume behavior

`get_eval_frame_export_path()` and `prepare_eval_frame_resume()` are shared by
BEVPVEpi and CTSD. They preserve the existing behavior: rank directories are
`rank_00`, `rank_01`, ... when `all_rank_preview` is enabled; each manifest is
counted by non-empty lines; distributed progress is the minimum rank count;
partial progress rewinds to a dataloader batch boundary; uneven manifests are
truncated with a timestamped backup; and `preview.py` skips completed batches
before calling `pipeline.preview_pipeline()`.

`CrossviewTemporalSD._prepare_eval_frame_resume()` is only a thin wrapper over
the shared helper.

## Config path changes

| Old path | New path |
| --- | --- |
| `.../camsim_lyh` (`global_state.nuscenes_fs`) | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nus_local` |
| `.../advanced-machine-learning/.../camsim_lyh/lyhmigration/stable-diffusion-3-medium` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/lyhmigration/stable-diffusion-3-medium` |
| `.../output/train_waymoplucker/checkpoints/18000.pth` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt/waymo/train_waymoplucker/18000.pth` |
| `.../camsim_lyh/lyhmigration/i3d_pretrained_400.pt` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/lyhmigration/i3d_pretrained_400.pt` |
| `.../camsim_lyh/waymo_link` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_open_dataset_v_1_4_3` |
| `.../camsim_lyh/waymo_balanced/validation.info.json` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_balanced/validation.info.json` |
| `.../camsim_lyh/waymo_caption_v2/waymo_caption_v2_val.json` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_caption_v2/waymo_caption_v2_val.json` |
| `.../camsim_lyh/waymo_caption_v2/waymo_caption_v2_times_val.json` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_caption_v2/waymo_caption_v2_times_val.json` |
| `.../output/eval/waymo/plucker` | `/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/eval/waymo/plucker_6hz_resume` |

The runtime paths above exist. The old training-only NuPlan, NuScenes, and
Argoverse paths remain in the JSON because `dwm.preview` does not instantiate
`training_dataset`; the preflight reports them as deferred missing paths.

Validation sampling remains `[6, 9]` with the original Waymo sensor order. The
current Waymo dataset accepts both two- and three-element `fps_stride_tuples`.

## Preview target and checks

The config sets `preview_item_count: 1000`, `eval_frame_resume: true`, and
`all_rank_preview: true`. With four ranks the logged per-rank target is 250.
The current validation dataset length is 404, so one run can export at most
those 404 validation clips; this is expected for the available validation
metadata and does not trigger sample duplication.

Static preflight results:

- JSON parsing and `CrossviewTemporalSD` import: PASS.
- SD3, Waymo root, validation info, captions, checkpoint, and I3D paths: PASS.
- Output parent creation/write check: PASS.
- Device mesh `[1, 4]`: PASS.
- Validation dataset construction: PASS, length 404.
- Resume simulations `[0,0,0,0] -> 0`, `[3,3,3,3] -> 3`, `[4,3,4,4] -> 3`: PASS.
- Waymo `corners8` record construction, ego-to-camera transform identity, and
  synthetic manifest export: PASS.
- A real first-batch read was attempted without constructing the diffusion
  pipeline; the host terminated that data decode for memory, so it is not
  reported as a passing batch check.

Waymo box parameter export is enabled with `corners8` in the current
vehicle/reference-ego frame. Each manifest frame now carries variable-length
`3dbox_records` and every view carries `T_reference_ego_to_camera`; existing
CTSD RGB/paired-real/manifest export remains enabled.

## Four-GPU command (do not launch until approved)

```bash
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/externals/waymo-open-dataset/src:$PYTHONPATH"
export PYTHONPATH="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/nuplan-devkit-master:$PYTHONPATH"
torchrun --standalone --nproc_per_node=4 -m dwm.preview \
  -c /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/OpenDWM/configs/camsim/waymo/waymopluckerpreview_quantum_resume.json \
  -o /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim/lyh_output/waymo_preview_quantum_resume
```

This command is recorded only; no diffusion preview was launched.
