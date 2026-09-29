#!/usr/bin/env bash
# SAM 3.1 only. CHECK_ONLY=1 builds config and validates inputs without inference.
set -eo pipefail
source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
set -u
CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
EVAL_ROOT="${CAMSIM_ROOT}/lyh_output/eval/nuplanhard1000"
SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"
CKPT_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt
OUT_ROOT="${EVAL_ROOT}/sam3_bev_pv_epipolar32000_generated1000_4gpu"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SAM31_CHECKPOINT="${CKPT_ROOT}/sam3.1/sam3.1_multiplex.pt"
export PYTHONPATH="${CAMSIM_ROOT}/OpenDWM/src:${SAM3_ROOT}/sam3:${PYTHONPATH:-}"
export EVAL_ROOT SAM3_ROOT OUT_ROOT
mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/results"
exec 9>"${OUT_ROOT}/evaluation.lock"
flock -n 9 || { echo '[ERROR] SAM evaluation already running'; exit 1; }
cd "${SAM3_ROOT}"

python - <<'PY'
import copy, json, os
from pathlib import Path
import yaml
from shared_box_projection import video_pose_signature

root = Path(os.environ['EVAL_ROOT'])
out = Path(os.environ['OUT_ROOT'])
preview = root / 'bev_pv_epipolar32000_merged1000'
boxes = root / 'uropetvtrack/box'
template = root / 'sam3_bev_pv_epipolar_merged999_box/configs/bev_pv_epipolar.yaml'
checkpoint = Path(os.environ['SAM31_CHECKPOINT'])
vocab = Path(os.environ['SAM3_ROOT']) / 'sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
if not checkpoint.is_file() or not vocab.is_file():
    raise SystemExit(f'Local checkpoint/tokenizer missing: {checkpoint}, {vocab}')
gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus):
    raise SystemExit('Exactly four distinct visible GPUs required')

# Match by complete ego-pose sequence, timestep and camera, never by local video ID.
required = set()
signatures = set()
count = 0
with (preview / 'stflow_manifest.jsonl').open() as handle:
    for line in handle:
        if not line.strip():
            continue
        item = json.loads(line)
        sig = video_pose_signature(item)
        if sig in signatures or not sig.startswith('pose:'):
            raise SystemExit(f'Duplicate/missing pose: {item["video_id"]}')
        signatures.add(sig)
        if item['video_id'] != f'nuplan_video_{count:06d}':
            raise SystemExit('Unexpected merged sample order')
        count += 1
        if item.get('reference_frame_count') != 3 or item.get('generate_frames_for_reference') is not False:
            raise SystemExit('Expected three ungenerated reference frames')
        if [f['frame_index'] for f in item['frames']] != list(range(19)):
            raise SystemExit('Expected ordered t0..t18')
        for frame in item['frames'][3:]:
            if [v['camera'] for v in frame['views']] != [f'CAM_{i:02d}' for i in range(8)]:
                raise SystemExit('Expected eight ordered cameras')
            for view in frame['views']:
                if not (preview / view['image_path']).is_file():
                    raise SystemExit(f'Missing generated image: {view["image_path"]}')
                required.add((sig, frame['frame_index'], view['camera']))
if count != 1000 or len(required) != 128000:
    raise SystemExit(f'Expected 1000 videos / 128000 generated views; got {count}/{len(required)}')
remaining = set(required)
for manifest in sorted(boxes.glob('rank_*/box_manifest.jsonl')):
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sig = video_pose_signature(item)
            for frame in item['frames']:
                if 'boxes_3d' not in frame and 'boxes_3d' not in item:
                    raise SystemExit(f'No 3D box annotation in {manifest}')
                for view in frame['views']:
                    remaining.discard((sig, frame['frame_index'], view['camera']))
if remaining:
    raise SystemExit(f'Missing shared-box matches: {len(remaining)}')
print('[CHECK] 1000 unique videos; 128000 generated views; all shared-box keys matched')
cfg = copy.deepcopy(yaml.safe_load(template.read_text()))
cfg['paths'].update(preview_root=str(preview), shared_box_root=str(boxes),
                    sam3_repo=str(Path(os.environ['SAM3_ROOT']) / 'sam3'),
                    checkpoint=str(checkpoint), output_dir=str(out / 'results'))
cfg['preview'].pop('box_source', None)
cfg['preview'].update(manifest_glob='stflow_manifest.jsonl', skip_reference_frames=True, strict_paths=True)
cfg['shared_box'].update(manifest_glob='rank_*/box_manifest.jsonl', strict_paths=True, strict_match=True)
cfg['sources'] = {'generated': {'type': 'preview_generated', 'group_by_manifest': False}}
cfg['runtime'].update(overwrite=False, limit_frames=0, backend='sam3.1')
config_path = out / 'configs/epipolar32000_generated_only.yaml'
text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
    raise SystemExit(f'Existing config differs; refusing unsafe resume: {config_path}')
config_path.write_text(text)
print(f'[CONFIG] {config_path}')
print(f'[WEIGHT] {checkpoint}')
print(f'[CLASSES] {cfg["model"]["prompts"]}')
PY

CONFIG="${OUT_ROOT}/configs/epipolar32000_generated_only.yaml"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  python run_eval.py --help >/dev/null
  echo '[CHECK] SAM entry imports passed; no GPU inference started'
  exit 0
fi
for rank in 0 1 2 3; do
  file="${OUT_ROOT}/results/records.rank$(printf '%03d' "$rank").jsonl"
  count=0
  if [[ -f "$file" ]]; then count=$(awk 'NF{n++} END{print n+0}' "$file"); fi
  echo "[RESUME] rank${rank}: ${count} / 32000 views"
done
torchrun --standalone --nproc_per_node=4 run_eval.py --config "$CONFIG" --resume \
  2>&1 | tee -a "${OUT_ROOT}/sam3_generated_box_4gpu.log"
echo "[DONE] ${OUT_ROOT}/results/summary.json"
