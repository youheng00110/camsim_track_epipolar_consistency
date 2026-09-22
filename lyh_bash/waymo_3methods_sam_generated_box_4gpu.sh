#!/usr/bin/env bash

set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"
WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"
TEMPLATE="${WAYMO_ROOT}/sam3_pvonly_6hz_generated1000_4gpu/configs/pvonly_6hz_generated_only.yaml"
EXPECTED_RANK_ITEMS="${EXPECTED_RANK_ITEMS:-32000}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OPENDWM_ROOT}/src:${SAM3_ROOT}/sam3:${PYTHONPATH:-}"
cd "${SAM3_ROOT}" || exit 1

prepare_manifest() {
  local method="$1"
  local src="${WAYMO_ROOT}/${method}_6hz_18000_1000"
  local dst="${WAYMO_ROOT}/${method}_6hz_18000_1000_merged1000"
  local manifest="${dst}/stflow_manifest.jsonl"
  if [ -f "${manifest}" ]; then
    echo "[MERGE] reuse ${manifest}"
    return 0
  fi
  if [ ! -d "${src}" ]; then
    echo "[ERROR] missing source root: ${src}"
    return 1
  fi
  if [ -e "${dst}" ]; then
    local stale="${dst}.stale_$(date +%Y%m%d_%H%M%S)"
    echo "[MERGE] moving incomplete target to ${stale}"
    mv "${dst}" "${stale}" || return 1
  fi
  python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "${src}" \
    --output-root "${dst}" \
    --dataset-name waymo \
    --max-videos 1000 \
    --overwrite
}

for method in box petr implicit; do
  prepare_manifest "${method}" || exit 1
done

cd "${SAM3_ROOT}" || exit 1

# Build generated-only configs for the current box / petr / implicit outputs.
TEMPLATE="${TEMPLATE}" WAYMO_ROOT="${WAYMO_ROOT}" python - <<'PY'
from pathlib import Path
import copy, os, yaml

template=Path(os.environ['TEMPLATE'])
root=Path(os.environ['WAYMO_ROOT'])
if not template.is_file():
    raise SystemExit(f'missing template: {template}')
base=yaml.safe_load(template.read_text(encoding='utf-8'))
for method in ('box','petr','implicit'):
    preview_root=root/f'{method}_6hz_18000_1000_merged1000'
    out_root=root/f'sam3_{method}_6hz_generated1000_4gpu'
    cfg=copy.deepcopy(base)
    cfg.setdefault('paths',{})['preview_root']=str(preview_root)
    cfg['paths']['output_dir']=str(out_root/'results')
    sources=cfg.get('sources',{})
    if 'generated' not in sources:
        raise SystemExit(f'generated source missing in template: {template}')
    cfg['sources']={'generated':copy.deepcopy(sources['generated'])}
    cfg.setdefault('preview',{})['box_source']='embedded_manifest'
    cfg['preview']['manifest_glob']='stflow_manifest.jsonl'
    cfg.setdefault('runtime',{})['overwrite']=False
    cfg['runtime']['limit_frames']=0
    (out_root/'configs').mkdir(parents=True,exist_ok=True)
    (out_root/'results').mkdir(parents=True,exist_ok=True)
    dst=out_root/'configs'/f'{method}_generated_only.yaml'
    dst.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(f'[CONFIG] {method}: {dst}')
    print(f'         preview_root={preview_root}')
    print(f'         output_dir={out_root/"results"}')
PY

print_resume_state() {
  local method="$1"
  local root="${WAYMO_ROOT}/sam3_${method}_6hz_generated1000_4gpu"
  echo "[RESUME STATE] ${method}"
  for rank in 0 1 2 3; do
    local f="${root}/results/records.rank$(printf '%03d' "${rank}").jsonl"
    local n=0
    if [ -f "${f}" ]; then n=$(awk 'NF{n++} END{print n+0}' "${f}"); fi
    echo "rank$(printf '%03d' "${rank}"): ${n} / ${EXPECTED_RANK_ITEMS}"
  done
}

run_method() {
  local method="$1"
  local root="${WAYMO_ROOT}/sam3_${method}_6hz_generated1000_4gpu"
  local config="${root}/configs/${method}_generated_only.yaml"
  local log="${root}/sam3_generated_box_4gpu.log"
  print_resume_state "${method}"
  if [ ! -f "${config}" ]; then echo "[ERROR] missing ${config}"; return 1; fi
  echo "[START] ${method}"
  torchrun --standalone --nproc_per_node=4 run_eval.py --config "${config}" --resume 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  echo "[DONE] ${method} status=${status} log=${log}"
  return "${status}"
}

S1=0; S2=0; S3=0
run_method box || S1=$?
run_method petr || S2=$?
run_method implicit || S3=$?
echo "[FINAL] box=${S1} petr=${S2} implicit=${S3}"
[ "${S1}" -eq 0 ] && [ "${S2}" -eq 0 ] && [ "${S3}" -eq 0 ]
