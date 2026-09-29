#!/usr/bin/env bash
set -eo pipefail

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate
set -u

CAMSIM_ROOT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim
WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"
OPENDWM_ROOT="${CAMSIM_ROOT}/OpenDWM"
SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"
PREVIEW_ROOT="${WAYMO_ROOT}/petr_6hz_18000_1000_original_0e57bb3b_merged1000"
OUTPUT_ROOT="${WAYMO_ROOT}/sam3_petr_original_0e57bb3b_generated1000_4gpu"
TEMPLATE="${WAYMO_ROOT}/sam3_pvonly_6hz_generated1000_4gpu/configs/pvonly_6hz_generated_only.yaml"
CONFIG="${OUTPUT_ROOT}/configs/petr_original_0e57bb3b_generated_only.yaml"
RESULTS="${OUTPUT_ROOT}/results"
CHECKPOINT=/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt/sam3.1/sam3.1_multiplex.pt
MANIFEST="${PREVIEW_ROOT}/stflow_manifest.jsonl"
EXPECTED_RANK_ITEMS=32000

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${OPENDWM_ROOT}/src:${SAM3_ROOT}/sam3:${PYTHONPATH:-}"

for required in "${MANIFEST}" "${TEMPLATE}" "${CHECKPOINT}"; do
  [[ -s "${required}" ]] || { echo "[ERROR] missing required file: ${required}" >&2; exit 1; }
done

PREVIEW_ROOT="${PREVIEW_ROOT}" MANIFEST="${MANIFEST}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PREVIEW_ROOT"])
manifest = Path(os.environ["MANIFEST"])
video_ids = set()
count = 0
missing = 0
with manifest.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        item = json.loads(line)
        count += 1
        video_id = item["video_id"]
        if video_id in video_ids:
            raise SystemExit(f"duplicate video_id: {video_id}")
        video_ids.add(video_id)
        frames = item.get("frames", [])
        if len(frames) != 19 or any(len(frame.get("views", [])) != 8 for frame in frames):
            raise SystemExit(f"invalid 19-frame/8-view layout: {video_id}")
        for frame in frames:
            for view in frame["views"]:
                image = Path(view["image_path"])
                if not image.is_absolute():
                    image = root / image
                missing += not image.is_file()
if count != 1000 or len(video_ids) != 1000 or missing:
    raise SystemExit(f"manifest check failed: videos={count}, unique={len(video_ids)}, missing_images={missing}")
print(f"[CHECK] videos={count}, unique={len(video_ids)}, frames=19, views=8, missing_images={missing}")
PY

mkdir -p "${OUTPUT_ROOT}/configs" "${RESULTS}"
TEMPLATE="${TEMPLATE}" PREVIEW_ROOT="${PREVIEW_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" CHECKPOINT="${CHECKPOINT}" CONFIG="${CONFIG}" python - <<'PY'
import copy
import os
from pathlib import Path
import yaml

template = Path(os.environ["TEMPLATE"])
config_path = Path(os.environ["CONFIG"])
cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
generated = cfg.get("sources", {}).get("generated")
if generated is None:
    raise SystemExit("template has no generated source")
cfg["sources"] = {"generated": copy.deepcopy(generated)}
cfg.setdefault("paths", {})["preview_root"] = os.environ["PREVIEW_ROOT"]
cfg["paths"]["output_dir"] = str(Path(os.environ["OUTPUT_ROOT"]) / "results")
cfg["paths"]["checkpoint"] = os.environ["CHECKPOINT"]
cfg.setdefault("preview", {})["box_source"] = "embedded_manifest"
cfg["preview"]["manifest_glob"] = "stflow_manifest.jsonl"
cfg.setdefault("runtime", {})["overwrite"] = False
cfg["runtime"]["limit_frames"] = 0
config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"[CONFIG] {config_path}")
print(f"[CONFIG] preview_root={cfg['paths']['preview_root']}")
print(f"[CONFIG] output_dir={cfg['paths']['output_dir']}")
print(f"[CONFIG] checkpoint={cfg['paths']['checkpoint']}")
PY

print_resume_state() {
  local rank records completed
  for rank in 0 1 2 3; do
    records="${RESULTS}/records.rank$(printf '%03d' "${rank}").jsonl"
    completed=0
    if [[ -f "${records}" ]]; then
      completed=$(awk 'NF{n++} END{print n+0}' "${records}")
    fi
    echo "[RESUME] rank$(printf '%03d' "${rank}"): ${completed}/${EXPECTED_RANK_ITEMS}"
  done
}

print_resume_state
cd "${SAM3_ROOT}"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  python run_eval.py --help >/dev/null
  echo "[CHECK] evaluator import passed; GPU evaluation not launched"
  exit 0
fi
torchrun --standalone --nproc_per_node=4 run_eval.py --config "${CONFIG}" --resume \
  2>&1 | tee -a "${OUTPUT_ROOT}/sam3_generated_box_4gpu.log"

print_resume_state
SUMMARY="${RESULTS}/summary.json"
[[ -s "${SUMMARY}" ]] || { echo "[ERROR] missing summary: ${SUMMARY}" >&2; exit 1; }
SUMMARY="${SUMMARY}" CHECKPOINT="${CHECKPOINT}" EXPECTED_RANK_ITEMS="${EXPECTED_RANK_ITEMS}" RESULTS="${RESULTS}" python - <<'PY'
import json
import os
from pathlib import Path

expected = int(os.environ["EXPECTED_RANK_ITEMS"])
results = Path(os.environ["RESULTS"])
counts = []
for rank in range(4):
    records = results / f"records.rank{rank:03d}.jsonl"
    count = sum(bool(line.strip()) for line in records.open(encoding="utf-8"))
    counts.append(count)
if counts != [expected] * 4:
    raise SystemExit(f"incomplete rank records: {counts}")
summary = json.loads(Path(os.environ["SUMMARY"]).read_text(encoding="utf-8"))
if summary.get("model", {}).get("checkpoint_path") != os.environ["CHECKPOINT"]:
    raise SystemExit("summary checkpoint path does not match requested local checkpoint")
generated = summary.get("sources", {}).get("generated", {})
if generated.get("frames") != 128000:
    raise SystemExit(f"unexpected generated frame count: {generated.get('frames')}")
print("[CHECK] complete rank counts:", counts)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[DONE] ${SUMMARY}"
