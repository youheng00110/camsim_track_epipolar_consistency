#!/usr/bin/env bash

set +e
set +u

source /inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/envs/lyhdwm/bin/activate

CAMSIM_ROOT="/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim"

WAYMO_ROOT="${CAMSIM_ROOT}/lyh_output/eval/waymo"

SAM3_ROOT="${CAMSIM_ROOT}/sam3-eval/sam3-eval"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Current generated-only Waymo shard size with four ranks.  Keep this as an
# explicit diagnostic target; it does not alter evaluator sharding.
EXPECTED_RANK_ITEMS="${EXPECTED_RANK_ITEMS:-32000}"

export PYTHONPATH="${SAM3_ROOT}/sam3:${PYTHONPATH:-}"

cd "${SAM3_ROOT}" || true


echo
echo "===================================================================================================="
echo "WAYMO GENERATED-ONLY SAM3 — 4 GPU — THREE METHODS SEQUENTIALLY"
echo "===================================================================================================="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo


# ================================================================================================
# Generate clean generated-only YAMLs with separate output directories.
# This does NOT touch the completed paired-real results.
# ================================================================================================

python - <<'PY'
from pathlib import Path
import copy
import yaml

root = Path(
    "/inspire/qb-ilm/project/quantum-artificial-intelligence/"
    "yanjunchi-24040/songbur/camsim/lyh_output/eval/waymo"
)

methods = [
    "pvonly_6hz",
    "nocondition_6hz",
    "plucker_6hz",
]

for method in methods:

    src = (
        root
        / f"sam3_{method}_merged1000_box"
        / "configs"
        / f"{method}.yaml"
    )

    new_root = (
        root
        / f"sam3_{method}_generated1000_4gpu"
    )

    dst = (
        new_root
        / "configs"
        / f"{method}_generated_only.yaml"
    )

    result_dir = (
        new_root
        / "results"
    )

    print()
    print("=" * 100)
    print(method)
    print("=" * 100)

    if not src.is_file():
        print("[ERROR] source YAML missing:")
        print(src)
        continue

    try:
        cfg = yaml.safe_load(
            src.read_text(
                encoding="utf-8"
            )
        )

        sources = cfg.get(
            "sources",
            {},
        )

        if "generated" not in sources:
            print("[ERROR] no generated source")
            print("sources =", list(sources))
            continue

        # --------------------------------------------------------
        # Only generated.
        # --------------------------------------------------------

        cfg["sources"] = {
            "generated": copy.deepcopy(
                sources["generated"]
            )
        }

        # --------------------------------------------------------
        # New independent output.
        # Never touch paired-real results.
        # --------------------------------------------------------

        paths = cfg.setdefault(
            "paths",
            {},
        )

        paths["output_dir"] = str(
            result_dir.resolve()
        )

        # Keep:
        # preview_root
        # embedded_manifest box_source
        # prompts
        # projection
        # class mapping
        # all Waymo geometry
        #
        # exactly as they currently are.

        runtime = cfg.setdefault(
            "runtime",
            {},
        )

        # Resume is the default for this generated-only workflow.  Do not
        # let prepare_output_directory remove existing rank records.
        runtime["overwrite"] = False
        runtime["limit_frames"] = 0

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dst.write_text(
            yaml.safe_dump(
                cfg,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        print("source YAML  =", src)
        print("new YAML     =", dst)
        print("preview_root =", paths.get("preview_root"))
        print("output_dir   =", paths.get("output_dir"))
        print("sources      =", list(cfg["sources"].keys()))
        print(
            "box_source   =",
            cfg.get(
                "box_source",
                cfg.get("embedded_boxes"),
            ),
        )

    except Exception as e:
        print("[YAML ERROR]", repr(e))
        print("CONTINUE")
PY


print_resume_state()
{
    METHOD="$1"
    ROOT="$2"

    echo "[RESUME STATE] ${METHOD}"
    for RANK in 0 1 2 3
    do
        RECORDS="${ROOT}/results/records.rank$(printf '%03d' "${RANK}").jsonl"
        if [ -f "${RECORDS}" ]; then
            COMPLETED=$(awk 'NF{n++} END{print n+0}' "${RECORDS}")
        else
            COMPLETED=0
        fi
        RANK_NAME=$(printf '%03d' "${RANK}")
        if [ "${COMPLETED}" -gt "${EXPECTED_RANK_ITEMS}" ]; then
            echo "[WARNING] rank${RANK_NAME}: ${COMPLETED} / ${EXPECTED_RANK_ITEMS} (exceeds expected total)"
        else
            echo "rank${RANK_NAME}: ${COMPLETED} / ${EXPECTED_RANK_ITEMS}"
        fi
    done
}


run_method()
{
    METHOD="$1"

    ROOT="${WAYMO_ROOT}/sam3_${METHOD}_generated1000_4gpu"

    CONFIG="${ROOT}/configs/${METHOD}_generated_only.yaml"

    LOG="${ROOT}/sam3_generated_4gpu.log"

    print_resume_state "${METHOD}" "${ROOT}"


    echo
    echo
    echo "===================================================================================================="
    echo "START: ${METHOD}"
    echo "===================================================================================================="
    echo "CONFIG=${CONFIG}"
    echo "LOG=${LOG}"
    echo


    if [ ! -f "${CONFIG}" ]; then
        echo "[ERROR] CONFIG MISSING"
        echo "${CONFIG}"
        echo "[CONTINUE TO NEXT METHOD]"
        return 0
    fi


    echo "COMMAND:"
    echo "torchrun --standalone --nproc_per_node=4 run_eval.py --config ${CONFIG} --resume"
    echo


    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    echo "START_TIME=${START_TIME}"


    set -x

    torchrun \
        --standalone \
        --nproc_per_node=4 \
        run_eval.py \
        --config "${CONFIG}" \
        --resume \
        2>&1 | tee "${LOG}"

    STATUS=${PIPESTATUS[0]}

    set +x


    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')


    echo
    echo "===================================================================================================="
    echo "FINISHED: ${METHOD}"
    echo "===================================================================================================="
    echo "STATUS     = ${STATUS}"
    echo "START_TIME = ${START_TIME}"
    echo "END_TIME   = ${END_TIME}"
    echo "LOG        = ${LOG}"
    echo


    if [ "${STATUS}" -ne 0 ]; then
        echo "[FAILED BUT CONTINUE TO NEXT METHOD]"
        echo
        echo "Last 100 lines:"
        tail -n 100 "${LOG}" 2>/dev/null || true
        echo
    else
        echo "[SUCCESS] ${METHOD}"
    fi


    # Important:
    # no exit here. Always continue to next method.
    return 0
}


# ================================================================================================
# Sequential 4-GPU runs
# ================================================================================================

run_method "pvonly_6hz"

run_method "nocondition_6hz"

run_method "plucker_6hz"


echo
echo
echo "===================================================================================================="
echo "ALL THREE METHODS ATTEMPTED"
echo "===================================================================================================="

for METHOD in \
    pvonly_6hz \
    nocondition_6hz \
    plucker_6hz
do

    ROOT="${WAYMO_ROOT}/sam3_${METHOD}_generated1000_4gpu"

    echo
    echo "${METHOD}:"
    echo "  results = ${ROOT}/results"
    echo "  log     = ${ROOT}/sam3_generated_4gpu.log"

done

echo
echo "DONE."
