#!/usr/bin/env bash

# Run all training accuracy benchmarks and log outputs to a directory including a timestamp, git hash, and a user-provided tag.
#
# Usage: ./scripts/benchmark_train_accuracy.sh <tag> [additional args forwarded to training/inference scripts]
#
# The tag is appended to the output directory name to help identify different
# runs. No separator is added, so you will likely want to prefix it with an
# underscore.
#
# Script output is tee-ed to a log file in the output directory. To avoid
# getting output in the terminal (e.g. if you want to run in the background),
# you can redirect stdout to /dev/null. The only output that is sent to stderr
# is an immediate error message if the output directory already exists (quite
# unlikely given it includes a timestamp).

set -euo pipefail

TAG="$1"
shift

RESULT_ROOT="results/"
SUBDIR="$(date +%F)/$(date +%H-%M-%S)_train_$(git rev-parse --short HEAD)${TAG}"
OUTPUT_DIR="${RESULT_ROOT}/${SUBDIR}"
CACHE_DIR_ROOT="/tmp/torchinductor_benchmark_cache"
CACHE_DIR_RUN_ROOT="${CACHE_DIR_ROOT}/${SUBDIR}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists. Exiting." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

LOGFILE="${OUTPUT_DIR}/train.log"

TASK=homo

FIXED_ARGS=(
    --amp
    --use_layer_norm
    --norm
    --task="${TASK}"
    --seed=42
    --batch_size=240
    --weight_decay=0.1
    --precompute_bases
    --log_dir="${OUTPUT_DIR}"
    "$@"
)

MULTI_GPU_ARGS=(
    "${FIXED_ARGS[@]}"
    --epochs 130
    --lr=0.01
    --min_lr=0.00001
    --save_ckpt_path="${OUTPUT_DIR}/model_qm9_${TASK}_multi.pth"
    --dllogger_name="dllogger_train_multi.json"
)

SINGLE_GPU_ARGS=(
    "${FIXED_ARGS[@]}"
    --epochs=100
    --lr=0.002
    --save_ckpt_path="${OUTPUT_DIR}/model_qm9_${TASK}.pth"
    --dllogger_name="dllogger_train.json"
)

RET=0

set +e
(
  set -x;
  TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR_RUN_ROOT}/train_multi" python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
    se3_transformer.runtime.training \
    "${MULTI_GPU_ARGS[@]}"
) 2>&1 | tee -a "${LOGFILE}"
ret=$?
set -e
if (( ret > RET )); then
    echo "Multi-GPU failed with exit code ${ret}. Continuing with other benchmarks..." 2>&1 | tee -a "${LOGFILE}"
    RET=${ret}
fi

(
    set -x;
    TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR_RUN_ROOT}/train_single" CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.training \
    "${SINGLE_GPU_ARGS[@]}"
) 2>&1 | tee -a "${LOGFILE}"
ret=$?
set -e
if (( ret > RET )); then
    echo "Single-GPU failed with exit code ${ret}. Continuing with other benchmarks..." 2>&1 | tee -a "${LOGFILE}"
    RET=${ret}
fi

if (( RET > 0 )); then
    echo "Some benchmarks failed. Please look for errors above." 2>&1 | tee -a "${LOGFILE}"
else
    echo "All benchmarks completed successfully." 2>&1 | tee -a "${LOGFILE}"
fi

python scripts/parse_training_outputs.py "${OUTPUT_DIR}" 2>&1 | tee -a "${LOGFILE}"
python scripts/parse_training_curves.py "${OUTPUT_DIR}" 2>&1 | tee -a "${LOGFILE}"

exit ${RET}
