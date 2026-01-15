#!/usr/bin/env bash

# Run all the performance benchmarks and log outputs to a directory including a timestamp, git hash, and a user-provided tag.
#
# Usage: ./scripts/benchmark_all.sh <tag> [additional args forwarded to training/inference scripts]
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
SUBDIR="$(date +%F)/$(date +%H-%M-%S)_perf_$(git rev-parse --short HEAD)${TAG}"
OUTPUT_DIR="${RESULT_ROOT}/${SUBDIR}"
CACHE_DIR_ROOT="/tmp/torchinductor_benchmark_cache"
CACHE_DIR_RUN_ROOT="${CACHE_DIR_ROOT}/${SUBDIR}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists. Exiting." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

LOGFILE="${OUTPUT_DIR}/benchmark.log"

echo "Logging to ${LOGFILE}" 2>&1 | tee -a "${LOGFILE}"

TRAINING_BATCH_SIZES=(120 240)
INFERENCE_BATCH_SIZES=(400 800 1600)

FIXED_ARGS=(
    --amp
    --use_layer_norm
    --norm
    --task=homo
    --seed=42
    --benchmark
    --log_dir="${OUTPUT_DIR}"
    --epochs=6
    "$@"
)

FIXED_TRAINING_ARGS=(
    --precompute_bases
)

RET=0

for batch_size in "${INFERENCE_BATCH_SIZES[@]}"; do
    config_name="inference_batch_${batch_size}"
    args=("${FIXED_ARGS[@]}")

    set +e
    (
        set -x;
        TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR_RUN_ROOT}/${config_name}" \
        CUDA_VISIBLE_DEVICES=0 \
        python -m se3_transformer.runtime.inference \
            "${args[@]}" \
            --batch_size="${batch_size}" \
            --dllogger_name="dllogger_${config_name}.json"
    ) 2>&1 | tee -a "${LOGFILE}"
    ret=$?
    set -e
    if (( ret > RET )); then
        echo "Failed with exit code ${ret}. Continuing with other benchmarks..." 2>&1 | tee -a "${LOGFILE}"
        RET=${ret}
    fi
done

for batch_size in "${TRAINING_BATCH_SIZES[@]}"; do
    args=(
        "${FIXED_ARGS[@]}"
        "${FIXED_TRAINING_ARGS[@]}"
        --batch_size="${batch_size}"
    )

    config_name="train_batch_${batch_size}"
    set +e
    (
        set -x;
        TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR_RUN_ROOT}/${config_name}" \
        CUDA_VISIBLE_DEVICES=0 \
        python -m se3_transformer.runtime.training \
            "${args[@]}" \
            --dllogger_name="dllogger_${config_name}.json"
    ) 2>&1 | tee -a "${LOGFILE}"
    ret=$?
    set -e
    if (( ret > RET )); then
        echo "Failed with exit code ${ret}. Continuing with other benchmarks..." 2>&1 | tee -a "${LOGFILE}"
        RET=${ret}
    fi

    config_name="train_multi_batch_${batch_size}"
    set +e
    (
        set -x;
        TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR_RUN_ROOT}/${config_name}" \
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts=0 --module \
            se3_transformer.runtime.training \
            "${args[@]}" \
            --dllogger_name="dllogger_${config_name}.json"
    ) 2>&1 | tee -a "${LOGFILE}"
    ret=$?
    set -e
    if (( ret > RET )); then
        echo "Failed with exit code ${ret}. Continuing with other benchmarks..." 2>&1 | tee -a "${LOGFILE}"
        RET=${ret}
    fi
done

if (( RET > 0 )); then
    echo "Some benchmarks failed. Please look for errors above." 2>&1 | tee -a "${LOGFILE}"
else
    echo "All benchmarks completed successfully." 2>&1 | tee -a "${LOGFILE}"
fi

echo "Processing log files..."
python scripts/parse_benchmark_outputs.py "${OUTPUT_DIR}" 2>&1 | tee -a "${LOGFILE}"

echo "Done benchmarking. Logs saved to ${LOGFILE}" 2>&1 | tee -a "${LOGFILE}"

exit ${RET}
