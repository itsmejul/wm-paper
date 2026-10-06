#!/usr/bin/env bash
# Submit Qwen3.5 full-parameter training on the Llama-watermarked corpus.
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 5 || "$1" != "capella" ]]; then
    echo "Usage: bash $0 capella [n_samples] [sample_type] [batch_size] [n_eval_samples]" >&2
    exit 64
fi

n_samples="${2:-1000}"
sample_type="${3:-abstracts_only}"
batch_size="${4:-32}"
n_eval_samples="${5:-1000}"

if [[ ! -f data/experiment_config_qwen.json || ! -f data/t_ws/llama/combined_t_ws.json ]]; then
    echo "Run from the repository root with the Llama-watermarked corpus available." >&2
    exit 1
fi

export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-experiment}"
if [[ ! -x "${RUN_VENV_DIR}/bin/python" ]]; then
    echo "Missing experiment environment: ${RUN_VENV_DIR}" >&2
    exit 1
fi
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /etc/profile
    fi
    module purge
    module load release/24.04 GCCcore/13.3.0 Python/3.12.3
fi
if ! "${RUN_VENV_DIR}/bin/python" -c \
    'from importlib.metadata import version; assert version("waterfall") == "0.3.4"; assert version("transformers") == "5.5.0"' \
    >/dev/null 2>&1; then
    echo "The experiment environment must contain waterfall==0.3.4 and transformers==5.5.0." >&2
    exit 1
fi

mkdir -p job_outputs
module_name="src.experiments.main.full_pipeline_pretrained"
args=(
    "$n_samples" "$sample_type" "$batch_size" "$n_eval_samples"
    --profile qwen_on_llama --resume
)

"${RUN_VENV_DIR}/bin/python" -m "$module_name" "${args[@]}" --preflight

submission=$(sbatch --parsable --time=24:00:00 \
    --job-name="qwen-full-llamawm-${n_samples}" \
    scripts/launch_capella.sh "$module_name" "${args[@]}")
job_id="${submission%%;*}"
echo "Submitted Qwen-on-Llama full-parameter job ${job_id}."
echo "Models: full_models/qwen_on_llama/${sample_type}/${n_samples}/${batch_size}"
echo "Results: results/_ablations/qwen_on_llama_full_finetuning/experiment*"
