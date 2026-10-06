#!/usr/bin/env bash
# Finish missing closed evaluation and auxiliary metrics for Llama EOS-fix.
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != "capella" ]]; then
    echo "Usage: bash $0 capella" >&2
    exit 64
fi
if [[ ! -f data/t_ws/llama/combined_t_ws.json ]]; then
    echo "Run from the repository root with the Llama-watermarked corpus available." >&2
    exit 1
fi

export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-experiment}"
if [[ ! -x "${RUN_VENV_DIR}/bin/python" ]]; then
    echo "Missing maintained experiment environment: ${RUN_VENV_DIR}" >&2
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
eval_job=""
if [[ ! -f results/llama/experiment2/titles/100/32/100/verification_closed.json ]]; then
    pipeline_args=()
    action="training and evaluation"
    if [[ -f lora_adapters/llama-on-llama/abstracts_and_titles/100/32/100/lora_adapter/adapter_config.json ]]; then
        pipeline_args=("--eval-only")
        action="evaluation"
    fi
    submission=$(sbatch --parsable --job-name="llama-eos-eval-titles-100" \
        scripts/launch_capella.sh \
        src.experiments.main.full_pipeline_llama_eosfix \
        100 abstracts_and_titles 32 1000 "${pipeline_args[@]}")
    eval_job="${submission%%;*}"
    echo "Submitted missing Experiment 2 N=100 ${action} as job ${eval_job}."
fi

for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        dependency=()
        if [[ "$sample_type" == "abstracts_and_titles" && "$n" == "100" && -n "$eval_job" ]]; then
            dependency=("--dependency=afterok:${eval_job}")
        fi
        sbatch "${dependency[@]}" \
            --job-name="llama-eos-metrics-${sample_type}-${n}" \
            scripts/launch_capella.sh \
            src.experiments.main.similarity_eval \
            --profile llama_eosfix \
            --metrics bm25 cosine bertscore bigram lcs \
            --n_samples "$n" --sample_type "$sample_type" \
            --batch_size 32 --skip-existing
    done
done
