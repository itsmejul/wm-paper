#!/usr/bin/env bash
# Submit open-keyspace evaluation plus auxiliary metrics for Qwen-on-Llama.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <capella|horeka>" >&2
    exit 64
fi

cluster="$1"
case "$cluster" in
    capella) launcher="scripts/launch_capella.sh" ;;
    horeka) launcher="scripts/launch_horeka.sh" ;;
    *) echo "Unknown cluster: $cluster" >&2; exit 64 ;;
esac

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
    if [[ "$cluster" == "capella" ]]; then
        if ! type module >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            source /etc/profile
        fi
        module purge
        module load release/24.04 GCCcore/13.3.0 Python/3.12.3
    fi
fi
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    echo "The runtime environment Python cannot start: ${RUN_VENV_DIR}/bin/python" >&2
    exit 1
fi

mkdir -p job_outputs

# Validate the 1,000-sample Experiment 1 open-keyspace ablation before
# creating any Slurm jobs.
"${RUN_VENV_DIR}/bin/python" -m src.experiments.main.qwen_on_llama_open_pipeline \
    1000 abstracts_only 32 1000 --preflight >/dev/null

# The open-keyspace ablation is intentionally only Experiment 1 at N=1,000.
# Avoid consuming a GPU allocation when all 20 checkpoints are already done.
open_complete=true
for epoch in $(seq 5 5 100); do
    epoch_dir="results/qwen_on_llama/experiment1/prefix_10/1000/32/${epoch}"
    for filename in answers_open.json eval_manifest_open.json verification_open.json; do
        [[ -f "${epoch_dir}/${filename}" ]] || open_complete=false
    done
done
if [[ "$open_complete" != true ]]; then
    sbatch --job-name="qol-open-abstracts_only-1000" "$launcher" \
        src.experiments.main.qwen_on_llama_open_pipeline \
        1000 abstracts_only 32 1000 --resume
else
    echo "Open-keyspace N=1000 is already complete; not submitting it again."
fi

# One auxiliary-metric job per configuration. Splitting the former three long
# sweeps prevents the 5k/50k configurations from exhausting one Slurm limit.
# Each job evaluates the best and final epochs and skips files already present.
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="qol-metrics-${sample_type}-${n}" "$launcher" \
            src.experiments.main.similarity_eval \
            --profile qwen_on_llama \
            --metrics bm25 cosine bertscore bigram lcs \
            --n_samples "$n" --sample_type "$sample_type" \
            --batch_size 32 --skip-existing
    done
done
