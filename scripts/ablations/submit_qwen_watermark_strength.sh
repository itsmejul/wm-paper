#!/usr/bin/env bash
# Reproduce the final Qwen watermark-strength sweep with the corrected length limit.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <capella|horeka>" >&2
    exit 64
fi

case "$1" in
    capella) launcher="scripts/launch_capella.sh" ;;
    horeka) launcher="scripts/launch_horeka.sh" ;;
    *) echo "Unknown cluster: $1" >&2; exit 64 ;;
esac

if [[ ! -f data/t_ws/config_qwen.json || ! -f data/keys.json ]]; then
    echo "Run this helper from the repository root." >&2
    exit 1
fi

export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-watermark}"
if [[ ! -x "${RUN_VENV_DIR}/bin/python" ]]; then
    echo "Missing watermark environment: ${RUN_VENV_DIR}" >&2
    exit 1
fi
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if [[ "$1" == "capella" ]]; then
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
module_name="src.experiments.ablations.qwen_kappa_ablation"
kappas=(2 3 3.5 4 6 8 10 12)

for kappa in "${kappas[@]}"; do
    result_dir="results/_ablations/watermark/qwen_sampling_source_lengthfix/kappa_${kappa}__temperature_1__top_p_1__top_k_0"
    args=("$kappa" --temperature 1.0 --top-p 1.0 --top-k 0 --token-length-limit)
    "${RUN_VENV_DIR}/bin/python" -m "$module_name" "${args[@]}" --dry-run >/dev/null
    if [[ -f "${result_dir}/summary.json" ]]; then
        echo "Kappa ${kappa} is already complete; not submitting it again."
        continue
    fi
    sbatch --job-name="qwen-lenfix-k${kappa}" "$launcher" "$module_name" "${args[@]}"
done
