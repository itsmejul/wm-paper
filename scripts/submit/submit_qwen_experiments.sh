#!/usr/bin/env bash
# Submit one experiment/corpus size per job; never shares a model across jobs.
set -Eeuo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <capella|horeka> [pipeline arguments, e.g. --resume]" >&2
    exit 64
fi
cluster="$1"
case "$cluster" in
    capella) launcher="scripts/launch_capella.sh" ;;
    horeka) launcher="scripts/launch_horeka.sh" ;;
    *) echo "Unknown cluster: $1" >&2; exit 64 ;;
esac
shift
for argument in "$@"; do
    case "$argument" in
        --smoke|--dry-run|--preflight)
            echo "Use the launcher directly for $argument; this helper submits 18 full configurations." >&2
            exit 64 ;;
    esac
done
if [[ ! -f data/experiment_config_qwen.json || \
      ! -f data/t_ws/qwen/combined_t_ws.json || \
      ! -f data/t_ws/qwen/combined_manifest.json || \
      ! -f data/prompts/qwen/prefix_10.json ]]; then
    echo "Missing the canonical Qwen corpus, manifest, prompts, or experiment config." >&2
    echo "Run from the repository root after preparing data/t_ws/qwen and data/prompts/qwen." >&2
    exit 1
fi
export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-experiment}"
if [[ ! -x "${RUN_VENV_DIR}/bin/python" ]]; then
    echo "Missing training environment: ${RUN_VENV_DIR}" >&2
    exit 1
fi
# Capella's Python module supplies libpython3.12.so to venv executables. The
# Slurm launcher loads it inside jobs; load it here too when the login-shell
# preflight cannot start without it.
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if [[ "$cluster" == "capella" ]]; then
        if ! type module >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            source /etc/profile
        fi
        if ! type module >/dev/null 2>&1; then
            echo "Capella's module command is unavailable; run this helper from a login shell." >&2
            exit 1
        fi
        module purge
        module load release/24.04 GCCcore/13.3.0 Python/3.12.3
    fi
fi
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    echo "The training-environment Python cannot start: ${RUN_VENV_DIR}/bin/python" >&2
    exit 1
fi
# Slurm opens stdout/stderr before the job script itself starts.
mkdir -p job_outputs
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        "${RUN_VENV_DIR}/bin/python" -m src.experiments.main.full_pipeline \
            "$n" "$sample_type" 32 1000 --profile qwen --preflight "$@" >/dev/null
    done
done
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="qwen-${sample_type}-${n}" "$launcher" \
            src.experiments.main.full_pipeline "$n" "$sample_type" 32 1000 --profile qwen "$@"
    done
done
