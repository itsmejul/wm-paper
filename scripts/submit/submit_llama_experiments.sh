#!/usr/bin/env bash
# Submit the isolated 3 experiments x 6 corpus sizes Llama EOS-fix grid.
set -Eeuo pipefail

if [[ $# -ne 1 || "$1" != "capella" ]]; then
    echo "Usage: bash $0 capella" >&2
    exit 64
fi

if [[ ! -f data/t_ws/config_llama.json || ! -f data/t_ws/llama/combined_t_ws.json ]]; then
    echo "Run from the repository root with the Llama-watermarked corpus available." >&2
    exit 1
fi

export RUN_VENV_DIR="${RUN_VENV_DIR:-${PWD}/.venv-experiment}"
if [[ ! -x "${RUN_VENV_DIR}/bin/python" ]]; then
    echo "Missing maintained experiment environment: ${RUN_VENV_DIR}" >&2
    exit 1
fi

# Capella's Python module supplies libpython3.12.so to venv executables.
if ! "${RUN_VENV_DIR}/bin/python" -c 'pass' >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /etc/profile
    fi
    if ! type module >/dev/null 2>&1; then
        echo "Capella's module command is unavailable; use a login shell." >&2
        exit 1
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
for sample_type in abstracts_only abstracts_and_titles questions; do
    for n in 100 500 1000 5000 10000 50000; do
        sbatch --job-name="llama-eos-${sample_type}-${n}" \
            scripts/launch_capella.sh \
            src.experiments.main.full_pipeline_llama_eosfix \
            "$n" "$sample_type" 32 1000
    done
done
