#!/bin/bash -l
#SBATCH --job-name=watermark-capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=108G
#SBATCH --licenses=cat
#SBATCH --time=40:00:00
#SBATCH --export=ALL
#SBATCH --output=job_outputs/slurm-%x-%j.out
#SBATCH --error=job_outputs/slurm-%x-%j.err

# Submit from the repository root with job_outputs/ already present. Slurm
# opens these files before running this script (Git tracks a .gitkeep there).

set -Eeuo pipefail

usage() {
    echo "Usage: sbatch $0 <python-module-path> [arguments ...]" >&2
    echo "Example: sbatch $0 src.data_creation.create_t_ws 1 --config data/t_ws/config_qwen.json" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 64
fi

readonly RUN_MODULE="$1"
shift

# Submit from the repository root. This avoids hard-coding whether the checkout
# directory on Capella is named cllm-waterfall or watermark-attr.
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly VENV_DIR="${RUN_VENV_DIR:-${PROJECT_DIR}/.venv-watermark}"
readonly LOG_DIR="${PROJECT_DIR}/logs"

if [[ ! -d "${PROJECT_DIR}/src" || ! -f "${PROJECT_DIR}/README.md" ]]; then
    echo "Submit this script from the repository root." >&2
    echo "SLURM_SUBMIT_DIR=${PROJECT_DIR}" >&2
    exit 1
fi

module purge
module load release/24.04 GCCcore/13.3.0 Python/3.12.3

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Runtime environment not found: ${VENV_DIR}" >&2
    echo "Create it once from req-watermark.txt before submitting." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"

mkdir -p "${LOG_DIR}" "${PROJECT_DIR}/.cache/huggingface" "${PROJECT_DIR}/.cache/torch"

export PYTHONPATH="${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor-${SLURM_JOB_ID}"
export UNSLOTH_COMPILE_LOCATION="${PROJECT_DIR}/.cache/unsloth-${SLURM_JOB_ID}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty; Slurm did not expose the requested GPU." >&2
    exit 1
fi

NVIDIA_SMI_PID=""
cleanup() {
    if [[ -n "${NVIDIA_SMI_PID}" ]]; then
        kill "${NVIDIA_SMI_PID}" 2>/dev/null || true
        wait "${NVIDIA_SMI_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-default}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Python executable: $(command -v python)"
echo "Python: $(python --version 2>&1)"
echo "Module: ${RUN_MODULE}"
echo "Arguments: $*"

nvidia-smi
nvidia-smi dmon -s u -d 10 > "${LOG_DIR}/gpu_util_${SLURM_JOB_ID}.log" &
NVIDIA_SMI_PID=$!

python -m "${RUN_MODULE}" "$@"
