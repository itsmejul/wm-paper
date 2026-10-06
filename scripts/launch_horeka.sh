#!/bin/bash -l
#SBATCH --job-name=watermark-horeka
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=19
#SBATCH --gres=gpu:1
#SBATCH --mem=124375mb
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL
#SBATCH --output=/hkfs/work/workspace/scratch/id_qry6439-watermark_paper/watermark-attr/job_outputs/slurm-%x-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/id_qry6439-watermark_paper/watermark-attr/job_outputs/slurm-%x-%j.err

# job_outputs/ must exist before submission; Git tracks a .gitkeep there.

set -Eeuo pipefail

readonly WORKSPACE_DIR="/hkfs/work/workspace/scratch/id_qry6439-watermark_paper"
readonly PROJECT_DIR="${WORKSPACE_DIR}/watermark-attr"
readonly VENV_DIR="${RUN_VENV_DIR:-${PROJECT_DIR}/.venv-watermark}"
readonly LOG_DIR="${PROJECT_DIR}/logs"

usage() {
    echo "Usage: sbatch $0 <python-module-path> [arguments ...]" >&2
    echo "Example: sbatch $0 src.experiments.main.full_pipeline --help" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 64
fi

readonly RUN_SCRIPT="$1"
shift

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Project directory not found: ${PROJECT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "runtime environment not found: ${VENV_DIR}" >&2
    echo "Create it once with a workspace-local Python >=3.10 before submitting experiments." >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"

mkdir -p "${LOG_DIR}" "${WORKSPACE_DIR}/.cache/huggingface" "${WORKSPACE_DIR}/.cache/torch"

export PYTHONPATH="${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export TORCH_HOME="${WORKSPACE_DIR}/.cache/torch"
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
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Python executable: $(command -v python)"
echo "Python: $(python --version 2>&1)"
echo "Module: ${RUN_SCRIPT}"
echo "Arguments: $*"

nvidia-smi
nvidia-smi dmon -s u -d 10 > "${LOG_DIR}/gpu_util_${SLURM_JOB_ID}.log" &
NVIDIA_SMI_PID=$!

python -m "${RUN_SCRIPT}" "$@"
