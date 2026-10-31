#!/bin/bash
# Submit from the repository root after creating output/vision_critic.
# Load the appropriate Python/Anaconda module before submitting this job.
# This job probes CUDA; it does not train or download a model.
#SBATCH --job-name=chameleon-gpu-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=output/vision_critic/slurm-probe-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the Chameleon repository root}"
CHAMELEON_VISION_PYTHON="${CHAMELEON_VISION_PYTHON:-$PWD/.venv-vision/bin/python}"
if [[ ! -x "$CHAMELEON_VISION_PYTHON" ]]; then
    printf '%s\n' 'Set CHAMELEON_VISION_PYTHON to the Python executable in your CUDA experiment environment.' >&2
    exit 2
fi
if [[ ! -f tools/vision_gpu_probe.py ]]; then
    printf '%s\n' 'Submit from the repository root; tools/vision_gpu_probe.py was not found.' >&2
    exit 2
fi
srun "$CHAMELEON_VISION_PYTHON" tools/vision_gpu_probe.py \
    --out "output/vision_critic/gpu-probe-${SLURM_JOB_ID}.json"
