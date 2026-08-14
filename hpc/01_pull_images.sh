#!/usr/bin/env bash
# STAGE 1a — run on the cluster LOGIN node (needs internet access).
# Pulls the container images:
#   1. NGC PyTorch (for AdsorbML stage 2 on the GPU nodes)
#   2. GPAW image for DFT — see the note below if you have not built one.
#
# MIND THE GPU NODE ARCHITECTURE: per the cluster documentation the GPU nodes
# are Grace Hopper (ARM64). Verify with:
#   srun -p gpu_short --gres=gpu:1 -t 5 uname -m
#   -> "aarch64" = keep GPU_ARCH=arm64 (default)
#   -> "x86_64"  = run with GPU_ARCH=amd64
set -euo pipefail

GPU_ARCH="${GPU_ARCH:-arm64}"
# 25.06 ships torch >= 2.6 with CUDA for ARM — required by fairchem-core >= 2.
NGC_TAG="${NGC_TAG:-25.06-py3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${REPO_ROOT}/hpc/pytorch_ngc_${NGC_TAG}.sif"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

echo ">>> NGC PyTorch ${NGC_TAG} (${GPU_ARCH})"
if [ ! -f "${SIF}" ]; then
  singularity pull --arch "${GPU_ARCH}" "${SIF}" \
    "docker://nvcr.io/nvidia/pytorch:${NGC_TAG}"
else
  echo "    $(basename "${SIF}") already present — skipping"
fi

echo
echo ">>> NO DFT container needed — DFT (stage 4) runs through a pyenv environment"
echo "    (bash scripts/setup_pyenv_env.sh; GPAW installed directly under ~/.pyenv)."
echo
echo "Next step:  sbatch --account=<PROJECT> hpc/02_setup_fairchem_venv.sbatch"
