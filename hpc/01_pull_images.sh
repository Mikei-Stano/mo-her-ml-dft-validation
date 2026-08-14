#!/usr/bin/env bash
# FÁZA 1a — spusti na PERUN LOGIN uzle (potrebuje internet).
# Stiahne kontajnerové obrazy:
#   1. NGC PyTorch (pre stage 2 AdsorbML na GPU uzloch)
#   2. GPAW obraz pre DFT — ak ho nemáš zbuildovaný, pozri poznámku nižšie.
#
# POZOR na architektúru GPU uzlov: podľa PERUN dokumentácie sú GPU uzly
# Grace Hopper (ARM64). Over si to:  srun -p gpu_short --gres=gpu:1 -t 5 uname -m
#   -> "aarch64" = nechaj GPU_ARCH=arm64 (default)
#   -> "x86_64"  = spusti s GPU_ARCH=amd64
set -euo pipefail

GPU_ARCH="${GPU_ARCH:-arm64}"
# 25.06 obsahuje torch >= 2.6 s CUDA pre ARM — nutne pre fairchem-core >= 2.
NGC_TAG="${NGC_TAG:-25.06-py3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${REPO_ROOT}/hpc/pytorch_ngc_${NGC_TAG}.sif"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

echo ">>> NGC PyTorch ${NGC_TAG} (${GPU_ARCH})"
if [ ! -f "${SIF}" ]; then
  singularity pull --arch "${GPU_ARCH}" "${SIF}" \
    "docker://nvcr.io/nvidia/pytorch:${NGC_TAG}"
else
  echo "    $(basename "${SIF}") už existuje — preskakujem"
fi

echo
echo ">>> DFT kontajner NETREBA — DFT (stage 4) beží cez pyenv prostredie"
echo "    (bash scripts/setup_pyenv_env.sh; GPAW nainštalovaný priamo v ~/.pyenv)."
echo
echo "Ďalší krok:  sbatch --account=<PROJEKT> hpc/02_setup_fairchem_venv.sbatch"
