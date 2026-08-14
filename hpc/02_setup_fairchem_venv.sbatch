#!/usr/bin/env bash
# STAGE 1b — one-off setup of the fairchem environment ON A GPU NODE.
# Must run on a GPU node: the environment inherits the node architecture and
# the script verifies CUDA and the UMA checkpoint before you rely on them.
#
# Usage:  sbatch --account=<PROJECT> hpc/02_setup_fairchem_venv.sbatch
#
#SBATCH --job-name=fairchem-setup
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=hpc/setup_%j.out
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
NGC_TAG="${NGC_TAG:-25.06-py3}"
SIF="${REPO_ROOT}/hpc/pytorch_ngc_${NGC_TAG}.sif"
# The container has no python3-venv, so we install with pip --user into a
# dedicated directory under $HOME (PYTHONUSERBASE), isolated from pyenv.
USERBASE="${USERBASE:-${HOME}/fairchem-userbase}"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

echo ">>> node: $(hostname), arch: $(uname -m)"
if [[ ! -f "${SIF}" ]]; then
  echo "Image not found: ${SIF} — run bash hpc/01_pull_images.sh first" >&2
  exit 1
fi

# Clean start: an older userbase may hold a CPU torch that shadows the container.
rm -rf "${USERBASE}"
mkdir -p "${USERBASE}"

singularity exec --nv "${SIF}" bash -c "
  set -e
  export PYTHONUSERBASE='${USERBASE}'
  echo '=== torch INSIDE THE CONTAINER (before installing anything) ==='
  python3 -c 'import torch; print(torch.__version__, \"CUDA:\", torch.cuda.is_available())'
  # CRITICAL: do NOT install torch — use the CUDA torch from the container.
  # fairchem pins torch~=2.8.0 while the container ships a pre-release
  # 2.8.0a0+nv that fails the pin formally but works fine, so fairchem is
  # installed with --no-deps and its dependencies listed explicitly (the list
  # below is taken from a verified working run).
  python3 -m pip install --user --no-cache-dir --no-deps fairchem-core==2.14.0
  python3 -m pip install --user --no-cache-dir \
    ase pandas pyyaml hydra-core omegaconf e3nn lmdb orjson monty pymatgen \
    submitit torchtnt wandb 'huggingface_hub[hf_xet]' requests tqdm \
    ray fastapi uvicorn clusterscope
  echo '=== verification ==='
  python3 - <<'PY'
import torch, fairchem
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
assert '+cpu' not in torch.__version__, 'pip replaced the container torch with a CPU build!'
assert torch.cuda.is_available(), 'CUDA is NOT available!'
print('GPU:', torch.cuda.get_device_name(0))
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator.from_model_checkpoint('uma-m-1p1', task_name='oc20', device='cuda')
print('UMA-M checkpoint loaded OK (on GPU)')
PY
"
echo
echo "Setup DONE. Next:"
echo "  1) python3 hpc/03_make_manifest.py        (login node is enough)"
echo "  2) sbatch --account=<PROJECT> hpc/04_submit_adsorbml.sbatch"
