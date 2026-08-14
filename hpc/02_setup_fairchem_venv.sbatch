#!/usr/bin/env bash
# FÁZA 1b — jednorazový setup fairchem prostredia NA GPU UZLE.
# Musí bežať na GPU uzle (venv zdieľa architektúru uzla + overí CUDA/UMA).
#
# Spustenie:  sbatch --account=<PROJEKT> hpc/02_setup_fairchem_venv.sbatch
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
# Kontajner nemá python3-venv -> inštalujeme cez pip --user do vyhradeného
# adresára v $HOME (PYTHONUSERBASE), izolovaného od pyenv prostredia.
USERBASE="${USERBASE:-${HOME}/fairchem-userbase}"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

echo ">>> uzol: $(hostname), arch: $(uname -m)"
if [[ ! -f "${SIF}" ]]; then
  echo "Obraz nenájdený: ${SIF} — spusti najprv: bash hpc/01_pull_images.sh" >&2
  exit 1
fi

# Čistý štart: staré userbase mohlo obsahovať CPU torch, ktorý tieni kontajner.
rm -rf "${USERBASE}"
mkdir -p "${USERBASE}"

singularity exec --nv "${SIF}" bash -c "
  set -e
  export PYTHONUSERBASE='${USERBASE}'
  echo '=== torch V KONTAJNERI (pred instalaciou cohokolvek) ==='
  python3 -c 'import torch; print(torch.__version__, \"CUDA:\", torch.cuda.is_available())'
  # KLUCOVE: torch NEINSTALUJEME - pouzivame CUDA torch z kontajnera.
  # fairchem pinuje torch~=2.8.0, ale kontajner ma pre-release 2.8.0a0+nv,
  # ktory formalne pin nesplna (funkcne ano) -> instalujeme fairchem --no-deps
  # a jeho zavislosti explicitne (zoznam overeny z funkcneho behu).
  python3 -m pip install --user --no-cache-dir --no-deps fairchem-core==2.14.0
  python3 -m pip install --user --no-cache-dir \
    ase pandas pyyaml hydra-core omegaconf e3nn lmdb orjson monty pymatgen \
    submitit torchtnt wandb 'huggingface_hub[hf_xet]' requests tqdm \
    ray fastapi uvicorn clusterscope
  echo '=== verifikácia ==='
  python3 - <<'PY'
import torch, fairchem
print('torch:', torch.__version__, '| CUDA dostupná:', torch.cuda.is_available())
assert '+cpu' not in torch.__version__, 'pip nahradil kontajnerovy torch CPU buildom!'
assert torch.cuda.is_available(), 'CUDA NIE JE dostupna!'
print('GPU:', torch.cuda.get_device_name(0))
from fairchem.core import FAIRChemCalculator
calc = FAIRChemCalculator.from_model_checkpoint('uma-m-1p1', task_name='oc20', device='cuda')
print('UMA-M model načítaný OK (na GPU)')
PY
"
echo
echo "Setup HOTOVÝ. Ďalej:"
echo "  1) python3 hpc/03_make_manifest.py        (login uzol stačí)"
echo "  2) sbatch --account=<PROJEKT> hpc/04_submit_adsorbml.sbatch"
