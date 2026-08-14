#!/usr/bin/env bash
# FÁZA 1d — stage 2 AdsorbML pre chýbajúce štruktúry (GPU).
# Skript 2-run_adsorbml.py si sám nájde všetky GPU, spustí 1 worker/GPU
# a preskočí štruktúry, ktoré už majú candidates.csv.
#
# Spustenie:  sbatch --account=<PROJEKT> hpc/04_submit_adsorbml.sbatch
# Ak 2 dni nestačia, po vypršaní jednoducho submitni znova — pokračuje tam,
# kde skončil (checkpointing cez hotové candidates.csv).
#
#SBATCH --job-name=adsorbml-stage2
#SBATCH --partition=gpu_medium
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=hpc/adsorbml_%j.out
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
NGC_TAG="${NGC_TAG:-25.06-py3}"
SIF="${REPO_ROOT}/hpc/pytorch_ngc_${NGC_TAG}.sif"
USERBASE="${USERBASE:-${HOME}/fairchem-userbase}"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

echo ">>> uzol: $(hostname), GPU: $(nvidia-smi -L 2>/dev/null | wc -l)"

singularity exec --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF}" bash -c "
    set -e
    export PYTHONUSERBASE='${USERBASE}'
    # doinstaluj fairchem-data-oc (Slab), --no-deps kvoli torch pinu:
    python3 -m pip install --user --no-cache-dir --no-deps -q fairchem-data-oc
    # kontrola vsetkych importov stage-2 PRED spustenim:
    python3 - <<'PY'
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator
print('importy stage-2 OK')
PY
    cd '${REPO_ROOT}'
    python3 -u scripts/adsorbml/2-run_adsorbml.py \
      --manifest data/adsorbml_manifest.csv
  "

echo
echo "Stage 2 HOTOVÝ. Ďalej:  bash hpc/05_extract_rank_fix.sh"
