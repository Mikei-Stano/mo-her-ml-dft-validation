#!/usr/bin/env bash
# STAGE 1d — AdsorbML stage 2 for the remaining structures (GPU).
# 2-run_adsorbml.py discovers every visible GPU, runs one worker per GPU and
# skips structures that already have a candidates.csv.
#
# Usage:  sbatch --account=<PROJECT> hpc/04_submit_adsorbml.sbatch
# If two days are not enough, simply resubmit after the walltime expires — the
# run resumes where it stopped, checkpointed through the finished
# candidates.csv files.
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

echo ">>> node: $(hostname), GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

singularity exec --nv \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF}" bash -c "
    set -e
    export PYTHONUSERBASE='${USERBASE}'
    # add fairchem-data-oc (provides Slab); --no-deps because of the torch pin:
    python3 -m pip install --user --no-cache-dir --no-deps -q fairchem-data-oc
    # verify every stage-2 import BEFORE starting the run:
    python3 - <<'PY'
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator
print('stage-2 imports OK')
PY
    cd '${REPO_ROOT}'
    python3 -u scripts/adsorbml/2-run_adsorbml.py \
      --manifest data/adsorbml_manifest.csv
  "

echo
echo "Stage 2 DONE. Next:  bash hpc/05_extract_rank_fix.sh"
