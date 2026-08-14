#!/usr/bin/env bash
# One SLURM array task = one structure = one call to
# scripts/gpaw_h_adsorption.py --adsorbml-candidates --structure-name <NAME>.
# Runs in the pyenv env built by scripts/setup_pyenv_env.sh (no container).
#
# Dva režimy behu:
#   USE_MPI=0 (default) : 1 proces + viacvláknový BLAS (súčasné správanie).
#   USE_MPI=1           : mpiexec -n N gpaw beh -> GPAW rozdelí k-body/domény
#                         cez N MPI rankov (EXAKTNÉ zrýchlenie, ~8-12x na SCF).
#                         VYŽADUJE GPAW postavený s MPI (hpc/rebuild_gpaw_mpi.sh;
#                         over: gpaw info | grep -i mpi). Bez MPI-buildu by N
#                         procesov počítalo to isté N-krát -> NEzapínať!
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"            # .../mo-h-adsorption-gpaw
: "${CANDIDATES_CSV:?CANDIDATES_CSV is required}"  # path-fixed ranked_candidates.csv
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"

PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-${HOME}/.local/mo_h_bootstrap}"
PYTHON_BIN="${PYTHON_BIN:-${PYENV_ROOT}/versions/${ENV_NAME}/bin/python}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
CORES_PER_CALC="${SLURM_CPUS_PER_TASK:-${CORES_PER_CALC:-11}}"

if [[ -z "${TASK_ID}" ]]; then
  echo "SLURM_ARRAY_TASK_ID or TASK_ID is required" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN} (run scripts/setup_pyenv_env.sh first)" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${BOOTSTRAP_PREFIX}/lib64:${BOOTSTRAP_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

STRUCTURE_NAME="$(sed -n "${TASK_ID}p" "${MANIFEST_PATH}")"
if [[ -z "${STRUCTURE_NAME}" ]]; then
  echo "No structure found for manifest line ${TASK_ID}" >&2
  exit 1
fi

USE_MPI="${USE_MPI:-0}"

if [[ "${USE_MPI}" == "1" ]]; then
  MPI_RANKS="${MPI_RANKS:-${CORES_PER_CALC}}"     # počet MPI rankov = pridelené jadrá
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  # Moderné GPAW (>=24) pri behu cez `mpiexec python` vyžaduje aktivovať
  # kompilovaný MPI backend (inak: "MPI parallelism is disabled"):
  export GPAW_MPI_BACKEND=cgpaw
  module load OpenMPI 2>/dev/null || module load openmpi 2>/dev/null || true
  # ── NUMA/core PINNING (kľúčové pre memory-bound GPAW na EPYC 9845) ────────
  # Uzol = 20 NUMA domén × 16 jadier (vlastný pamäťový kanál + L3 každá).
  # Bez pinningu OS migruje ranky medzi NUMA doménami → prístup do cudzej
  # pamäte cez Infinity Fabric → 15–40× spomalenie pri plnom uzle.
  # `--bind-to core` pripne každý rank na jadro (first-touch → lokálna pamäť);
  # 16-jadrový job tak sedí v 1 NUMA doméne = plná lokálna priepustnosť,
  # 20 takých jobov/uzol beží bez vzájomnej interferencie (nezávislé kanály).
  # --map-by numa rozprestrie ranky cez NUMA domény prideleného bloku (viac
  # pamäťových kanálov na job); --bind-to core zabráni migrácii. Ak launcher
  # tieto flagy nepozná, ticho ich ignoruje (job beží ďalej).
  MPI_BIND="${MPI_BIND:---bind-to core --map-by numa}"
  [[ "${MPI_REPORT_BINDINGS:-0}" == "1" ]] && MPI_BIND="${MPI_BIND} --report-bindings"
  echo "[$(date --iso-8601=seconds)] Task ${TASK_ID}: ${STRUCTURE_NAME}  (MPI ranks=${MPI_RANKS}, bind='${MPI_BIND}')"
  exec mpiexec ${MPI_BIND} -n "${MPI_RANKS}" "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/gpaw_h_adsorption.py" \
    --adsorbml-candidates "${CANDIDATES_CSV}" \
    --structure-name "${STRUCTURE_NAME}" \
    --workers 1 \
    --cores-per-calc 1
else
  echo "[$(date --iso-8601=seconds)] Task ${TASK_ID}: ${STRUCTURE_NAME}  (serial, cores-per-calc=${CORES_PER_CALC})"
  exec "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/gpaw_h_adsorption.py" \
    --adsorbml-candidates "${CANDIDATES_CSV}" \
    --structure-name "${STRUCTURE_NAME}" \
    --workers 1 \
    --cores-per-calc "${CORES_PER_CALC}"
fi
