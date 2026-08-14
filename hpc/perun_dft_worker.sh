#!/usr/bin/env bash
# One SLURM array task = one structure = one call to
# scripts/gpaw_h_adsorption.py --adsorbml-candidates --structure-name <NAME>.
# Runs in the pyenv env built by scripts/setup_pyenv_env.sh (no container).
#
# Two run modes:
#   USE_MPI=0 (default) : one process with multi-threaded BLAS.
#   USE_MPI=1           : mpiexec -n N -> GPAW distributes k-points/domains
#                         across N MPI ranks (an exact speed-up, roughly
#                         8-12x on the SCF cycle).
#                         REQUIRES a GPAW built with MPI
#                         (hpc/rebuild_gpaw_mpi.sh; check with
#                         `gpaw info | grep -i mpi`). Without an MPI build the
#                         N processes would each repeat the same calculation N
#                         times — do NOT enable it in that case.
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
  MPI_RANKS="${MPI_RANKS:-${CORES_PER_CALC}}"     # MPI ranks = allocated cores
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  # Modern GPAW (>=24) launched through `mpiexec python` needs the compiled
  # MPI backend switched on explicitly, otherwise it reports
  # "MPI parallelism is disabled":
  export GPAW_MPI_BACKEND=cgpaw
  module load OpenMPI 2>/dev/null || module load openmpi 2>/dev/null || true
  # ── NUMA / core PINNING (critical for memory-bound GPAW on EPYC 9845) ────
  # A node is 20 NUMA domains x 16 cores, each with its own memory channel and
  # L3 slice. Without pinning the OS migrates ranks between NUMA domains, so
  # memory is reached across the Infinity Fabric — a 15-40x slowdown on a
  # fully occupied node.
  # `--bind-to core` pins each rank to a core (first-touch then keeps memory
  # local), so a 16-core job sits inside one NUMA domain at full local
  # bandwidth and 20 such jobs per node run without interfering, each on its
  # own channels.
  # `--map-by numa` spreads ranks across the NUMA domains of the allocated
  # block, giving a job more memory channels; `--bind-to core` then prevents
  # migration. A launcher that does not know these flags ignores them quietly
  # and the job still runs.
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
