#!/usr/bin/env bash
# Submit stage 4 (GPAW AdsorbML DFT validation) as a SLURM array on PERUN,
# one array task per structure in the (path-fixed) ranked_candidates.csv.
#
# Prerequisites:
#   1. pyenv env built:  bash scripts/setup_pyenv_env.sh   (rootless, ~30-60 min)
#   2. hpc/fix_ranked_candidates_paths.py already run to produce a
#      path-fixed CSV + manifest.txt for THIS machine
#
# Usage:
#   ACCOUNT=myproject bash hpc/submit_perun_dft_array.sh
#
# Env overrides (all optional except ACCOUNT):
#   PARTITION=cpu_short   TIME_LIMIT=1-00:00:00   CORES_PER_CALC=11
#   ARRAY_LIMIT=%40   (throttle concurrent tasks so you don't hog the queue)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATES_CSV="${CANDIDATES_CSV:-${REPO_ROOT}/data/adsorbml_results/ranked_candidates.perun.csv}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/adsorbml_results/manifest.txt}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-${HOME}/.local/mo_h_bootstrap}"
PYTHON_BIN="${PYTHON_BIN:-${PYENV_ROOT}/versions/${ENV_NAME}/bin/python}"

ACCOUNT="${ACCOUNT:-}"
# Projekt p2061-26-2 ma CPU alokaciu na FAT (high-memory) particiach, nie na
# standardnom cpu_short (tam ma 0 core-h -> QOSGrpBillingMinutes). Preto FAT:
PARTITION="${PARTITION:-cpu_hm_short}"
QOS="${QOS:-}"                 # napr. QOS=fat ak FAT alokacia vyzaduje vlastnu QOS
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
CORES_PER_CALC="${CORES_PER_CALC:-11}"
MEM_PER_CPU="${MEM_PER_CPU:-3500M}"
# GPAW je memory-bandwidth-bound; plný uzol (320 jadier) saturuje zdieľanú DDR5
# zbernicu (12 kanálov/socket) → každý job spomalí ~20x. Pamäť na klastri NIE je
# consumable (SelectType=CR_CORE) → --mem packing NEobmedzí. Počet jobov/uzol
# preto riadime REZERVÁCIOU JADIER (cpus-per-task): job si vyžiada
# NODE_CORES/JOBS_PER_NODE jadier, ale spustí len CORES_PER_CALC MPI rankov →
# zvyšné jadrá ostanú voľné = viac pamäťovej priepustnosti na job.
# Napr. JOBS_PER_NODE=4, CORES_PER_CALC=16 → 80 jadier/job → 4 joby/uzol.
JOBS_PER_NODE="${JOBS_PER_NODE:-}"
NODE_CORES="${NODE_CORES:-320}"
MEM_FLAG="--mem-per-cpu=${MEM_PER_CPU}"
JOB_NAME="${JOB_NAME:-gpaw-dft-perun}"
ARRAY_LIMIT="${ARRAY_LIMIT:-}"   # e.g. "%40" to cap 40 concurrent tasks

if [[ -z "${ACCOUNT}" ]]; then
  echo "ACCOUNT is required. Example: ACCOUNT=myproject bash hpc/submit_perun_dft_array.sh" >&2
  exit 1
fi
if [[ ! -f "${CANDIDATES_CSV}" ]]; then
  echo "Candidates CSV not found: ${CANDIDATES_CSV}" >&2
  echo "Run hpc/fix_ranked_candidates_paths.py first." >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  echo "Run first:  bash scripts/setup_pyenv_env.sh" >&2
  exit 1
fi

TASK_COUNT="$(grep -c . "${MANIFEST_PATH}")"
if [[ "${TASK_COUNT}" -lt 1 ]]; then
  echo "Manifest is empty: ${MANIFEST_PATH}" >&2
  exit 1
fi

LOG_DIR="${REPO_ROOT}/data/outputs/slurm_logs"
mkdir -p "${LOG_DIR}"

# Rozsah array úloh. Default = všetky štruktúry (1..N s limitom súbežnosti).
# Kalibrácia jednej štruktúry:  ARRAY_SPEC='1-1' bash hpc/submit_perun_dft_array.sh
ARRAY_SPEC="${ARRAY_SPEC:-1-${TASK_COUNT}${ARRAY_LIMIT}}"

# SLURM rozdelenie zdrojov podľa režimu:
#   USE_MPI=1 -> N MPI rankov (--ntasks=N, 1 vlákno/rank), MPI rieši k-body
#   USE_MPI=0 -> 1 úloha + N vlákien BLAS (--cpus-per-task=N)
USE_MPI="${USE_MPI:-0}"
if [[ "${USE_MPI}" == "1" ]]; then
  NTASKS="${CORES_PER_CALC}"; CPUS_PER_TASK=1
  export MPI_RANKS="${CORES_PER_CALC}"
  # rezervuj (aj nevyužité) jadrá → obmedz joby/uzol → uvoľni pamäťovú zbernicu
  if [[ -n "${JOBS_PER_NODE}" ]]; then
    CPUS_PER_TASK=$(( NODE_CORES / (JOBS_PER_NODE * CORES_PER_CALC) ))
    [[ "${CPUS_PER_TASK}" -lt 1 ]] && CPUS_PER_TASK=1
  fi
else
  NTASKS=1; CPUS_PER_TASK="${CORES_PER_CALC}"
fi

echo "Submitting array ${ARRAY_SPEC} (z ${TASK_COUNT} štruktúr)  part=${PARTITION} qos=${QOS:-<default>} MPI=${USE_MPI} ntasks=${NTASKS} cpt=${CPUS_PER_TASK} mem=[${MEM_FLAG}]${JOBS_PER_NODE:+ (~${JOBS_PER_NODE} jobov/uzol)}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  ${QOS:+--qos="${QOS}"} \
  --job-name="${JOB_NAME}" \
  --chdir="${REPO_ROOT}" \
  --time="${TIME_LIMIT}" \
  --nodes=1 \
  --ntasks="${NTASKS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --distribution=block:block \
  ${MEM_FLAG} \
  --output="${LOG_DIR}/slurm_%A_%a.out" \
  --error="${LOG_DIR}/slurm_%A_%a.err" \
  --array="${ARRAY_SPEC}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",CANDIDATES_CSV="${CANDIDATES_CSV}",MANIFEST_PATH="${MANIFEST_PATH}",PYENV_ROOT="${PYENV_ROOT}",ENV_NAME="${ENV_NAME}",BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX}",PYTHON_BIN="${PYTHON_BIN}",CORES_PER_CALC="${CORES_PER_CALC}",USE_MPI="${USE_MPI:-0}",MPI_RANKS="${MPI_RANKS:-}",GPAW_XC="${GPAW_XC:-}",GPAW_MODE="${GPAW_MODE:-}",GPAW_PW_ECUT="${GPAW_PW_ECUT:-}",GPAW_H="${GPAW_H:-}",GPAW_KPTS="${GPAW_KPTS:-}",GPAW_SIGMA="${GPAW_SIGMA:-}",GPAW_SYMMETRY="${GPAW_SYMMETRY:-}",GPAW_SPINPOL="${GPAW_SPINPOL:-}",GPAW_SCALAPACK="${GPAW_SCALAPACK:-}",HUBBARD_U="${HUBBARD_U:-}",ADSORBML_RELAX="${ADSORBML_RELAX:-}",RELAX_FMAX="${RELAX_FMAX:-}",RELAX_STEPS="${RELAX_STEPS:-}",ADSORBML_OUTPUT_CSV="${ADSORBML_OUTPUT_CSV:-}" \
  "${REPO_ROOT}/hpc/perun_dft_worker.sh"
