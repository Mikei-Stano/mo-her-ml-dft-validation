#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  monitor_camp.sh — production monitor for the DFT campaign (UMA ΔG_H validation)
#
#  Fourth of the family:
#     monitor_uma.sh   →  UMA/AdsorbML on GPU
#     monitor_val.sh   →  convergence validations (k-mesh, h, σ)
#     monitor_dft.sh   →  the earlier DFT run
#     monitor_camp.sh  →  THIS campaign + crystallographic description
#
#  Why the core is Python: surface termination, supercell, adsorption site and
#  coverage cannot be read off a filename — they are computed from atomic
#  positions, the cell and the FixAtoms mask through ASE. Bash would be the
#  wrong tool. This wrapper keeps the same interface as the other monitors
#  (REFRESH, --once).
#
#  Usage:
#     bash hpc/monitor_camp.sh              # refresh every 30 s
#     bash hpc/monitor_camp.sh --once       # single snapshot, e.g. to paste
#     REFRESH=10 bash hpc/monitor_camp.sh
#     MONITOR_COLOR=1 bash hpc/monitor_camp.sh --once   # keep colour into a pipe
#
#  Crystallography is cached in data/outputs/campaign_dft/campaign_meta.json
#  (computed once, not on every refresh).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# Optional bootstrap library (if GPAW/ASE were built outside the system paths).
BOOT="${MO_H_BOOTSTRAP:-${HOME}/.local/mo_h_bootstrap}"
[[ -d "$BOOT" ]] && export LD_LIBRARY_PATH="${BOOT}/lib64:${BOOT}/lib:${LD_LIBRARY_PATH:-}"
PY="${PYTHON_BIN:-python3}"
[[ -x "$PY" ]] || PY=python3
exec "$PY" -u "${REPO_ROOT}/hpc/monitor_camp.py" "$@"
