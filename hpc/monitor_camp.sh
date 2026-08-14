#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  monitor_camp.sh — produkčný monitor DFT kampane (ΔG_H validácia UMA)
#
#  Štvrtý do rodiny:
#     monitor_uma.sh   →  UMA/AdsorbML na GPU
#     monitor_val.sh   →  konvergenčné validácie (k-mriežka, h, σ)
#     monitor_dft.sh   →  starý DFT beh
#     monitor_camp.sh  →  TÁTO kampaň + kryštalografický popis
#
#  Prečo je jadro v Pythone: terminácia povrchu, supercela, adsorpčné miesto
#  a pokrytie sa nedajú prečítať z názvu súboru — počítajú sa z polôh atómov,
#  bunky a FixAtoms cez ASE. Bash by na to bol nesprávny nástroj. Tento wrapper
#  drží rovnaké rozhranie ako ostatné monitory (REFRESH, --once).
#
#  Použitie:
#     bash hpc/monitor_camp.sh              # refresh 30 s
#     bash hpc/monitor_camp.sh --once       # jeden snímok (na skopírovanie)
#     REFRESH=10 bash hpc/monitor_camp.sh
#     MONITOR_COLOR=1 bash hpc/monitor_camp.sh --once   # farby aj do pipe
#
#  Kryštalografia sa cachuje do data/outputs/campaign_dft/campaign_meta.json
#  (počíta sa raz, nie pri každom refreshi).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# Voliteľná bootstrap knižnica (ak si GPAW/ASE staval mimo systémových ciest).
BOOT="${MO_H_BOOTSTRAP:-${HOME}/.local/mo_h_bootstrap}"
[[ -d "$BOOT" ]] && export LD_LIBRARY_PATH="${BOOT}/lib64:${BOOT}/lib:${LD_LIBRARY_PATH:-}"
PY="${PYTHON_BIN:-python3}"
[[ -x "$PY" ]] || PY=python3
exec "$PY" -u "${REPO_ROOT}/hpc/monitor_camp.py" "$@"
