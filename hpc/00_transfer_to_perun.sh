#!/usr/bin/env bash
# STAGE 0 — run LOCALLY (on your own machine).
# Transfers the working workspace plus the Hugging Face cache. The UMA
# checkpoint is gated, so it cannot be downloaded on the cluster without
# authenticating there; copying the local cache avoids that.
#
# Usage:
#   bash hpc/00_transfer_to_perun.sh
#   (override via env: PERUN_USER=..., PERUN_HOST=..., PERUN_PORT=..., SSH_KEY=...)
set -euo pipefail

PERUN_USER="${PERUN_USER:?set PERUN_USER=<your-login>}"
PERUN_HOST="${PERUN_HOST:-login.perun.sav.sk}"
PERUN_PORT="${PERUN_PORT:-5522}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"   # override with SSH_KEY=...
PERUN_DEST="${PERUN_DEST:-/home/${PERUN_USER}/cemea}"

SSH_CMD="ssh -p ${PERUN_PORT} -i ${SSH_KEY}"

LOCAL_CEMEA="${LOCAL_CEMEA:-$(cd "$(dirname "$0")/../.." && pwd)}"

echo ">>> 0/3 Creating the destination directory on the cluster"
${SSH_CMD} "${PERUN_USER}@${PERUN_HOST}" "mkdir -p '${PERUN_DEST}'"

echo ">>> 1/3 Repository mo-h-adsorption-gpaw (code + data, backups excluded)"
rsync -av --progress -e "${SSH_CMD}" \
  --exclude '.git' \
  --exclude 'LogsAndResults.tar.gz' \
  --exclude 'Logs&Results' \
  "${LOCAL_CEMEA}/mo-h-adsorption-gpaw" \
  "${PERUN_USER}@${PERUN_HOST}:${PERUN_DEST}/"

echo ">>> 2/3 ranked_structures.csv (ML screening reference)"
rsync -av -e "${SSH_CMD}" "${LOCAL_CEMEA}/ranked_structures.csv" \
  "${PERUN_USER}@${PERUN_HOST}:${PERUN_DEST}/"

echo ">>> 3/3 Hugging Face cache holding the UMA model (gated checkpoint)"
${SSH_CMD} "${PERUN_USER}@${PERUN_HOST}" "mkdir -p ~/.cache"
for CACHE in "${HOME}/.cache/huggingface" "${HOME}/.cache/fairchem"; do
  if [ -d "${CACHE}" ]; then
    rsync -av --progress -e "${SSH_CMD}" "${CACHE}" "${PERUN_USER}@${PERUN_HOST}:.cache/"
  else
    echo "    (${CACHE} does not exist — skipping)"
  fi
done

echo
echo "DONE. Continue on the cluster:"
echo "  ${SSH_CMD} ${PERUN_USER}@${PERUN_HOST}"
echo "  cd ${PERUN_DEST}/mo-h-adsorption-gpaw && bash hpc/01_pull_images.sh"
