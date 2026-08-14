#!/usr/bin/env bash
# FÁZA 0 — spusti LOKÁLNE (na tomto PC).
# Prenesie pracovný workspace + HuggingFace cache (UMA model je gated,
# na PERUNe sa nedá stiahnuť bez prihlásenia — preto kopírujeme lokálnu cache).
#
# Použitie:
#   bash hpc/00_transfer_to_perun.sh
#   (prípadné zmeny cez env: PERUN_USER=..., PERUN_HOST=..., PERUN_PORT=..., SSH_KEY=...)
set -euo pipefail

PERUN_USER="${PERUN_USER:?nastav PERUN_USER=<tvoj-login>}"
PERUN_HOST="${PERUN_HOST:-login.perun.sav.sk}"
PERUN_PORT="${PERUN_PORT:-5522}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
PERUN_DEST="${PERUN_DEST:-/home/${PERUN_USER}/cemea}"

SSH_CMD="ssh -p ${PERUN_PORT} -i ${SSH_KEY}"

LOCAL_CEMEA="${LOCAL_CEMEA:-$(cd "$(dirname "$0")/../.." && pwd)}"

echo ">>> 0/3 Vytváram cieľový adresár na PERUNe"
${SSH_CMD} "${PERUN_USER}@${PERUN_HOST}" "mkdir -p '${PERUN_DEST}'"

echo ">>> 1/3 Repo mo-h-adsorption-gpaw (kód + dáta, bez záloh)"
rsync -av --progress -e "${SSH_CMD}" \
  --exclude '.git' \
  --exclude 'LogsAndResults.tar.gz' \
  --exclude 'Logs&Results' \
  "${LOCAL_CEMEA}/mo-h-adsorption-gpaw" \
  "${PERUN_USER}@${PERUN_HOST}:${PERUN_DEST}/"

echo ">>> 2/3 ranked_structures.csv (referencia ML výsledkov)"
rsync -av -e "${SSH_CMD}" "${LOCAL_CEMEA}/ranked_structures.csv" \
  "${PERUN_USER}@${PERUN_HOST}:${PERUN_DEST}/"

echo ">>> 3/3 HuggingFace cache s UMA modelom (gated checkpoint)"
${SSH_CMD} "${PERUN_USER}@${PERUN_HOST}" "mkdir -p ~/.cache"
for CACHE in "${HOME}/.cache/huggingface" "${HOME}/.cache/fairchem"; do
  if [ -d "${CACHE}" ]; then
    rsync -av --progress -e "${SSH_CMD}" "${CACHE}" "${PERUN_USER}@${PERUN_HOST}:.cache/"
  else
    echo "    (${CACHE} neexistuje — preskakujem)"
  fi
done

echo
echo "HOTOVO. Pokračuj na PERUNe:"
echo "  ${SSH_CMD} ${PERUN_USER}@${PERUN_HOST}"
echo "  cd ${PERUN_DEST}/mo-h-adsorption-gpaw && bash hpc/01_pull_images.sh"
