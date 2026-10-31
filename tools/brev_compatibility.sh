#!/usr/bin/env bash
# Synthetic three-image infrastructure run only. Root/local controls own billing.
set -euo pipefail
umask 077
CHAMELEON_ROOT="${CHAMELEON_ROOT:-$HOME/workspace/chameleon}"
cd "$CHAMELEON_ROOT"
source "$CHAMELEON_ROOT/vision-env.sh"
: "${CHAMELEON_BASE_REVISION:?Set the exact 40-hex Qwen3-VL base commit}"
if [[ ! "$CHAMELEON_BASE_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  printf '%s\n' 'The base revision must be an exact 40-hex commit.' >&2
  exit 2
fi
test -f tools/vision_lora_smoke.py
test -f experiments/vision_critic/train_sft.py
test -f experiments/vision_critic/contracts.py
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIRECTORY="${CHAMELEON_COMPAT_OUT:-training_runs/source-ab-compatibility-$RUN_STAMP}"
mkdir -p output/vision_critic
DOWNLOAD_FLAGS=(--local-files-only)
if [[ "${CHAMELEON_COMPAT_ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_FLAGS=()
fi
printf '%s\n' 'Synthetic compatibility only; this process cannot stop Brev billing.'
timeout --signal=TERM --kill-after=30s 18m \
  python -m tools.vision_lora_smoke \
  --scenario source-ab --steps 2 --timeout-minutes 15 \
  --revision "$CHAMELEON_BASE_REVISION" "${DOWNLOAD_FLAGS[@]}" \
  --out-dir "$RUN_DIRECTORY" \
  2>&1 | tee "output/vision_critic/source-ab-compatibility-$RUN_STAMP.log"
