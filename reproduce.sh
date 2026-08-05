#!/usr/bin/env bash
# One-command reproduction of the two headline claims from the paper on a
# real 24 GiB consumer GPU (the numbers in the paper are from an RTX 3090,
# PCIe 4.0 x16):
#
#   1. byte-identity  — WiSP decode == vanilla vLLM token-for-token (temp 0)
#   2. iso-VRAM Pareto — WiSP pages Qwen3-30B-A3B onto a card that cannot
#                        hold it, where vanilla --cpu-offload-gb OOMs at the
#                        same VRAM budget.
#
# Requirements: one CUDA GPU with >= ~16 GiB free, ~80 GiB host RAM (Qwen3
# master weights live pinned in DRAM), and the model cached locally or a
# working HF connection.
#
# Usage:
#   bash reproduce.sh                 # full: byte-identity + iso-VRAM sweep
#   STEP=identity bash reproduce.sh   # just the byte-identity check
#   STEP=isovram  bash reproduce.sh   # just the iso-VRAM sweep
set -euo pipefail

export VLLM_USE_V1=1
export WISP_MODE=paged
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
OUTDIR="${OUTDIR:-./results}"
STEP="${STEP:-all}"
mkdir -p "$OUTDIR"

run_identity() {
  echo "=== [1/2] byte-identity: WiSP (paged, cap=8) vs vanilla vLLM ==="
  # Vanilla can't hold a >VRAM model on its own, so it streams weights from
  # CPU to produce the reference (same MoE math, true vanilla output).
  python scripts/check_byte_identity.py --mode vanilla --model "$MODEL" \
      --cpu-offload-gb "${OFFLOAD_GB:-48}" --gpu-memory-utilization 0.90 \
      --out "$OUTDIR/identity_vanilla.json"
  python scripts/check_byte_identity.py --mode wisp --cap-experts 8 --model "$MODEL" \
      --gpu-memory-utilization 0.45 --out "$OUTDIR/identity_wisp.json"
  python scripts/check_byte_identity.py \
      --compare "$OUTDIR/identity_vanilla.json" "$OUTDIR/identity_wisp.json"
}

run_isovram() {
  echo "=== [2/2] iso-VRAM Pareto sweep on this card ==="
  MODEL="$MODEL" bash scripts/m3_3090_isovram_sweep.sh "$OUTDIR/isovram"
  echo "--- plotting Figure 3 (iso-VRAM) ---"
  python scripts/m3_fig1_plot.py "$OUTDIR/isovram" || \
      echo "[note] plot step skipped (matplotlib missing or no rows)"
}

case "$STEP" in
  identity) run_identity ;;
  isovram)  run_isovram ;;
  all)      run_identity; run_isovram ;;
  *) echo "unknown STEP=$STEP (use: identity | isovram | all)"; exit 2 ;;
esac

echo
echo "Done. Artifacts under $OUTDIR/"
