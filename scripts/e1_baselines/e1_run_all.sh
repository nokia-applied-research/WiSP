#!/usr/bin/env bash
# E1 orchestrator: run every engine's sweep, then render the cross-engine
# Pareto with the existing plot script. Resumable — each launcher skips rows
# whose JSON already exists.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR=${OUTDIR:-/workspace/e1}
V=${V:-/workspace/venv/bin}
mkdir -p "$OUTDIR"

bash "$HERE/e1_llamacpp.sh"
bash "$HERE/e1_ktransformers.sh"

# WiSP + vanilla rows: reuse the m3 sweep (writes the same schema).
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B} bash "$HERE/../m3_3090_isovram_sweep.sh" "$OUTDIR" || true

"$V/python" "$HERE/../m3_fig1_plot.py" --in-dir "$OUTDIR" --out "$OUTDIR/e1_pareto" \
  --title "iso-VRAM decode Pareto, Qwen3-30B-A3B (RTX 3090, all engines)" || true
echo E1-ALL-DONE
