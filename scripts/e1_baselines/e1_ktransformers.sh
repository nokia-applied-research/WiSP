#!/usr/bin/env bash
# E1 baseline: KTransformers, current (2026) architecture = sglang-kt frontend
# + kt-kernel CPU expert backend. GPU holds attention/dense + --kt-num-gpu-experts
# hot experts; the rest run on CPU from GGUF (LLAMAFILE backend, works on any
# AVX2+ CPU; AMX backends need Sapphire Rapids+, which rented Ice Lake pods lack).
#
# --kt-num-gpu-experts is the VRAM lever — sweep it for iso-VRAM points.
#
# Smoke-day findings encoded here (2026-08-30): the PyPI `ktransformers` meta
# package's old `ktransformers.server.main` entry point is gone; the supported
# path is `pip install kt-kernel sglang-kt` and `python -m sglang.launch_server
# --kt-*` per kt-kernel/README.md. Their README's Qwen3-30B-A3B example uses
# --kt-num-gpu-experts 32 on a 24 GB card.
#
# Env knobs:
#   KT_VENV (default /workspace/venv-kt)   — dedicated venv (own torch pins)
#   KT_KERNEL_PIN / SGLANG_KT_PIN          — versions; empty = latest (freeze after smoke)
#   KT_MODEL      (default Qwen/Qwen3-30B-A3B)      safetensors side
#   KT_GGUF_REPO  (default Qwen/Qwen3-30B-A3B-GGUF) CPU expert side
#   KT_GGUF_PATTERN (default Q8_0)
#   KT_METHOD     (default LLAMAFILE)
#   GPU_EXPERTS_SWEEP (default "32")        — sweep list for iso-VRAM points
#   THREADS (default nproc)   PORT (8392)   OUTDIR (/workspace/e1)
#   KT_MEM_FRAC (default 0.85)
set -uo pipefail
V=${V:-/workspace/venv/bin}
KT_VENV=${KT_VENV:-/workspace/venv-kt}
KT_KERNEL_PIN=${KT_KERNEL_PIN:-}
SGLANG_KT_PIN=${SGLANG_KT_PIN:-}
KT_MODEL=${KT_MODEL:-Qwen/Qwen3-30B-A3B}
KT_GGUF_REPO=${KT_GGUF_REPO:-Qwen/Qwen3-30B-A3B-GGUF}
KT_GGUF_PATTERN=${KT_GGUF_PATTERN:-Q8_0}
KT_METHOD=${KT_METHOD:-LLAMAFILE}
GPU_EXPERTS_SWEEP=${GPU_EXPERTS_SWEEP:-"32"}
THREADS=${THREADS:-$(nproc)}
PORT=${PORT:-8392}
OUTDIR=${OUTDIR:-/workspace/e1}
KT_MEM_FRAC=${KT_MEM_FRAC:-0.85}
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- env (once) -----------------------------------------------------------
if [ ! -x "$KT_VENV/bin/python" ] || ! "$KT_VENV/bin/python" -c "import sglang" 2>/dev/null; then
  rm -rf "$KT_VENV"
  python3 -m venv "$KT_VENV"
  "$KT_VENV/bin/pip" install -q --upgrade pip
  "$KT_VENV/bin/pip" install -q "kt-kernel${KT_KERNEL_PIN:+==$KT_KERNEL_PIN}" \
                              "sglang-kt${SGLANG_KT_PIN:+==$SGLANG_KT_PIN}" \
    || { echo "E1-KT-INSTALL-FAIL"; exit 1; }
  # sgl_kernel's compiled ops (per-SM wheels) are not always pulled in as a
  # dependency; without them sglang aborts with "No module named common_ops".
  "$KT_VENV/bin/pip" install -q --upgrade sgl-kernel || echo "[e1] warn: sgl-kernel install failed"
  echo "[e1] installed: $("$KT_VENV/bin/pip" list 2>/dev/null | grep -Ei 'kt-kernel|sglang')"
fi

# GGUF dir for the CPU expert path (reuses the llama.cpp download if present).
"$V/hf" download "$KT_GGUF_REPO" --include "*${KT_GGUF_PATTERN}*.gguf" >/dev/null 2>&1 || true
GGUF_FILE="$(find "$HF_HOME/hub" -path "*${KT_GGUF_REPO##*/}*" -name "*${KT_GGUF_PATTERN}*.gguf" | sort | head -1)"
[ -z "$GGUF_FILE" ] && { echo "E1-KT-NO-GGUF"; exit 1; }
KT_GGUF_DIR="$(dirname "$GGUF_FILE")"
echo "[e1] gguf dir: $KT_GGUF_DIR"

# Safetensors side: make sure the HF snapshot exists locally.
"$V/hf" download "$KT_MODEL" >/dev/null 2>&1 || true

# --- sweep ----------------------------------------------------------------
for NG in $GPU_EXPERTS_SWEEP; do
  OUT="$OUTDIR/ktransformers_ge${NG}.json"
  [ -s "$OUT" ] && { echo "[e1] skip gpu-experts=$NG (exists)"; continue; }
  "$KT_VENV/bin/python" -m sglang.launch_server \
      --host 127.0.0.1 --port "$PORT" \
      --model "$KT_MODEL" --trust-remote-code \
      --mem-fraction-static "$KT_MEM_FRAC" \
      --kt-method "$KT_METHOD" \
      --kt-weight-path "$KT_GGUF_DIR" \
      --kt-cpuinfer "$THREADS" \
      --kt-num-gpu-experts "$NG" \
      > "$OUTDIR/ktransformers_ge${NG}.server.log" 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "127.0.0.1:$PORT/v1/models")" = "200" ] && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "E1-KT-SERVE-FAIL gpu-experts=$NG — tail of server log:"
    tail -20 "$OUTDIR/ktransformers_ge${NG}.server.log"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; continue
  fi
  "$V/python" "$HERE/e1_endpoint_bench.py" --url "http://127.0.0.1:$PORT" \
      --model "$KT_MODEL" --backend ktransformers \
      --extra-config "{\"kt_method\": \"$KT_METHOD\", \"gpu_experts\": $NG, \"cpu_threads\": $THREADS, \"gguf_pattern\": \"$KT_GGUF_PATTERN\"}" \
      --out "$OUT" || echo "E1-KT-BENCH-FAIL gpu-experts=$NG"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  sleep 5
done
echo E1-KT-DONE
