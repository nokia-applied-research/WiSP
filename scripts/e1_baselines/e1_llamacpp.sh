#!/usr/bin/env bash
# E1 baseline: llama.cpp on an iso-VRAM budget, via llama-server + the shared
# endpoint driver. llama.cpp's memory lever is -ngl (layers on GPU); we sweep
# it to land near the target VRAM budgets and record actual peak VRAM per run.
#
# Env knobs:
#   LLAMA_DIR      where to clone/build (default /workspace/llama.cpp)
#   LLAMA_TAG      git tag to pin (default: b6100 — set explicitly for the paper)
#   GGUF_REPO      HF repo with the GGUF (default Qwen/Qwen3-30B-A3B-GGUF)
#   GGUF_PATTERN   filename pattern preference (default "F16" then "Q8_0")
#   NGL_SWEEP      space-separated -ngl values (default "8 16 24 32 40 48")
#   CTX            context size (default 4096)   PORT (default 8391)
#   OUTDIR         results dir (default /workspace/e1)
set -uo pipefail
V=${V:-/workspace/venv/bin}
LLAMA_DIR=${LLAMA_DIR:-/workspace/llama.cpp}
LLAMA_TAG=${LLAMA_TAG:-b6100}
GGUF_REPO=${GGUF_REPO:-Qwen/Qwen3-30B-A3B-GGUF}
GGUF_PATTERN=${GGUF_PATTERN:-F16}
NGL_SWEEP=${NGL_SWEEP:-"8 16 24 32 40 48"}
CTX=${CTX:-4096}
PORT=${PORT:-8391}
OUTDIR=${OUTDIR:-/workspace/e1}
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- build (once) ---------------------------------------------------------
if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" 2>/dev/null || true
  git -C "$LLAMA_DIR" fetch -q --tags && git -C "$LLAMA_DIR" checkout -q "$LLAMA_TAG"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-server \
    || { echo "E1-LLAMACPP-BUILD-FAIL"; exit 1; }
fi

# --- model (once): pick GGUF by pattern, fall back to Q8_0 ----------------
pick_gguf() {
  "$V/hf" download "$GGUF_REPO" --include "*${1}*.gguf" >/dev/null 2>&1 || return 1
  find "$HF_HOME/hub" -path "*${GGUF_REPO##*/}*" -name "*${1}*.gguf" | sort | head -1
}
GGUF="$(pick_gguf "$GGUF_PATTERN")"
[ -z "$GGUF" ] && { echo "[e1] $GGUF_PATTERN not found in $GGUF_REPO, trying Q8_0"; GGUF="$(pick_gguf Q8_0)"; }
[ -z "$GGUF" ] && { echo "E1-LLAMACPP-NO-GGUF (ls the repo and set GGUF_PATTERN)"; exit 1; }
echo "[e1] gguf: $GGUF"
# multi-part GGUFs: llama-server takes the first shard and finds the rest.
GGUF="$(echo "$GGUF" | head -1)"

# --- sweep ----------------------------------------------------------------
for NGL in $NGL_SWEEP; do
  OUT="$OUTDIR/llamacpp_ngl${NGL}.json"
  [ -s "$OUT" ] && { echo "[e1] skip ngl=$NGL (exists)"; continue; }
  "$LLAMA_DIR/build/bin/llama-server" -m "$GGUF" -ngl "$NGL" -c "$CTX" \
      --port "$PORT" --host 127.0.0.1 -t "$(nproc)" --no-warmup \
      > "$OUTDIR/llamacpp_ngl${NGL}.server.log" 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 180); do
    curl -s -o /dev/null "127.0.0.1:$PORT/health" && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "E1-LLAMACPP-SERVE-FAIL ngl=$NGL (likely VRAM: see server log)"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; continue
  fi
  "$V/python" "$HERE/e1_endpoint_bench.py" --url "http://127.0.0.1:$PORT" \
      --model "$(basename "$GGUF")" --backend llamacpp \
      --extra-config "{\"ngl\": $NGL, \"gguf\": \"$(basename "$GGUF")\", \"tag\": \"$LLAMA_TAG\"}" \
      --out "$OUT" || echo "E1-LLAMACPP-BENCH-FAIL ngl=$NGL"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  sleep 3
done
echo E1-LLAMACPP-DONE
