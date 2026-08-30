#!/usr/bin/env bash
# E1 baseline: KTransformers (GPU attention/dense + CPU expert kernels).
#
# KTransformers' CLI and injection-config surface has churned across releases,
# so everything here is env-overridable and the script fails loudly with the
# server log rather than guessing. First smoke day: pin KT_VERSION, confirm the
# server command + model layout, then freeze them here for the paper.
#
# Env knobs:
#   KT_VENV       separate venv (their torch pin may fight vLLM's; default
#                 /workspace/venv-kt, created on first run)
#   KT_VERSION    pip version to pin (default 0.3.2 — CONFIRM on smoke day)
#   KT_MODEL      HF id of the safetensors model (default Qwen/Qwen3-30B-A3B)
#   KT_GGUF_REPO  GGUF repo for expert weights if the KT path needs one
#                 (default Qwen/Qwen3-30B-A3B-GGUF, pattern KT_GGUF_PATTERN)
#   KT_CMD        full server command template override; {port} substituted.
#   PORT (8392)   OUTDIR (/workspace/e1)   THREADS (default nproc)
set -uo pipefail
V=${V:-/workspace/venv/bin}
KT_VENV=${KT_VENV:-/workspace/venv-kt}
KT_VERSION=${KT_VERSION:-0.3.2}
KT_MODEL=${KT_MODEL:-Qwen/Qwen3-30B-A3B}
KT_GGUF_REPO=${KT_GGUF_REPO:-Qwen/Qwen3-30B-A3B-GGUF}
KT_GGUF_PATTERN=${KT_GGUF_PATTERN:-Q8_0}
PORT=${PORT:-8392}
OUTDIR=${OUTDIR:-/workspace/e1}
THREADS=${THREADS:-$(nproc)}
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- env (once) -----------------------------------------------------------
if [ ! -x "$KT_VENV/bin/python" ]; then
  python3 -m venv "$KT_VENV"
  "$KT_VENV/bin/pip" install -q --upgrade pip
  "$KT_VENV/bin/pip" install -q "ktransformers==$KT_VERSION" \
    || { echo "E1-KT-INSTALL-FAIL (try a different KT_VERSION or their prebuilt wheel index)"; exit 1; }
fi

# GGUF for the CPU expert path (KTransformers' usual layout for MoE).
"$V/hf" download "$KT_GGUF_REPO" --include "*${KT_GGUF_PATTERN}*.gguf" >/dev/null 2>&1 || true
KT_GGUF_DIR="$(dirname "$(find "$HF_HOME/hub" -path "*${KT_GGUF_REPO##*/}*" -name "*${KT_GGUF_PATTERN}*.gguf" | head -1)" 2>/dev/null)"

# --- serve ----------------------------------------------------------------
OUT="$OUTDIR/ktransformers.json"
[ -s "$OUT" ] && { echo "[e1] skip ktransformers (exists)"; echo E1-KT-DONE; exit 0; }
DEFAULT_CMD="$KT_VENV/bin/python -m ktransformers.server.main \
  --model_path $KT_MODEL --gguf_path ${KT_GGUF_DIR:-MISSING} \
  --port {port} --host 127.0.0.1 --cpu_infer $THREADS"
CMD="${KT_CMD:-$DEFAULT_CMD}"
CMD="${CMD//\{port\}/$PORT}"
echo "[e1] kt cmd: $CMD"
$CMD > "$OUTDIR/ktransformers.server.log" 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 240); do
  curl -s -o /dev/null "127.0.0.1:$PORT/v1/models" && { ok=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "E1-KT-SERVE-FAIL — tail of server log:"
  tail -20 "$OUTDIR/ktransformers.server.log"
  kill "$SPID" 2>/dev/null
  exit 1
fi
"$V/python" "$HERE/e1_endpoint_bench.py" --url "http://127.0.0.1:$PORT" \
    --model "$KT_MODEL" --backend ktransformers \
    --extra-config "{\"kt_version\": \"$KT_VERSION\", \"cpu_threads\": $THREADS, \"gguf_pattern\": \"$KT_GGUF_PATTERN\"}" \
    --out "$OUT" || echo "E1-KT-BENCH-FAIL"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
echo E1-KT-DONE
