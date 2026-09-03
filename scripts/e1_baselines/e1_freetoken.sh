#!/usr/bin/env bash
# E1 baseline: FreeToken (arXiv 2608.16157) via its OpenAI-compatible server.
#
# Smoke-day-2 findings encoded here (2026-09-03, RunPod eu-cz 3090):
#  - entry point is `python -m freetoken` (no console script); pip name
#    freetoken[accel], tested version 0.1.2.
#  - `ninja` is a hard runtime dep for their JIT kernel builds but is NOT
#    declared — install it explicitly or the backend worker dies.
#  - their torch wheel may be a newer CUDA flavor than the pod's nvcc
#    (2.11+cu130 vs nvcc 12.8); FREETOKEN_ALLOW_CUDA_MISMATCH=1 is their
#    documented override and worked (kernels built, CUDA graphs captured).
#  - /v1/models returns 200 BEFORE the model finishes loading; real
#    readiness = a tiny /v1/completions probe returning 200 (they answer
#    503 {"error":"model is still loading"} until then). First boot is slow
#    (weight-bank conversion + JIT + CUDA graph capture: ~50 min cold).
#  - the served model id is the BASENAME of --model-path (e.g.
#    "Qwen3-30B-A3B", not "Qwen/Qwen3-30B-A3B").
#
# Env knobs:
#   FT_VENV (/root/venv-ft)   FT_PIN (0.1.2)   FT_MODEL (Qwen/Qwen3-30B-A3B)
#   FT_BACKEND (hybrid; also: offload=fetch-only, cpu, fused)
#   MEMRATIO_SWEEP (default "0.85")   PORT (8395)   OUTDIR (/workspace/e1)
set -uo pipefail
V=${V:-/workspace/venv/bin}
FT_VENV=${FT_VENV:-/root/venv-ft}
FT_PIN=${FT_PIN:-0.1.2}
FT_MODEL=${FT_MODEL:-Qwen/Qwen3-30B-A3B}
FT_BACKEND=${FT_BACKEND:-hybrid}
MEMRATIO_SWEEP=${MEMRATIO_SWEEP:-"0.85"}
PORT=${PORT:-8395}
OUTDIR=${OUTDIR:-/workspace/e1}
export HF_HOME=${HF_HOME:-/root/hf}
export FREETOKEN_ALLOW_CUDA_MISMATCH=${FREETOKEN_ALLOW_CUDA_MISMATCH:-1}
mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="$(basename "$FT_MODEL")"

if [ ! -x "$FT_VENV/bin/python" ] || ! "$FT_VENV/bin/python" -c "import freetoken" 2>/dev/null; then
  python3 -m venv "$FT_VENV"
  "$FT_VENV/bin/pip" install -q --upgrade pip
  "$FT_VENV/bin/pip" install -q "freetoken[accel]==$FT_PIN" ninja packaging \
    || { echo "E1-FT-INSTALL-FAIL"; exit 1; }
fi
command -v ninja >/dev/null || ln -sf "$FT_VENV/bin/ninja" /usr/local/bin/ninja

for MR in $MEMRATIO_SWEEP; do
  OUT="$OUTDIR/freetoken_${FT_BACKEND}_mr${MR}.json"
  [ -s "$OUT" ] && { echo "[e1] skip freetoken $FT_BACKEND mr=$MR"; continue; }
  "$FT_VENV/bin/python" -m freetoken --model-path "$FT_MODEL" \
      --moe-backend "$FT_BACKEND" --memory-ratio "$MR" \
      --host 127.0.0.1 --port "$PORT" \
      > "${OUT%.json}.server.log" 2>&1 &
  SPID=$!
  ready=0
  for i in $(seq 1 400); do   # cold boot can take ~50 min (JIT + conversion)
    CODE=$(curl -s -m 60 -o /dev/null -w '%{http_code}' "127.0.0.1:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0}")
    [ "$CODE" = "200" ] && { ready=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    echo "E1-FT-SERVE-FAIL backend=$FT_BACKEND mr=$MR"; tail -15 "${OUT%.json}.server.log"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; continue
  fi
  "$V/python" "$HERE/e1_endpoint_bench.py" --url "http://127.0.0.1:$PORT" \
      --model "$MODEL_ID" --backend freetoken \
      --extra-config "{\"moe_backend\": \"$FT_BACKEND\", \"memory_ratio\": $MR, \"ft_version\": \"$FT_PIN\"}" \
      --out "$OUT" || echo "E1-FT-BENCH-FAIL mr=$MR"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  sleep 5
done
echo E1-FT-DONE
