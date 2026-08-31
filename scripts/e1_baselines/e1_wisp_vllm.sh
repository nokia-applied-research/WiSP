#!/usr/bin/env bash
# E1 rows for WiSP and vanilla vLLM, measured with the SAME endpoint driver and
# protocol as the other engines (streamed TTFT, 8 prompts x 256 tokens, temp 0).
#
# Sweeps are "tag:env" configs:
#   WISP_SWEEP    default "cap8:0.45 cap16:0.60 cap32:0.80"  (cap:gpu-mem-util)
#   VANILLA_SWEEP default "off44:0.85 off50:0.85"            (cpu-offload-gb:util)
# Env: MODEL (Qwen/Qwen3-30B-A3B), PORT (8394), OUTDIR (/workspace/e1), CTX (4096)
set -uo pipefail
V=${V:-/workspace/venv/bin}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
PORT=${PORT:-8394}
OUTDIR=${OUTDIR:-/workspace/e1}
CTX=${CTX:-4096}
WISP_SWEEP=${WISP_SWEEP:-"cap8:0.45 cap16:0.60 cap32:0.80"}
VANILLA_SWEEP=${VANILLA_SWEEP:-"off44:0.85 off50:0.85"}
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

serve_and_bench() { # $1 backend-tag $2 out.json $3 extra-config-json; server already launched as $SPID
  local ok=0
  for i in $(seq 1 240); do
    curl -s -o /dev/null "127.0.0.1:$PORT/v1/models" && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "E1-VLLM-SERVE-FAIL $1 ($2)"; tail -8 "${2%.json}.server.log"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; return 1
  fi
  "$V/python" "$HERE/e1_endpoint_bench.py" --url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --backend "$1" --extra-config "$3" --out "$2" \
      || echo "E1-VLLM-BENCH-FAIL $1"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  sleep 5
}

for CFG in $WISP_SWEEP; do
  CAP="${CFG%%:*}"; CAP="${CAP#cap}"; UTIL="${CFG##*:}"
  OUT="$OUTDIR/wisp_cap${CAP}.json"
  [ -s "$OUT" ] && { echo "[e1] skip wisp cap$CAP"; continue; }
  VLLM_USE_V1=1 WISP_MODE=paged WISP_CAP_EXPERTS="$CAP" \
    "$V/vllm" serve "$MODEL" --enforce-eager --gpu-memory-utilization "$UTIL" \
    --max-model-len "$CTX" --port "$PORT" --host 127.0.0.1 \
    > "${OUT%.json}.server.log" 2>&1 &
  SPID=$!
  serve_and_bench wisp_plugin "$OUT" "{\"cap_experts\": $CAP, \"gpu_mem_util\": $UTIL}"
done

for CFG in $VANILLA_SWEEP; do
  OFF="${CFG%%:*}"; OFF="${OFF#off}"; UTIL="${CFG##*:}"
  OUT="$OUTDIR/vanilla_off${OFF}.json"
  [ -s "$OUT" ] && { echo "[e1] skip vanilla off$OFF"; continue; }
  VLLM_USE_V1=1 WISP_PLUGIN_DISABLE=1 \
    "$V/vllm" serve "$MODEL" --enforce-eager --cpu-offload-gb "$OFF" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$CTX" --port "$PORT" --host 127.0.0.1 \
    > "${OUT%.json}.server.log" 2>&1 &
  SPID=$!
  serve_and_bench vanilla "$OUT" "{\"cpu_offload_gb\": $OFF, \"gpu_mem_util\": $UTIL}"
done
echo E1-VLLM-DONE
