#!/usr/bin/env bash
# iso-VRAM Pareto + byte-identity on a PHYSICALLY constrained RTX 3090 (24 GiB,
# real PCIe 4.0 x16). This is the C2 experiment: reproduce the headline iso-VRAM
# curve on a real consumer card instead of an emulated H100 gpu-mem-util cap.
#
# Unlike scripts/m3_fig1_isovram_sweep.sh (whose gpu-mem-util knobs were sized
# for a 94 GiB H100), every arm here is sized for 24 GiB. Each config runs in
# its OWN python process (fresh CUDA context) via the day5 in-process bench.
# NVML measures the actual peak VRAM each one lands at; the plot is then
# (measured peak VRAM) -> (decode tok/s).
#
# Qwen3-30B-A3B is ~57 GiB BF16 and does NOT fit 24 GiB, so paging here is
# forced by real hardware, not by a utilization cap.
#
# Usage:
#   export VLLM_USE_V1=1 WISP_MODE=paged   # see reproduce.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/m3_3090_isovram_sweep.sh ./results/3090_fig1
set -uo pipefail

OUTDIR="${1:-./results/3090_fig1}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
MAXTOK="${MAXTOK:-128}"
NPROMPTS="${NPROMPTS:-8}"
MAXLEN="${MAXLEN:-2048}"
BENCH="scripts/m3_5_plugin_day5_bench.py"

mkdir -p "$OUTDIR"
echo "=== 3090 iso-VRAM sweep ==="
echo "  outdir=$OUTDIR model=$MODEL max_tokens=$MAXTOK n_prompts=$NPROMPTS gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo

run() {
  # args: tag backend extra_flags...
  local tag="$1"; shift
  local backend="$1"; shift
  local out="$OUTDIR/${tag}.json"
  local log="$OUTDIR/${tag}.log"
  if [[ -f "$out" ]]; then echo "[skip] $tag (exists)"; return; fi
  echo "[run ] $tag  ($backend $*)"
  python3 "$BENCH" --backend "$backend" --model "$MODEL" --num-prompts "$NPROMPTS" \
      --max-tokens "$MAXTOK" --max-model-len "$MAXLEN" \
      --output "$out" "$@" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "       FAILED (rc=$rc) — see $log"
    grep -iE "out of memory|no available memory|cannot|valueerror|assert" "$log" | tail -2 | sed 's/^/         /'
  else
    grep "SUMMARY" "$log" | tail -1 | sed 's/^/       /'
  fi
}

# ---- WiSP plug-in (paged mode): sweep cap_experts. gpu-mem-util sized so
#      each arm fits 24 GiB and leaves headroom for the KV pool. ----
run "wisp_cap8_g45"  wisp_plugin --cap-experts 8  --gpu-memory-utilization 0.45
run "wisp_cap8_g80"  wisp_plugin --cap-experts 8  --gpu-memory-utilization 0.80
run "wisp_cap16_g80" wisp_plugin --cap-experts 16 --gpu-memory-utilization 0.80
run "wisp_cap32_g90" wisp_plugin --cap-experts 32 --gpu-memory-utilization 0.90

# ---- vanilla vLLM: sweep --cpu-offload-gb (model ~57 GiB BF16). To fit the
#      resident weights on a 24 GiB card it must offload >= ~40 GiB; smaller
#      offload is expected to FAIL to start (the consumer-card floor evidence).
run "van_off50_g85"  vanilla --cpu-offload-gb 50 --gpu-memory-utilization 0.85
run "van_off44_g85"  vanilla --cpu-offload-gb 44 --gpu-memory-utilization 0.85
run "van_off30_g85"  vanilla --cpu-offload-gb 30 --gpu-memory-utilization 0.85

echo
echo "=== sweep done; results in $OUTDIR ==="
ls "$OUTDIR"/*.json 2>/dev/null
