"""MV-WSA startup configurator for the live vLLM plugin (Phase B, Option A).

A personal device has a fixed GPU budget. WiSP pages MoE experts (CPU master +
GPU scratch of ``cap_experts`` per layer) and vLLM holds the KV cache on GPU
with the rest spilling to CPU via native KV offload. MV-WSA's job, *at launch*
for a given workload, is to pick the expert<->KV split that minimises agentic
latency -- the offline-validated equimarginal point.

This module turns a target split fraction ``f`` (= bytes to experts) into the
two concrete vLLM/WiSP knobs, holding the *total* GPU footprint fixed so the
arms are iso-VRAM:

  - ``WISP_CAP_EXPERTS``      : resident experts per layer  (expert GPU bytes)
  - ``--kv-cache-memory-bytes``: GPU KV pool size           (KV GPU bytes)

Both arms also get identical CPU KV offload via vLLM's built-in swap space
(``--swap-space``) so KV beyond the GPU pool spills to CPU -- this is the
apples-to-apples baseline the user asked for (vLLM's own KV offload ON for
everyone). We deliberately avoid the experimental ``OffloadingConnector``
(``--kv-offloading-*`` / ``--kv-transfer-config``): in vLLM 0.11.2 it ships a
wiring bug (num_cpu_blocks stuck at 0) and serializes the engine to one request
at a time, which breaks concurrent agentic serving. See ``serve_flags``.

Usage:
  python3 mvwsa_configure.py --model qwen3 --total-gpu-gib 24 --gpu-util 0.9 \
      --split 0.65 --arm mvwsa --print-flags
  # or derive the split from an offline result file:
  python3 mvwsa_configure.py --model qwen3 --total-gpu-gib 24 \
      --from-results results/mvwsa/v2/qwen3_os_cs16.json --print-flags
"""
from __future__ import annotations
import argparse, json, os

GIB = 1 << 30

# Per-model facts. e_bytes = bytes of ONE (layer,expert) MLP in bf16
# (= 3 * hidden * moe_inter * 2). non_expert_gib = attention+embed+norm
# weights resident on GPU (experts are offloaded to CPU by the plugin).
MODELS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "n_layers": 48, "n_experts": 128, "top_k": 8,
        "e_bytes": 9.4e6, "kv_bytes_per_token": 98304.0,
        "non_expert_gib": 3.0,   # ~1.5B non-expert params bf16 + lm_head
        "weights_gib": 61.0,     # ~30.5B params bf16 (for naive --cpu-offload-gb sizing)
    },
    "olmoe": {
        "model_id": "allenai/OLMoE-1B-7B-0924",
        "n_layers": 16, "n_experts": 64, "top_k": 8,
        "e_bytes": 12.0e6, "kv_bytes_per_token": 32768.0,
        "non_expert_gib": 0.6,
        "weights_gib": 13.8,     # ~6.9B params bf16
    },
    # MiniMax-M2.5: 229B FP8 MoE, full softmax attention on every layer
    # (attn_type_list all 1s -> KV grows normally, so the split applies).
    # e_bytes = 3 * hidden(3072) * moe_inter(1536) * 1 byte (FP8); FP8 block
    # scales add <1%. kv = 2 * n_kv(8) * head_dim(128) * L(62) * 2 (bf16 KV).
    "minimax": {
        "model_id": "MiniMaxAI/MiniMax-M2.5",
        "n_layers": 62, "n_experts": 256, "top_k": 8,
        "e_bytes": 14.2e6, "kv_bytes_per_token": 253952.0,
        "non_expert_gib": 8.0,    # bf16 attention + embed + lm_head + norms
        "weights_gib": 218.0,     # ~210 GiB experts (FP8) + ~8 non-expert
    },
}


def split_to_config(model: str, total_gpu_gib: float, card_total_gib: float,
                    f: float, *, overhead_gib: float = 2.5,
                    kv_floor_gib: float = 0.5, kv_admission_tokens: float = 0.0):
    """Map split fraction f (-> experts) to (cap_experts, kv_cache_bytes).

    ``total_gpu_gib`` is the *simulated device* budget. On a larger physical
    card we constrain vLLM with ``--gpu-memory-utilization = total/card``.
    After reserving the non-expert weights and a fixed activation/cuda-graph
    overhead, the remaining *pageable pool* is divided between expert scratch
    and KV at fraction ``f``.

    KV-admission floor (the live-regime correction). The offline cost model is
    blind to one hard fact of a live engine: a KV pool too small to *admit* the
    concurrent working set does not pay "one extra miss" -- it cliffs (the
    scheduler cannot place the batch, so it preempts/re-prefills every step, or
    refuses to start). In the expert-bound regime the offline oracle therefore
    drifts to f->1 (experts buy everything, KV ~0), which is *infeasible* live.
    MV-WSA corrects this with ``kv_admission_tokens``: KV is reserved to hold
    ``concurrency`` working contexts before any byte goes to experts, so the
    expert cap is the largest one that still leaves an admissible KV pool. This
    is what separates MV-WSA from a KV-blind expert-proportional prior (Flux),
    which sizes KV for a single nominal context and starves under concurrency.
    Pass ``0`` to disable (the Flux/50-50 baselines).
    """
    m = MODELS[model]
    total_bytes = total_gpu_gib * GIB
    gpu_util = min(0.95, total_gpu_gib / card_total_gib)
    non_expert = m["non_expert_gib"] * GIB
    overhead = overhead_gib * GIB
    pool = total_bytes - non_expert - overhead
    if pool <= 0:
        raise ValueError(f"budget too small: pool={pool/GIB:.2f} GiB <= 0")

    per_layer_expert = m["n_layers"] * m["e_bytes"]
    # KV bytes that MUST stay resident to admit the concurrent working set.
    kv_floor_bytes = max(kv_floor_gib * GIB,
                         kv_admission_tokens * m["kv_bytes_per_token"])
    # experts get f of the pool, snapped to an integer per-layer cap ...
    cap = round(f * pool / per_layer_expert)
    cap = min(cap, m["n_experts"])
    # ... but never so many that the admission floor is violated.
    cap_max_floor = int((pool - kv_floor_bytes) // per_layer_expert)
    if cap_max_floor >= m["top_k"]:
        cap = min(cap, cap_max_floor)
    cap = max(m["top_k"], cap)
    expert_bytes = cap * per_layer_expert
    kv_bytes = max(pool - expert_bytes, kv_floor_gib * GIB)

    realized_f = expert_bytes / (expert_bytes + kv_bytes)
    kv_tokens = int(kv_bytes / m["kv_bytes_per_token"])
    return {
        "model": model, "model_id": m["model_id"],
        "target_split": f, "realized_split": realized_f,
        "cap_experts": int(cap), "n_experts": m["n_experts"],
        "kv_cache_memory_bytes": int(kv_bytes),
        "kv_cache_gib": kv_bytes / GIB,
        "kv_cache_tokens_est": kv_tokens,
        "kv_admission_tokens": int(kv_admission_tokens),
        "expert_gpu_gib": expert_bytes / GIB,
        "pool_gib": pool / GIB,
        "total_gpu_gib": total_bytes / GIB,
        "gpu_util": gpu_util,
    }


def arm_split(model: str, arm: str, mvwsa_f: float):
    """The split fraction for a baseline arm."""
    if arm == "mvwsa":
        return mvwsa_f
    if arm == "fifty":
        return 0.5
    if arm == "flux":
        # working-set-proportional prior (model-only, context-agnostic):
        # all experts vs a nominal KV working set. We approximate with the
        # full expert WS share -> expert-heavy, matching the offline Flux.
        m = MODELS[model]
        we = m["n_experts"] * m["n_layers"] * m["e_bytes"]
        # nominal KV WS ~ one full max_model_len context (set small here;
        # Flux is intentionally a naive static prior).
        wk = 8192 * m["kv_bytes_per_token"]
        return we / (we + wk)
    raise ValueError(arm)


def admission_tokens(arm: str, concurrency: int, max_model_len: int,
                     *, override: float | None = None) -> float:
    """KV-admission floor in tokens. MV-WSA reserves enough GPU KV to admit the
    concurrent working set (``concurrency`` contexts at full length) before
    spending on experts; Flux/50-50 do not (they pass 0)."""
    if override is not None:
        return override
    if arm == "mvwsa":
        return float(concurrency) * float(max_model_len)
    return 0.0


def serve_flags(cfg: dict, *, kv_offload_gib: float, max_model_len: int,
                port: int, gpu_idx: int, block_size: int = 16):
    """vLLM serve CLI flags + env for one arm.

    KV offload: we use vLLM's built-in CPU swap space (``--swap-space``) as the
    KV-offload mechanism, enabled identically on every arm (apples-to-apples,
    per the user's requirement). We deliberately do NOT use the experimental
    ``OffloadingConnector`` (``--kv-offloading-*`` / ``--kv-transfer-config``):
    in vLLM 0.11.2 it (a) ships a wiring bug (num_cpu_blocks defaults to 0 and
    is never recomputed) and (b) serializes the engine to ONE request at a time,
    making concurrent agentic serving impossible. Swap space gives real CPU KV
    offload under pressure while preserving continuous batching.

    The expert<->KV split is realized by ``cap_experts`` (expert GPU bytes) +
    ``--kv-cache-memory-bytes`` (pinned GPU KV pool); when the GPU KV pool is
    exceeded under concurrency, vLLM swaps/recomputes -- the KV-heavy arms ride
    this out, the expert-heavy arms thrash.
    """
    flags = [
        "vllm", "serve", cfg["model_id"],
        "--port", str(port), "--host", "127.0.0.1",
        "--dtype", "bfloat16",
        "--block-size", str(block_size),
        "--gpu-memory-utilization", str(cfg["gpu_util"]),
        "--kv-cache-memory-bytes", str(cfg["kv_cache_memory_bytes"]),
        # vLLM built-in CPU KV offload (swap), identical across arms
        "--swap-space", str(kv_offload_gib),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    cfg["swap_space_gib"] = kv_offload_gib
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu_idx),
        "VLLM_USE_V1": "1",
        "WISP_MODE": "paged",
        "WISP_CAP_EXPERTS": str(cfg["cap_experts"]),
        # Force the Triton FP8 MoE backend. WiSP's FP8 paging slices each
        # expert's (weight, scale) per-block layout straight from the pinned-CPU
        # master into a scratch slot; DeepGEMM pre-processes/realigns the layout,
        # which both breaks bit-exact slicing and crashes its MoE warmup under a
        # paged expert_map (randint(0, 0) when a scratch slot is momentarily
        # empty). Triton consumes the raw block layout, so paging is exact.
        "VLLM_USE_DEEP_GEMM": "0",
    }
    return flags, env


# ---------------------------------------------------------------------------
# Naive (vanilla vLLM) baselines -- the serving stack's own offload knobs, with
# NO WiSP routing-aware paging and NO joint split policy. Three variants:
#   naive_kv     : weights fully resident + --swap-space (KV->CPU only).
#                  Fails to start when the model does not fit the budget (the
#                  MV-WSA regime) -- a measured capability boundary.
#   naive_expert : --cpu-offload-gb (static, routing-blind weight offload),
#                  KV gets whatever GPU is left. == the main-table naive baseline.
#   naive_both   : --cpu-offload-gb + --swap-space.
# All are iso-VRAM with the WiSP arms (same gpu_memory_utilization cap) and run
# with WISP_PLUGIN_DISABLE=1 so the engine is pure vanilla vLLM.
# ---------------------------------------------------------------------------
NAIVE_VARIANTS = ("naive_kv", "naive_expert", "naive_both")


def naive_config(model: str, total_gpu_gib: float, card_total_gib: float,
                 variant: str, *, overhead_gib: float = 2.5,
                 kv_floor_gib: float = 0.5):
    if variant not in NAIVE_VARIANTS:
        raise ValueError(variant)
    m = MODELS[model]
    gpu_util = min(0.95, total_gpu_gib / card_total_gib)
    weights = m["weights_gib"]
    expert_off = variant in ("naive_expert", "naive_both")
    kv_off = variant in ("naive_kv", "naive_both")
    if expert_off:
        # offload enough weight so resident weights + overhead + a minimal KV
        # pool fit inside the budget; vLLM then auto-sizes KV with the leftover.
        resident_target = total_gpu_gib - overhead_gib - kv_floor_gib
        cpu_offload_gb = max(0.0, weights - resident_target)
    else:
        cpu_offload_gb = 0.0
    return {
        "model": model, "model_id": m["model_id"], "variant": variant, "arm": variant,
        "cpu_offload_gb": round(cpu_offload_gb, 1), "swap": kv_off,
        "gpu_util": gpu_util, "total_gpu_gib": total_gpu_gib,
        "weights_gib": weights,
        "resident_weights_gib": round(max(0.0, weights - cpu_offload_gb), 1),
        "fits_budget": (weights - cpu_offload_gb) <= (total_gpu_gib - overhead_gib),
    }


def naive_serve_flags(cfg: dict, *, kv_offload_gib: float, max_model_len: int,
                      port: int, gpu_idx: int, block_size: int = 16):
    """Vanilla vLLM serve flags (no WiSP) for a naive baseline variant."""
    flags = [
        "vllm", "serve", cfg["model_id"],
        "--port", str(port), "--host", "127.0.0.1",
        "--dtype", "bfloat16",
        "--block-size", str(block_size),
        "--gpu-memory-utilization", str(cfg["gpu_util"]),
        "--max-model-len", str(max_model_len),
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    if cfg["cpu_offload_gb"] > 0:
        flags += ["--cpu-offload-gb", str(cfg["cpu_offload_gb"])]
    if cfg["swap"]:
        flags += ["--swap-space", str(kv_offload_gib)]
        cfg["swap_space_gib"] = kv_offload_gib
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu_idx),
        "VLLM_USE_V1": "1",
        "WISP_PLUGIN_DISABLE": "1",   # pure vanilla vLLM
    }
    return flags, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="qwen3")
    ap.add_argument("--total-gpu-gib", type=float, required=True,
                    help="simulated device GPU budget")
    ap.add_argument("--card-total-gib", type=float, default=94.0,
                    help="physical GPU memory (for --gpu-memory-utilization)")
    ap.add_argument("--overhead-gib", type=float, default=2.5)
    ap.add_argument("--split", type=float, default=None, help="explicit MV-WSA split f")
    ap.add_argument("--from-results", default=None,
                    help="offline result json; use its converged/oracle split")
    ap.add_argument("--arm", choices=["mvwsa", "fifty", "flux"], default="mvwsa")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="concurrent sessions; sizes MV-WSA's KV-admission floor")
    ap.add_argument("--kv-admission-tokens", type=float, default=None,
                    help="override KV-admission floor (default conc*max_model_len for mvwsa)")
    ap.add_argument("--kv-offload-gib", type=float, default=40.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-idx", type=int, default=0)
    ap.add_argument("--print-flags", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.from_results:
        d = json.load(open(args.from_results))
        mvwsa_f = d.get("converged_split", d.get("oracle_split"))
    elif args.split is not None:
        mvwsa_f = args.split
    else:
        raise SystemExit("need --split or --from-results")

    f = arm_split(args.model, args.arm, mvwsa_f)
    # Only MV-WSA reserves a concurrency-aware KV-admission floor; Flux/50-50
    # are blind to it (Flux deliberately so -- that is the baseline gap).
    adm = admission_tokens(args.arm, args.concurrency, args.max_model_len,
                           override=args.kv_admission_tokens)
    cfg = split_to_config(args.model, args.total_gpu_gib, args.card_total_gib, f,
                          overhead_gib=args.overhead_gib, kv_admission_tokens=adm)
    cfg["arm"] = args.arm
    flags, env = serve_flags(cfg, kv_offload_gib=args.kv_offload_gib,
                             max_model_len=args.max_model_len,
                             port=args.port, gpu_idx=args.gpu_idx)
    cfg["serve_flags"] = flags
    cfg["serve_env"] = env

    print(f"[configure] {args.model} arm={args.arm} target_f={f:.3f} "
          f"realized_f={cfg['realized_split']:.3f} "
          f"cap_experts={cfg['cap_experts']}/{cfg['n_experts']} "
          f"KV={cfg['kv_cache_gib']:.2f}GiB (~{cfg['kv_cache_tokens_est']} tok) "
          f"expert={cfg['expert_gpu_gib']:.2f}GiB pool={cfg['pool_gib']:.2f}GiB")
    if args.print_flags:
        print("  ENV:", " ".join(f"{k}={v}" for k, v in env.items()))
        print("  CMD:", " ".join(flags))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(cfg, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
