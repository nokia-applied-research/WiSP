"""Day 5 benchmark: vanilla vLLM vs WiSP plug-in across cap_experts.

What this measures
------------------

Each ``(backend, cap)`` configuration runs ``--num-prompts`` short
prompts at ``--max-tokens`` each, greedy decode, ``temperature=0``.
For each prompt we capture (ttft, decode_wall, completion_tokens).
The script emits one JSON per run in a subset of the M3
``wisp.bench.m3`` schema so the existing plotting can ingest it.

Key metrics (in ``derived``):

- ``decode_tok_per_s_per_stream_mean`` — robust per-prompt mean,
  filters degenerate <50 ms decode walls.
- ``decode_tok_per_s_aggregate`` — total completion tokens /
  total decode wall.
- ``ttft_p50_seconds`` / ``ttft_p95_seconds``.
- ``peak_vram_gib`` — best-effort poll from NVML during decode.
- ``resident_moe_gib`` (NEW for the plug-in) — sum of scratch sizes
  reported by the patch; for vanilla this is the full weight.

The prompts are intentionally simple Q&A — agentic ReAct traces
are saved for the full M3.5 harness run later. We just need clean
decode tok/s numbers to validate the cap-vs-throughput Pareto.

Why this exists (vs the existing M3 harness)
--------------------------------------------

The M3 harness wraps backends via subprocess + HTTP. Our plug-in
runs in-process and requires the patch to be installed *before*
vLLM imports the model. Adding it as a subprocess backend means
wrapping the patch install into a launcher script — doable, but
deferred to Day 6. For tonight we want the headline number on a
common prompt set.

Usage::

    python scripts/m3_5_plugin_day5_bench.py \\
        --backend wisp_plugin \\
        --cap-experts 16 \\
        --output /tmp/day5_wisp_cap16.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any


# A small, deterministic prompt set. Mixed lengths to exercise both
# prefill-bound and decode-bound paths.
DEFAULT_PROMPTS: list[str] = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Explain photosynthesis in two sentences.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Give a brief definition of machine learning.",
    "What's 17 * 23? Show your work briefly.",
    "Describe what a black hole is in two sentences.",
    "Write one sentence about the moon.",
]

# Coherent single-topic prompt sets — for the working-set-locality experiment
# (Fig 3). A same-topic session should warm the resident expert set so later
# prompts page-fault less and run faster; a diverse session never converges.
CODE_PROMPTS: list[str] = [
    "Write a Python function to reverse a linked list.",
    "Write a Python function that checks if a string is a palindrome.",
    "Implement binary search in Python with comments.",
    "Write a Python class for a simple stack with push and pop.",
    "Write a Python function to merge two sorted lists.",
    "Implement quicksort in Python.",
    "Write a Python decorator that times a function's execution.",
    "Write a Python function to count word frequencies in a string.",
]

MATH_PROMPTS: list[str] = [
    "Compute the derivative of x^3 + 2x and explain each step.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step.",
    "What is the integral of 2x dx? Show the steps.",
    "Find the sum of the first 20 positive integers and show the formula.",
    "Compute 12 factorial and explain how.",
    "Solve for x: 3x + 7 = 22, step by step.",
    "What is the greatest common divisor of 48 and 36? Show the method.",
    "Compute the area of a circle with radius 5, showing the formula.",
]

PROMPT_SETS = {"diverse": DEFAULT_PROMPTS, "code": CODE_PROMPTS, "math": MATH_PROMPTS}


_MIN_DEC_WALL_S = 0.05


def _gpu_used_gib(idx: int) -> float:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(int(mem.used) / (1 << 30))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return float("nan")


def _resolve_gpu_idx() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return 0
    try:
        return int(cvd.split(",")[0])
    except Exception:
        return 0


def _aggregate(rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    """Mirror ``wisp.bench.m3.metrics.aggregate_derived`` (subset)."""
    completed = [r for r in rows if r["completion_tokens"] > 0]
    if not completed:
        return {
            "n_prompts": len(rows),
            "n_completed": 0,
            "ttft_p50_seconds": None,
            "ttft_p95_seconds": None,
            "wall_p50_seconds": None,
            "wall_mean_seconds": None,
            "decode_tok_per_s_per_stream_mean": None,
            "decode_tok_per_s_aggregate": None,
            "completion_tokens_total": 0,
            "concurrency": concurrency,
        }

    def _pctile(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        i = int(round(q * (len(s) - 1)))
        return s[max(0, min(len(s) - 1, i))]

    ttfts = [r["ttft_seconds"] for r in completed if r.get("ttft_seconds") is not None]
    walls = [r["wall_seconds"] for r in completed]
    total_completion = sum(r["completion_tokens"] for r in completed)
    total_decode_wall = 0.0
    max_decode_wall = 0.0
    per_stream_rates: list[float] = []
    skipped = 0
    counted = 0
    for r in completed:
        dec_wall = r["wall_seconds"] - (r.get("ttft_seconds") or 0.0)
        dec_wall = max(dec_wall, 0.0)
        if dec_wall < _MIN_DEC_WALL_S:
            skipped += 1
            continue
        per_stream_rates.append(r["completion_tokens"] / dec_wall)
        counted += 1
        total_decode_wall += dec_wall
        if dec_wall > max_decode_wall:
            max_decode_wall = dec_wall
    decode_per_stream_mean = (
        sum(per_stream_rates) / len(per_stream_rates) if per_stream_rates else None
    )
    if concurrency > 1 and max_decode_wall > 0:
        agg_denom = max_decode_wall
    else:
        agg_denom = total_decode_wall
    decode_aggregate = (
        total_completion / agg_denom if agg_denom > 0 else None
    )

    return {
        "n_prompts": len(rows),
        "n_completed": len(completed),
        "ttft_p50_seconds": _pctile(ttfts, 0.5) if ttfts else None,
        "ttft_p95_seconds": _pctile(ttfts, 0.95) if ttfts else None,
        "wall_p50_seconds": _pctile(walls, 0.5),
        "wall_mean_seconds": sum(walls) / len(walls),
        "wall_seconds_total": sum(walls),
        "decode_tok_per_s_per_stream_mean": decode_per_stream_mean,
        "decode_tok_per_s_per_stream_n_counted": counted,
        "decode_tok_per_s_per_stream_n_skipped": skipped,
        "decode_tok_per_s_aggregate": decode_aggregate,
        "completion_tokens_total": total_completion,
        "concurrency": concurrency,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="m3_5_plugin_day5_bench")
    ap.add_argument(
        "--backend",
        choices=["vanilla", "wisp_plugin"],
        required=True,
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B",
    )
    ap.add_argument(
        "--cap-experts",
        type=int,
        default=None,
        help="WiSP-only. Defaults to env WISP_CAP_EXPERTS or num_experts.",
    )
    ap.add_argument(
        "--wisp-mode",
        default="paged",
        choices=["resident", "paged", "day2a", "day2b"],  # day* = legacy aliases
    )
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--prompt-set",
        default="diverse",
        choices=["diverse", "code", "math"],
        help="diverse = mixed topics (working set never converges); "
             "code/math = coherent single-topic session (working set warms).",
    )
    ap.add_argument(
        "--repeat-prompts",
        type=int,
        default=1,
        help="Repeat the prompt set N times back-to-back (longer session to expose warm-up).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Per-prompt completion budget. Longer = better decode-tok/s signal.",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    ap.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="vanilla-only: GiB of weights to offload to CPU (the iso-VRAM knob for the vLLM baseline).",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Where to write the result JSON.",
    )
    ap.add_argument(
        "--prefetch",
        action="store_true",
        help="WiSP-only. Enable cooccur prefetch (default off; net-negative on the H100 sandbox).",
    )
    ap.add_argument(
        "--cooccur-path",
        default=None,
        help="WiSP-only. Path to warm cooccur JSON; only used when --prefetch.",
    )
    args = ap.parse_args(argv)

    log_level = os.environ.get("WISP_BENCH_LOG_LEVEL", "info").upper()
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s  %(name)s  %(message)s")
    log = logging.getLogger("wisp.day5.bench")

    gpu_idx = _resolve_gpu_idx()
    log.info("starting day5 bench: backend=%s model=%s cap=%s prefetch=%s gpu=%d",
             args.backend, args.model, args.cap_experts, args.prefetch, gpu_idx)
    log.info("VRAM before any import: %.2f GiB", _gpu_used_gib(gpu_idx))

    if args.backend == "vanilla":
        # CRITICAL: the wisp plug-in auto-registers via the
        # `vllm.general_plugins` entry point on `import vllm`. Without
        # this, the "vanilla" baseline would be silently patched by WiSP
        # too, invalidating the comparison. Inert the plug-in explicitly.
        os.environ["WISP_PLUGIN_DISABLE"] = "1"
        log.info("vanilla backend: WISP_PLUGIN_DISABLE=1 (plug-in inerted)")

    if args.backend == "wisp_plugin":
        os.environ.pop("WISP_PLUGIN_DISABLE", None)
        os.environ["WISP_MODE"] = args.wisp_mode
        if args.cap_experts is not None:
            os.environ["WISP_CAP_EXPERTS"] = str(args.cap_experts)
        if args.prefetch:
            os.environ["WISP_PREFETCH"] = "1"
            if args.cooccur_path:
                os.environ["WISP_COOCCUR_PATH"] = args.cooccur_path
        else:
            os.environ.pop("WISP_PREFETCH", None)
        from wisp.integrations.vllm import install_wisp_moe, is_installed
        install_wisp_moe()
        log.info("WiSP patch installed=%s", is_installed())

    os.environ.setdefault("VLLM_USE_V1", "0")

    log.info("importing vllm...")
    from vllm import LLM, SamplingParams

    log.info("loading %s with gpu_memory_utilization=%.2f max_model_len=%d ...",
             args.model, args.gpu_memory_utilization, args.max_model_len)
    t_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        cpu_offload_gb=float(args.cpu_offload_gb),
        tensor_parallel_size=1,
    )
    load_wall = time.perf_counter() - t_load
    log.info("LLM() finished in %.1fs", load_wall)

    vram_after_load = _gpu_used_gib(gpu_idx)
    log.info("VRAM after model load: %.2f GiB", vram_after_load)

    base_prompts = PROMPT_SETS[args.prompt_set][: args.num_prompts]
    prompts = base_prompts * max(1, args.repeat_prompts)
    log.info("prompt_set=%s n=%d (repeat=%d -> %d total)",
             args.prompt_set, len(base_prompts), args.repeat_prompts, len(prompts))
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Warm-up: run one short prompt to JIT-compile the kernel + prime
    # the scratch with steady-state experts; first call always pays
    # prefill overhead which would inflate ttft on prompt 0.
    log.info("warm-up generation...")
    _ = llm.generate([prompts[0]], SamplingParams(max_tokens=8, temperature=0.0))

    # Polling peak VRAM during decode (best-effort thread).
    import threading
    peak_vram = {"v": _gpu_used_gib(gpu_idx)}
    poll_stop = threading.Event()

    def _poll() -> None:
        while not poll_stop.is_set():
            try:
                v = _gpu_used_gib(gpu_idx)
                if v > peak_vram["v"]:
                    peak_vram["v"] = v
            except Exception:
                pass
            poll_stop.wait(0.25)

    poll_t = threading.Thread(target=_poll, daemon=True)
    poll_t.start()

    # ----- per-prompt run loop -----
    rows: list[dict[str, Any]] = []
    log.info("running %d prompts at max_tokens=%d, temp=0", len(prompts), args.max_tokens)
    overall_t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        # vLLM's LLM.generate doesn't expose ttft natively. Use streaming
        # to time the first token; fall back to wall_seconds if not
        # available. For our purposes we'll approximate ttft via a
        # second short call with max_tokens=1; this duplicates prefill
        # so it's an upper bound but consistent across configs.
        t_ttft0 = time.perf_counter()
        ttft_out = llm.generate([p], SamplingParams(max_tokens=1, temperature=0.0))
        ttft_wall = time.perf_counter() - t_ttft0

        t_full0 = time.perf_counter()
        full_out = llm.generate([p], sp)
        full_wall = time.perf_counter() - t_full0

        completion_tokens = (
            len(full_out[0].outputs[0].token_ids) if full_out and full_out[0].outputs else 0
        )
        text = full_out[0].outputs[0].text if full_out and full_out[0].outputs else ""
        row = {
            "prompt_idx": i,
            "prompt": p,
            "completion_text": text,
            "completion_tokens": completion_tokens,
            "wall_seconds": full_wall,
            "ttft_seconds": ttft_wall,
        }
        rows.append(row)
        log.info(
            "  [%d/%d] tokens=%d wall=%.2fs ttft=%.2fs dec_tok/s=%.2f",
            i + 1, len(prompts), completion_tokens, full_wall, ttft_wall,
            completion_tokens / max(full_wall - ttft_wall, _MIN_DEC_WALL_S),
        )

    overall_wall = time.perf_counter() - overall_t0
    poll_stop.set()
    poll_t.join(timeout=1)

    derived = _aggregate(rows, concurrency=1)
    derived["peak_vram_gib"] = float(peak_vram["v"]) if peak_vram["v"] == peak_vram["v"] else None
    derived["vram_after_load_gib"] = float(vram_after_load)
    derived["model_load_seconds"] = float(load_wall)
    derived["wall_seconds_total"] = float(overall_wall)

    # Best-effort: count pinned MoE layers from log scan (the plug-in
    # path doesn't expose its state via in-process attrs in V1).
    derived["cap_experts"] = args.cap_experts

    result = {
        "backend": args.backend,
        "workload": "day5_static_prompts",
        "schema_version": 1,
        "config": {
            "backend": args.backend,
            "model": args.model,
            "dtype": args.dtype,
            "cap_experts": args.cap_experts,
            "cpu_offload_gb": float(args.cpu_offload_gb),
            "prompt_set": args.prompt_set,
            "repeat_prompts": args.repeat_prompts,
            "wisp_mode": args.wisp_mode if args.backend == "wisp_plugin" else None,
            "prefetch": bool(args.prefetch),
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "temperature": 0.0,
            "concurrency": 1,
        },
        "derived": derived,
        "per_prompt": rows,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("wrote %s", args.output)
    log.info(
        "SUMMARY backend=%s cap=%s decode_tok/s(per-stream-mean)=%s ttft_p50=%s peak_vram=%.2f",
        args.backend, args.cap_experts,
        f"{derived['decode_tok_per_s_per_stream_mean']:.2f}" if derived.get("decode_tok_per_s_per_stream_mean") else "n/a",
        f"{derived['ttft_p50_seconds']:.2f}" if derived.get("ttft_p50_seconds") else "n/a",
        derived.get("peak_vram_gib", float("nan")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
