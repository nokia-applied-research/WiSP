#!/usr/bin/env python3
"""E1 cross-engine baseline driver: benchmark ANY OpenAI-compatible endpoint.

Times single-stream decode against a running server (llama.cpp `llama-server`,
KTransformers server, vLLM, vLLM+WiSP — anything speaking /v1/completions with
stream=true). TTFT is measured as time-to-first-streamed-token (no double-run
approximation); decode rate as (ntok-1)/(t_last - t_first). Peak VRAM is polled
via nvidia-smi in a side thread.

Output JSON matches the m3_5 bench schema (config/derived) so
scripts/m3_fig1_plot.py pairs rows across engines unchanged.

Protocol defaults mirror the paper's iso-VRAM measurement: the m3_5 "diverse"
prompt set, temperature 0, max_tokens 256, single stream, one warmup prompt.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time

import requests

# Same spirit as m3_5_plugin_day5_bench.py prompt sets: a coherent-session set
# (working-set locality) and a diverse set. Keep them short and fixed — these
# are throughput probes, not quality evals.
PROMPTS = {
    "diverse": [
        "Explain the difference between a mutex and a semaphore.",
        "Write a haiku about autumn rain.",
        "What causes inflation in an economy?",
        "Describe how a rocket engine achieves thrust.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "How does a hash table handle collisions?",
        "What is the significance of the Krebs cycle?",
        "Give three tips for improving photography composition.",
    ],
    "code": [
        "Write a Python function that merges two sorted lists.",
        "Refactor this idea into a class: a counter with increment and reset.",
        "Write a bash loop that renames every .txt file to .md.",
        "Implement binary search in Python with tests.",
        "Write a SQL query returning the top 5 customers by total order value.",
        "Explain and fix a Python UnboundLocalError in a closure.",
        "Write a regex that matches ISO-8601 dates.",
        "Implement an LRU cache in Python using OrderedDict.",
    ],
}


class VramPoller:
    def __init__(self, period_s: float = 0.25):
        self.period = period_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                self.peak_mib = max(self.peak_mib, int(out))
            except Exception:
                pass
            time.sleep(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


def stream_completion(url: str, model: str, prompt: str, max_tokens: int,
                      timeout_s: float) -> tuple[float, float, int]:
    """Returns (ttft_s, decode_tps, ntok). Counts streamed chunks as tokens
    (engines emit one token per SSE data line on this endpoint)."""
    t0 = time.perf_counter()
    t_first = None
    t_last = t0
    ntok = 0
    with requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, timeout=timeout_s,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                choice = json.loads(payload)["choices"][0]
            except Exception:
                continue
            if choice.get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                ntok += 1
    if t_first is None or ntok < 2:
        raise RuntimeError(f"no streamed tokens (ntok={ntok})")
    return t_first - t0, (ntok - 1) / max(t_last - t_first, 1e-9), ntok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8391")
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--backend", required=True,
                    help="tag for the config.backend field, e.g. llamacpp / ktransformers")
    ap.add_argument("--prompt-set", choices=sorted(PROMPTS), default="diverse")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-request cap; offloaded 30B decode can be very slow")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--extra-config", default="{}",
                    help="JSON merged into config (e.g. '{\"ngl\": 24}')")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = PROMPTS[args.prompt_set]
    for p in prompts[: args.warmup]:
        try:
            stream_completion(args.url, args.model, p, 16, args.timeout)
        except Exception as e:
            print(f"[warmup] {e} (continuing)")

    rows = []
    with VramPoller() as poller:
        for i, p in enumerate(prompts):
            ttft, tps, ntok = stream_completion(
                args.url, args.model, p, args.max_tokens, args.timeout)
            rows.append({"prompt_idx": i, "ttft_s": ttft,
                         "decode_tok_per_s": tps, "ntok": ntok})
            print(f"[{i+1}/{len(prompts)}] ttft={ttft:.2f}s decode={tps:.2f} tok/s ntok={ntok}")

    config = {"backend": args.backend, "model": args.model,
              "prompt_set": args.prompt_set, "max_tokens": args.max_tokens,
              "concurrency": 1}
    config.update(json.loads(args.extra_config))
    result = {
        "config": config,
        "rows": rows,
        "derived": {
            "decode_tok_per_s_per_stream_mean":
                statistics.mean(r["decode_tok_per_s"] for r in rows),
            "decode_tok_per_s_per_stream_median":
                statistics.median(r["decode_tok_per_s"] for r in rows),
            "ttft_p50_seconds":
                statistics.median(r["ttft_s"] for r in rows),
            "peak_vram_gib": poller.peak_mib / 1024.0,
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    d = result["derived"]
    print(f"[done] {args.backend}: decode_mean={d['decode_tok_per_s_per_stream_mean']:.2f} "
          f"tok/s ttft_p50={d['ttft_p50_seconds']:.2f}s "
          f"peak_vram={d['peak_vram_gib']:.1f}GiB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
