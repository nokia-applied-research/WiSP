"""Verify WiSP decode is byte-identical to vanilla vLLM at temperature 0.

The claim: on the unquantized fused-MoE path, per-token combine is a
weighted sum over each token's routed expert set *in routing order*, so
it does not depend on which physical scratch slot an expert lands in.
WiSP only moves where the weights live (host pinned vs GPU resident);
the math is unchanged. Therefore the generated token IDs must match
vanilla vLLM exactly.

Because the WiSP plug-in auto-patches vLLM on engine startup (via its
``vllm.general_plugins`` entry point), the two arms cannot share one
process. Run each arm separately, then compare:

    # 1) vanilla baseline (plug-in inert)
    python scripts/check_byte_identity.py --mode vanilla \\
        --model Qwen/Qwen3-30B-A3B --out vanilla.json

    # 2) WiSP paging (cap forces eviction)
    python scripts/check_byte_identity.py --mode wisp --cap-experts 8 \\
        --model Qwen/Qwen3-30B-A3B --gpu-memory-utilization 0.45 \\
        --out wisp.json

    # 3) compare token IDs
    python scripts/check_byte_identity.py --compare vanilla.json wisp.json

Exit code is non-zero on any mismatch, so this is CI-friendly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROMPTS = [
    "Working-Set Paging is a memory-management technique that",
    "Explain, step by step, why a single decode step of a Mixture-of-Experts model is bound by",
    "def fib(n):\n    # iterative Fibonacci\n",
    "The capital of France is",
    "List three properties that make a cache eviction policy optimal:",
]


def _generate(args) -> dict:
    """Load the model under the requested mode and return token IDs."""
    if args.mode == "vanilla":
        # Make the auto-registered plug-in inert so this is a true baseline.
        os.environ["WISP_PLUGIN_DISABLE"] = "1"
    else:  # wisp
        os.environ.pop("WISP_PLUGIN_DISABLE", None)
        os.environ["WISP_MODE"] = "paged"
        if args.cap_experts is not None:
            os.environ["WISP_CAP_EXPERTS"] = str(args.cap_experts)
        # Explicit install in case the package was run from source without
        # `pip install` (no entry point registered).
        try:
            from wisp.integrations.vllm import install_wisp_moe
            install_wisp_moe()
        except Exception as e:  # pragma: no cover
            print(f"[warn] explicit install_wisp_moe failed ({e}); "
                  "relying on entry-point auto-load", file=sys.stderr)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    # The vanilla reference cannot fit a model larger than VRAM on its own,
    # so it streams weights from CPU (vLLM's own static offload). This does
    # not change the fused-MoE math, so its output is the true vanilla
    # reference. WiSP (paged) does not need this knob.
    if args.cpu_offload_gb and args.cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    outs = llm.generate(PROMPTS, sp)
    # vLLM may reorder; key by prompt to be safe.
    by_prompt = {o.prompt: list(o.outputs[0].token_ids) for o in outs}
    return {
        "mode": args.mode,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "cap_experts": args.cap_experts,
        "token_ids": [by_prompt[p] for p in PROMPTS],
    }


def _compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    ta, tb = a["token_ids"], b["token_ids"]
    if len(ta) != len(tb):
        print(f"FAIL: prompt count differs ({len(ta)} vs {len(tb)})")
        return 1
    all_match = True
    for i, (xa, xb) in enumerate(zip(ta, tb)):
        ok = xa == xb
        all_match = all_match and ok
        n = min(len(xa), len(xb))
        first_diff = next((j for j in range(n) if xa[j] != xb[j]), None)
        mark = "ok" if ok else f"DIFF@tok{first_diff if first_diff is not None else n}"
        print(f"  prompt {i}: {mark} (len {len(xa)} vs {len(xb)})")
    print()
    print(f"{a['mode']} vs {b['mode']}:",
          "BYTE-IDENTICAL" if all_match else "MISMATCH")
    return 0 if all_match else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["vanilla", "wisp"])
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cap-experts", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--cpu-offload-gb", type=float, default=0.0,
                   help="vanilla-only: GiB of weights to stream from CPU so a "
                        "model larger than VRAM can produce the reference.")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.compare:
        return _compare(*args.compare)

    if not args.mode:
        p.error("either --mode {vanilla,wisp} or --compare A B is required")

    res = _generate(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out} ({len(res['token_ids'])} prompts, "
              f"mode={res['mode']})")
    else:
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
