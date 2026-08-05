"""M3.11 — Anvil credibility microbenchmark (1/3): bandwidth probe.

Issues a series of pinned-CPU → GPU :func:`torch.cuda.Stream.synchronize`'d
:func:`Tensor.copy_` ops and reports the achieved bandwidth. Run this
on the pod's H100 (which natively sees ~3 TB/s HBM and PCIe5 ~64 GB/s
host->device); the throttle profiles in :mod:`wisp.harness.profiles`
claim PCIe4 = 32 GB/s and PCIe5 = 64 GB/s. The point of this script
is to **prove that the Anvil's nominal numbers are within an order
of magnitude of what cudaMemcpyAsync can deliver under similar
constraints** — the first leg of the paper's M3.11 credibility
defence.

Output: a JSON envelope at ``$WISP_M3_OUT/anvil_bandwidth_probe.json``
with one row per probe size + iteration. The chart-builder under
``scripts/m3_11_plot.py`` (lands with M3.13) reads this directly.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/m3_11_anvil_bandwidth_probe.py \
        --out ./results/bandwidth.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Dict, List, Optional


logger = logging.getLogger("wisp.m3.bandwidth_probe")


def _gib(n: int) -> float:
    return n / float(1 << 30)


def _probe(
    *,
    nbytes: int,
    iters: int,
    device: str,
) -> Dict[str, object]:
    """Time ``iters`` copies of ``nbytes`` from pinned-CPU to GPU."""
    import torch

    n_elems = nbytes // 2  # bfloat16 elements
    cpu = torch.empty(n_elems, dtype=torch.bfloat16, pin_memory=True)
    gpu = torch.empty(n_elems, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream(device=device)
    # Warmup — first copy includes CUDA-context spin-up.
    with torch.cuda.stream(stream):
        gpu.copy_(cpu, non_blocking=True)
    stream.synchronize()

    seconds: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            gpu.copy_(cpu, non_blocking=True)
        stream.synchronize()
        seconds.append(time.perf_counter() - t0)

    bw_gbps = [_gib(nbytes) / s for s in seconds]
    return {
        "nbytes": nbytes,
        "nbytes_gib": _gib(nbytes),
        "iters": iters,
        "device": device,
        "seconds_mean": statistics.fmean(seconds),
        "seconds_p50": statistics.median(seconds),
        "seconds_min": min(seconds),
        "seconds_max": max(seconds),
        "gbps_mean": statistics.fmean(bw_gbps),
        "gbps_p50": statistics.median(bw_gbps),
        "gbps_max": max(bw_gbps),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sizes-gib",
        default="0.0625,0.25,1,4,16",
        help="comma-separated probe sizes in GiB (default: 64 MiB .. 16 GiB)",
    )
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--out",
        required=True,
        help="output JSON path",
    )
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(message)s",
    )

    try:
        import torch
    except ImportError:
        print("torch not installed", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    sizes_gib = [float(s) for s in args.sizes_gib.split(",") if s.strip()]
    sizes_bytes = [int(g * (1 << 30)) for g in sizes_gib]

    # Probe pcie/hbm side; the result encoding doesn't try to label
    # the bus — that's an analyst's job (compare gbps_p50 to
    # PROFILES[name].bytes_per_sec).
    results: List[Dict[str, object]] = []
    for nb in sizes_bytes:
        logger.info("probe %.3f GiB", _gib(nb))
        results.append(_probe(nbytes=nb, iters=args.iters, device=args.device))

    # Compare against the Anvil profile catalogue so the JSON is
    # self-explanatory if you read it with `cat`.
    try:
        from wisp.harness.profiles import PROFILES

        profile_summary = {
            name: {
                "bytes_per_sec": p.bytes_per_sec,
                "gbps": p.bytes_per_sec / (1 << 30),
            }
            for name, p in PROFILES.items()
        }
    except Exception:  # pragma: no cover
        profile_summary = {}

    envelope = {
        "schema_version": "m3_anvil_bandwidth_probe.v1",
        "device": args.device,
        "results": results,
        "anvil_profiles": profile_summary,
    }

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print(f"  [ok] wrote {out_path}")
    for r in results:
        print(
            f"  {r['nbytes_gib']:>6.3f} GiB  {r['gbps_p50']:>7.2f} GB/s p50  "
            f"({r['seconds_p50']*1e3:>6.2f} ms)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
