#!/usr/bin/env python3
"""MV-WSA live dual-dynamic split on the *same* AgentInstruct workload as the
static `tab:mvwsa-live` table (see paper_draft/md/exp_tables/mvwsa_live_split.md).

Same dataset / trace / metrics as `m3_mvwsa_live_bench.py`:
  * THUDM/AgentInstruct, `os` split, first 12 conversations, <=3 timed assistant
    turns each (= 36 timed turns), each turn = full history up to that turn,
    600-char per-turn cap, a distinct ~1.5k-token pseudo-document prepended per
    session (KV pressure), concurrency 4, temperature 0, max_tokens 128.
  * Metrics: TTFT, decode tok/s, end-to-end wall (lower is better).

The ONLY difference from the static harness: that one drives `vllm serve` over
HTTP (multi-process); live KV-pool resize needs the in-process v1 EngineCore
(VLLM_ENABLE_V1_MULTIPROCESSING=0). So BOTH arms run in this single-process
harness and are compared apples-to-apples:

  * ``fixed``   — static MV-WSA: cap_experts + KV pool fixed at the offline split
                  (conservative admission floor = concurrency x max_model_len).
                  This is exactly the arxiv behaviour.
  * ``dynamic`` — same split at t=0, then the live MVWSAController re-solves the
                  expert<->KV allocation after every (drained) round, tracking the
                  *actual* KV working set: this workload's contexts sit far below
                  the conservative floor, so the controller reclaims idle KV bytes
                  into resident experts -- automatically finding the tighter
                  operating point the static floor leaves on the table.

Concurrency-4 is realised with the engine step loop: up to 4 sessions' current
turns are in flight together (scheduler batches them); we drain between rounds so
the resize is a clean barrier.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import threading
import time
from pathlib import Path


def _set_env(cap_experts: int) -> None:
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("WISP_MODE", "day2b")
    os.environ["WISP_CAP_EXPERTS"] = str(cap_experts)
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")


# Reuse the EXACT workload construction from the static harness.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "m3_bench", os.path.join(_here, "m3_mvwsa_live_bench.py"))
_m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m3)
load_conversations = _m3.load_conversations
_flatten = _m3._flatten


class _KVSampler:
    """Peak KV blocks used during a round."""

    def __init__(self, block_pool, period_s: float = 0.02):
        self.bp = block_pool
        self.period = period_s
        self.total = int(block_pool.num_gpu_blocks)
        self.min_free = self.total
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.total = int(self.bp.num_gpu_blocks)
        self.min_free = self.total
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                f = self.bp.get_num_free_blocks()
                if f < self.min_free:
                    self.min_free = f
            except Exception:
                pass
            time.sleep(self.period)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)

    @property
    def peak_used(self) -> int:
        return int(self.total - self.min_free)


def build_round_plan(convs, concurrency):
    """Concurrency-`concurrency` schedule over multi-turn sessions with drain
    barriers. Returns a list of rounds; each round is a list of (sid, turn_idx,
    prompt_text) for the sessions currently active in this wave.

    A pool of `concurrency` slots; each slot advances one session through its
    turns; when a session is exhausted a queued one takes the slot. A round is
    'one turn from each currently-active slot', so the engine sees up to
    `concurrency` concurrent requests, and we drain (and resize) between rounds.
    """
    queue = list(range(len(convs)))          # session ids waiting
    slots = [None] * concurrency             # each: [sid, next_turn_idx]
    rounds = []
    while True:
        # fill empty slots
        for i in range(concurrency):
            if slots[i] is None and queue:
                slots[i] = [queue.pop(0), 0]
        active = [(i, s) for i, s in enumerate(slots) if s is not None]
        if not active:
            break
        this_round = []
        for i, s in active:
            sid, ti = s
            prompt = _flatten(convs[sid][ti])
            this_round.append((sid, ti, prompt))
            s[1] += 1
            if s[1] >= len(convs[sid]):       # session done -> free slot
                slots[i] = None
        rounds.append(this_round)
    return rounds


def run_arm(args, arm: str):
    from vllm import LLM, SamplingParams

    from wisp.dynamic import locate_handles
    from wisp.dynamic.controller import MVWSAController
    from wisp.dynamic.kv_resize import current_kv_blocks, kv_page_bytes
    from wisp.integrations.vllm import fused_moe as fm

    convs = load_conversations(args.workflow, args.max_conv, args.max_turns,
                               prefix_tokens=args.prefix_tokens)
    n_turns = sum(len(c) for c in convs)
    rounds = build_round_plan(convs, args.concurrency)
    print(f"[{arm}] {len(convs)} sessions, {n_turns} timed turns, "
          f"{len(rounds)} rounds, conc={args.concurrency}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        kv_cache_memory_bytes=args.kv_bytes,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    engine = llm.llm_engine
    handles = locate_handles(llm)
    page = kv_page_bytes(handles)

    ctrl = None
    if arm == "dynamic":
        ctrl = MVWSAController(
            handles, cap_min=args.cap_min, cap_max=args.cap_max,
            kv_floor_blocks=args.kv_floor_blocks, headroom=args.headroom,
            deadzone_cap=1,
        )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # warmup (not timed): one short request
    _wu = SamplingParams(temperature=0.0, max_tokens=4)
    engine.add_request("warmup", rounds[0][0][2], _wu)
    while engine.has_unfinished_requests():
        engine.step()

    turn_recs = []
    t_start = time.time()
    for ri, rnd in enumerate(rounds):
        first_tok = {}
        finish = {}
        ntok = {}
        meta = {}
        with _KVSampler(handles.block_pool) as smp:
            r0 = time.time()
            for (sid, ti, prompt) in rnd:
                rid = f"r{ri}_s{sid}_t{ti}"
                meta[rid] = (sid, ti)
                engine.add_request(rid, prompt, sp)
            while engine.has_unfinished_requests():
                for out in engine.step():
                    rid = out.request_id
                    if rid not in meta:
                        continue
                    toks = len(out.outputs[0].token_ids) if out.outputs else 0
                    if rid not in first_tok and toks > 0:
                        first_tok[rid] = time.time() - r0
                    if out.finished:
                        finish[rid] = time.time() - r0
                        ntok[rid] = toks
        for rid, (sid, ti) in meta.items():
            wall = finish.get(rid)
            ttft = first_tok.get(rid)
            nt = ntok.get(rid, 0)
            dtps = (nt / (wall - ttft)) if (wall and ttft and wall > ttft and nt) else None
            turn_recs.append({"round": ri, "sid": sid, "turn": ti,
                              "ttft": ttft, "wall": wall, "ntok": nt,
                              "decode_tps": dtps})
        cap_now = fm.current_layer_cap()
        kv_now = current_kv_blocks(handles)
        line = (f"[{arm}] round {ri:2d} n={len(rnd)} "
                f"kv_peak={smp.peak_used:5d}/{kv_now} cap={cap_now}")
        if ctrl is not None:
            d = ctrl.step(smp.peak_used)
            line += (f" -> cap {d.cap_from}->{d.cap_to}, kv {d.kv_from}->{d.kv_to}"
                     f" ({d.reason})")
        print(line, flush=True)
    e2e = time.time() - t_start

    import statistics as st
    ok = [r for r in turn_recs if r["ttft"] is not None and r["wall"] is not None]
    ttfts = sorted(r["ttft"] for r in ok)
    walls = sorted(r["wall"] for r in ok)
    tps = [r["decode_tps"] for r in ok if r["decode_tps"]]

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    summ = {
        "arm": arm,
        "e2e_wall_s": round(e2e, 2),
        "n_turns_ok": len(ok),
        "n_turns_total": len(turn_recs),
        "ttft_med": round(st.median(ttfts), 3) if ttfts else None,
        "ttft_p90": round(pct(ttfts, 0.9), 3) if ttfts else None,
        "turn_wall_med": round(st.median(walls), 3) if walls else None,
        "decode_tps_med": round(st.median(tps), 2) if tps else None,
        "total_tokens": sum(r["ntok"] for r in ok),
        "kv_page_bytes": page,
        "final_cap": fm.current_layer_cap(),
        "final_kv_blocks": current_kv_blocks(handles),
        "turns": turn_recs,
    }
    print(f"\n[{arm}] E2E {summ['e2e_wall_s']}s  TTFTmed={summ['ttft_med']}  "
          f"decode_tps={summ['decode_tps_med']}  cap0={args.init_cap}->"
          f"{summ['final_cap']}  kv->{summ['final_kv_blocks']}blk", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--arm", choices=["fixed", "dynamic"], required=True)
    ap.add_argument("--workflow", default="os")
    ap.add_argument("--max-conv", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--prefix-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--init-cap", type=int, default=36)
    ap.add_argument("--cap-min", type=int, default=16)
    ap.add_argument("--cap-max", type=int, default=48)
    ap.add_argument("--kv-bytes", type=int, default=3_650_000_000)
    ap.add_argument("--kv-floor-blocks", type=int, default=640)
    ap.add_argument("--headroom", type=float, default=0.20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _set_env(args.init_cap)
    res = run_arm(args, args.arm)
    res["args"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[{args.arm}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
