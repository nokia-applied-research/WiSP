"""Phase B live bench: MV-WSA iso-VRAM expert<->KV split on real vLLM serve.

For each arm (mvwsa / fifty / flux) we launch ``vllm serve`` with the WiSP
plugin at a FIXED total GPU budget, differing only in the expert<->KV split
(cap_experts + --kv-cache-memory-bytes from mvwsa_configure). All arms run with
identical native KV offload, so the only variable is where the fixed GPU bytes
go. We then replay a workflow's multi-turn agent conversations concurrently
(personal-device multitasking) and measure, on the wire:

  - TTFT (prefill latency)        -> sensitive to the KV pool / offload
  - decode throughput (tok/s)     -> sensitive to expert paging (PCIe)
  - end-to-end turn latency

Claim: at iso-VRAM, MV-WSA's split gives the best end-to-end agentic latency;
expert-heavy (flux) and 50/50 pay more on one side or the other.

  python3 m3_mvwsa_live_bench.py --model qwen3 --workflow os \
      --total-gpu-gib 16 --gpu-idx 0 --concurrency 4 --max-conv 8 \
      --from-results results/mvwsa/v2/qwen3_os_cs16.json \
      --out results/mvwsa/live/qwen3_os.json
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, threading, time
import concurrent.futures as cf
import importlib.util
import urllib.request, urllib.error

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "mvwsa_configure", os.path.join(_here, "mvwsa_configure.py"))
conf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(conf)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def wait_ready(base_url, proc, log_path, timeout=900):
    url = f"{base_url}/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = open(log_path).read()[-2000:] if os.path.exists(log_path) else ""
            raise RuntimeError(f"vllm exited rc={proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError("vllm not ready")


_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
          "lima mike november oscar papa quebec romeo sierra tango uniform victor "
          "whiskey xray yankee zulu zero one two three four five six seven eight "
          "nine ten config socket buffer kernel thread mutex packet daemon").split()


def _make_prefix(session_idx: int, n_tokens: int) -> str:
    """A DISTINCT long pseudo-document per session (simulates a retrieved doc /
    large tool output). Distinct content prevents cross-session prefix-cache
    dedup, so each concurrent session genuinely consumes KV -- reproducing the
    offline context-scale regime where KV competes with experts. ~0.75 tok/word,
    so we emit ~1.4*n_tokens words."""
    import random
    rng = random.Random(1000 + session_idx)
    nwords = int(n_tokens * 1.4)
    body = " ".join(rng.choice(_WORDS) for _ in range(nwords))
    return (f"[Session {session_idx} reference material — use only if relevant]\n"
            f"{body}\n[end of reference material]")


def load_conversations(workflow, max_conv, max_turns, prefix_tokens=0):
    """Return list of conversations; each is a list of {role,content} messages
    ending at each assistant turn we will time. Optionally prepend a distinct
    long context per session to create KV pressure."""
    from datasets import load_dataset
    d = load_dataset("THUDM/AgentInstruct", split=workflow)
    convs = []
    for si, row in enumerate(list(d)[:max_conv]):
        msgs, timed = [], []
        if prefix_tokens > 0:
            msgs.append({"role": "user", "content": _make_prefix(si, prefix_tokens)})
            msgs.append({"role": "assistant", "content": "Understood. Ready."})
        for t in row["conversations"]:
            role = t.get("from") or t.get("role")
            content = str(t.get("value") or t.get("content"))[:600]  # bound turn len
            r = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, "user")
            if r == "assistant":
                # record the prefix (everything before this assistant turn)
                timed.append(list(msgs))
            msgs.append({"role": r, "content": content})
        if timed:
            convs.append(timed[:max_turns])
    return convs


def _flatten(messages):
    """Flatten chat messages into a single prompt. We use /v1/completions
    (not /v1/chat) so the harness works on base models without a chat template
    (e.g. OLMoE) AND on chat models -- identical tokenization across arms."""
    parts = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}")
    parts.append("assistant:")
    return "\n".join(parts)


def stream_turn(client, model_id, messages, max_tokens):
    prompt = _flatten(messages)
    t0 = time.perf_counter(); ttft = None; ntok = 0
    try:
        stream = client.completions.create(
            model=model_id, prompt=prompt, max_tokens=max_tokens,
            temperature=0.0, stream=True,
            stream_options={"include_usage": True})
        for chunk in stream:
            if chunk.choices:
                piece = (chunk.choices[0].text or "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
            if getattr(chunk, "usage", None):
                ntok = int(getattr(chunk.usage, "completion_tokens", 0) or 0)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "wall": time.perf_counter() - t0, "ttft": ttft, "ntok": ntok}
    wall = time.perf_counter() - t0
    decode_tps = (ntok / (wall - ttft)) if (ttft and wall > ttft and ntok) else None
    return {"wall": wall, "ttft": ttft, "ntok": ntok, "decode_tps": decode_tps}


def _run_session_inproc(base_url, model_id, max_tokens, timed_prefixes):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)
    return [stream_turn(client, model_id, messages, max_tokens)
            for messages in timed_prefixes]


def run_client(base_url, model_id, convs, concurrency, max_tokens):
    """Launch one INDEPENDENT OS subprocess per session, up to `concurrency`
    in flight. Threads serialize on the GIL during openai's SSE loop and
    process pools deadlock/hang under fork-after-threads / spawn; independent
    subprocesses give true concurrency (verified: matches N parallel curls).
    Each worker prints a JSON list of per-turn results to stdout.
    """
    import json as _json, tempfile
    self_path = os.path.abspath(__file__)
    results = []
    pending = list(enumerate(convs))
    running = {}  # proc -> (taskfile, outfile)

    def launch(idx, conv):
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        _json.dump({"base_url": base_url, "model_id": model_id,
                    "max_tokens": max_tokens, "timed_prefixes": conv}, tf)
        tf.close()
        of = tf.name + ".out"
        ef = tf.name + ".err"
        p = subprocess.Popen([sys.executable, self_path, "--worker-task", tf.name,
                              "--worker-out", of], stdout=subprocess.DEVNULL,
                             stderr=open(ef, "w"))
        running[p] = (tf.name, of, ef)

    while pending or running:
        while pending and len(running) < concurrency:
            idx, conv = pending.pop(0)
            launch(idx, conv)
        time.sleep(0.3)
        for p in list(running):
            if p.poll() is not None:
                tfn, ofn, efn = running.pop(p)
                try:
                    results.extend(_json.load(open(ofn)))
                except Exception:
                    err = ""
                    try:
                        err = open(efn).read()[-500:]
                    except Exception:
                        pass
                    print(f"[worker rc={p.returncode}] {err}", flush=True)
                for fn in (tfn, ofn, efn):
                    try:
                        os.unlink(fn)
                    except Exception:
                        pass
    return results


def scrape_server_stats(log_path):
    """Pull KV-pressure evidence from the vLLM engine log."""
    import re
    kv_use, ext_hit, preempt = [], [], 0
    try:
        for line in open(log_path):
            m = re.search(r"GPU KV cache usage: ([\d.]+)%", line)
            if m:
                kv_use.append(float(m.group(1)))
            m = re.search(r"External prefix cache hit rate: ([\d.]+)%", line)
            if m:
                ext_hit.append(float(m.group(1)))
            if "Preempt" in line or "preempt" in line:
                preempt += 1
    except FileNotFoundError:
        pass
    return {
        "kv_usage_peak_pct": max(kv_use) if kv_use else None,
        "ext_prefix_hit_max_pct": max(ext_hit) if ext_hit else None,
        "preempt_log_lines": preempt,
    }


def summarize(results):
    ok = [r for r in results if "error" not in r and r.get("ttft")]
    if not ok:
        return {"n": 0, "errors": len(results)}
    import statistics as st
    ttfts = sorted(r["ttft"] for r in ok)
    walls = sorted(r["wall"] for r in ok)
    tps = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    def pct(xs, p): return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "n_turns": len(ok), "errors": len(results) - len(ok),
        "ttft_med": st.median(ttfts), "ttft_p90": pct(ttfts, 0.9),
        "turn_wall_med": st.median(walls), "turn_wall_p90": pct(walls, 0.9),
        "decode_tps_med": (st.median(tps) if tps else None),
        "total_tokens": sum(r["ntok"] for r in ok),
    }


def _kill_group(proc):
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(5)  # let GPU memory free before next arm


def run_arm(args, arm, convs):
    is_naive = arm in conf.NAIVE_VARIANTS
    port = free_port()
    if is_naive:
        cfg = conf.naive_config(args.model, args.total_gpu_gib, args.card_total_gib,
                                arm, overhead_gib=args.overhead_gib)
        flags, env_extra = conf.naive_serve_flags(
            cfg, kv_offload_gib=args.kv_offload_gib, max_model_len=args.max_model_len,
            port=port, gpu_idx=args.gpu_idx)
        desc = (f"cpu_offload={cfg['cpu_offload_gb']}GiB swap={cfg['swap']} "
                f"resident_w={cfg['resident_weights_gib']}GiB")
    else:
        mvwsa_f = json.load(open(args.from_results)).get(
            "converged_split", 0.5) if args.from_results else args.split
        f = conf.arm_split(args.model, arm, mvwsa_f)
        adm = conf.admission_tokens(arm, args.concurrency, args.max_model_len)
        cfg = conf.split_to_config(args.model, args.total_gpu_gib, args.card_total_gib,
                                   f, overhead_gib=args.overhead_gib,
                                   kv_admission_tokens=adm)
        cfg["arm"] = arm
        flags, env_extra = conf.serve_flags(
            cfg, kv_offload_gib=args.kv_offload_gib, max_model_len=args.max_model_len,
            port=port, gpu_idx=args.gpu_idx)
        desc = f"f={f:.3f} cap_experts={cfg['cap_experts']} KV={cfg['kv_cache_gib']:.2f}GiB"

    base_url = f"http://127.0.0.1:{port}/v1"
    log_path = f"/tmp/vllm_mvwsa_{arm}_{port}.log"
    env = os.environ.copy(); env.update(env_extra)
    if not is_naive and args.vanilla_experts:
        env["WISP_PLUGIN_DISABLE"] = "1"   # ablation: no expert paging

    print(f"\n[arm {arm}] {desc} -> launching vllm (log {log_path})", flush=True)
    proc = subprocess.Popen(flags, env=env, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    t_launch = time.time()
    try:
        wait_ready(base_url, proc, log_path, timeout=args.ready_timeout)
    except (RuntimeError, TimeoutError) as e:
        # launch failure (e.g. naive_kv: model does not fit the budget) -- record
        # as a measured capability failure rather than aborting the whole run.
        _kill_group(proc)
        print(f"[arm {arm}] FAILED to start: {str(e)[:140]}", flush=True)
        return {"arm": arm, "config": cfg, "failed": True,
                "error": str(e)[-500:], "server_stats": scrape_server_stats(log_path)}

    print(f"[arm {arm}] ready in {time.time()-t_launch:.0f}s; "
          f"running client (conc={args.concurrency})", flush=True)
    try:
        run_client(base_url, cfg["model_id"], convs[:1], 1, 8)  # warmup
        t0 = time.time()
        results = run_client(base_url, cfg["model_id"], convs,
                             args.concurrency, args.max_tokens)
        wall = time.time() - t0
    finally:
        _kill_group(proc)
    summ = summarize(results)
    summ.update({"arm": arm, "config": cfg, "bench_wall_s": wall,
                 "server_stats": scrape_server_stats(log_path)})
    ss = summ.get("server_stats", {})
    print(f"[arm {arm}] n_turns={summ.get('n_turns')} errors={summ.get('errors')}  "
          f"TTFT med={summ.get('ttft_med')} p90={summ.get('ttft_p90')}  "
          f"decode_tps={summ.get('decode_tps_med')}  wall={wall:.0f}s  "
          f"KVpeak={ss.get('kv_usage_peak_pct')}% extHit={ss.get('ext_prefix_hit_max_pct')}%",
          flush=True)
    return summ


def worker_main(task_path, out_path):
    """One session, run as an independent process (see run_client)."""
    t = json.load(open(task_path))
    res = _run_session_inproc(t["base_url"], t["model_id"],
                              t["max_tokens"], t["timed_prefixes"])
    json.dump(res, open(out_path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-task", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=list(conf.MODELS), default="qwen3")
    ap.add_argument("--workflow", default="os")
    ap.add_argument("--total-gpu-gib", type=float, default=16.0,
                    help="simulated device GPU budget")
    ap.add_argument("--card-total-gib", type=float, default=94.0)
    ap.add_argument("--overhead-gib", type=float, default=2.5)
    ap.add_argument("--gpu-idx", type=int, default=0)
    ap.add_argument("--arms", default="mvwsa,fifty,flux")
    ap.add_argument("--from-results", default=None)
    ap.add_argument("--split", type=float, default=0.5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-conv", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prefix-tokens", type=int, default=0,
                    help="distinct long context per session (KV pressure)")
    ap.add_argument("--kv-offload-gib", type=float, default=40.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--vanilla-experts", action="store_true",
                    help="disable WiSP expert paging (KV-split ablation only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.worker_task:
        worker_main(args.worker_task, args.worker_out)
        return

    convs = load_conversations(args.workflow, args.max_conv, args.max_turns,
                               prefix_tokens=args.prefix_tokens)
    print(f"[live] {args.model}/{args.workflow}: {len(convs)} sessions, "
          f"{sum(len(c) for c in convs)} timed turns, conc={args.concurrency}, "
          f"prefix_tokens={args.prefix_tokens}", flush=True)

    arms = args.arms.split(",")
    out = {"model": args.model, "workflow": args.workflow,
           "total_gpu_gib": args.total_gpu_gib, "concurrency": args.concurrency,
           "arms": {}}
    for arm in arms:
        out["arms"][arm] = run_arm(args, arm, convs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[live] wrote {args.out}")
    # quick comparison
    base = out["arms"].get("fifty", {}).get("turn_wall_med")
    for arm, s in out["arms"].items():
        tw = s.get("turn_wall_med")
        rel = f"{100*(tw-base)/base:+.1f}% vs 50/50" if base and tw else ""
        print(f"  {arm:6s} TTFT_med={s.get('ttft_med')}  "
              f"turn_wall_med={tw}  decode_tps={s.get('decode_tps_med')}  {rel}")


if __name__ == "__main__":
    main()
