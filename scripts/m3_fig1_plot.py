"""Plot Figure 1 — iso-VRAM decode-throughput Pareto: WiSP plug-in vs vanilla vLLM.

Reads the JSONs emitted by m3_5_plugin_day5_bench.py (one per config) from a
directory and produces:
  (a) a Pareto scatter/line plot: peak VRAM (x) vs decode tok/s (y), two series.
  (b) a markdown table pairing configs by matched VRAM budget, with speedups.

Usage:
  python scripts/m3_fig1_plot.py --in-dir results/m3/fig1_isovram_pull \
      --out results/m3/fig1_isovram
"""
from __future__ import annotations
import argparse
import glob
import json
import os


def load_rows(in_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        der = d.get("derived", {})
        cfg = d.get("config", {})
        rows.append({
            "tag": os.path.basename(f).replace(".json", ""),
            "backend": cfg.get("backend"),
            "cap": cfg.get("cap_experts"),
            "offload": cfg.get("cpu_offload_gb"),
            "tok_s": der.get("decode_tok_per_s_per_stream_mean"),
            "ttft": der.get("ttft_p50_seconds"),
            "peak_vram": der.get("peak_vram_gib"),
        })
    return [r for r in rows if r["tok_s"] is not None and r["peak_vram"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (no ext)")
    ap.add_argument("--title", default="iso-VRAM decode Pareto (eager)",
                    help="plot title, e.g. 'Qwen3-30B-A3B (RTX 3090, eager)'")
    args = ap.parse_args()

    rows = load_rows(args.in_dir)
    wisp = sorted([r for r in rows if r["backend"] == "wisp_plugin"], key=lambda r: r["peak_vram"])
    vllm = sorted([r for r in rows if r["backend"] == "vanilla"], key=lambda r: r["peak_vram"])

    # markdown table: pair by nearest peak VRAM
    lines = ["| peak VRAM (GiB) | WiSP tok/s | vanilla tok/s | WiSP speedup |",
             "|---|---|---|---|"]
    for w in wisp:
        # nearest vanilla by peak vram (within 4 GiB)
        cand = [v for v in vllm if abs(v["peak_vram"] - w["peak_vram"]) < 4.0]
        if cand:
            v = min(cand, key=lambda v: abs(v["peak_vram"] - w["peak_vram"]))
            sp = w["tok_s"] / v["tok_s"] if v["tok_s"] else float("nan")
            lines.append(f"| ~{w['peak_vram']:.0f} | {w['tok_s']:.2f} (cap{w['cap']}) | "
                         f"{v['tok_s']:.2f} (off{int(v['offload'])}) | {sp:.2f}x |")
        else:
            lines.append(f"| ~{w['peak_vram']:.0f} | {w['tok_s']:.2f} (cap{w['cap']}) | — | (no vanilla pt) |")
    table = "\n".join(lines)
    print(table)
    with open(args.out + "_table.md", "w") as fp:
        fp.write(table + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); wrote table only")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["peak_vram"] for r in wisp], [r["tok_s"] for r in wisp],
            "o-", color="#1f77b4", label="WiSP plug-in (paged experts)", linewidth=2, markersize=7)
    ax.plot([r["peak_vram"] for r in vllm], [r["tok_s"] for r in vllm],
            "s--", color="#d62728", label="vanilla vLLM (--cpu-offload-gb)", linewidth=2, markersize=7)
    for r in wisp:
        ax.annotate(f"cap{r['cap']}", (r["peak_vram"], r["tok_s"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=8, color="#1f77b4")
    for r in vllm:
        ax.annotate(f"off{int(r['offload'])}", (r["peak_vram"], r["tok_s"]),
                    textcoords="offset points", xytext=(5, -12), fontsize=8, color="#d62728")
    ax.set_xlabel("Peak GPU memory (GiB)")
    ax.set_ylabel("Decode throughput (tok/s, per-stream)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    fig.savefig(args.out + ".pdf")
    print(f"[plot] wrote {args.out}.png / .pdf / _table.md")


if __name__ == "__main__":
    raise SystemExit(main())
