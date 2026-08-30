# E1 — cross-engine iso-VRAM baselines (paper experiment 1)

Head-to-head decode/TTFT comparison on one memory-constrained GPU for a model
that does not fit: WiSP (vLLM plug-in) vs vanilla vLLM `--cpu-offload-gb` vs
**llama.cpp** (layer-split, `-ngl`) vs **KTransformers** (GPU attention + CPU
expert kernels). Default subject: Qwen3-30B-A3B on a 24 GiB RTX 3090.

## Design

One shared driver, `e1_endpoint_bench.py`, benchmarks any OpenAI-compatible
`/v1/completions` endpoint with streaming: TTFT = time to first streamed token
(measured, not approximated), decode = (ntok−1)/(t_last−t_first), peak VRAM
polled via nvidia-smi. Output JSON uses the m3_5 schema (`config` + `derived`)
so `scripts/m3_fig1_plot.py` renders the cross-engine Pareto unchanged.

Per-engine launchers own install/model/serve/sweep:

- `e1_llamacpp.sh` — pins a llama.cpp tag, builds with CUDA, downloads the
  GGUF (F16 preferred for precision parity, `Q8_0` fallback — record which!),
  sweeps `-ngl` to hit VRAM budgets. Points that fail to start ARE data
  (the capability floor), same as the m3 sweep convention.
- `e1_ktransformers.sh` — separate venv (their torch pin), pinned version,
  serve command overridable via `KT_CMD` because their CLI churns.
- WiSP / vanilla-vLLM rows come from the existing `m3_3090_isovram_sweep.sh`;
  rerun with `--max-tokens 256` matching this driver, or re-derive from the
  same protocol for the paper table.

## Status / smoke checklist (September)

Nothing here is smoke-tested yet. On the first pod day:
1. `bash e1_llamacpp.sh` with `NGL_SWEEP="16"` and a small `--max-tokens`
   (edit driver call) → confirm build, GGUF pick (check multi-part shards),
   health endpoint, JSON row.
2. `bash e1_ktransformers.sh` → expect to iterate on `KT_VERSION`/`KT_CMD`;
   freeze what works into this script.
3. Decide the precision story for the paper: llama.cpp F16 GGUF (~57 GB) keeps
   weights bit-comparable to BF16 safetensors in spirit but not identical;
   KTransformers' CPU path typically wants quantized GGUF experts. Whatever is
   chosen, the `config` JSON records it per row — the paper table footnotes it.
4. Full protocol (all prompts, max_tokens 256, full sweeps) is the sprint run,
   not the smoke run.

## Fairness rules

- Same prompts, temp 0, same max_tokens, single stream, warmup 1 across all engines.
- Record everything that differs (precision, threads, context) in `config`.
- Iso-VRAM pairing by measured `peak_vram_gib` (the plot script's ±4 GiB rule),
  never by nominal knob values.
- Every engine gets its best honest configuration: llama.cpp with full thread
  count, KTransformers with `--cpu_infer` = physical cores. If an engine can't
  serve the model at a budget, that row is reported as ✗ (capability), not dropped.
