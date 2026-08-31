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

## Status after smoke day 1 (2026-08-30, RTX 3090 pod)

- **llama.cpp: FULL PIPELINE PASS.** Build (CUDA), Q8_0 GGUF download, `llama-server`
  at ngl=16, streamed bench, JSON row: 1.42 tok/s mean @ 11.2 GiB peak, TTFT p50 1.75 s.
  Sweep the full NGL grid in the sprint.
- **KTransformers: parked with findings** (five layers deep, each documented so the
  sprint doesn't re-derive them):
  1. PyPI has no `ktransformers` 0.3.x; versions jump 0.2.1 → 0.5.2 → … → 0.7.0.post3.
  2. The 0.7 meta package has no `ktransformers.server.main`; the 2026 architecture is
     **kt-kernel (CPU ops) + sglang-kt (forked SGLang)**, launched via
     `python -m sglang.launch_server --kt-*` (this script now encodes that).
  3. `sglang-kt` does NOT depend on `sgl-kernel`, but imports it unconditionally
     (`ModuleNotFoundError: common_ops` when absent).
  4. Installing PyPI `sgl-kernel` (0.3.21, 0.3.16, 0.3.9, 0.3.4, 0.2.9 all probed) fails
     against the venv's torch 2.9.1 with
     `undefined symbol: _ZNK3c106SymInt6sym_neERKS0_` — **torch ABI mismatch**, not a
     missing SM86 variant (the sm100 dir naming is a red herring; the .so exists and
     fails to load).
  5. Next moves for the sprint: install sgl-kernel from kvcache-ai's own wheel index /
     match torch to sgl-kernel's build matrix (their pip extra likely pins torch), or run
     the baseline from their official Docker image instead of a venv.
- **Ops note:** the pod image presets `HF_HOME=/workspace/.cache/huggingface`, which
  silently duplicated 74 GB of model downloads next to `/workspace/hf`. Fixed on the pod
  with `rm + ln -s`; keep the two paths unified on any new box.

## Remaining smoke checklist
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
