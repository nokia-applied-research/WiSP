# MV-WSA dynamic dual-resize — reproduction run (2026-08-30)

Reproduces the paper's dynamic-vs-fixed table (arXiv:2606.21868 v2, `tab:mvwsa-dynamic`)
with `scripts/m5_dynamic_agentinstruct.py` on a rented RTX 3090 (24 GiB, PCIe 4.0 x16,
pinned H2D ~23-24 GB/s — see probe.json), vLLM 0.11.2, Qwen3-30B-A3B, AgentInstruct
os/db, concurrency 4, 36 timed turns, init cap 32 / KV 2.79 GB, temperature 0.

| workflow | e2e wall (fixed -> dynamic) | TTFT med | decode med | controller move |
|---|---|---|---|---|
| os | 1173 -> 1057 s = **1.11x** | 85.4 -> 74.6 s (1.15x) | 3.37 -> 3.57 t/s | cap 32->34, KV 1773->1197 blk |
| db | 1191 -> 1066 s = **1.12x** | 83.9 -> 73.4 s (1.14x) | 3.62 -> 3.88 t/s | cap 32->34, KV 1773->1197 blk |

Paper reported 1.07x (os) / 1.19x (db) on its original 3090 host; this independent
rerun lands both workflows at ~1.11-1.12x — same mechanism (one decisive
reclaim-idle-KV resize at round 0, then deadzone-stable), same direction, magnitude
within the paper's band. All 36/36 turns completed in every arm.

Repro: `python scripts/m5_dynamic_agentinstruct.py --arm {fixed,dynamic} --workflow {os,db} --init-cap 32 --kv-bytes 2790000000 --out <json>`
