"""WiSP — routing-aware expert paging for memory-constrained MoE serving.

One-liner: *WiSP pages experts like vLLM pages KV.*

This public release is a drop-in plug-in for vLLM (tested against
vLLM 0.11.2) that lets a Mixture-of-Experts model run on a GPU whose
VRAM cannot hold the full expert weights. The resident experts are
treated as a cache; the rest are paged in from pinned host memory on
demand, evicted by an LRU/cost-aware policy under a single shared VRAM
budget that the KV cache also draws from.

Decode output is *byte-identical* to vanilla vLLM at temperature 0 on
the unquantized fused-MoE path: per-token combine is a weighted sum
over each token's routed expert set in routing order, so it is
independent of where an expert physically sits in the scratch buffer.
Use ``scripts/check_byte_identity.py`` to verify this on your box.

Public modules
--------------
- :mod:`wisp.integrations.vllm` — the vLLM plug-in (``install_wisp_moe``).
- :mod:`wisp.oracle.cooccur`     — layer-conditioned routing co-occurrence
                                   used by the (negative) prefetch path.

See ``README.md`` for the quickstart and ``reproduce.sh`` for the
paper's iso-VRAM and byte-identity results on a 24 GiB RTX 3090.
"""

__version__ = "0.0.1"
__all__ = [
    "__version__",
]
