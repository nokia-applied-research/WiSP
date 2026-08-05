"""WiSP integration shim for vLLM.

Usage::

    from wisp.integrations.vllm import install_wisp_moe
    install_wisp_moe()  # MUST be called before vLLM imports the model

    from vllm import LLM
    llm = LLM(model="Qwen/Qwen3-30B-A3B", enforce_eager=True)

``install_wisp_moe`` patches vLLM's unquantized fused-MoE method
(:class:`UnquantizedFusedMoEMethod`) so that:

1. Master expert weights are allocated in pinned **host** memory at
   model-construction time, not on the GPU. (Upstream allocates the
   full ``(num_experts, ...)`` tensors directly on-device, which OOMs
   on a 24 GiB card before anything can be moved off.)
2. On each MoE forward, only the experts in the current working set are
   staged into a small GPU scratch buffer (``WISP_CAP_EXPERTS`` slots)
   via ``cudaMemcpyAsync`` from the pinned host copy; the upstream
   kernel runs unchanged.

Modes (``WISP_MODE``, default ``paged``):

- ``paged``    — LRU paging: scratch holds ``cap_experts`` slots, experts
  are paged in/out on demand. This is the memory-saving regime.
- ``resident`` — full residency in a permanent GPU scratch (no paging);
  useful as an upper-bound / correctness control when VRAM allows.
- ``copy``     — naive full-tensor copy every step (slow; correctness only).

(Legacy aliases ``day2b`` / ``day2a`` / ``day1`` are still accepted.)

Resident output is byte-identical to vanilla vLLM at temperature 0 on
this path (see ``scripts/check_byte_identity.py``).
"""

from .fused_moe import install_wisp_moe, is_installed

__all__ = ["install_wisp_moe", "is_installed"]
