"""WiSP MLSys "super-final" variant: live MV-WSA dual-dynamic split.

This subpackage is the *only* code that is new relative to the static
(arxiv) WiSP. It implements the two physical resize primitives the live
MV-WSA controller needs — usable both from a single-process offline
``vllm.LLM`` (the PoC harnesses) *and* from inside the multi-process
``vllm serve`` loop (:mod:`wisp.dynamic.serve_hook`, which drives the
controller from the EngineCore subprocess busy loop; MLSys gap C3):

* **expert scratch resize** — ``wisp.integrations.vllm.fused_moe.resize_layer_caps``
  (defined next to the static paging code, behind a no-op guard);
* **KV pool resize** — :mod:`wisp.dynamic.kv_resize`, which reallocates the
  worker's per-layer KV tensors and rebuilds the scheduler block pool at a
  drained barrier.

:class:`wisp.dynamic.controller.MVWSAController` ties them together: it
reads the expert miss signal and KV utilisation between engine steps and
moves GPU bytes across the expert<->KV boundary to equalise marginal value,
exactly the equimarginal rule of the paper, but now realised *online*
instead of only as a startup configuration.

Everything here is gated behind ``WISP_DYNAMIC_SPLIT`` / explicit calls, so
importing this package never changes static behaviour.
"""

from wisp.dynamic.engine_access import EngineHandles, locate_handles
from wisp.dynamic.serve_hook import (
    install_wisp_serve_controller,
    is_serve_controller_installed,
)

__all__ = [
    "EngineHandles",
    "locate_handles",
    "install_wisp_serve_controller",
    "is_serve_controller_installed",
]
