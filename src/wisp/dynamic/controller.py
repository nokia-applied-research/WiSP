"""The live MV-WSA controller: online dual-dynamic expert<->KV split.

The static (arxiv) WiSP solves the equimarginal allocation problem *once*,
offline, and bakes the resulting split ``f`` into the launch flags. The
limitation is obvious: a single ``f`` is the best *compromise* across the
whole trace, but a real agentic session moves between regimes — long-context
turns that are KV-bound, then tool-call bursts that are expert-bound. No
fixed split is optimal for both.

This controller keeps the same equimarginal objective but solves it online,
between drained engine steps, using two physical levers:

* **KV pool** (:func:`wisp.dynamic.kv_resize.resize_kv_blocks`)
* **expert scratch** (``wisp.integrations.vllm.fused_moe.resize_layer_caps``)

Allocation rule. KV's miss curve is, to first order, a *step*: once the GPU
KV pool covers the live working set (plus a small headroom) and the
admission floor ``concurrency x max_model_len``, extra KV blocks have
≈zero marginal value. Every byte beyond KV's need therefore has higher
marginal value as an expert slot (it removes expert page faults). So the
controller sizes KV to ``max(floor, peak_used x (1+headroom))`` and hands
the entire remainder of the fixed byte budget to the experts. As the
workload shifts KV-heavy <-> expert-heavy, the split tracks it. This is the
same κ_min admission floor the paper's configurator uses, now enforced
continuously.

All moves preserve the total GPU byte budget (iso-VRAM), happen only at a
drained barrier, and are wrapped so a failed move is skipped rather than
crashing the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from wisp.dynamic.kv_resize import (
    current_kv_blocks,
    kv_page_bytes,
    resize_kv_blocks,
)
from wisp.integrations.vllm import fused_moe as _fm


@dataclass
class ControlDecision:
    cap_from: int
    cap_to: int
    kv_from: int
    kv_to: int
    kv_peak_used: int
    applied: bool
    reason: str = ""


@dataclass
class MVWSAController:
    handles: object
    cap_min: int = 4
    cap_max: int = 64
    kv_floor_blocks: int = 64
    headroom: float = 0.15
    deadzone_cap: int = 1
    log: list = field(default_factory=list)

    # cached invariants
    _page: int = 0
    _ebpc: int = 0

    def __post_init__(self) -> None:
        self._page = kv_page_bytes(self.handles)
        self._ebpc = _fm.expert_bytes_per_cap()
        if self._ebpc <= 0:
            raise RuntimeError(
                "MVWSAController: no WiSP MoE layers registered; the live "
                "split needs the unquantized WiSP paging path."
            )

    @property
    def budget_bytes(self) -> int:
        """The fixed GPU byte budget shared by KV and experts (iso-VRAM)."""
        cur_kv = current_kv_blocks(self.handles)
        cur_cap = _fm.current_layer_cap() or self.cap_min
        return cur_kv * self._page + cur_cap * self._ebpc

    def plan(self, kv_peak_used_blocks: int) -> ControlDecision:
        """Compute the target split for the just-finished epoch without
        applying it (useful for logging / dry-run)."""
        cur_kv = current_kv_blocks(self.handles)
        cur_cap = _fm.current_layer_cap() or self.cap_min
        budget = cur_kv * self._page + cur_cap * self._ebpc

        # KV sized to the larger of admission floor and observed peak need.
        kv_target = max(
            self.kv_floor_blocks,
            math.ceil(kv_peak_used_blocks * (1.0 + self.headroom)),
        )
        # Never starve experts below cap_min.
        max_kv_by_cap = (budget - self.cap_min * self._ebpc) // self._page
        kv_target = int(max(2, min(kv_target, max_kv_by_cap)))

        # Experts get the entire remainder, clamped to [cap_min, cap_max].
        cap_target = (budget - kv_target * self._page) // self._ebpc
        cap_target = int(max(self.cap_min, min(self.cap_max, cap_target)))
        # Recompute KV from the clamped cap so we never overcommit bytes.
        kv_target = int((budget - cap_target * self._ebpc) // self._page)
        kv_target = max(2, kv_target)

        applied = abs(cap_target - cur_cap) >= self.deadzone_cap
        return ControlDecision(
            cap_from=cur_cap,
            cap_to=cap_target,
            kv_from=cur_kv,
            kv_to=kv_target,
            kv_peak_used=int(kv_peak_used_blocks),
            applied=applied,
        )

    def step(self, kv_peak_used_blocks: int) -> ControlDecision:
        """Plan and (if outside the deadzone) apply the resize. Must be
        called at a drained barrier."""
        d = self.plan(kv_peak_used_blocks)
        if not d.applied:
            d.reason = "within deadzone"
            self.log.append(d)
            return d
        try:
            if d.cap_to > d.cap_from:
                # Experts grow -> free KV first.
                resize_kv_blocks(self.handles, d.kv_to)
                _fm.resize_layer_caps(d.cap_to)
            else:
                # Experts shrink -> free experts first, then grow KV.
                _fm.resize_layer_caps(d.cap_to)
                resize_kv_blocks(self.handles, d.kv_to)
            d.reason = "ok"
        except Exception as exc:  # never crash a run on a control move
            d.applied = False
            d.reason = f"skipped: {type(exc).__name__}: {exc}"
        self.log.append(d)
        return d
