"""WiSP-vLLM fused-MoE method patch: host-pinned expert weights + paging.

This module patches
:class:`vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod`
to back its MoE weights with a *pinned-CPU master copy* and a
*GPU scratch buffer* of configurable size.

Modes
-----

``WISP_MODE`` (env var, read once at ``install_wisp_moe()``):

- ``copy`` — staged full copy per forward (kept for regression).
  Throughput is terrible; only useful for proving the weight-storage
  interception itself. (Legacy alias: ``day1``.)
- ``resident`` — full residency in a permanent GPU scratch
  of shape ``(num_experts, *, *)``. expert_map is identity. The
  CPU pinning is purely a backup; perf and output should match
  vanilla byte-for-byte. (Legacy alias: ``day2a``.)
- ``paged`` (default) — partial residency: scratch holds ``cap_experts``
  slots, routed experts are staged from pinned CPU on demand,
  ``expert_map`` redirects ``topk_ids`` from the global space to
  scratch slots. Output should still byte-match vanilla because
  the math is unchanged (only the physical storage moves).
  (Legacy alias: ``day2b``.)

Why monkey-patch instead of subclass + registry override?

  vLLM picks the ``quant_method`` for each :class:`FusedMoE` layer
  via a dispatch keyed on the model's quant config (see
  ``FusedMoE.__init__`` → ``self.quant_method = quant_config.get_…``).
  For unquantised models the dispatch returns
  ``UnquantizedFusedMoEMethod``. Replacing it via subclass would
  require either intercepting that dispatch (private, varies by
  quant flavour) or registering a custom
  ``CustomOp.register("unquantized_fused_moe")`` *after* vLLM has
  registered its upstream version — order-dependent and fragile.
  A class-level monkey-patch is robust to vLLM's internal dispatch
  and easy to undo (we keep the originals).

Caveats
-------

- Single-process or fork-inherited multiprocess only. The
  ``vllm.general_plugins`` entry point path will be wired in
  Day 4 so multi-worker spawn works.
- Only the unquantised BF16 path is patched. FP8 / AWQ / GPTQ MoE
  methods are untouched.
- ``forward_native`` / ``forward_cpu`` are untouched — H100 uses
  ``forward_cuda``.
- ``rocm_aiter_moe`` and ``flashinfer_cutlass_moe`` paths in
  upstream ``forward_cuda`` are *not* patched. They're only hit on
  ROCm or with ``VLLM_USE_FLASHINFER_MOE_FP16=1``; the H100
  sandbox always lands in the plain ``fused_experts`` branch.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger("wisp.integrations.vllm.fused_moe")


_PATCHED_FLAG = "_wisp_patched"


# ---------------------------------------------------------------------------
# Mode handling
# ---------------------------------------------------------------------------


class _Mode:
    COPY = "copy"          # full-tensor copy per forward (correctness only)
    RESIDENT = "resident"  # full residency in a permanent GPU scratch (no paging)
    PAGED = "paged"        # LRU paging of a small scratch (default; memory-saving)


# Legacy milestone aliases accepted for backward compatibility.
_LEGACY_MODE_ALIASES = {"day1": _Mode.COPY, "day2a": _Mode.RESIDENT, "day2b": _Mode.PAGED}


def _resolve_mode() -> str:
    raw = os.environ.get("WISP_MODE", _Mode.PAGED).strip().lower()
    raw = _LEGACY_MODE_ALIASES.get(raw, raw)
    if raw not in (_Mode.COPY, _Mode.RESIDENT, _Mode.PAGED):
        logger.warning("WISP_MODE=%r unrecognised; defaulting to paged", raw)
        return _Mode.PAGED
    return raw


def _resolve_cap_experts(num_experts: int) -> int:
    """paged-mode only. Read ``WISP_CAP_EXPERTS`` and clamp to
    ``[top_k_estimated, num_experts]``."""
    raw = os.environ.get("WISP_CAP_EXPERTS")
    if raw is None:
        return num_experts
    try:
        cap = int(raw)
    except ValueError:
        logger.warning(
            "WISP_CAP_EXPERTS=%r is not an int; using num_experts=%d", raw, num_experts
        )
        return num_experts
    # Safety floor: at minimum we must hold the union of routed experts
    # for a single token (top_k). top_k is unknown at this init stage, so
    # we use a small floor (2) that admits low-expert-count MoEs such as
    # Mixtral/Jamba (8 experts, top_k=2); if the caller picks cap < top_k
    # for a given model, ensure_resident raises a clear error at the first
    # step rather than producing wrong output (P1 is unaffected). The
    # caller is responsible for picking cap >= top_k.
    cap = max(2, min(cap, num_experts))
    return cap


# ---------------------------------------------------------------------------
# Per-layer state
# ---------------------------------------------------------------------------


class WispMoEState:
    """Per-layer paging state attached as ``layer._wisp_state``.

    Holds the pinned-CPU master copies, the GPU scratch buffers,
    and the slot ↔ expert mapping. All fields are mutated in place
    on each forward (paged-mode); resident-mode treats the state as immutable
    after init.
    """

    __slots__ = (
        "layer_idx",
        "num_experts",
        "cap_experts",
        "mode",
        "cpu_w13",
        "cpu_w2",
        "scratch_w13",
        "scratch_w2",
        "slot_to_expert",
        "expert_to_slot",
        "expert_map_device",
        "lru_tick",
        "lru_clock",
        "stats_forward",
        "stats_miss",
        "stats_evict",
        "stats_hits",
        "stats_pred_hits",
        "stats_pred_total",
        "copy_stream",
        "copy_event",
        "last_topk_ids_cpu",
        "last_unique_experts",
    )

    def __init__(
        self,
        cpu_w13: torch.Tensor,
        cpu_w2: torch.Tensor,
        cap_experts: int,
        mode: str,
        device: torch.device,
        layer_idx: int,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.num_experts = int(cpu_w13.shape[0])
        self.cap_experts = int(cap_experts)
        self.mode = mode
        self.cpu_w13 = cpu_w13
        self.cpu_w2 = cpu_w2

        # Slot-shaped GPU scratch buffers. For resident-mode (cap = num_experts),
        # these match the upstream weight shape exactly.
        self.scratch_w13 = torch.empty(
            (cap_experts, cpu_w13.shape[1], cpu_w13.shape[2]),
            dtype=cpu_w13.dtype,
            device=device,
        )
        self.scratch_w2 = torch.empty(
            (cap_experts, cpu_w2.shape[1], cpu_w2.shape[2]),
            dtype=cpu_w2.dtype,
            device=device,
        )

        # Mapping bookkeeping (host-side).
        # slot_to_expert[s] = expert id currently in slot s, or -1 if free
        # expert_to_slot[e] = slot id for expert e, or absent if not resident
        self.slot_to_expert: list[int] = [-1] * cap_experts
        self.expert_to_slot: dict[int, int] = {}

        # Device-side expert_map buffer: shape [num_experts], int32, -1
        # means "not resident this step". Allocated once, mutated in
        # place every forward in paged-mode.
        self.expert_map_device = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.int32,
            device=device,
        )

        # LRU clock — incremented on every touch. lru_tick[slot] = clock
        # at last touch; eviction picks the smallest tick.
        self.lru_tick = [0] * cap_experts
        self.lru_clock = 0

        # Light telemetry.
        self.stats_forward = 0
        self.stats_miss = 0
        self.stats_evict = 0
        self.stats_hits = 0
        # Day 4: cooccur prediction quality.
        # stats_pred_total = total expert lookups attempted via predict
        # stats_pred_hits  = of those, how many appeared in the
        #                   observed-needed set on the next forward.
        self.stats_pred_total = 0
        self.stats_pred_hits = 0

        # Side stream for H2D copies; lets ensure_resident overlap with
        # whatever the default (compute) stream is running. Re-uses
        # PyTorch's default allocator on the same device. We create the
        # stream lazily on first use to avoid touching CUDA when this
        # module is imported on a non-CUDA box (the smoke runs in a
        # fork-inherited worker; create here is safe).
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copy_event = torch.cuda.Event(enable_timing=False, blocking=False)

        # Day 4 layer-level history predictor: remember last step's
        # observed topk so the next step can pre-load while the
        # previous layer is still running.
        self.last_topk_ids_cpu: torch.Tensor | None = None
        self.last_unique_experts: set[int] = set()

    # --------------------------------------------------------------
    # resident-mode path: prime with identity
    # --------------------------------------------------------------

    def prime_identity(self) -> None:
        """resident-mode: copy all experts from CPU pin into scratch in
        identity order (slot i ↔ expert i). Set the expert_map to
        the identity permutation."""
        assert self.cap_experts == self.num_experts, (
            "prime_identity requires cap == num_experts"
        )
        # Async H2D — torch.cuda will synchronise implicitly when the
        # caller's next op (the kernel) runs on the default stream.
        self.scratch_w13.copy_(self.cpu_w13, non_blocking=True)
        self.scratch_w2.copy_(self.cpu_w2, non_blocking=True)
        for i in range(self.num_experts):
            self.slot_to_expert[i] = i
            self.expert_to_slot[i] = i
        # Identity expert_map.
        idx = torch.arange(self.num_experts, dtype=torch.int32, device=self.expert_map_device.device)
        self.expert_map_device.copy_(idx)

    # --------------------------------------------------------------
    # paged-mode path: dynamic admission via LRU
    # --------------------------------------------------------------

    def ensure_resident(self, needed_experts: torch.Tensor) -> None:
        """Ensure all unique experts in ``needed_experts`` are in
        scratch. Copies are issued on the **compute** stream so they
        serialise with the kernel that follows; there is no real
        overlap opportunity at this point because the caller has
        already forced a host sync to read ``topk_ids``. The side
        stream + speculative prefetch path (``prefetch_async`` +
        cooccur predictor) is where Day-4 overlap actually happens.

        ``needed_experts`` may contain duplicates; we ``.unique()``
        internally.
        """
        # Move to host to drive the bookkeeping; the unique() result is
        # very small (≤ top_k × num_tokens unique entries; in practice
        # ≤ num_experts).
        needed = torch.unique(needed_experts).to(device="cpu", dtype=torch.long).tolist()
        needed_set = set(int(e) for e in needed)

        # Identify which needed experts are missing.
        missing = [e for e in needed_set if e not in self.expert_to_slot]
        self.stats_hits += len(needed_set) - len(missing)
        if not missing:
            # All needed are resident; just bump LRU and (optionally)
            # absorb any in-flight side-stream prefetches into the
            # compute stream.
            self.lru_clock += 1
            for e in needed_set:
                self.lru_tick[self.expert_to_slot[e]] = self.lru_clock
            self.copy_event.record(self.copy_stream)
            torch.cuda.current_stream().wait_event(self.copy_event)
            return

        # We need slots for ``len(missing)`` experts. Pick eviction
        # victims from slots whose current expert is NOT in needed_set,
        # in LRU order.
        free_slots: list[int] = [
            s for s in range(self.cap_experts) if self.slot_to_expert[s] == -1
        ]
        # Candidates for eviction (not currently needed).
        evict_candidates: list[tuple[int, int]] = sorted(
            (
                (self.lru_tick[s], s)
                for s in range(self.cap_experts)
                if self.slot_to_expert[s] != -1
                and self.slot_to_expert[s] not in needed_set
            ),
            key=lambda x: x[0],
        )

        need_slot = len(missing) - len(free_slots)
        if need_slot > 0:
            if need_slot > len(evict_candidates):
                # Caller violated the sub-batching contract — the
                # working set within a SINGLE kernel call exceeds the
                # scratch. The forward path is supposed to chunk
                # tokens to avoid this; if we land here it's a bug
                # upstream, not a recoverable runtime condition.
                raise RuntimeError(
                    f"wisp.vllm: per-kernel-call working set "
                    f"({len(needed_set)} unique experts) exceeds "
                    f"cap_experts ({self.cap_experts}). The forward "
                    f"path failed to sub-batch correctly; this is a "
                    f"bug in _wisp_forward_cuda_paged."
                )
            for _tick, s in evict_candidates[:need_slot]:
                evicted = self.slot_to_expert[s]
                if evicted != -1:
                    self.expert_to_slot.pop(evicted, None)
                self.slot_to_expert[s] = -1
                self.expert_map_device[evicted] = -1
                free_slots.append(s)
                self.stats_evict += 1

        # First absorb any in-flight side-stream prefetches.
        self.copy_event.record(self.copy_stream)
        torch.cuda.current_stream().wait_event(self.copy_event)

        # Copy missing experts on the compute stream (the kernel
        # implicitly waits for these). Each H2D is a single ~1 MiB
        # row; with non_blocking=True they pipeline through the
        # PCIe queue.
        self.lru_clock += 1
        for expert in missing:
            slot = free_slots.pop()
            self.slot_to_expert[slot] = int(expert)
            self.expert_to_slot[int(expert)] = slot
            self.lru_tick[slot] = self.lru_clock
            self.scratch_w13[slot].copy_(self.cpu_w13[expert], non_blocking=True)
            self.scratch_w2[slot].copy_(self.cpu_w2[expert], non_blocking=True)
            self.expert_map_device[expert] = slot
            self.stats_miss += 1

        # Touch LRU for the experts that were already resident too.
        for e in needed_set:
            self.lru_tick[self.expert_to_slot[e]] = self.lru_clock

    def prefetch_async(self, predicted_experts: set[int]) -> None:
        """Day 4: speculatively load ``predicted_experts`` into scratch
        on the copy stream so they're (probably) resident by the time
        ensure_resident is next called on this layer.

        We do NOT update ``expert_map_device`` here — the prediction
        could be wrong, and committing the map before ensure_resident
        sees the real ``topk_ids`` would risk wrong routing if the
        kernel ran on stale state. Instead, we copy into scratch slots
        and update the host-side ``slot_to_expert`` / ``expert_to_slot``
        so ensure_resident's "already resident" check succeeds.

        Eviction policy: same LRU as ensure_resident, but biased
        against evicting *currently-resident* experts that the most-
        recent observed topk contained, because they may still be
        needed for the immediately-following decode step on this
        same layer (history bias).
        """
        if not predicted_experts:
            return
        to_load = [e for e in predicted_experts if e not in self.expert_to_slot]
        if not to_load:
            return
        free_slots: list[int] = [
            s for s in range(self.cap_experts) if self.slot_to_expert[s] == -1
        ]
        # Bias eviction: avoid slots that hold an expert in the
        # last_unique_experts set (those are likely to be re-needed in
        # the next decode step that lands on this layer).
        evict_candidates: list[tuple[int, int]] = sorted(
            (
                (self.lru_tick[s], s)
                for s in range(self.cap_experts)
                if self.slot_to_expert[s] != -1
                and self.slot_to_expert[s] not in self.last_unique_experts
                and self.slot_to_expert[s] not in predicted_experts
            ),
            key=lambda x: x[0],
        )
        need_slot = len(to_load) - len(free_slots)
        if need_slot > 0:
            if need_slot > len(evict_candidates):
                # Don't have enough non-history slots to satisfy the
                # whole prediction; partially load what we can. We're
                # speculative — leaving experts unprefetched is fine
                # (the next ensure_resident will fall back to the slow
                # path).
                evict_candidates = evict_candidates  # already sorted
                to_load = to_load[: len(evict_candidates) + len(free_slots)]
                need_slot = len(to_load) - len(free_slots)
            for _tick, s in evict_candidates[:need_slot]:
                evicted = self.slot_to_expert[s]
                if evicted != -1:
                    self.expert_to_slot.pop(evicted, None)
                self.slot_to_expert[s] = -1
                self.expert_map_device[evicted] = -1
                free_slots.append(s)
                self.stats_evict += 1
        self.lru_clock += 1
        with torch.cuda.stream(self.copy_stream):
            for expert in to_load:
                if not free_slots:
                    break
                slot = free_slots.pop()
                self.slot_to_expert[slot] = int(expert)
                self.expert_to_slot[int(expert)] = slot
                self.lru_tick[slot] = self.lru_clock
                self.scratch_w13[slot].copy_(self.cpu_w13[expert], non_blocking=True)
                self.scratch_w2[slot].copy_(self.cpu_w2[expert], non_blocking=True)
                self.expert_map_device[expert] = slot
        # Don't sync the compute stream here — the prefetch is
        # *speculative* and meant to overlap with current compute.
        # ensure_resident will sync via its own event when the
        # prediction is "consumed".

    # --------------------------------------------------------------
    # MLSys "super-final" variant: live expert<->KV split resize
    # --------------------------------------------------------------

    def resize_cap(self, new_cap: int) -> None:
        """Hot-resize this layer's GPU expert scratch to ``new_cap`` slots.

        This is the expert half of the live MV-WSA controller's split move
        (the KV half is done by the controller against the in-process
        engine's KV pool). It physically frees/allocates GPU memory:

        * **grow** — allocate a larger scratch and copy the live
          ``cap_experts`` resident slots forward; the new slots start free
          and fill on demand. The freed bytes come from the controller
          having just shrunk the KV pool.
        * **shrink** — evict the experts living in the tail slots
          ``[new_cap, cap_experts)`` (they re-page on demand) and copy the
          head into a smaller scratch; the freed bytes go to the KV pool.

        MUST be called only at a *drained barrier* (no MoE forward in
        flight) — the live controller runs it between engine steps. The
        caller is responsible for ``torch.cuda.empty_cache()`` after
        resizing all layers so the allocator returns shrunk bytes to the
        driver before the KV pool reallocates.
        """
        new_cap = max(2, min(int(new_cap), self.num_experts))
        old_cap = self.cap_experts
        if new_cap == old_cap:
            return
        dev = self.scratch_w13.device
        new_w13 = torch.empty(
            (new_cap, *self.scratch_w13.shape[1:]),
            dtype=self.scratch_w13.dtype, device=dev,
        )
        new_w2 = torch.empty(
            (new_cap, *self.scratch_w2.shape[1:]),
            dtype=self.scratch_w2.dtype, device=dev,
        )
        keep = min(old_cap, new_cap)
        new_w13[:keep].copy_(self.scratch_w13[:keep])
        new_w2[:keep].copy_(self.scratch_w2[:keep])
        if new_cap < old_cap:
            # Drop experts in evicted tail slots; they re-page on demand.
            for s in range(new_cap, old_cap):
                e = self.slot_to_expert[s]
                if e != -1:
                    self.expert_to_slot.pop(e, None)
                    self.expert_map_device[e] = -1
            del self.slot_to_expert[new_cap:]
            del self.lru_tick[new_cap:]
        else:
            self.slot_to_expert.extend([-1] * (new_cap - old_cap))
            self.lru_tick.extend([0] * (new_cap - old_cap))
        self.scratch_w13 = new_w13
        self.scratch_w2 = new_w2
        self.cap_experts = new_cap


# ---------------------------------------------------------------------------
# Per-instance config (set on the bound UnquantizedFusedMoEMethod class)
# ---------------------------------------------------------------------------

_WISP_RUNTIME_CONFIG: dict[str, Any] = {
    "mode": _Mode.RESIDENT,
    "cap_experts_override": None,
}


# ---------------------------------------------------------------------------
# Cross-layer prefetch state (Day 4)
# ---------------------------------------------------------------------------
#
# We need three pieces of process-global state:
#   * a registry of per-layer WispMoEState objects, indexed by the
#     order in which ``process_weights_after_loading`` is called (this
#     equals the model's MoE-block depth order for all current
#     Qwen3-MoE / Mixtral / DeepSeek configs);
#   * a single CooccurTable shared by all layers, optionally
#     pre-loaded from ``WISP_COOCCUR_PATH`` (the offline warm-up
#     produced by ``scripts/m3_4_warm_cooccur.py``);
#   * a per-step observation buffer so cross-layer conditioning can
#     use experts observed at earlier depths.
#
# All of these are reset to empty at module import; they're populated
# lazily as layers are initialised and forwards run.

_LAYER_STATES: list["WispMoEState"] = []
_LAYER_BY_ID: dict[int, "WispMoEState"] = {}  # id(layer_module) -> state

_COOCCUR_TABLE: Any = None  # CooccurTable | None — lazy-imported
_CURRENT_STEP_OBS: dict[int, set[int]] = {}  # layer_idx -> set(expert_ids)


# ---------------------------------------------------------------------------
# MLSys "super-final" variant: live-controller control surface
# ---------------------------------------------------------------------------
#
# The live MV-WSA controller (wisp.dynamic.controller) reaches the loaded
# MoE layers through these process-global helpers. They are no-ops unless
# the unquantized WiSP path actually registered layers, so the static
# variant (and the FP8 path) are unaffected.

def resize_layer_caps(new_cap: int) -> int:
    """Resize every registered unquantized MoE layer's expert scratch to
    ``new_cap`` slots. Drained-barrier only. Returns #layers resized."""
    n = 0
    for st in _LAYER_STATES:
        st.resize_cap(new_cap)
        n += 1
    if n:
        import torch as _torch
        _torch.cuda.empty_cache()
    return n


def current_layer_cap() -> "int | None":
    """The current per-layer expert cap (uniform across layers), or None."""
    return _LAYER_STATES[0].cap_experts if _LAYER_STATES else None


def num_registered_layers() -> int:
    return len(_LAYER_STATES)


def num_experts_per_layer() -> "int | None":
    """Global expert count per registered MoE layer (uniform across layers), or
    None if no WiSP paging layers are registered. The live controller uses this
    as the natural upper bound for the expert cap."""
    return _LAYER_STATES[0].num_experts if _LAYER_STATES else None


def expert_bytes_per_cap() -> int:
    """GPU bytes freed/claimed by changing the per-layer expert cap by one
    slot, summed across all registered layers. This is the conversion factor
    the live controller uses to trade expert slots against KV blocks."""
    total = 0
    for st in _LAYER_STATES:
        w13 = st.scratch_w13
        w2 = st.scratch_w2
        total += w13[0].numel() * w13.element_size()
        total += w2[0].numel() * w2.element_size()
    return total


def snapshot_expert_stats(reset: bool = True) -> dict:
    """Aggregate the expert-paging telemetry the controller uses as its
    miss-curve signal, summed over layers. ``ws`` is the largest per-layer
    unique-expert working set observed on the most recent step (a proxy for
    routing spread)."""
    miss = sum(s.stats_miss for s in _LAYER_STATES)
    hit = sum(s.stats_hits for s in _LAYER_STATES)
    fwd = sum(s.stats_forward for s in _LAYER_STATES)
    ws = max((len(s.last_unique_experts) for s in _LAYER_STATES), default=0)
    if reset:
        for s in _LAYER_STATES:
            s.stats_miss = 0
            s.stats_hits = 0
            s.stats_forward = 0
    return {"miss": miss, "hit": hit, "forward": fwd, "ws": ws}


def _maybe_load_cooccur() -> Any:
    """Lazy-load the global cooccur table. Returns ``None`` if no warm
    table is configured AND online learning is off (in which case the
    cooccur prefetch is disabled and we fall back to LRU-only paged-mode
    behaviour, which is what cap=8/16 already demonstrates)."""
    global _COOCCUR_TABLE
    if _COOCCUR_TABLE is not None:
        return _COOCCUR_TABLE
    path = os.environ.get("WISP_COOCCUR_PATH", "").strip()
    online = os.environ.get("WISP_COOCCUR_ONLINE", "0").strip().lower() in (
        "1", "true", "yes",
    )
    if not path and not online:
        return None
    try:
        from wisp.oracle.cooccur import CooccurTable

        if path and os.path.isfile(path):
            _COOCCUR_TABLE = CooccurTable.load(path)
            logger.info(
                "wisp.vllm: loaded warm cooccur table from %s "
                "(%d observed steps, %d source keys)",
                path,
                getattr(_COOCCUR_TABLE, "num_steps_observed", -1),
                getattr(_COOCCUR_TABLE, "n_source_keys", -1),
            )
        else:
            _COOCCUR_TABLE = CooccurTable()
            logger.info(
                "wisp.vllm: cooccur table starts cold (online=%s)", online
            )
    except Exception as e:  # pragma: no cover
        logger.warning("wisp.vllm: cooccur init failed: %s — disabling", e)
        _COOCCUR_TABLE = False  # sentinel: disabled
        return None
    return _COOCCUR_TABLE


def _cooccur_predict_top_k(target_layer: int, k: int) -> list[int]:
    """Predict the top-k experts to prefetch into ``target_layer``'s
    scratch using all observations from the current step (layers
    < target_layer). Returns [] if the table is cold or no signal."""
    table = _COOCCUR_TABLE
    if table is None or table is False:
        return []
    if not _CURRENT_STEP_OBS:
        return []
    from wisp.oracle.cooccur import predict_layer
    return predict_layer(
        table=table,
        seen_so_far=_CURRENT_STEP_OBS,
        target_layer=target_layer,
        k=k,
        normalize=True,
        decay=0.5,
        only_prev_layer=False,
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _move_to_pinned_cpu(t: torch.Tensor) -> torch.Tensor:
    if t.device.type == "cpu" and t.is_pinned():
        return t
    pinned = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    pinned.copy_(t, non_blocking=False)
    return pinned


# ---------------------------------------------------------------------------
# Patched methods
# ---------------------------------------------------------------------------


def _patched_create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
) -> None:
    """Override: allocate the master MoE weight Parameters directly on
    pinned CPU memory so load-time peak VRAM is bounded by the
    non-MoE footprint + scratch.

    Upstream's ``create_weights`` calls ``torch.empty(...)`` without a
    device kwarg, which honors PyTorch's default-device context. vLLM
    sets that context to ``cuda`` during model construction, so the
    upstream code lands the full ``(num_experts, *, *)`` tensor on
    the GPU before the safetensors loader writes into it. On a
    consumer GPU with < 24 GiB that OOMs before our
    ``process_weights_after_loading`` can free it.

    We replace the constructor with ``torch.empty(..., device="cpu",
    pin_memory=True)``. The model loader calls
    ``param.data.copy_(safetensors_tensor)`` to populate weights;
    ``.copy_`` honors the destination device, so the load becomes a
    pure CPU→CPU memcpy. No GPU weight allocation occurs during load.

    Bias paging not yet supported — most current MoE checkpoints
    (Qwen3-MoE, Mixtral, DeepSeek-V2/V3) have ``has_bias=False`` so
    this is mostly a future-proofing concern.
    """
    from vllm.model_executor.utils import set_weight_attrs

    if self.moe.has_bias:
        # We could page biases too (they're tiny — a few KiB per expert)
        # but they need their own slot-space scratch buffer aligned with
        # the weight scratch. Defer until we hit a checkpoint that
        # needs it.
        raise NotImplementedError(
            "wisp.vllm: has_bias=True MoE checkpoints are not supported "
            "yet. Open an issue / extend WispMoEState to also page bias."
        )

    if self.moe.is_act_and_mul:
        w13_up_dim = 2 * intermediate_size_per_partition
    else:
        w13_up_dim = intermediate_size_per_partition

    # Allocate non-pinned CPU first; we pin in process_weights_after_loading.
    # Pinning at alloc time interacts oddly with vLLM's post-load
    # ``model.to("cuda")`` — empirically the loader writes were not
    # visible on the pinned CPU mirror after the move (Plan A v1 ate
    # 6 hours debugging this).
    w13_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            w13_up_dim,
            hidden_size,
            dtype=params_dtype,
            device="cpu",
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    set_weight_attrs(w13_weight, extra_weight_attrs)
    # Stash a reference to the CPU pinned storage on the layer. vLLM
    # calls ``model.to("cuda")`` after weight load, which rebinds
    # ``Parameter.data`` to a fresh GPU tensor; the original CPU pin
    # would otherwise be GC'd. Holding a Python reference here keeps
    # the storage alive and the loaded weights accessible to
    # ``process_weights_after_loading`` without paying the GPU →
    # CPU bandwidth round-trip.
    layer._wisp_cpu_pin_w13 = w13_weight.data

    w2_weight = torch.nn.Parameter(
        torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype,
            device="cpu",
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2_weight)
    set_weight_attrs(w2_weight, extra_weight_attrs)
    layer._wisp_cpu_pin_w2 = w2_weight.data


def _patched_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Run vLLM's post-load processing, then materialise the GPU
    scratch.

    Three paths are handled:

    1. The common path (``create_weights`` patch ran): w13/w2 are
       already on pinned CPU; we just take them as the master copy.
    2. Fallback (``create_weights`` patch didn't fire for this layer,
       e.g. a quant-flavour MoE method that doesn't route through
       our override): w13/w2 are on GPU; do the original swing-out.
    3. Already-patched (idempotent re-call): ``layer._wisp_state``
       exists; bail.
    """
    if getattr(layer, "_wisp_state", None) is not None or getattr(layer, "_wisp_cpu_w13", None) is not None:
        return

    type(self)._wisp_original_process_weights_after_loading(self, layer)

    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    if w13 is None or w2 is None:
        return

    gpu_device = torch.device("cuda", torch.cuda.current_device())

    pin_w13 = getattr(layer, "_wisp_cpu_pin_w13", None)
    pin_w2 = getattr(layer, "_wisp_cpu_pin_w2", None)

    if pin_w13 is not None and pin_w2 is not None:
        # Plan A path — ``create_weights`` allocated w13/w2 on CPU.
        #
        # vLLM's ``device_loading_context`` (which wraps us) has
        # temporarily moved the Parameter onto the GPU via
        # ``p.data = p.data.to(cuda)`` so that GPU-only post-load
        # ops (padding kernels, repack, etc.) can run. On exit it
        # will restore CPU pinned storage from whatever ``p.data``
        # ends up as.
        #
        # Our stash still references the original CPU storage that
        # the weight loader wrote into. Pin it now (skipped at
        # alloc-time to avoid pin/move/restore corner cases) and use
        # it as the source for ensure_resident / prime_identity.
        cpu_w13 = pin_w13 if pin_w13.is_pinned() else pin_w13.pin_memory()
        cpu_w2 = pin_w2 if pin_w2.is_pinned() else pin_w2.pin_memory()
        layer._wisp_cpu_pin_w13 = cpu_w13
        layer._wisp_cpu_pin_w2 = cpu_w2
    elif w13.device.type == "cuda" and w2.device.type == "cuda":
        # Legacy path (Plan A patch disabled, or quant flavour bypasses
        # create_weights). Swing GPU → pinned CPU now.
        cpu_w13 = _move_to_pinned_cpu(w13.data)
        cpu_w2 = _move_to_pinned_cpu(w2.data)
        layer.w13_weight.data = torch.empty(0, dtype=cpu_w13.dtype, device=gpu_device)
        layer.w2_weight.data = torch.empty(0, dtype=cpu_w2.dtype, device=gpu_device)
        torch.cuda.empty_cache()
    elif w13.device.type == "cpu" and w2.device.type == "cpu":
        # Edge case: param is still on CPU at process time AND no
        # stash. Use it directly.
        cpu_w13 = w13.data if w13.data.is_pinned() else w13.data.pin_memory()
        cpu_w2 = w2.data if w2.data.is_pinned() else w2.data.pin_memory()
    else:
        return

    mode: str = _WISP_RUNTIME_CONFIG["mode"]

    if mode == _Mode.COPY:
        # Keep only the pinned-CPU mirror; the swap-in happens
        # per-forward in _patched_forward_cuda.
        layer._wisp_cpu_w13 = cpu_w13
        layer._wisp_cpu_w2 = cpu_w2
        layer._wisp_state = None
        logger.info(
            "wisp.vllm[copy]: pinned MoE weights to CPU "
            "(w13=%.2f GiB, w2=%.2f GiB, prefix=%s)",
            cpu_w13.numel() * cpu_w13.element_size() / (1 << 30),
            cpu_w2.numel() * cpu_w2.element_size() / (1 << 30),
            getattr(layer, "prefix", "?"),
        )
        return

    num_experts = int(cpu_w13.shape[0])
    cap_override = _WISP_RUNTIME_CONFIG["cap_experts_override"]
    if mode == _Mode.RESIDENT:
        cap_experts = num_experts
    elif mode == _Mode.PAGED:
        cap_experts = cap_override if cap_override is not None else _resolve_cap_experts(num_experts)
    else:  # pragma: no cover  (unreachable; _resolve_mode covers it)
        cap_experts = num_experts

    # Register layer in the cross-layer registry. The ordering of
    # ``process_weights_after_loading`` calls matches the model's
    # ``nn.Module.named_modules()`` iteration order, which for all
    # current MoE checkpoints (Qwen3-MoE, Mixtral, DeepSeek) is the
    # depth order of MoE blocks. If a future model breaks this, the
    # cooccur predictor still works (predictions just won't match
    # canonical layer indices) — telemetry would flag this via low
    # stats_pred_hits.
    layer_idx = len(_LAYER_STATES)
    state = WispMoEState(
        cpu_w13=cpu_w13,
        cpu_w2=cpu_w2,
        cap_experts=cap_experts,
        mode=mode,
        device=gpu_device,
        layer_idx=layer_idx,
    )
    _LAYER_STATES.append(state)
    _LAYER_BY_ID[id(layer)] = state
    if mode == _Mode.RESIDENT:
        state.prime_identity()
    # In paged-mode the scratch starts empty; the first forward will
    # populate it from the routing decisions.

    layer._wisp_state = state
    # IMPORTANT: do NOT set ``layer.w13_weight.data = state.scratch_w13``.
    #
    # vLLM wraps our ``process_weights_after_loading`` in a context
    # manager (``device_loading_context`` in
    # vllm/model_executor/model_loader/utils.py). For each Parameter
    # that started on CPU, the context enters by doing
    # ``p.data = p.data.to(target_device)`` and on exit copies
    # ``p.data`` back into a freshly-allocated pinned CPU tensor of
    # ``p.data.size()``. With Plan A active, w13/w2 start on CPU, so
    # if we left ``p.data = scratch`` here, the exit handler would
    # PCIe-copy the entire scratch back to CPU pinned memory — both
    # wrecking VRAM accounting and leaving the Parameter referencing
    # a CPU tensor that the next forward would feed to a CUDA kernel
    # (garbage logits, see test history 2026-06-04).
    #
    # The fix is to leave Parameter.data as a tiny GPU placeholder.
    # The exit handler then allocates a tiny CPU pinned tensor —
    # wasting nothing — and our forward path bypasses
    # ``layer.w13_weight`` entirely, reading ``state.scratch_w13``
    # directly.
    _placeholder = torch.empty(0, dtype=state.scratch_w13.dtype, device=gpu_device)
    layer.w13_weight.data = _placeholder
    layer.w2_weight.data = torch.empty(0, dtype=state.scratch_w2.dtype, device=gpu_device)

    # Release the per-layer GPU temporary that ``device_loading_context``
    # allocated to call us. Without this, vLLM's ``MemorySnapshot``
    # (which reads ``cuda.mem_get_info()`` for accounting) sees the
    # CUDA caching allocator's cached blocks and concludes the model
    # used full MoE weight VRAM — even though our scratch is the only
    # thing actually live. That mis-accounting then makes KV cache
    # budgeting fail at small ``gpu_memory_utilization`` and defeats
    # the whole point of Plan A.
    torch.cuda.empty_cache()

    scratch_bytes = (
        state.scratch_w13.numel() * state.scratch_w13.element_size()
        + state.scratch_w2.numel() * state.scratch_w2.element_size()
    )
    pinned_bytes = (
        cpu_w13.numel() * cpu_w13.element_size()
        + cpu_w2.numel() * cpu_w2.element_size()
    )
    logger.info(
        "wisp.vllm[%s]: layer init: num_experts=%d cap_experts=%d "
        "pinned=%.2f GiB gpu_scratch=%.2f GiB prefix=%s",
        mode,
        num_experts,
        cap_experts,
        pinned_bytes / (1 << 30),
        scratch_bytes / (1 << 30),
        getattr(layer, "prefix", "?"),
    )


def _patched_forward_cuda(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    state: WispMoEState | None = getattr(layer, "_wisp_state", None)
    mode_attr = getattr(state, "mode", None) if state is not None else None

    # ------------------------------------------------------------------
    # copy-mode fallback (legacy): full-copy staging per forward
    # ------------------------------------------------------------------
    if state is None and getattr(layer, "_wisp_cpu_w13", None) is not None:
        cpu_w13 = layer._wisp_cpu_w13
        cpu_w2 = layer._wisp_cpu_w2
        device = x.device
        w13_gpu = torch.empty(cpu_w13.shape, dtype=cpu_w13.dtype, device=device)
        w2_gpu = torch.empty(cpu_w2.shape, dtype=cpu_w2.dtype, device=device)
        w13_gpu.copy_(cpu_w13, non_blocking=True)
        w2_gpu.copy_(cpu_w2, non_blocking=True)
        saved_w13 = layer.w13_weight.data
        saved_w2 = layer.w2_weight.data
        layer.w13_weight.data = w13_gpu
        layer.w2_weight.data = w2_gpu
        try:
            out = type(self)._wisp_original_forward_cuda(self, layer, x, *args, **kwargs)
        finally:
            layer.w13_weight.data = saved_w13
            layer.w2_weight.data = saved_w2
            del w13_gpu, w2_gpu
        return out

    # ------------------------------------------------------------------
    # Untouched layer (no WiSP state) — pass through verbatim.
    # ------------------------------------------------------------------
    if state is None:
        return type(self)._wisp_original_forward_cuda(self, layer, x, *args, **kwargs)

    # ------------------------------------------------------------------
    # resident-mode: scratch is the full identity-mapped weight. Just call
    # upstream. expert_map kwargs from caller is left alone (None →
    # the kernel will treat scratch directly).
    # ------------------------------------------------------------------
    if mode_attr == _Mode.RESIDENT:
        state.stats_forward += 1
        return type(self)._wisp_original_forward_cuda(self, layer, x, *args, **kwargs)

    # ------------------------------------------------------------------
    # paged-mode: partial residency. We need the topk_ids BEFORE the
    # kernel runs, then bring missing experts in, then call the
    # kernel with our remap.
    #
    # The upstream forward_cuda computes topk_ids inside via
    # ``layer.select_experts(...)``. To avoid duplicating that whole
    # body we re-implement the small piece around the call.
    # ------------------------------------------------------------------
    return _wisp_forward_cuda_paged(self, layer, x, state, *args, **kwargs)


def _wisp_forward_cuda_paged(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    state: WispMoEState,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """paged-mode forward: replicates the relevant slice of upstream
    ``UnquantizedFusedMoEMethod.forward_cuda`` so we can inject our
    expert_map between ``select_experts`` and ``fused_experts``."""
    state.stats_forward += 1

    # If the caller already gave us an expert_map (e.g. real EP), we
    # don't yet support stacking. Fall back to full-copy correctness.
    if expert_map is not None:
        logger.warning(
            "wisp.vllm[paged]: caller-supplied expert_map detected; "
            "paged-mode partial residency not yet composed with EP. "
            "Falling back to identity scratch (whole working set must fit)."
        )
        # If we don't have full coverage, this will silently zero
        # tokens for the missing experts. For PoC, force prime_identity.
        if state.cap_experts == state.num_experts and -1 in state.slot_to_expert:
            state.prime_identity()
        return type(self)._wisp_original_forward_cuda(
            self, layer, x,
            use_grouped_topk=use_grouped_topk, top_k=top_k,
            router_logits=router_logits, renormalize=renormalize,
            topk_group=topk_group, num_expert_group=num_expert_group,
            global_num_experts=global_num_experts, expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func, routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation, enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    # Defer kernel imports to here so importing this module doesn't
    # require vLLM (e.g. on a Mac dev box).
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)

    topk_weights, topk_ids, zero_expert_result = layer.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        indices_type=self.topk_indices_dtype,
        enable_eplb=enable_eplb,
        expert_map=None,
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        global_num_experts=global_num_experts,
        zero_expert_num=zero_expert_num,
        zero_expert_type=zero_expert_type,
        num_fused_shared_experts=layer.num_fused_shared_experts,
    )

    if global_num_experts == -1:
        global_num_experts = state.num_experts

    # ------------------------------------------------------------------
    # Working-set check + adaptive sub-batching
    # ------------------------------------------------------------------
    # vLLM's fused_experts kernel processes all rows in ``hidden_states``
    # in one launch. The set of experts the kernel needs is
    # ``unique(topk_ids)``; if that exceeds our scratch capacity we
    # cannot fit the working set in one call.
    #
    # Fix (paper-quality, transparent to the engine): partition tokens
    # greedily into groups whose post-routing working set fits in
    # ``cap_experts``, then call the kernel once per group. This works
    # because the MoE compute for token ``t`` only depends on row ``t``
    # of ``hidden_states`` and ``topk_*[t]`` — there's no
    # cross-token coupling inside a single MoE layer. Decoding is
    # therefore *embarrassingly* sub-batchable.
    #
    # Cost: O(num_groups) kernel launches per layer instead of one,
    # plus a small CPU-side packing pass (top_k=8, prompt length ≤
    # thousands → microseconds). The PCIe copies in
    # ``ensure_resident`` are amortised across groups by the LRU
    # policy.
    unique_global = int(torch.unique(topk_ids).numel())
    if unique_global <= state.cap_experts:
        # Fast path: whole batch fits in scratch.
        state.ensure_resident(topk_ids)
        result = fused_experts(
            hidden_states=x,
            w1=state.scratch_w13,
            w2=state.scratch_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=state.expert_map_device,
        )
    else:
        # Sub-batch path. Pack greedily — first-fit-decreasing on
        # token-expert-set size would be marginally better but the
        # variance in top_k is zero (always = top_k), so first-fit
        # is optimal in expectation.
        topk_ids_cpu = topk_ids.to(device="cpu", dtype=torch.long)
        T = topk_ids_cpu.shape[0]
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_set: set[int] = set()
        for t in range(T):
            token_experts: set[int] = {int(e) for e in topk_ids_cpu[t].tolist()}
            merged = current_set | token_experts
            if len(merged) <= state.cap_experts or not current_group:
                current_group.append(t)
                current_set = merged
            else:
                groups.append(current_group)
                current_group = [t]
                current_set = token_experts
        if current_group:
            groups.append(current_group)

        logger.debug(
            "wisp.vllm[paged]: sub-batching: %d tokens, %d unique experts, "
            "cap=%d, packed into %d groups",
            T, unique_global, state.cap_experts, len(groups),
        )

        result = torch.empty_like(x)
        for tok_idxs in groups:
            idx = torch.tensor(tok_idxs, dtype=torch.long, device=x.device)
            x_sub = x.index_select(0, idx)
            tw_sub = topk_weights.index_select(0, idx)
            ti_sub = topk_ids.index_select(0, idx)
            state.ensure_resident(ti_sub)
            y_sub = fused_experts(
                hidden_states=x_sub,
                w1=state.scratch_w13,
                w2=state.scratch_w2,
                topk_weights=tw_sub,
                topk_ids=ti_sub,
                inplace=False,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=state.expert_map_device,
            )
            result.index_copy_(0, idx, y_sub)

    # ------------------------------------------------------------------
    # Day 4 post-forward: cooccur observe + cross-layer prefetch
    # ------------------------------------------------------------------
    try:
        last_set = {int(e) for e in torch.unique(topk_ids).cpu().tolist()}
    except Exception:
        last_set = set()
    state.last_unique_experts = last_set

    # Account prediction quality: how many of last step's predictions
    # for *this* layer landed in this step's actual needed set.
    if state.stats_pred_total > 0:
        # Conceptual: we'd need to have remembered last step's
        # prediction for this layer. We don't (yet) — leave 0/0
        # for now; Day 5 wires this with proper bookkeeping in
        # prefetch_async.
        pass

    # Cooccur prefetch is OFF by default. Empirically (see
    # doc/m3_5_plugin_day4_status_2026_06_04.md) speculative H2D
    # copies issued on a side stream compete with ensure_resident's
    # mandatory copies on the same PCIe bus and net-slow the engine
    # at every cap we measured (cap ∈ {8, 16, 32, 64}). The code is
    # kept behind a flag so we can ablate vs LRU-only in the paper.
    prefetch_on = os.environ.get("WISP_PREFETCH", "0").strip().lower() in (
        "1", "true", "yes",
    )
    table = _maybe_load_cooccur() if prefetch_on else None
    if table is not None and last_set:
        _CURRENT_STEP_OBS[state.layer_idx] = last_set

        # Predict experts for the NEXT MoE block (layer_idx + 1) using
        # everything we've observed so far in the current step. Issue
        # async H2D into that layer's scratch on its side stream —
        # overlaps with the remaining layers of compute we'll do
        # before that layer fires again.
        next_idx = state.layer_idx + 1
        if next_idx < len(_LAYER_STATES):
            next_state = _LAYER_STATES[next_idx]
            # Predict ``cap_experts // 2`` experts: leaves half the
            # capacity for the actual ensure_resident to consume,
            # so over-confident prediction doesn't evict slots that
            # the real routing turns out to need.
            k_predict = max(8, next_state.cap_experts // 2)
            predicted = _cooccur_predict_top_k(next_idx, k_predict)
            if predicted:
                try:
                    next_state.prefetch_async(set(predicted))
                    next_state.stats_pred_total += len(predicted)
                except Exception as e:  # pragma: no cover
                    logger.debug("prefetch_async failed: %s", e)

        # End-of-step bookkeeping. observe_step builds the cross-
        # layer pair set for the full step, so we must call it once
        # with all layers' observations rather than 48 times with one.
        if state.layer_idx == len(_LAYER_STATES) - 1:
            if os.environ.get("WISP_COOCCUR_ONLINE", "0").strip().lower() in (
                "1", "true", "yes",
            ):
                try:
                    table.observe_step(_CURRENT_STEP_OBS)
                except Exception:
                    pass
            _CURRENT_STEP_OBS.clear()

    if zero_expert_num != 0 and zero_expert_type is not None:
        return result, zero_expert_result
    return result


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_wisp_moe(*, mode: str | None = None, cap_experts: int | None = None) -> bool:
    """Idempotent install. Reads ``WISP_MODE`` from env if ``mode`` is
    omitted, ``WISP_CAP_EXPERTS`` for paged-mode cap if ``cap_experts``
    is omitted.

    Returns True iff this call applied the patch (False = was
    already patched). Reapplying the patch with a different mode is
    NOT supported; restart Python instead. (We keep this simple
    on purpose; the engine subprocess fork happens after our patch,
    so cross-process consistency is what matters.)
    """
    try:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
            UnquantizedFusedMoEMethod,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "wisp.integrations.vllm requires vllm to be installed."
        ) from e

    _WISP_RUNTIME_CONFIG["mode"] = mode if mode is not None else _resolve_mode()
    _WISP_RUNTIME_CONFIG["cap_experts_override"] = cap_experts

    if getattr(UnquantizedFusedMoEMethod, _PATCHED_FLAG, False):
        logger.debug(
            "wisp.vllm: UnquantizedFusedMoEMethod already patched; "
            "runtime config refreshed to mode=%s cap_experts=%s",
            _WISP_RUNTIME_CONFIG["mode"], _WISP_RUNTIME_CONFIG["cap_experts_override"],
        )
        return False

    UnquantizedFusedMoEMethod._wisp_original_create_weights = (
        UnquantizedFusedMoEMethod.create_weights
    )
    UnquantizedFusedMoEMethod._wisp_original_process_weights_after_loading = (
        UnquantizedFusedMoEMethod.process_weights_after_loading
    )
    UnquantizedFusedMoEMethod._wisp_original_forward_cuda = (
        UnquantizedFusedMoEMethod.forward_cuda
    )
    UnquantizedFusedMoEMethod.create_weights = _patched_create_weights
    UnquantizedFusedMoEMethod.process_weights_after_loading = (
        _patched_process_weights_after_loading
    )
    UnquantizedFusedMoEMethod.forward_cuda = _patched_forward_cuda
    setattr(UnquantizedFusedMoEMethod, _PATCHED_FLAG, True)

    logger.info(
        "wisp.vllm: patched UnquantizedFusedMoEMethod (mode=%s cap_experts=%s)",
        _WISP_RUNTIME_CONFIG["mode"], _WISP_RUNTIME_CONFIG["cap_experts_override"],
    )

    # Also patch the FP8 block-quant MoE path (MiniMax-M2, DeepSeek-V3, ...).
    # Best-effort: a failure here must not break the unquantized path.
    try:
        from .fused_moe_fp8 import install_wisp_fp8
        install_wisp_fp8()
    except Exception as e:  # pragma: no cover
        logger.warning("wisp.vllm: FP8 MoE patch not applied (%s)", e)

    return True


def is_installed() -> bool:
    try:
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
            UnquantizedFusedMoEMethod,
        )
    except ImportError:
        return False
    return bool(getattr(UnquantizedFusedMoEMethod, _PATCHED_FLAG, False))


__all__ = ["install_wisp_moe", "is_installed", "WispMoEState"]
