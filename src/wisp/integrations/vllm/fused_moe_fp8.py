"""WiSP FP8 MoE paging patch (block-quantized FP8, e.g. MiniMax-M2, DeepSeek-V3).

The base WiSP patch (``fused_moe.py``) only handles the unquantized BF16/FP16
MoE method. Production-scale MoEs ship FP8 block-quantized. This module adds the
same working-set paging to ``Fp8MoEMethod`` so a 240B-class FP8 MoE can be served
on a single GPU.

Design (mirrors the proven Day-2b unquantized path):

- A block-FP8 expert is FOUR tensors, not one:
    w13_weight            [E, 2I, H]              float8_e4m3fn
    w2_weight             [E, H, I]               float8_e4m3fn
    w13_weight_scale_inv  [E, 2*ceil(I/bn), ceil(H/bk)]  float32  (per-block scale)
    w2_weight_scale_inv   [E, ceil(H/bn), ceil(I/bk)]    float32
  We page the (weight, scale) pair for each expert together.

- We force the **Triton** FP8 MoE backend (``VLLM_USE_DEEP_GEMM=0`` at launch),
  which consumes the raw block layout — no DeepGEMM post-process realignment, so
  a per-expert CPU→scratch slice is bit-exact.

- The fused-FP8 kernel reads weights from its ``w1``/``w2`` args and scales from
  ``quant_config`` (a ``FusedMoEQuantConfig``). Because our GPU scratch buffers
  are allocated once and reused (only contents change per forward), we build a
  single ``FusedMoEQuantConfig`` that references the scratch scale tensors and
  reuse it every step. The kernel's ``expert_map`` remaps global expert ids to
  scratch slots, exactly as in expert-parallel deployments.

Correctness: outputs are identical to unpaged FP8 inference up to kernel
reproducibility — the same FP8 weights/scales are read, only their physical
residence moves. (P1/P2 as in the unquantized path.)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger("wisp.integrations.vllm.fused_moe_fp8")

_PATCHED_FLAG = "_wisp_fp8_patched"

# Registry of FP8 layer states (separate from the unquantized registry).
_FP8_LAYER_STATES: list["WispMoEStateFP8"] = []


def _resolve_cap_experts(num_experts: int) -> int:
    raw = os.environ.get("WISP_CAP_EXPERTS")
    if raw is None:
        return num_experts
    try:
        cap = int(raw)
    except ValueError:
        return num_experts
    return max(8, min(cap, num_experts))


class WispMoEStateFP8:
    """Per-layer FP8 paging state: pages (weight, scale) pairs for w13 and w2."""

    __slots__ = (
        "layer_idx", "num_experts", "cap_experts",
        "cpu_w13", "cpu_w2", "cpu_s13", "cpu_s2",
        "scratch_w13", "scratch_w2", "scratch_s13", "scratch_s2",
        "slot_to_expert", "expert_to_slot", "expert_map_device",
        "lru_tick", "lru_clock",
        "stats_forward", "stats_miss", "stats_hits", "stats_evict",
        "quant_config",
    )

    def __init__(self, cpu_w13, cpu_w2, cpu_s13, cpu_s2, cap_experts, device, layer_idx):
        self.layer_idx = int(layer_idx)
        self.num_experts = int(cpu_w13.shape[0])
        self.cap_experts = int(cap_experts)
        self.cpu_w13, self.cpu_w2 = cpu_w13, cpu_w2
        self.cpu_s13, self.cpu_s2 = cpu_s13, cpu_s2

        C = cap_experts
        self.scratch_w13 = torch.empty((C, *cpu_w13.shape[1:]), dtype=cpu_w13.dtype, device=device)
        self.scratch_w2 = torch.empty((C, *cpu_w2.shape[1:]), dtype=cpu_w2.dtype, device=device)
        self.scratch_s13 = torch.empty((C, *cpu_s13.shape[1:]), dtype=cpu_s13.dtype, device=device)
        self.scratch_s2 = torch.empty((C, *cpu_s2.shape[1:]), dtype=cpu_s2.dtype, device=device)

        self.slot_to_expert: list[int] = [-1] * C
        self.expert_to_slot: dict[int, int] = {}
        self.expert_map_device = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
        self.lru_tick = [0] * C
        self.lru_clock = 0
        self.stats_forward = self.stats_miss = self.stats_hits = self.stats_evict = 0
        self.quant_config: Any = None  # set after scratch is bound

    def ensure_resident(self, needed_experts: torch.Tensor) -> None:
        """Page in every unique expert in ``needed_experts`` (≤ cap). Copies the
        (weight, scale) pair for w13 and w2 on the compute stream so the kernel
        that follows is correctly ordered. Caller guarantees the unique count
        ≤ cap_experts (via sub-batching)."""
        needed = torch.unique(needed_experts).to(device="cpu", dtype=torch.long).tolist()
        needed_set = set(int(e) for e in needed)
        missing = [e for e in needed_set if e not in self.expert_to_slot]
        self.stats_hits += len(needed_set) - len(missing)
        if not missing:
            self.lru_clock += 1
            for e in needed_set:
                self.lru_tick[self.expert_to_slot[e]] = self.lru_clock
            return

        free_slots = [s for s in range(self.cap_experts) if self.slot_to_expert[s] == -1]
        evict_candidates = sorted(
            ((self.lru_tick[s], s) for s in range(self.cap_experts)
             if self.slot_to_expert[s] != -1 and self.slot_to_expert[s] not in needed_set),
            key=lambda x: x[0],
        )
        need_slot = len(missing) - len(free_slots)
        if need_slot > 0:
            if need_slot > len(evict_candidates):
                raise RuntimeError(
                    f"wisp.fp8: working set {len(needed_set)} > cap {self.cap_experts} "
                    f"(sub-batching failed) layer={self.layer_idx}"
                )
            for _t, s in evict_candidates[:need_slot]:
                ev = self.slot_to_expert[s]
                if ev != -1:
                    self.expert_to_slot.pop(ev, None)
                self.slot_to_expert[s] = -1
                self.expert_map_device[ev] = -1
                free_slots.append(s)
                self.stats_evict += 1

        self.lru_clock += 1
        for e in missing:
            slot = free_slots.pop()
            self.slot_to_expert[slot] = int(e)
            self.expert_to_slot[int(e)] = slot
            self.lru_tick[slot] = self.lru_clock
            self.scratch_w13[slot].copy_(self.cpu_w13[e], non_blocking=True)
            self.scratch_w2[slot].copy_(self.cpu_w2[e], non_blocking=True)
            self.scratch_s13[slot].copy_(self.cpu_s13[e], non_blocking=True)
            self.scratch_s2[slot].copy_(self.cpu_s2[e], non_blocking=True)
            self.expert_map_device[e] = slot
            self.stats_miss += 1
        for e in needed_set:
            self.lru_tick[self.expert_to_slot[e]] = self.lru_clock


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t if (t.device.type == "cpu") else t.to("cpu")


def _patched_create_weights(self, layer, num_experts, hidden_size,
                            intermediate_size_per_partition, params_dtype,
                            **extra_weight_attrs):
    """Allocate the FP8 master tensors on CPU instead of GPU, so load-time peak
    VRAM is bounded. Mirrors Fp8MoEMethod.create_weights for the block-quant
    branch but with device='cpu'."""
    from vllm.model_executor.utils import set_weight_attrs

    if not self.block_quant:
        # Only block-quant FP8 is supported by WiSP paging for now; fall back.
        return type(self)._wisp_original_create_weights(
            self, layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype, **extra_weight_attrs)

    layer.intermediate_size_per_partition = intermediate_size_per_partition
    layer.hidden_size = hidden_size
    layer.num_experts = num_experts
    layer.orig_dtype = params_dtype
    layer.weight_block_size = self.weight_block_size
    block_n, block_k = self.weight_block_size[0], self.weight_block_size[1]
    fp8_dtype = torch.float8_e4m3fn

    w13_weight = torch.nn.Parameter(
        torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size,
                    dtype=fp8_dtype, device="cpu"), requires_grad=False)
    layer.register_parameter("w13_weight", w13_weight)
    set_weight_attrs(w13_weight, extra_weight_attrs)
    layer._wisp_cpu_pin_w13 = w13_weight.data

    w2_weight = torch.nn.Parameter(
        torch.empty(num_experts, hidden_size, intermediate_size_per_partition,
                    dtype=fp8_dtype, device="cpu"), requires_grad=False)
    layer.register_parameter("w2_weight", w2_weight)
    set_weight_attrs(w2_weight, extra_weight_attrs)
    layer._wisp_cpu_pin_w2 = w2_weight.data

    w13_scale = torch.nn.Parameter(
        torch.ones(num_experts,
                   2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                   (hidden_size + block_k - 1) // block_k,
                   dtype=torch.float32, device="cpu"), requires_grad=False)
    layer.register_parameter("w13_weight_scale_inv", w13_scale)
    layer._wisp_cpu_pin_s13 = w13_scale.data

    w2_scale = torch.nn.Parameter(
        torch.ones(num_experts,
                   (hidden_size + block_n - 1) // block_n,
                   (intermediate_size_per_partition + block_k - 1) // block_k,
                   dtype=torch.float32, device="cpu"), requires_grad=False)
    layer.register_parameter("w2_weight_scale_inv", w2_scale)
    layer._wisp_cpu_pin_s2 = w2_scale.data

    from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported as _WSS
    extra_weight_attrs.update({"quant_method": _WSS.BLOCK.value})
    set_weight_attrs(w13_scale, extra_weight_attrs)
    set_weight_attrs(w2_scale, extra_weight_attrs)

    layer.w13_input_scale = None
    layer.w2_input_scale = None
    self.rocm_aiter_moe_enabled = False


def _patched_process_weights_after_loading(self, layer) -> None:
    if getattr(layer, "_wisp_fp8_state", None) is not None:
        return
    if not getattr(self, "block_quant", False):
        return type(self)._wisp_original_process_weights_after_loading(self, layer)

    cpu_w13 = _to_cpu(getattr(layer, "_wisp_cpu_pin_w13", layer.w13_weight.data))
    cpu_w2 = _to_cpu(getattr(layer, "_wisp_cpu_pin_w2", layer.w2_weight.data))
    cpu_s13 = _to_cpu(getattr(layer, "_wisp_cpu_pin_s13", layer.w13_weight_scale_inv.data))
    cpu_s2 = _to_cpu(getattr(layer, "_wisp_cpu_pin_s2", layer.w2_weight_scale_inv.data))
    # Do NOT pin the weight masters. pin_memory() allocates a *second* copy while
    # the original is still referenced; for a 200GB+ master that doubles memory
    # and OOM-kills the engine on a memory-capped container. We keep the weight
    # masters in pageable CPU memory (slightly slower H2D, fully correct) and
    # pin only the tiny per-block scales.
    def _try_pin(t):
        try:
            return t.pin_memory() if not t.is_pinned() else t
        except Exception:
            return t
    cpu_s13, cpu_s2 = _try_pin(cpu_s13), _try_pin(cpu_s2)

    gpu = torch.device("cuda", torch.cuda.current_device())
    num_experts = int(cpu_w13.shape[0])
    cap = _resolve_cap_experts(num_experts)
    layer_idx = len(_FP8_LAYER_STATES)
    st = WispMoEStateFP8(cpu_w13, cpu_w2, cpu_s13, cpu_s2, cap, gpu, layer_idx)
    _FP8_LAYER_STATES.append(st)
    layer._wisp_fp8_state = st

    # IMPORTANT: do NOT point layer.w*_weight at the scratch. vLLM's loader wraps
    # process_weights_after_loading in a device-loading context that copies each
    # param's .data back to a CPU tensor of matching size on exit — pointing at
    # scratch would round-trip it and corrupt accounting (see the unquantized
    # path's 6-hour debug note). Our apply() reads st.scratch_* directly, so the
    # layer params are never consulted; leave tiny GPU placeholders.
    ph = torch.empty(0, dtype=st.scratch_w13.dtype, device=gpu)
    layer.w13_weight.data = ph
    layer.w2_weight.data = torch.empty(0, dtype=st.scratch_w2.dtype, device=gpu)
    layer.w13_weight_scale_inv.data = torch.empty(0, dtype=st.scratch_s13.dtype, device=gpu)
    layer.w2_weight_scale_inv.data = torch.empty(0, dtype=st.scratch_s2.dtype, device=gpu)

    # Build a quant_config that references the (persistent) scratch scales.
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    st.quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=st.scratch_s13, w2_scale=st.scratch_s2,
        a1_scale=None, a2_scale=None, block_shape=self.weight_block_size,
    )
    # Force the method's cached config to rebuild from the (now scratch) layer.
    self.moe_quant_config = st.quant_config

    torch.cuda.empty_cache()
    scratch_gib = sum(t.numel() * t.element_size() for t in
                      (st.scratch_w13, st.scratch_w2, st.scratch_s13, st.scratch_s2)) / (1 << 30)
    logger.info("wisp.fp8: layer %d init num_experts=%d cap=%d gpu_scratch=%.2f GiB",
                layer_idx, num_experts, cap, scratch_gib)


def _patched_apply(self, layer, x, router_logits, top_k, renormalize,
                   use_grouped_topk=False, topk_group=None, num_expert_group=None,
                   global_num_experts=-1, expert_map=None, custom_routing_function=None,
                   scoring_func="softmax", routed_scaling_factor=1.0,
                   e_score_correction_bias=None, apply_router_weight_on_input=False,
                   activation="silu", enable_eplb=False, expert_load_view=None,
                   logical_to_physical_map=None, logical_replica_count=None):
    st: WispMoEStateFP8 | None = getattr(layer, "_wisp_fp8_state", None)
    if st is None:
        # Not a WiSP-paged FP8 layer (e.g. non-block-quant) — defer to original.
        return type(self)._wisp_original_apply(
            self, layer, x, router_logits, top_k, renormalize,
            use_grouped_topk=use_grouped_topk, topk_group=topk_group,
            num_expert_group=num_expert_group, global_num_experts=global_num_experts,
            expert_map=expert_map, custom_routing_function=custom_routing_function,
            scoring_func=scoring_func, routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation, enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count)

    if expert_map is not None:
        raise NotImplementedError("wisp.fp8: EP expert_map + paging not composed yet")

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe import fused_experts

    st.stats_forward += 1
    if global_num_experts == -1:
        global_num_experts = st.num_experts

    # Global routing ids (expert_map=None → global space), mirror paged mode.
    topk_weights, topk_ids, _zero = FusedMoE.select_experts(
        hidden_states=x, router_logits=router_logits,
        use_grouped_topk=use_grouped_topk, top_k=top_k, renormalize=renormalize,
        topk_group=topk_group, num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function, scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        indices_type=self.topk_indices_dtype, enable_eplb=False, expert_map=None,
        expert_load_view=None, logical_to_physical_map=None, logical_replica_count=None,
        global_num_experts=global_num_experts,
        zero_expert_num=getattr(layer, "zero_expert_num", 0),
        zero_expert_type=getattr(layer, "zero_expert_type", None),
        num_fused_shared_experts=layer.num_fused_shared_experts,
    )

    def _call(xs, tw, ti):
        return fused_experts(
            hidden_states=xs, w1=st.scratch_w13, w2=st.scratch_w2,
            topk_weights=tw, topk_ids=ti, inplace=False, activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=st.expert_map_device, quant_config=st.quant_config,
            allow_deep_gemm=False, allow_cutlass_block_scaled_grouped_gemm=False,
        )

    unique_global = int(torch.unique(topk_ids).numel())
    if unique_global <= st.cap_experts:
        st.ensure_resident(topk_ids)
        return _call(x, topk_weights, topk_ids)

    # Sub-batch tokens so each group's expert union fits the cap (first-fit).
    ti_cpu = topk_ids.to(device="cpu", dtype=torch.long)
    T = ti_cpu.shape[0]
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_set: set[int] = set()
    for t in range(T):
        te = {int(e) for e in ti_cpu[t].tolist()}
        merged = cur_set | te
        if len(merged) <= st.cap_experts or not cur:
            cur.append(t); cur_set = merged
        else:
            groups.append(cur); cur = [t]; cur_set = te
    if cur:
        groups.append(cur)

    result = torch.empty_like(x)
    for idxs in groups:
        idx = torch.tensor(idxs, dtype=torch.long, device=x.device)
        ti_sub = topk_ids.index_select(0, idx)
        st.ensure_resident(ti_sub)
        y = _call(x.index_select(0, idx), topk_weights.index_select(0, idx), ti_sub)
        result.index_copy_(0, idx, y)
    return result


def install_wisp_fp8() -> bool:
    """Patch Fp8MoEMethod (block-quant path) for working-set paging. Idempotent."""
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    except ImportError as e:  # pragma: no cover
        logger.warning("wisp.fp8: cannot import Fp8MoEMethod (%s); FP8 paging disabled", e)
        return False
    if getattr(Fp8MoEMethod, _PATCHED_FLAG, False):
        return False
    Fp8MoEMethod._wisp_original_create_weights = Fp8MoEMethod.create_weights
    Fp8MoEMethod._wisp_original_process_weights_after_loading = Fp8MoEMethod.process_weights_after_loading
    Fp8MoEMethod._wisp_original_apply = Fp8MoEMethod.apply
    Fp8MoEMethod.create_weights = _patched_create_weights
    Fp8MoEMethod.process_weights_after_loading = _patched_process_weights_after_loading
    Fp8MoEMethod.apply = _patched_apply
    setattr(Fp8MoEMethod, _PATCHED_FLAG, True)
    logger.info("wisp.fp8: patched Fp8MoEMethod (block-quant working-set paging)")
    return True


def is_fp8_installed() -> bool:
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    except ImportError:
        return False
    return bool(getattr(Fp8MoEMethod, _PATCHED_FLAG, False))


__all__ = ["install_wisp_fp8", "is_fp8_installed", "WispMoEStateFP8"]
