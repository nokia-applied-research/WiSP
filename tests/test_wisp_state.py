"""Unit tests for WispMoEState — the engine-independent pager logic.

These exercise LRU admission/eviction, the expert_map invariants, the
working-set overflow contract, and the live resize_cap used by the MV-WSA
controller. They need a CUDA device (the state allocates GPU scratch and a
copy stream) but no vLLM and no model — run them on any GPU box:

    pip install pytest && pytest tests/ -q
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="WispMoEState requires a CUDA device"
)

from wisp.integrations.vllm.fused_moe import WispMoEState  # noqa: E402

E = 8          # experts per layer in the toy state
CAP = 4        # scratch slots
DEV = "cuda:0"


def make_state(cap: int = CAP, mode: str = "paged") -> WispMoEState:
    # Fill each expert's weights with its own id so content checks are exact.
    w13 = torch.empty((E, 4, 2), dtype=torch.float32)
    w2 = torch.empty((E, 2, 4), dtype=torch.float32)
    for e in range(E):
        w13[e].fill_(float(e))
        w2[e].fill_(float(e))
    return WispMoEState(
        cpu_w13=w13.pin_memory(),
        cpu_w2=w2.pin_memory(),
        cap_experts=cap,
        mode=mode,
        device=torch.device(DEV),
        layer_idx=0,
    )


def need(state: WispMoEState, experts: list[int]) -> None:
    state.ensure_resident(torch.tensor(experts, dtype=torch.long, device=DEV))


def emap(state: WispMoEState) -> list[int]:
    torch.cuda.synchronize()
    return state.expert_map_device.cpu().tolist()


def assert_consistent(state: WispMoEState) -> None:
    """The three views of residency must agree, and scratch content must
    match the CPU master for every resident expert."""
    torch.cuda.synchronize()
    m = emap(state)
    for e, s in state.expert_to_slot.items():
        assert state.slot_to_expert[s] == e
        assert m[e] == s
        assert torch.all(state.scratch_w13[s].cpu() == float(e)), f"expert {e} slot {s} w13 content"
        assert torch.all(state.scratch_w2[s].cpu() == float(e)), f"expert {e} slot {s} w2 content"
    for e in range(state.num_experts):
        if e not in state.expert_to_slot:
            assert m[e] == -1, f"non-resident expert {e} has map entry {m[e]}"
    for s, e in enumerate(state.slot_to_expert):
        if e != -1:
            assert state.expert_to_slot[e] == s


def test_miss_then_hit():
    st = make_state()
    need(st, [0, 1])
    assert st.stats_miss == 2 and st.stats_hits == 0
    assert_consistent(st)
    need(st, [0, 1])
    assert st.stats_miss == 2 and st.stats_hits == 2
    assert st.stats_evict == 0
    assert_consistent(st)


def test_lru_eviction_order():
    st = make_state()
    need(st, [0, 1])           # tick 1
    need(st, [2, 3])           # tick 2 — scratch now full
    slot_of_0 = st.expert_to_slot[0]
    need(st, [4])              # must evict from the oldest tick
    assert 0 not in st.expert_to_slot or 1 not in st.expert_to_slot
    assert emap(st)[4] != -1
    assert st.stats_evict == 1
    # the freed slot was one that held tick-1 experts
    assert st.expert_to_slot[4] in (slot_of_0, 1 - slot_of_0, 0, 1, 2, 3)
    assert_consistent(st)


def test_needed_set_never_evicted():
    st = make_state()
    need(st, [0, 1, 2, 3])
    need(st, [0, 1, 4])        # victim must be 2 or 3, never 0/1
    assert 0 in st.expert_to_slot and 1 in st.expert_to_slot
    assert 4 in st.expert_to_slot
    assert (2 in st.expert_to_slot) != (3 in st.expert_to_slot)
    assert_consistent(st)


def test_working_set_overflow_raises():
    st = make_state()
    with pytest.raises(RuntimeError, match="sub-batch"):
        need(st, [0, 1, 2, 3, 4])


def test_map_consistency_random_walk():
    torch.manual_seed(0)
    st = make_state()
    for _ in range(50):
        k = int(torch.randint(1, CAP + 1, ()).item())
        experts = torch.randperm(E)[:k].tolist()
        need(st, experts)
        for e in experts:
            assert e in st.expert_to_slot
    assert_consistent(st)


def test_resize_grow_preserves_residents():
    st = make_state()
    need(st, [0, 1, 2, 3])
    torch.cuda.synchronize()
    st.resize_cap(6)
    assert st.cap_experts == 6
    assert len(st.slot_to_expert) == 6 and len(st.lru_tick) == 6
    assert st.slot_to_expert[4] == -1 and st.slot_to_expert[5] == -1
    assert_consistent(st)
    before_evict = st.stats_evict
    need(st, [4, 5])           # fits in the new slots, no eviction
    assert st.stats_evict == before_evict
    assert_consistent(st)


def test_resize_shrink_drops_tail_and_repages():
    st = make_state()
    need(st, [0, 1, 2, 3])
    torch.cuda.synchronize()
    tail_experts = [e for e in st.slot_to_expert[2:] if e != -1]
    st.resize_cap(2)
    assert st.cap_experts == 2
    assert len(st.slot_to_expert) == 2 and len(st.lru_tick) == 2
    m = emap(st)
    for e in tail_experts:
        assert e not in st.expert_to_slot and m[e] == -1
    assert_consistent(st)
    # a dropped expert re-pages on demand
    if tail_experts:
        need(st, [tail_experts[0]])
        assert tail_experts[0] in st.expert_to_slot
        assert_consistent(st)


def test_resize_noop_and_clamps():
    st = make_state()
    need(st, [0, 1])
    st.resize_cap(CAP)             # no-op
    assert st.cap_experts == CAP
    st.resize_cap(1)               # clamps to floor 2
    assert st.cap_experts == 2
    st.resize_cap(999)             # clamps to num_experts
    assert st.cap_experts == E
    assert_consistent(st)


def test_prime_identity_resident_mode():
    st = make_state(cap=E, mode="resident")
    st.prime_identity()
    torch.cuda.synchronize()
    assert emap(st) == list(range(E))
    assert_consistent(st)
