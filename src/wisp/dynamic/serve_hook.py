"""Wire the live MV-WSA controller into the real multi-process ``vllm serve`` loop.

Background
----------
The in-process PoC (``scripts/m4_dynamic_poc.py`` / ``scripts/m5_dynamic_agentinstruct.py``)
hand-drives ``engine.step()`` and re-solves the expert<->KV split between rounds,
using ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the v1 ``EngineCore`` lives in the
caller's process. That is *not* how ``vllm serve`` runs: the OpenAI API server is
asyncio, and vLLM forbids ``asyncio + not multiprocess`` (``EngineCoreClient.make_client``),
so serving always runs ``EngineCore`` in a **background subprocess** driven by
:meth:`EngineCoreProc.run_busy_loop`.

The key enabling fact for single-GPU WiSP (the target regime, ``tp=1``): that
EngineCore subprocess uses :class:`UniProcExecutor`, so the **worker**
(``model_runner`` + WiSP expert scratch + the KV tensors) is *co-located in the
EngineCore process* with the scheduler and its block pool. Therefore the exact
in-process resize primitives the PoC uses (:mod:`wisp.dynamic.kv_resize` +
``fused_moe.resize_layer_caps``) work unchanged here — we just have to drive the
controller from *inside* the busy loop instead of from a hand-written harness.

What this module does
---------------------
It patches :meth:`vllm.v1.engine.core.EngineCoreProc._process_engine_step`
(idempotently, installed from the ``vllm.general_plugins`` entry point, which
vLLM runs inside the EngineCore subprocess) to, on every step:

1. sample the KV block-pool occupancy (no polling thread — one cheap read per
   step), tracking the per-epoch peak, and
2. at every **drained barrier** (the scheduler just went from "has requests" to
   "empty" — no request in flight, GPU idle, the only safe point), re-solve and
   apply the expert<->KV split via :class:`MVWSAController`, exactly as the PoC
   does between rounds.

Because the resize runs in the busy-loop thread at a drained barrier, no request
is in flight and the move is iso-VRAM + byte-identical, same invariants as the
offline path. A failed move is swallowed by the controller and the run continues.

Enabled by ``WISP_DYNAMIC=1``. Tunables (all optional):

==============================  ==========================================
``WISP_DYN_CAP_MIN``            min per-layer expert cap (default 4)
``WISP_DYN_CAP_MAX``            max per-layer expert cap (default = num_experts)
``WISP_DYN_KV_FLOOR_BLOCKS``    KV admission floor in blocks (default 64)
``WISP_DYN_HEADROOM``           KV sizing headroom fraction (default 0.15)
``WISP_DYN_DEADZONE``           min cap delta to act on (default 1)
``WISP_DYN_MIN_INTERVAL_S``     min seconds between resizes (default 0.0)
``WISP_DYN_LOG``                ``1`` to log each decision (default 1)
==============================  ==========================================

Multi-GPU serving (``tp>1``, remote worker processes) is **not** wired here: the
worker-side ``model_runner`` is then in a different process and the resize would
need ``collective_rpc``. We detect that case (handles not locatable in this
process) and stay inert with a clear log, rather than corrupting a run.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("wisp.dynamic.serve_hook")

_PATCHED_FLAG = "_wisp_serve_patched"
_STATE_ATTR = "_wisp_serve_state"


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _default_kv_floor_blocks(handles) -> int:
    """A *safe* KV admission floor derived from the engine config: enough GPU KV
    blocks to hold one full ``max_model_len`` sequence. Shrinking KV below this
    can starve even a single request (it cannot be scheduled -> the loop stalls),
    so this is the hard lower bound the controller must never cross. The reactive
    controller reclaims KV *above* this floor when observed usage is lower.

    Falls back to 256 if the config can't be read."""
    try:
        cfg = handles.model_runner.vllm_config
        max_len = int(cfg.model_config.max_model_len)
        block = int(cfg.cache_config.block_size)
        return max(2, -(-max_len // block))  # ceil(max_len / block)
    except Exception:
        return 256


def _append_log(path: str, record: dict) -> None:
    """Append one resize decision as a JSON line. Best-effort telemetry; never
    raises into the serve loop."""
    try:
        import json

        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # pragma: no cover
        pass


class _ServeCtrlState:
    """Per-EngineCore live-controller state (attached as ``engine_core._wisp_serve_state``)."""

    __slots__ = (
        "enabled",
        "controller",
        "handles",
        "scheduler",
        "block_pool",
        "epoch_total",
        "epoch_min_free",
        "epoch_pressure",
        "had_requests",
        "last_fire",
        "min_interval_s",
        "do_log",
        "log_file",
        "n_resizes",
        "n_drains",
    )

    def __init__(self) -> None:
        self.enabled = False
        self.controller = None
        self.handles = None
        self.scheduler = None
        self.block_pool = None
        self.epoch_total = 0
        self.epoch_min_free = 0
        self.epoch_pressure = False
        self.had_requests = False
        self.last_fire = 0.0
        self.min_interval_s = 0.0
        self.do_log = True
        self.log_file = ""
        self.n_resizes = 0
        self.n_drains = 0


def _init_state(engine_core) -> _ServeCtrlState:
    """Locate handles + build the controller for this EngineCore. On any problem
    (no WiSP MoE layers, handles not reachable in this process, tp>1 remote
    worker) we return a disabled state and log why — the engine keeps serving,
    just without live resizing."""
    st = _ServeCtrlState()
    if not _truthy("WISP_DYNAMIC"):
        return st  # disabled; should not be reached (install gated on this)

    try:
        from wisp.dynamic.engine_access import locate_handles
        from wisp.dynamic.controller import MVWSAController
        from wisp.integrations.vllm import fused_moe as fm
    except Exception as e:  # pragma: no cover
        logger.warning("wisp.serve_hook: imports failed (%s); live split disabled", e)
        return st

    if fm.num_registered_layers() == 0:
        logger.warning(
            "wisp.serve_hook: WISP_DYNAMIC=1 but no WiSP MoE layers are registered "
            "(WISP_MODE must be day2b and the unquantized paging path active). "
            "Live expert<->KV split disabled."
        )
        return st

    try:
        # The EngineCore object graph contains the scheduler + (for tp=1
        # UniProcExecutor) the co-located model_runner; the BFS locator finds
        # both. For tp>1 (remote workers) the runner is not in this process and
        # this raises -> we stay inert.
        handles = locate_handles(engine_core)
    except Exception as e:
        logger.warning(
            "wisp.serve_hook: could not locate in-process engine handles (%s). "
            "This is expected for multi-GPU serving (tp>1, remote workers), which "
            "the live split does not support yet. Disabled.",
            e,
        )
        return st

    cap_max_default = fm.num_experts_per_layer() or 64
    kv_floor_default = _default_kv_floor_blocks(handles)
    try:
        controller = MVWSAController(
            handles,
            cap_min=_env_int("WISP_DYN_CAP_MIN", 8),
            cap_max=_env_int("WISP_DYN_CAP_MAX", cap_max_default),
            kv_floor_blocks=_env_int("WISP_DYN_KV_FLOOR_BLOCKS", kv_floor_default),
            headroom=_env_float("WISP_DYN_HEADROOM", 0.15),
            deadzone_cap=_env_int("WISP_DYN_DEADZONE", 1),
        )
    except Exception as e:
        logger.warning("wisp.serve_hook: controller init failed (%s); disabled", e)
        return st

    st.enabled = True
    st.controller = controller
    st.handles = handles
    st.scheduler = engine_core.scheduler
    st.block_pool = handles.block_pool
    st.min_interval_s = _env_float("WISP_DYN_MIN_INTERVAL_S", 0.0)
    st.do_log = _truthy("WISP_DYN_LOG", "1")
    st.log_file = os.environ.get("WISP_DYN_LOG_FILE", "").strip()
    st.epoch_total = int(st.block_pool.num_gpu_blocks)
    st.epoch_min_free = st.epoch_total
    logger.info(
        "wisp.serve_hook: live MV-WSA controller armed in EngineCore "
        "(layers=%d, cap=%s, kv_blocks=%d, cap_min=%d cap_max=%d floor=%d headroom=%.2f)",
        fm.num_registered_layers(),
        fm.current_layer_cap(),
        st.epoch_total,
        controller.cap_min,
        controller.cap_max,
        controller.kv_floor_blocks,
        controller.headroom,
    )
    return st


def _on_step(engine_core) -> None:
    """Per-step hook: sample KV occupancy, and at a drained barrier re-solve and
    apply the expert<->KV split. Must be exception-free — a control problem must
    never take down the serving loop."""
    st = getattr(engine_core, _STATE_ATTR, None)
    if st is None:
        st = _init_state(engine_core)
        setattr(engine_core, _STATE_ATTR, st)
    if not st.enabled:
        return

    try:
        free = int(st.block_pool.get_num_free_blocks())
    except Exception:
        return
    if free < st.epoch_min_free:
        st.epoch_min_free = free

    # KV-pressure signal: if any request is queued in `waiting` while others run,
    # the KV pool is too small to admit the offered load -> observed usage
    # *understates* true KV demand (the pool caps it). Without this, sizing KV to
    # observed peak is a one-way ratchet (KV can shrink but never grow back). On
    # pressure we treat the pool as saturated so the controller grows KV.
    try:
        if st.scheduler.waiting and len(st.scheduler.waiting) > 0:
            st.epoch_pressure = True
    except Exception:
        pass

    try:
        has_req = bool(st.scheduler.has_requests())
    except Exception:
        return

    # Drained-barrier transition: the scheduler just emptied. No request is in
    # flight, the GPU is idle -> the one safe point to physically resize.
    if st.had_requests and not has_req:
        st.n_drains += 1
        peak_used = max(0, st.epoch_total - st.epoch_min_free)
        if st.epoch_pressure:
            # Unmet KV demand this epoch -> drive KV up toward (a bit beyond) the
            # full pool so the equimarginal step reclaims expert bytes for KV.
            peak_used = st.epoch_total
        now = time.monotonic()
        if st.min_interval_s <= 0.0 or (now - st.last_fire) >= st.min_interval_s:
            try:
                d = st.controller.step(peak_used)
            except Exception as e:  # never crash the serve loop on a control move
                logger.warning("wisp.serve_hook: controller.step raised (%s); skipped", e)
                d = None
            if d is not None:
                st.last_fire = now
                if d.applied:
                    st.n_resizes += 1
                if st.do_log and (d.applied or st.n_drains <= 3):
                    logger.info(
                        "wisp.serve_hook: drain#%d kv_peak=%d/%d -> cap %d->%d, "
                        "kv %d->%d (%s)",
                        st.n_drains, peak_used, st.epoch_total,
                        d.cap_from, d.cap_to, d.kv_from, d.kv_to, d.reason,
                    )
                if st.log_file:
                    _append_log(st.log_file, {
                        "drain": st.n_drains,
                        "kv_peak_used": peak_used,
                        "kv_total": st.epoch_total,
                        "pressure": st.epoch_pressure,
                        "cap_from": d.cap_from, "cap_to": d.cap_to,
                        "kv_from": d.kv_from, "kv_to": d.kv_to,
                        "applied": d.applied, "reason": d.reason,
                        "t": now,
                    })
        # New epoch: re-read total (it may have changed if we just resized).
        try:
            st.epoch_total = int(st.block_pool.num_gpu_blocks)
        except Exception:
            pass
        st.epoch_min_free = st.epoch_total
        st.epoch_pressure = False

    st.had_requests = has_req


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_wisp_serve_controller() -> bool:
    """Idempotently patch ``EngineCoreProc._process_engine_step`` so the live
    MV-WSA controller runs inside the ``vllm serve`` busy loop. Returns True iff
    this call applied the patch. Safe to call in every process (the patched
    method only ever runs in the EngineCore subprocess)."""
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception as e:  # pragma: no cover
        logger.warning("wisp.serve_hook: cannot import EngineCoreProc (%s); not installed", e)
        return False

    if getattr(EngineCoreProc, _PATCHED_FLAG, False):
        return False

    EngineCoreProc._wisp_original_process_engine_step = (
        EngineCoreProc._process_engine_step
    )

    def _patched_process_engine_step(self):
        executed = type(self)._wisp_original_process_engine_step(self)
        try:
            _on_step(self)
        except Exception as e:  # pragma: no cover - belt and suspenders
            logger.debug("wisp.serve_hook: _on_step failed (%s)", e)
        return executed

    EngineCoreProc._process_engine_step = _patched_process_engine_step
    setattr(EngineCoreProc, _PATCHED_FLAG, True)
    logger.info("wisp.serve_hook: patched EngineCoreProc._process_engine_step (live MV-WSA)")
    return True


def is_serve_controller_installed() -> bool:
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception:
        return False
    return bool(getattr(EngineCoreProc, _PATCHED_FLAG, False))


__all__ = ["install_wisp_serve_controller", "is_serve_controller_installed"]
