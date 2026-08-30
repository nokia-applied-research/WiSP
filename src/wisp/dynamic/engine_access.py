"""Locate the in-process vLLM v1 internals the live controller mutates.

The live MV-WSA controller needs three handles out of an offline
``vllm.LLM`` object:

* the **GPU model runner** (owns ``kv_caches`` + the KV-tensor (re)alloc
  methods and the static forward context that binds tensors to attention
  layers);
* the **scheduler block pool** (owns ``num_gpu_blocks`` + the free-block
  doubly-linked list the scheduler hands out);
* the **cache config** (so ``num_gpu_blocks`` stays consistent for metrics).

These only exist in the same process when the v1 EngineCore is *not* run
in a subprocess, i.e. ``VLLM_ENABLE_V1_MULTIPROCESSING=0``. We discover the
objects by a small bounded BFS over ``__dict__`` rather than hard-coding an
attribute path, so the locator survives minor vLLM refactors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _dig(root: Any, path: str):
    """Follow a dotted attribute path; return None if any hop is missing."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_platform(obj: Any) -> bool:
    # vLLM's current_platform proxy returns None (not AttributeError) from
    # __getattr__, so it spuriously satisfies hasattr() for any attribute.
    # Never descend into or match it.
    return "Platform" in type(obj).__name__


def _bfs_find(root: Any, predicate, max_depth: int = 7, max_nodes: int = 6000):
    """Return the first reachable object satisfying ``predicate``.

    Walks attribute graphs (object ``__dict__`` and list/tuple/dict values)
    breadth-first from ``root``. Avoids cycles via an id-set and skips
    vLLM platform proxies. Returns ``None`` if nothing matches.
    """
    seen: set[int] = set()
    frontier: list[tuple[Any, int]] = [(root, 0)]
    visited = 0
    while frontier:
        obj, depth = frontier.pop(0)
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        visited += 1
        if visited > max_nodes:
            break
        if _is_platform(obj):
            continue
        try:
            if predicate(obj):
                return obj
        except Exception:
            pass
        if depth >= max_depth:
            continue
        children: list[Any] = []
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            children.extend(d.values())
        if isinstance(obj, (list, tuple)):
            children.extend(obj)
        elif isinstance(obj, dict):
            children.extend(obj.values())
        for c in children:
            if c is None or isinstance(c, (int, float, str, bytes, bool)):
                continue
            if id(c) not in seen:
                frontier.append((c, depth + 1))
    return None


def _looks_like_runner(o: Any) -> bool:
    return (
        not _is_platform(o)
        and callable(getattr(o, "initialize_kv_cache_tensors", None))
        and isinstance(getattr(o, "kv_caches", None), list)
        and getattr(o, "kv_cache_config", None) is not None
    )


def _looks_like_block_pool(o: Any) -> bool:
    return (
        not _is_platform(o)
        and isinstance(getattr(o, "num_gpu_blocks", None), int)
        and getattr(o, "num_gpu_blocks", 0) > 0
        and isinstance(getattr(o, "blocks", None), list)
        and getattr(o, "free_block_queue", None) is not None
    )


@dataclass
class EngineHandles:
    """Live references into a single-process vLLM v1 engine."""

    model_runner: Any
    block_pool: Any
    cache_config: Any
    kv_cache_manager: Any

    @property
    def num_gpu_blocks(self) -> int:
        return int(self.block_pool.num_gpu_blocks)


def locate_handles(llm: Any) -> EngineHandles:
    """Find the live engine internals inside an offline ``vllm.LLM``.

    Raises a clear error if the EngineCore is running in a subprocess
    (the handles are then simply not reachable in this process).
    """
    engine = getattr(llm, "llm_engine", llm)

    # Fast path: the stable v1 in-process attribute layout.
    model_runner = _dig(engine, "model_executor.driver_worker.worker.model_runner")
    if not _looks_like_runner(model_runner):
        model_runner = _bfs_find(engine, _looks_like_runner)

    kv_cache_manager = _dig(
        engine, "engine_core.engine_core.scheduler.kv_cache_manager"
    )
    block_pool = getattr(kv_cache_manager, "block_pool", None)
    if not _looks_like_block_pool(block_pool):
        block_pool = _bfs_find(engine, _looks_like_block_pool)
    if kv_cache_manager is None:
        kv_cache_manager = _bfs_find(
            engine, lambda o: getattr(o, "block_pool", None) is block_pool
        )

    cache_config = getattr(
        getattr(model_runner, "vllm_config", None), "cache_config", None
    )

    if model_runner is None or block_pool is None:
        raise RuntimeError(
            "wisp.dynamic: could not locate the in-process model_runner / "
            "block_pool. The live dual-dynamic split needs the v1 EngineCore "
            "in *this* process. Launch with VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "and an offline vllm.LLM (not the API server)."
        )
    return EngineHandles(
        model_runner=model_runner,
        block_pool=block_pool,
        cache_config=cache_config,
        kv_cache_manager=kv_cache_manager,
    )
