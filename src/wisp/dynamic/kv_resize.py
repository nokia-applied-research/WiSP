"""Physically resize the vLLM v1 GPU KV pool at a drained barrier.

Two coordinated mutations, both done in-process between engine steps when
the scheduler is fully drained (no running/waiting request, every block
free):

1. **Worker KV tensors** — vLLM allocates one ``int8`` buffer per layer of
   ``num_blocks * page_size_bytes`` and exposes it as a reshaped view bound
   to each attention layer. We drop the old buffers (clear the runner list
   and each attention layer's ``.kv_cache``), shrink/grow ``num_blocks`` in
   the cached ``kv_cache_config``, and re-run vLLM's own
   ``initialize_kv_cache_tensors`` (allocate + reshape + ``bind_kv_cache``).
   Freeing before allocating keeps the transient peak at ``max(old, new)``.

2. **Scheduler block pool** — rebuilt in place (it is a single shared
   object referenced by the coordinator and every single-type manager). We
   reuse the existing ``null_block`` *object identity* because the
   single-type managers cached it at construction.

The freed bytes (on KV shrink) are returned to the allocator via
``torch.cuda.empty_cache`` so the paired expert-scratch grow can claim
them; on KV grow the caller must have shrunk experts first.
"""

from __future__ import annotations

from typing import Any

import torch


def kv_page_bytes(handles) -> int:
    """Total GPU bytes of one KV *block*, summed across all layers.

    A logical KV block reserves one slot in *every* layer's KV tensor, so its
    real cost is the sum of all per-layer page sizes, not just one layer's.
    (Using one layer's page here under-counts KV by the layer count and
    starves the controller's budget math.) ``kv_cache_tensors`` lists one
    entry per distinct physical buffer (shared layers appear once), so summing
    their sizes gives the total KV bytes; dividing by num_blocks gives the
    per-block cost the controller trades against expert slots."""
    cfg = handles.model_runner.kv_cache_config
    old_num = int(cfg.num_blocks)
    if old_num <= 0 or not cfg.kv_cache_tensors:
        raise RuntimeError("wisp.dynamic: empty KV cache config")
    total = sum(int(t.size) for t in cfg.kv_cache_tensors)
    assert total % old_num == 0, "total KV size not a multiple of num_blocks"
    return total // old_num


def current_kv_blocks(handles) -> int:
    return int(handles.block_pool.num_gpu_blocks)


def _assert_drained(handles) -> None:
    bp = handles.block_pool
    free = bp.get_num_free_blocks()
    # null_block is permanently popped, so a drained pool has num-1 free.
    if free < bp.num_gpu_blocks - 1:
        raise RuntimeError(
            f"wisp.dynamic: KV resize requires a drained pool "
            f"(free={free}, total={bp.num_gpu_blocks}); "
            "run only between fully-finished batches."
        )


def _free_worker_kv_tensors(mr) -> None:
    """Drop every reference to the current KV buffers so the allocator can
    reclaim them before we allocate the new ones."""
    fctx = mr.compilation_config.static_forward_context
    for layer in fctx.values():
        if hasattr(layer, "kv_cache"):
            try:
                layer.kv_cache = []
            except Exception:
                pass
    # ``self.kv_caches`` is the runner list bind_kv_cache appends into; it
    # must be empty for bind_kv_cache's assertion to pass.
    try:
        mr.kv_caches.clear()
    except Exception:
        mr.kv_caches = []
    torch.cuda.empty_cache()


def _realloc_worker_kv_tensors(mr, new_num_blocks: int) -> None:
    cfg = mr.kv_cache_config
    old_num = int(cfg.num_blocks)
    if new_num_blocks == old_num:
        return
    _free_worker_kv_tensors(mr)
    cfg.num_blocks = int(new_num_blocks)
    for t in cfg.kv_cache_tensors:
        page = int(t.size) // old_num
        t.size = page * int(new_num_blocks)
    kernel_block_sizes = mr._prepare_kernel_block_sizes(cfg)
    mr.initialize_kv_cache_tensors(cfg, kernel_block_sizes)


def _rebuild_block_pool(bp, new_num_blocks: int) -> None:
    """Rebuild the shared block pool for ``new_num_blocks``, preserving the
    ``null_block`` object identity that single-type managers cached."""
    from vllm.v1.core.block_pool import BlockHashToBlockMap
    from vllm.v1.core.kv_cache_utils import (
        FreeKVCacheBlockQueue,
        KVCacheBlock,
    )

    null = bp.null_block
    null.prev_free_block = None
    null.next_free_block = None
    null.ref_cnt = 0
    new_blocks = [KVCacheBlock(idx) for idx in range(new_num_blocks)]
    new_blocks[0] = null  # keep identity for managers' cached _null_block
    bp.blocks = new_blocks
    bp.num_gpu_blocks = int(new_num_blocks)
    bp.free_block_queue = FreeKVCacheBlockQueue(new_blocks)
    popped = bp.free_block_queue.popleft()
    assert popped is null, "wisp.dynamic: null_block identity lost on rebuild"
    null.is_null = True
    bp.cached_block_hash_to_block = BlockHashToBlockMap()
    bp.kv_event_queue = []


def resize_kv_blocks(handles, new_num_blocks: int) -> dict[str, Any]:
    """Resize the GPU KV pool to ``new_num_blocks``. Drained-barrier only.

    Returns a small telemetry dict. On any failure the function attempts to
    leave the engine in its prior, usable state and re-raises, so the live
    controller can simply skip the move.
    """
    new_num_blocks = int(new_num_blocks)
    if new_num_blocks < 2:
        raise ValueError("new_num_blocks must be >= 2")
    _assert_drained(handles)
    mr = handles.model_runner
    bp = handles.block_pool
    old = int(bp.num_gpu_blocks)
    if new_num_blocks == old:
        return {"old": old, "new": old, "changed": False}

    if new_num_blocks < old:
        # Shrink: rebuild pool first (scheduler stops handing high ids),
        # then free KV bytes so experts can grow into them.
        _rebuild_block_pool(bp, new_num_blocks)
        _realloc_worker_kv_tensors(mr, new_num_blocks)
    else:
        # Grow: experts must already be shrunk by the caller; allocate the
        # bigger KV tensors first, then expose the new blocks.
        _realloc_worker_kv_tensors(mr, new_num_blocks)
        _rebuild_block_pool(bp, new_num_blocks)

    if handles.cache_config is not None:
        try:
            handles.cache_config.num_gpu_blocks = new_num_blocks
        except Exception:
            pass
    torch.cuda.empty_cache()
    return {"old": old, "new": new_num_blocks, "changed": True}
