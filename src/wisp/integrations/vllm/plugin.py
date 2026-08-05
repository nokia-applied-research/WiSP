"""Entry-point shim for the ``vllm.general_plugins`` discovery group.

vLLM auto-imports any function registered under the
``vllm.general_plugins`` entry-point group at engine bring-up. Critically,
this happens **in every process** that vLLM spawns — the parent
``LLM`` process, the engine-core subprocess, and each worker — and
*before* the model class is imported. That's exactly the order we need
for the WiSP MoE plug-in: ``UnquantizedFusedMoEMethod.create_weights``
must be patched before the first ``FusedMoE.__init__`` runs, otherwise
expert weights land on the GPU and the consumer-card load-time VRAM
ceiling (the whole point of Plan A) is blown.

Usage after ``pip install -e .`` of the wisp package::

    # Python API — patch happens automatically when vllm imports.
    from vllm import LLM
    llm = LLM(model="Qwen/Qwen3-30B-A3B", enforce_eager=True)

    # OpenAI HTTP server — same, no extra flags needed:
    # $ vllm serve Qwen/Qwen3-30B-A3B --enforce-eager \\
    #       --gpu-memory-utilization 0.25 --max-model-len 4096
    # On a 24 GiB GPU this would have OOM'd without the plug-in; with
    # it, the model loads in ~10 GiB and serves over HTTP.

Configuration:

- ``WISP_MODE``: ``copy`` | ``resident`` | ``paged`` (default: ``paged``).
  (Legacy aliases ``day1`` / ``day2a`` / ``day2b`` are still accepted.)
- ``WISP_CAP_EXPERTS``: int, scratch size for paged mode
  (default: ``min(num_experts, 24)``).
- ``WISP_PLUGIN_DISABLE``: set to ``1`` to leave the plug-in inert.
  Useful for A/B benchmarking on the same vLLM install without
  uninstalling the package.

The function is registered via the entry point declared in
``pyproject.toml`` (``[project.entry-points."vllm.general_plugins"]``).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register() -> None:
    """vLLM plug-in entry point.

    Idempotent — vLLM calls plug-in entry points multiple times across
    process boundaries (parent + engine core + workers), and warns plug-in
    authors to handle that explicitly. We forward to
    ``install_wisp_moe()`` which already guards on a class attribute
    on ``UnquantizedFusedMoEMethod``.
    """
    disabled = os.environ.get("WISP_PLUGIN_DISABLE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if disabled:
        logger.info(
            "wisp.vllm.plugin: WISP_PLUGIN_DISABLE=1 — skipping patch"
        )
        return

    # Import lazily so a bare ``pip install wisp`` on a torch-less box
    # doesn't crash when something else in the host environment
    # imports ``vllm.plugins`` for discovery.
    try:
        from .fused_moe import install_wisp_moe, is_installed
    except ImportError as e:  # pragma: no cover
        logger.warning(
            "wisp.vllm.plugin: import failed (%s); plug-in not installed. "
            "This usually means the host process has no torch / vllm. "
            "Discovery is silent — no functionality is lost.",
            e,
        )
        return

    if is_installed():
        logger.debug(
            "wisp.vllm.plugin: already patched in this process; nothing to do"
        )
        return

    install_wisp_moe()
    logger.info(
        "wisp.vllm.plugin: patched vLLM via entry point "
        "(WISP_MODE=%s WISP_CAP_EXPERTS=%s WISP_PREFETCH=%s)",
        os.environ.get("WISP_MODE", "paged"),
        os.environ.get("WISP_CAP_EXPERTS", "auto"),
        os.environ.get("WISP_PREFETCH", "0"),
    )

    # Optionally arm the live MV-WSA controller inside the (multi-process)
    # `vllm serve` EngineCore loop. Best-effort: a failure here must never
    # break the MoE paging patch above. The patched busy-loop method only
    # ever runs in the EngineCore subprocess; installing it in other
    # processes is a harmless no-op.
    if os.environ.get("WISP_DYNAMIC", "0").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from wisp.dynamic.serve_hook import install_wisp_serve_controller

            install_wisp_serve_controller()
            # The effective cap/floor bounds are resolved later, inside the
            # EngineCore process when the controller is built (cap_max ->
            # num_experts, kv_floor -> ceil(max_model_len/block) when unset);
            # only echo explicit overrides here to avoid implying a value.
            logger.info(
                "wisp.vllm.plugin: live MV-WSA serve controller armed "
                "(WISP_DYNAMIC=1; cap_min=%s cap_max=%s kv_floor_blocks=%s headroom=%s)",
                os.environ.get("WISP_DYN_CAP_MIN", "8 (default)"),
                os.environ.get("WISP_DYN_CAP_MAX", "num_experts (default)"),
                os.environ.get("WISP_DYN_KV_FLOOR_BLOCKS", "auto (default)"),
                os.environ.get("WISP_DYN_HEADROOM", "0.15 (default)"),
            )
        except Exception as e:  # pragma: no cover
            logger.warning(
                "wisp.vllm.plugin: live serve controller not armed (%s); "
                "static paging unaffected",
                e,
            )
