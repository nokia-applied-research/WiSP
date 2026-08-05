"""Offline routing co-occurrence used by the speculative-prefetch path.

The public v1 release ships only the pieces the vLLM plug-in actually
imports: the layer-conditioned co-occurrence table that backs the
(intentionally negative) prefetch experiments in the paper. The
offline upper-bound oracles (Belady, multi-turn replay) used for the
analysis figures are not part of this release.
"""

from .cooccur import CooccurTable, predict_layer, predict_marginal

__all__ = [
    "CooccurTable",
    "predict_layer",
    "predict_marginal",
]
