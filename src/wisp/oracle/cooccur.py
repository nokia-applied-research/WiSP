"""Expert co-activation table + cooccurrence-based predictor.

User insight (S1.7 brainstorm):

    "Layer L's router takes layer L-1's hidden state as input. So an
    expert's activation at layer L is determined by the hidden state,
    which is itself a function of the experts that fired in layers <
    L. Therefore P(expert e at L | experts seen at layers < L) is a
    non-uniform distribution we can estimate from a corpus of step
    records, and use to prefetch layer L while layer L-1 is still
    computing."

This module implements the table + the predictor in two layers:

- :class:`CooccurTable` — append-only Counter-of-Counters. Sparse
  storage; updates O(per-step expert-count squared) which is ~150k
  ops on Qwen3-30B-A3B (negligible). Used both offline (build from a
  recorded corpus) and online (refine during replay or live serving).
- :func:`predict_layer` — given a partial "what's been activated in
  layers 0..L-1 of this step?" plus the table, return a ranked list
  of expert ids predicted for a target layer L.

The within-step lookahead (predict layer L from layers 0..L-1 in the
SAME forward pass) is the strong signal — it directly reflects the
hidden-state Markov chain. Cross-step / cross-turn extension lives in
:mod:`wisp.oracle.prefetch_cooccur`.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Iterable, Mapping, Union

if TYPE_CHECKING:  # offline trace types; not needed by the runtime plug-in path
    from ..trace import StepRecord, Trace


@dataclass
class CooccurTable:
    """Per-layer-pair expert co-activation counts.

    Two equivalent ways to read the table:

    - **Logical**:
      ``count_logical[(L_src, e_src)][(L_dst, e_dst)] = N`` means: in
      ``N`` distinct forward-pass step records, expert ``e_src`` fired
      at layer ``L_src`` AND expert ``e_dst`` fired at layer ``L_dst``
      (with ``L_dst > L_src``).

    - **Physical (post-S2.1c refactor)**:
      ``count[(L_src, e_src)][L_dst][e_dst] = N``. Nested dict so
      ``predict_layer(target_layer=L_dst)`` is a single O(1) dict
      lookup instead of the prior O(rows_per_src) linear scan that
      filtered on ``L_dst``. ~50× speedup on Qwen3-30B-A3B's
      48-layer table per S2.1b's perf-risk note.

    ``layer_total[(L, e)] = N`` is the marginal — how many step
    records saw expert ``e`` at layer ``L``. Used to normalise
    ``count`` into a conditional probability when scoring.
    """

    count: dict[tuple[int, int], dict[int, Counter]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    layer_total: dict[tuple[int, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    num_steps_observed: int = 0

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def observe_step(self, expert_ids_by_layer: Mapping[int, Iterable[int]]) -> None:
        """Update from one step's per-layer activation lists.

        Only counts cross-layer pairs (``L_dst > L_src``). Same-layer
        pairs would also be informative for "co-activation" but they
        are not useful for within-token *lookahead* (you see them
        simultaneously, not sequentially).
        """
        # Materialise to lists once so we can iterate twice.
        per_layer = {
            L: list(eids) for L, eids in expert_ids_by_layer.items() if eids
        }
        # Marginals
        for L_src, src_experts in per_layer.items():
            for e_src in src_experts:
                self.layer_total[(L_src, e_src)] += 1
        # Pairs — nested by L_dst so predict_layer can skip irrelevant
        # destinations in one hash lookup.
        for L_src, src_experts in per_layer.items():
            for e_src in src_experts:
                per_L_dst = self.count[(L_src, e_src)]
                for L_dst, dst_experts in per_layer.items():
                    if L_dst <= L_src:
                        continue
                    row = per_L_dst[L_dst]
                    for e_dst in dst_experts:
                        row[e_dst] += 1
        self.num_steps_observed += 1

    def observe_trace(self, trace: Trace, *, decode_only: bool = True) -> None:
        """Update from every step in ``trace``.

        ``decode_only=True`` skips the prefill record (where layer L
        accumulates ~``8 * prompt_len`` experts and the
        within-token-lookahead semantics break down).
        """
        for s in trace.steps:
            if decode_only and s.step_idx < 0:
                continue
            self.observe_step(s.expert_ids_by_layer)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_source_keys(self) -> int:
        return len(self.count)

    @property
    def n_pairs(self) -> int:
        """Total ``(L_src, e_src, L_dst, e_dst)`` quadruples observed
        (= sum of Counter sizes across all destinations)."""
        return sum(
            sum(len(row) for row in per_L.values())
            for per_L in self.count.values()
        )

    def memory_estimate_bytes(self) -> int:
        """Very rough: 40 bytes per Counter entry under CPython 3.11."""
        return self.n_pairs * 40

    # ------------------------------------------------------------------
    # Serialisation.
    # ------------------------------------------------------------------
    #
    # The nested ``count[(L_src, e_src)][L_dst][e_dst]`` Counter is
    # not directly JSON-able, so we flatten to a list of records:
    # ``[L_src, e_src, L_dst, e_dst, n]``. ``layer_total`` flattens
    # to ``[L, e, n]``. With Qwen3-30B-A3B's 48 layers × 128 experts
    # × top-k=8 the pair count is bounded at ~150k after a few
    # hundred steps; serialisation is ~5 MB un-gzipped, ~500 KB
    # gzipped — small enough to ship next to the workload script.

    SCHEMA_VERSION = 1

    def to_dict(self) -> dict:
        pairs: list[list[int]] = []
        for (L_src, e_src), per_L_dst in self.count.items():
            for L_dst, row in per_L_dst.items():
                for e_dst, n in row.items():
                    if n > 0:
                        pairs.append([int(L_src), int(e_src), int(L_dst), int(e_dst), int(n)])
        marginals: list[list[int]] = []
        for (L, e), n in self.layer_total.items():
            if n > 0:
                marginals.append([int(L), int(e), int(n)])
        return {
            "schema_version": self.SCHEMA_VERSION,
            "num_steps_observed": int(self.num_steps_observed),
            "pairs": pairs,
            "marginals": marginals,
            "stats": {
                "n_source_keys": self.n_source_keys,
                "n_pairs": self.n_pairs,
                "memory_estimate_bytes": self.memory_estimate_bytes(),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CooccurTable":
        version = int(data.get("schema_version", 1))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"CooccurTable.from_dict: schema_version={version} "
                f"unsupported (expected {cls.SCHEMA_VERSION})"
            )
        t = cls()
        for row in data.get("pairs") or []:
            L_src, e_src, L_dst, e_dst, n = row  # type: ignore[misc]
            t.count[(int(L_src), int(e_src))][int(L_dst)][int(e_dst)] = int(n)
        for row in data.get("marginals") or []:
            L, e, n = row  # type: ignore[misc]
            t.layer_total[(int(L), int(e))] = int(n)
        t.num_steps_observed = int(data.get("num_steps_observed", 0))
        return t

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Persist to ``path``. ``.json.gz`` triggers gzip; anything
        else writes plain JSON.
        """
        spath = os.fspath(path)
        os.makedirs(os.path.dirname(spath) or ".", exist_ok=True)
        payload = self.to_dict()
        text = json.dumps(payload, separators=(",", ":"))
        if spath.endswith(".gz"):
            with gzip.open(spath, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(spath, "w", encoding="utf-8") as f:
                f.write(text)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "CooccurTable":
        spath = os.fspath(path)
        if spath.endswith(".gz"):
            with gzip.open(spath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(spath, encoding="utf-8") as f:
                data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


def predict_layer(
    table: CooccurTable,
    seen_so_far: Mapping[int, Iterable[int]],
    target_layer: int,
    *,
    k: int = 8,
    normalize: bool = True,
    decay: float = 1.0,
    only_prev_layer: bool = False,
) -> list[int]:
    """Predict top-``k`` expert ids at ``target_layer``.

    Parameters
    ----------
    seen_so_far
        ``layer_idx -> list[expert_id]`` for layers that have already
        fired in the current step. Only entries with ``layer_idx <
        target_layer`` are used as conditioning.
    normalize
        If True, divide pair counts by ``layer_total[(L_src, e_src)]``
        — i.e. score by conditional probability ``P(e_dst|e_src)``
        rather than raw co-occurrence count. Strongly recommended;
        prevents popular ``e_src`` from dominating.
    decay
        Multiplicative weight ``decay ** (target_layer - L_src - 1)``
        applied to contributions from earlier layers. ``decay < 1``
        emphasises recent layers (closer in depth ≈ closer in hidden-
        state evolution); ``decay = 1`` weighs all conditioning
        equally. Empirically the first few layers below the target
        carry most of the signal so ``decay ∈ [0.3, 0.7]`` often beats
        ``1.0``.
    only_prev_layer
        Hard-truncate the conditioning to just ``target_layer - 1``.
        Ablation: tests whether multi-layer lookback adds anything.

    Returns
    -------
    list[int]
        Up to ``k`` expert ids, ranked highest-score-first. Empty list
        if no conditioning experts have any rows in the table (cold
        start).
    """
    scores: defaultdict[int, float] = defaultdict(float)
    for L_src, src_experts in seen_so_far.items():
        if L_src >= target_layer:
            continue
        if only_prev_layer and L_src != target_layer - 1:
            continue
        dist = target_layer - L_src
        w = decay ** (dist - 1)
        for e_src in src_experts:
            per_L_dst = table.count.get((L_src, e_src))
            if not per_L_dst:
                continue
            row = per_L_dst.get(target_layer)
            if not row:
                continue
            total = table.layer_total.get((L_src, e_src), 0) if normalize else 1
            denom = total if (normalize and total > 0) else 1
            for e_dst, c in row.items():
                scores[e_dst] += w * c / denom
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [e for e, _ in ranked[:k]]


def predict_marginal(
    table: CooccurTable, target_layer: int, num_experts: int, k: int
) -> list[int]:
    """Top-``k`` most-frequent experts at ``target_layer`` (no
    conditioning). Equivalent to ``history_top_k`` but per-layer."""
    counts = [
        (table.layer_total.get((target_layer, e), 0), e)
        for e in range(num_experts)
    ]
    counts.sort(reverse=True)
    return [e for _, e in counts[:k]]


__all__ = [
    "CooccurTable",
    "predict_layer",
    "predict_marginal",
]
