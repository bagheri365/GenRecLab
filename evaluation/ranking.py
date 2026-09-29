"""Ranking metrics for single- and multi-positive recommendation evaluation."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def _relevant_set(
    relevant_items: Hashable | Iterable[Hashable],
) -> set[Hashable]:
    """Normalize single- or multi-positive relevance into a set.

    The original GenRecLab metric API accepted a single item ID, while the
    multi-positive protocol accepts an iterable of item IDs. Supporting both
    preserves backward compatibility.
    """
    if isinstance(relevant_items, (str, bytes)):
        relevant = {relevant_items}
    else:
        try:
            relevant = set(relevant_items)
        except TypeError:
            relevant = {relevant_items}

    if not relevant:
        raise ValueError("relevant_items must contain at least one item")

    return relevant


def recall_at_k(
    ranked_items: Sequence[Hashable],
    relevant_items: Iterable[Hashable],
    k: int,
) -> float:
    """Fraction of relevant items retrieved in the first k positions."""
    _validate_k(k)
    relevant = _relevant_set(relevant_items)
    hits = len(relevant.intersection(ranked_items[:k]))
    return hits / len(relevant)


def mrr_at_k(
    ranked_items: Sequence[Hashable],
    relevant_items: Iterable[Hashable],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant item in the first k positions."""
    _validate_k(k)
    relevant = _relevant_set(relevant_items)

    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_items: Sequence[Hashable],
    relevant_items: Iterable[Hashable],
    k: int,
) -> float:
    """Binary-relevance NDCG@k supporting multiple relevant items."""
    _validate_k(k)
    relevant = _relevant_set(relevant_items)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked_items[:k], start=1)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg


def rank_by_score(
    scores: dict[Hashable, float],
) -> list[Hashable]:
    """Rank descending by score with deterministic item-id tie breaking.

    MovieLens item IDs are integers, but repr-based tie breaking keeps this
    helper deterministic for other hashable identifiers used in tests.
    """
    return [
        item
        for item, _ in sorted(
            scores.items(),
            key=lambda pair: (-pair[1], repr(pair[0])),
        )
    ]
