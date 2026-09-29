"""Reusable full-catalog temporal evaluation protocol."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable

import pandas as pd

from evaluation.candidates import full_catalog_candidates
from evaluation.ranking import (
    mrr_at_k,
    ndcg_at_k,
    rank_by_score,
    recall_at_k,
)


ScoreFunction = Callable[
    [Iterable[Hashable]],
    dict[Hashable, float],
]


def build_user_histories(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
) -> dict[Hashable, set[Hashable]]:
    """Build observed histories used to filter candidates."""
    history: dict[Hashable, set[Hashable]] = defaultdict(set)

    for row in train.itertuples(index=False):
        history[row.user_id].add(row.movie_id)

    if validation is not None:
        for row in validation.itertuples(index=False):
            history[row.user_id].add(row.movie_id)

    return dict(history)


def held_out_groups(
    interactions: pd.DataFrame,
) -> dict[Hashable, set[Hashable]]:
    """Collect one held-out timestamp group per user."""
    groups: dict[Hashable, set[Hashable]] = {}

    for user_id, frame in interactions.groupby(
        "user_id",
        sort=False,
    ):
        timestamps = frame["timestamp"].unique()

        if len(timestamps) != 1:
            raise ValueError(
                f"user {user_id!r} has {len(timestamps)} "
                "held-out timestamps; expected exactly one"
            )

        groups[user_id] = set(frame["movie_id"].tolist())

    return groups


def evaluate_full_catalog(
    *,
    catalog_items: Iterable[Hashable],
    history_by_user: dict[Hashable, set[Hashable]],
    relevant_by_user: dict[Hashable, set[Hashable]],
    score_items: ScoreFunction,
    k: int = 10,
) -> dict[str, float | int]:
    """Evaluate under the frozen full-catalog contract."""
    if k < 1:
        raise ValueError("k must be >= 1")

    if not relevant_by_user:
        raise ValueError("relevant_by_user must not be empty")

    catalog = tuple(catalog_items)

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for user_id, relevant in relevant_by_user.items():
        candidates = full_catalog_candidates(
            catalog,
            history_by_user.get(user_id, set()),
            relevant,
        )

        scores = score_items(candidates)

        if set(scores) != set(candidates):
            raise ValueError(
                "score_items must return exactly one score "
                "per candidate"
            )

        ranked = rank_by_score(scores)

        recalls.append(
            recall_at_k(ranked, relevant, k)
        )
        mrrs.append(
            mrr_at_k(ranked, relevant, k)
        )
        ndcgs.append(
            ndcg_at_k(ranked, relevant, k)
        )

    users = len(relevant_by_user)

    return {
        "users": users,
        f"recall@{k}": sum(recalls) / users,
        f"mrr@{k}": sum(mrrs) / users,
        f"ndcg@{k}": sum(ndcgs) / users,
    }
