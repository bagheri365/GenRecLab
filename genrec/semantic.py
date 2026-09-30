"""Frozen semantic representations for recommendation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def movie_text(title: str, genres: str) -> str:
    """Canonical text representation for a MovieLens item."""
    genre_text = ", ".join(genres.split("|"))
    return f"Title: {title}\nGenres: {genre_text}"


def mean_history_embedding(
    history: Iterable[int],
    embeddings: dict[int, np.ndarray],
    *,
    max_history: int = 50,
) -> np.ndarray:
    """Mean-pool the most recent known item embeddings."""
    if max_history < 1:
        raise ValueError("max_history must be >= 1")

    recent = list(history)[-max_history:]
    vectors = [
        embeddings[item_id]
        for item_id in recent
        if item_id in embeddings
    ]

    if not vectors:
        raise ValueError("history has no known embeddings")

    pooled = np.mean(vectors, axis=0)
    norm = np.linalg.norm(pooled)

    if norm == 0.0:
        raise ValueError("history embedding has zero norm")

    return pooled / norm


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """L2-normalize an embedding matrix row-wise."""
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0.0):
        raise ValueError("embedding contains zero-norm row")

    return embeddings / norms


class FrozenSemanticRecommender:
    """Parameter-free recommender over frozen item embeddings."""

    def __init__(
        self,
        movie_ids: np.ndarray,
        embeddings: np.ndarray,
        *,
        max_history: int = 50,
    ) -> None:
        movie_ids = np.asarray(movie_ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if movie_ids.ndim != 1:
            raise ValueError("movie_ids must be 1D")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(movie_ids) != len(embeddings):
            raise ValueError("movie_ids and embeddings must align")
        if len(set(movie_ids.tolist())) != len(movie_ids):
            raise ValueError("movie_ids must be unique")
        if max_history < 1:
            raise ValueError("max_history must be >= 1")

        self.movie_ids = movie_ids
        self.embeddings = normalize_embeddings(embeddings)
        self.max_history = max_history
        self._index = {
            int(movie_id): index
            for index, movie_id in enumerate(movie_ids)
        }

    @classmethod
    def from_npz(
        cls,
        path: str,
        *,
        max_history: int = 50,
    ) -> "FrozenSemanticRecommender":
        data = np.load(path)
        return cls(
            data["movie_ids"],
            data["embeddings"],
            max_history=max_history,
        )

    def user_embedding(
        self,
        history: Iterable[int],
        *,
        truncate: bool = True,
    ) -> np.ndarray:
        items = list(history)
        if truncate:
            items = items[-self.max_history:]

        indices = [
            self._index[int(item_id)]
            for item_id in items
            if int(item_id) in self._index
        ]

        if not indices:
            raise ValueError("history has no known embeddings")

        pooled = self.embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(pooled)

        if norm == 0.0:
            raise ValueError("history embedding has zero norm")

        return pooled / norm

    def score_user_items(
        self,
        history: Iterable[int],
        item_ids: Iterable[int],
        *,
        truncate: bool = True,
    ) -> dict[int, float]:
        user = self.user_embedding(
            history,
            truncate=truncate,
        )
        scores: dict[int, float] = {}

        for item_id in item_ids:
            item_id = int(item_id)
            index = self._index.get(item_id)
            if index is None:
                scores[item_id] = 0.0
            else:
                scores[item_id] = float(
                    self.embeddings[index] @ user
                )

        return scores



def timestamp_group_histories(
    interactions,
) -> dict[int, list[list[int]]]:
    """Build chronological histories without ordering timestamp ties."""
    required = {"user_id", "movie_id", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(
            f"interactions must contain {sorted(required)}"
        )

    result: dict[int, list[list[int]]] = {}

    for user_id, frame in interactions.groupby(
        "user_id",
        sort=False,
    ):
        groups = []
        for _, group in frame.groupby(
            "timestamp",
            sort=True,
        ):
            groups.append(
                sorted(group["movie_id"].astype(int).tolist())
            )
        result[int(user_id)] = groups

    return result


def recent_complete_groups(
    groups: Iterable[Iterable[int]],
    *,
    min_items: int = 50,
) -> list[int]:
    """Select recent complete timestamp groups covering min_items."""
    if min_items < 1:
        raise ValueError("min_items must be >= 1")

    groups = [list(group) for group in groups]
    selected: list[list[int]] = []
    count = 0

    for group in reversed(groups):
        selected.append(group)
        count += len(group)
        if count >= min_items:
            break

    return [
        item
        for group in reversed(selected)
        for item in group
    ]
