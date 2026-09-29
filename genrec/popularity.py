"""Popularity recommender baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable

import pandas as pd


class PopularityRecommender:
    """Rank items by interaction frequency in the training set."""

    def __init__(self) -> None:
        self._counts: Counter[Hashable] = Counter()

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        if "movie_id" not in interactions.columns:
            raise ValueError("interactions must contain movie_id")

        self._counts = Counter(interactions["movie_id"].tolist())
        return self

    def score(self, movie_id: Hashable) -> float:
        return float(self._counts.get(movie_id, 0))

    def scores(
        self,
        movie_ids: Iterable[Hashable],
    ) -> dict[Hashable, float]:
        return {
            movie_id: self.score(movie_id)
            for movie_id in movie_ids
        }

    @property
    def counts(self) -> dict[Hashable, int]:
        return dict(self._counts)
