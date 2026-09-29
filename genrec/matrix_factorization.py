"""Implicit-feedback matrix factorization with BPR training."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BPRMatrixFactorization:
    """CPU-friendly Bayesian Personalized Ranking matrix factorization."""

    def __init__(
        self,
        factors: int = 32,
        learning_rate: float = 0.05,
        regularization: float = 0.0025,
        epochs: int = 10,
        seed: int = 42,
    ) -> None:
        if factors < 1 or epochs < 1:
            raise ValueError("factors and epochs must be >= 1")

        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed

        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame) -> "BPRMatrixFactorization":
        required = {"user_id", "movie_id"}
        if not required.issubset(interactions.columns):
            raise ValueError("interactions must contain user_id and movie_id")

        users = sorted(interactions["user_id"].unique())
        items = sorted(interactions["movie_id"].unique())

        self.user_to_index = {u: i for i, u in enumerate(users)}
        self.item_to_index = {m: i for i, m in enumerate(items)}

        rng = np.random.default_rng(self.seed)

        self.user_factors = rng.normal(
            0.0, 0.1, (len(users), self.factors)
        )
        self.item_factors = rng.normal(
            0.0, 0.1, (len(items), self.factors)
        )

        positives = {
            self.user_to_index[user]: {
                self.item_to_index[item]
                for item in frame["movie_id"]
                if item in self.item_to_index
            }
            for user, frame in interactions.groupby("user_id")
        }

        pairs = [
            (self.user_to_index[row.user_id], self.item_to_index[row.movie_id])
            for row in interactions.itertuples(index=False)
        ]

        all_items = np.arange(len(items))

        for _ in range(self.epochs):
            order = rng.permutation(len(pairs))

            for pair_index in order:
                user_index, positive_index = pairs[pair_index]

                if len(positives[user_index]) == len(items):
                    continue

                while True:
                    negative_index = int(rng.choice(all_items))
                    if negative_index not in positives[user_index]:
                        break

                user_vector = self.user_factors[user_index].copy()
                positive_vector = self.item_factors[positive_index].copy()
                negative_vector = self.item_factors[negative_index].copy()

                x = float(
                    user_vector
                    @ (positive_vector - negative_vector)
                )
                gradient = 1.0 / (1.0 + np.exp(np.clip(x, -35.0, 35.0)))

                self.user_factors[user_index] += self.learning_rate * (
                    gradient * (positive_vector - negative_vector)
                    - self.regularization * user_vector
                )
                self.item_factors[positive_index] += self.learning_rate * (
                    gradient * user_vector
                    - self.regularization * positive_vector
                )
                self.item_factors[negative_index] += self.learning_rate * (
                    -gradient * user_vector
                    - self.regularization * negative_vector
                )

        return self

    def score(self, user_id: int, movie_id: int) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("model must be fit before scoring")

        user_index = self.user_to_index.get(user_id)
        item_index = self.item_to_index.get(movie_id)

        if user_index is None or item_index is None:
            return 0.0

        return float(
            self.user_factors[user_index]
            @ self.item_factors[item_index]
        )

    def scores(self, user_id: int, movie_ids) -> dict[int, float]:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("model must be fit before scoring")

        movie_ids = list(movie_ids)
        user_index = self.user_to_index.get(user_id)

        if user_index is None:
            return {movie_id: 0.0 for movie_id in movie_ids}

        known_positions = []
        known_indices = []

        for position, movie_id in enumerate(movie_ids):
            item_index = self.item_to_index.get(movie_id)
            if item_index is not None:
                known_positions.append(position)
                known_indices.append(item_index)

        values = np.zeros(len(movie_ids), dtype=float)

        if known_indices:
            values[known_positions] = (
                self.item_factors[known_indices]
                @ self.user_factors[user_index]
            )

        return {
            movie_id: float(value)
            for movie_id, value in zip(movie_ids, values)
        }
