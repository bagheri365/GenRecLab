"""Training utilities for the M4 frozen-Qwen ranking head."""

from __future__ import annotations

import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def sample_training_batch(
    targets,
    item_to_index,
    *,
    catalog_items,
    batch_indices,
    num_negatives: int,
    seed: int,
):
    """Build deterministic positive/negative item indices."""
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    catalog = tuple(int(x) for x in catalog_items)
    rng = random.Random(seed)
    rows = []

    for row_index in batch_indices:
        positives = tuple(
            int(x) for x in targets[int(row_index)]
        )
        if not positives:
            raise ValueError("training target group is empty")

        positive = rng.choice(positives)
        positive_set = set(positives)
        candidates = [
            item for item in catalog
            if item not in positive_set
        ]

        if num_negatives > len(candidates):
            raise ValueError(
                "num_negatives exceeds available items"
            )

        negatives = rng.sample(
            candidates,
            num_negatives,
        )
        rows.append(
            [
                item_to_index[positive],
                *[
                    item_to_index[item]
                    for item in negatives
                ],
            ]
        )

    return np.asarray(rows, dtype=np.int32)


class RankingHeadTrainer:
    def __init__(
        self,
        model,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        def loss_fn(model, reps, items):
            scores = model(reps, items)
            labels = mx.zeros(
                (scores.shape[0],),
                dtype=mx.int32,
            )
            return mx.mean(
                nn.losses.cross_entropy(
                    scores,
                    labels,
                )
            )

        self._loss_and_grad = nn.value_and_grad(
            model,
            loss_fn,
        )

    def train_epoch(
        self,
        representations: np.ndarray,
        targets,
        item_to_index,
        *,
        catalog_items,
        batch_size: int = 256,
        num_negatives: int = 20,
        seed: int = 42,
    ) -> float:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        order = list(range(len(representations)))
        random.Random(seed).shuffle(order)

        total = 0.0
        batches = 0

        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            item_indices = sample_training_batch(
                targets,
                item_to_index,
                catalog_items=catalog_items,
                batch_indices=indices,
                num_negatives=num_negatives,
                seed=seed + start,
            )

            reps = mx.array(
                representations[indices],
                dtype=mx.float32,
            )
            items = mx.array(
                item_indices,
                dtype=mx.int32,
            )

            loss, grads = self._loss_and_grad(
                self.model,
                reps,
                items,
            )
            self.optimizer.update(
                self.model,
                grads,
            )
            mx.eval(
                self.model.parameters(),
                self.optimizer.state,
                loss,
            )

            total += float(loss.item())
            batches += 1

        return total / batches
