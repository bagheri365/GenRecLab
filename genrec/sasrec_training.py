"""Training utilities for the MLX SASRec baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genrec.sequential import (
    ItemVocabulary,
    SequenceExample,
    encode_history,
)


@dataclass(frozen=True)
class TrainingRow:
    history: tuple[int, ...]
    positive: int
    negatives: tuple[int, ...]


def sample_training_rows(
    examples: list[SequenceExample],
    vocabulary: ItemVocabulary,
    *,
    max_length: int,
    num_negatives: int,
    seed: int,
) -> list[TrainingRow]:
    """Expand target groups and sample deterministic negatives."""
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    rng = np.random.default_rng(seed)
    all_items = np.array(
        sorted(vocabulary.item_to_index),
        dtype=np.int64,
    )

    rows: list[TrainingRow] = []

    for example in examples:
        excluded = set(example.history)
        excluded.update(example.targets)

        available = len(all_items) - len(
            excluded.intersection(
                vocabulary.item_to_index
            )
        )

        if available < num_negatives:
            raise ValueError(
                "not enough negative items for requested sample"
            )

        encoded_history = encode_history(
            example.history,
            vocabulary,
            max_length=max_length,
        )

        for positive in example.targets:
            negatives: list[int] = []
            chosen: set[int] = set()

            while len(negatives) < num_negatives:
                item = int(
                    all_items[
                        rng.integers(len(all_items))
                    ]
                )

                if item in excluded or item in chosen:
                    continue

                chosen.add(item)
                negatives.append(item)

            rows.append(
                TrainingRow(
                    history=encoded_history,
                    positive=vocabulary.encode(positive),
                    negatives=tuple(
                        vocabulary.encode(item)
                        for item in negatives
                    ),
                )
            )

    return rows

import mlx.core as mx
import mlx.nn as nn


def sampled_softmax_loss(
    model,
    histories,
    candidates,
):
    """Cross-entropy where candidate column zero is positive."""
    user = model(histories)

    # candidates: (batch, 1 + negatives)
    item_vectors = model.item_embedding(candidates)

    logits = mx.sum(
        user[:, None, :] * item_vectors,
        axis=-1,
    )

    targets = mx.zeros(
        (histories.shape[0],),
        dtype=mx.int32,
    )

    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )


def batch_arrays(rows: list[TrainingRow]):
    """Convert training rows into MLX batch arrays."""
    if not rows:
        raise ValueError("rows must not be empty")

    histories = mx.array(
        [row.history for row in rows],
        dtype=mx.int32,
    )

    candidates = mx.array(
        [
            (row.positive,) + row.negatives
            for row in rows
        ],
        dtype=mx.int32,
    )

    return histories, candidates

import mlx.optimizers as optim


class SASRecTrainer:
    """Persistent optimizer state for epoch-wise training."""

    def __init__(
        self,
        model,
        *,
        learning_rate: float,
        weight_decay: float,
    ):
        self.model = model
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_and_grad = nn.value_and_grad(
            model,
            sampled_softmax_loss,
        )

    def train_epoch(
        self,
        examples: list[SequenceExample],
        vocabulary: ItemVocabulary,
        *,
        max_length: int,
        batch_size: int,
        num_negatives: int,
        seed: int,
    ) -> float:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        total_loss = 0.0
        batches = 0

        for histories, candidates in iter_training_batches(
            examples,
            vocabulary,
            max_length=max_length,
            batch_size=batch_size,
            num_negatives=num_negatives,
            seed=seed,
        ):
            loss, gradients = self.loss_and_grad(
                self.model,
                histories,
                candidates,
            )

            self.optimizer.update(
                self.model,
                gradients,
            )

            mx.eval(
                loss,
                self.model.parameters(),
                self.optimizer.state,
            )

            total_loss += float(loss.item())
            batches += 1

        if batches == 0:
            raise ValueError(
                "training produced no batches"
            )

        return total_loss / batches


def train_sasrec(
    model,
    examples: list[SequenceExample],
    vocabulary: ItemVocabulary,
    *,
    max_length: int,
    epochs: int,
    batch_size: int,
    num_negatives: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    """Train SASRec with streaming negative sampling."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    trainer = SASRecTrainer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    return [
        trainer.train_epoch(
            examples,
            vocabulary,
            max_length=max_length,
            batch_size=batch_size,
            num_negatives=num_negatives,
            seed=seed + epoch,
        )
        for epoch in range(epochs)
    ]

def iter_training_batches(
    examples: list[SequenceExample],
    vocabulary: ItemVocabulary,
    *,
    max_length: int,
    batch_size: int,
    num_negatives: int,
    seed: int,
):
    """Yield MLX batches without materializing an epoch of rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    rng = np.random.default_rng(seed)
    all_items = np.array(
        sorted(vocabulary.item_to_index),
        dtype=np.int64,
    )

    # Shuffle timestamp-group examples, then positives within groups
    # remain unordered evidence rather than sequential events.
    order = rng.permutation(len(examples))

    histories: list[tuple[int, ...]] = []
    candidates: list[tuple[int, ...]] = []

    for example_index in order:
        example = examples[int(example_index)]

        excluded = set(example.history)
        excluded.update(example.targets)

        available = len(all_items) - len(
            excluded.intersection(
                vocabulary.item_to_index
            )
        )

        if available < num_negatives:
            raise ValueError(
                "not enough negative items for requested sample"
            )

        encoded = encode_history(
            example.history,
            vocabulary,
            max_length=max_length,
        )

        target_order = rng.permutation(
            len(example.targets)
        )

        for target_index in target_order:
            positive = example.targets[int(target_index)]

            negatives: list[int] = []
            chosen: set[int] = set()

            while len(negatives) < num_negatives:
                item = int(
                    all_items[
                        rng.integers(len(all_items))
                    ]
                )

                if item in excluded or item in chosen:
                    continue

                chosen.add(item)
                negatives.append(item)

            histories.append(encoded)
            candidates.append(
                (
                    vocabulary.encode(positive),
                    *(
                        vocabulary.encode(item)
                        for item in negatives
                    ),
                )
            )

            if len(histories) == batch_size:
                yield (
                    mx.array(
                        histories,
                        dtype=mx.int32,
                    ),
                    mx.array(
                        candidates,
                        dtype=mx.int32,
                    ),
                )
                histories = []
                candidates = []

    if histories:
        yield (
            mx.array(histories, dtype=mx.int32),
            mx.array(candidates, dtype=mx.int32),
        )
