"""Ranking head for frozen M4 Qwen representations."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class FrozenQwenRankingHead(nn.Module):
    """Project frozen user representations into item-score space."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_items: int,
        dims: int = 64,
    ):
        super().__init__()
        if hidden_size < 1 or num_items < 1 or dims < 1:
            raise ValueError("model dimensions must be positive")

        self.user_projection = nn.Linear(
            hidden_size,
            dims,
            bias=False,
        )
        self.item_embeddings = nn.Embedding(
            num_items,
            dims,
        )

    def __call__(
        self,
        representations,
        item_indices,
    ):
        users = self.user_projection(representations)
        items = self.item_embeddings(item_indices)
        return mx.sum(
            mx.expand_dims(users, axis=1) * items,
            axis=-1,
        )


def load_training_targets(
    manifest: dict,
    example_ids: np.ndarray,
    *,
    example_id_fn,
) -> list[tuple[int, ...]]:
    """Align manifest targets to cached representation rows."""
    by_id = {
        example_id_fn(example): tuple(
            int(x) for x in example["targets"]
        )
        for example in manifest["examples"]
    }

    targets = []
    for raw_id in example_ids.tolist():
        key = str(raw_id)
        if key not in by_id:
            raise ValueError(
                f"cache example id not found in manifest: {key}"
            )
        targets.append(by_id[key])

    return targets
