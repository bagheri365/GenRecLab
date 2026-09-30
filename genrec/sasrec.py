"""Compact SASRec-style sequential recommender using MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        max_length: int,
        dims: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        mlp_dims: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_items < 1:
            raise ValueError("num_items must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if dims % num_heads != 0:
            raise ValueError("dims must be divisible by num_heads")

        self.num_items = num_items
        self.max_length = max_length
        self.dims = dims

        # Index 0 is padding.
        self.item_embedding = nn.Embedding(
            num_items + 1,
            dims,
        )
        self.position_embedding = nn.Embedding(
            max_length,
            dims,
        )

        self.encoder = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=dims,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            norm_first=True,
        )

        self.output_norm = nn.LayerNorm(dims)

    def _attention_mask(self, item_indices):
        batch_size, length = item_indices.shape

        causal = (
            nn.MultiHeadAttention
            .create_additive_causal_mask(length)
        )

        # Prevent attention to padding keys.
        valid_keys = item_indices != 0
        padding = mx.where(
            valid_keys,
            mx.array(0.0),
            mx.array(-mx.inf),
        )
        padding = padding[:, None, None, :]

        return causal[None, None, :, :] + padding

    def encode(self, item_indices):
        if item_indices.ndim != 2:
            raise ValueError(
                "item_indices must have shape (batch, sequence)"
            )

        batch_size, length = item_indices.shape

        if length != self.max_length:
            raise ValueError(
                f"expected sequence length {self.max_length}, "
                f"got {length}"
            )

        positions = mx.arange(length)
        x = (
            self.item_embedding(item_indices)
            + self.position_embedding(positions)[None, :, :]
        )

        # Remove content/position signal from padded query slots.
        valid = (item_indices != 0)[..., None]
        x = mx.where(valid, x, mx.zeros_like(x))

        mask = self._attention_mask(item_indices)
        hidden = self.encoder(x, mask)
        hidden = self.output_norm(hidden)

        return hidden

    def __call__(self, item_indices):
        """Return one user representation per history."""
        if mx.any(item_indices[:, -1] == 0).item():
            raise ValueError(
                "each history must end with a non-padding item"
            )

        hidden = self.encode(item_indices)
        return hidden[:, -1, :]

    def score_items(self, item_indices, candidate_indices):
        """Score candidate item indices for each history.

        candidate_indices has shape (candidates,) and returns
        shape (batch, candidates).
        """
        user = self(item_indices)
        items = self.item_embedding(candidate_indices)

        return user @ items.T
