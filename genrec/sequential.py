"""Leakage-safe sequential recommendation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SequenceExample:
    user_id: int
    history: tuple[int, ...]
    targets: tuple[int, ...]
    timestamp: int


def build_sequence_examples(
    interactions: pd.DataFrame,
    *,
    max_history: int | None = None,
    min_history: int = 1,
) -> list[SequenceExample]:
    """Build next-timestamp-group examples.

    Items at the same timestamp are treated as unordered. A target group
    can use only interactions from strictly earlier timestamps.
    """
    required = {
        "user_id",
        "movie_id",
        "timestamp",
    }
    if not required.issubset(interactions.columns):
        raise ValueError(
            "interactions must contain user_id, movie_id, timestamp"
        )

    if min_history < 1:
        raise ValueError("min_history must be >= 1")

    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be >= 1")

    examples: list[SequenceExample] = []

    ordered = interactions.sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )

    for user_id, user_frame in ordered.groupby(
        "user_id",
        sort=False,
    ):
        history: list[int] = []

        for timestamp, group in user_frame.groupby(
            "timestamp",
            sort=True,
        ):
            targets = tuple(
                int(movie_id)
                for movie_id in group["movie_id"].tolist()
            )

            if len(history) >= min_history:
                context = history

                if max_history is not None:
                    context = history[-max_history:]

                examples.append(
                    SequenceExample(
                        user_id=int(user_id),
                        history=tuple(context),
                        targets=targets,
                        timestamp=int(timestamp),
                    )
                )

            # Critical leakage rule: add the complete timestamp group
            # only after its prediction example has been constructed.
            history.extend(targets)

    return examples


@dataclass(frozen=True)
class ItemVocabulary:
    """Map raw movie IDs to model indices.

    Index 0 is reserved for padding.
    """

    item_to_index: dict[int, int]
    index_to_item: dict[int, int]

    @classmethod
    def from_items(cls, items) -> "ItemVocabulary":
        unique = sorted({int(item) for item in items})

        item_to_index = {
            item: index
            for index, item in enumerate(
                unique,
                start=1,
            )
        }

        index_to_item = {
            index: item
            for item, index in item_to_index.items()
        }

        return cls(
            item_to_index=item_to_index,
            index_to_item=index_to_item,
        )

    @property
    def size(self) -> int:
        """Embedding-table size including padding."""
        return len(self.item_to_index) + 1

    def encode(self, item_id: int) -> int:
        try:
            return self.item_to_index[int(item_id)]
        except KeyError as exc:
            raise ValueError(
                f"unknown item_id: {item_id}"
            ) from exc

    def decode(self, index: int) -> int:
        if index == 0:
            raise ValueError(
                "padding index does not map to an item"
            )

        try:
            return self.index_to_item[int(index)]
        except KeyError as exc:
            raise ValueError(
                f"unknown item index: {index}"
            ) from exc


def encode_history(
    history,
    vocabulary: ItemVocabulary,
    *,
    max_length: int,
) -> tuple[int, ...]:
    """Left-pad the most recent history to fixed length."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    encoded = [
        vocabulary.encode(item_id)
        for item_id in history[-max_length:]
    ]

    padding = [0] * (
        max_length - len(encoded)
    )

    return tuple(padding + encoded)
