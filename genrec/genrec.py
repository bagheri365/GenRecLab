"""Data contracts for LM-native recommendation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class GenRecExample:
    user_id: int
    history: tuple[int, ...]
    targets: tuple[int, ...]
    timestamp: int


def build_genrec_examples(
    interactions: pd.DataFrame,
    *,
    min_history: int = 50,
) -> list[GenRecExample]:
    """Build leakage-safe examples using complete timestamp groups."""
    if min_history < 1:
        raise ValueError("min_history must be >= 1")

    required = {"user_id", "movie_id", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(
            f"interactions must contain {sorted(required)}"
        )

    ordered = interactions.sort_values(
        ["user_id", "timestamp", "movie_id"],
        kind="stable",
    )

    examples: list[GenRecExample] = []

    for user_id, frame in ordered.groupby("user_id", sort=False):
        groups = [
            (
                int(timestamp),
                tuple(sorted(group["movie_id"].astype(int))),
            )
            for timestamp, group in frame.groupby(
                "timestamp",
                sort=True,
            )
        ]

        history_groups: list[tuple[int, ...]] = []
        history_count = 0

        for timestamp, targets in groups:
            if history_count >= min_history:
                selected: list[tuple[int, ...]] = []
                selected_count = 0

                for group in reversed(history_groups):
                    selected.append(group)
                    selected_count += len(group)
                    if selected_count >= min_history:
                        break

                history = tuple(
                    item
                    for group in reversed(selected)
                    for item in group
                )

                examples.append(
                    GenRecExample(
                        user_id=int(user_id),
                        history=history,
                        targets=targets,
                        timestamp=timestamp,
                    )
                )

            history_groups.append(targets)
            history_count += len(targets)

    return examples


def sample_user_balanced(
    examples: Iterable[GenRecExample],
    *,
    n: int,
    seed: int = 42,
) -> list[GenRecExample]:
    """Sample examples in deterministic user-balanced rounds."""
    examples = list(examples)

    if n < 1:
        raise ValueError("n must be >= 1")
    if n > len(examples):
        raise ValueError("n cannot exceed number of examples")

    by_user: dict[int, list[GenRecExample]] = defaultdict(list)

    for example in examples:
        by_user[example.user_id].append(example)

    rng = random.Random(seed)

    for user_examples in by_user.values():
        rng.shuffle(user_examples)

    selected: list[GenRecExample] = []
    round_index = 0

    while len(selected) < n:
        users = [
            user_id
            for user_id, user_examples in by_user.items()
            if round_index < len(user_examples)
        ]

        if not users:
            break

        rng.shuffle(users)

        for user_id in users:
            selected.append(by_user[user_id][round_index])
            if len(selected) == n:
                break

        round_index += 1

    return selected


def history_text(
    history: Iterable[int],
    movies: pd.DataFrame,
) -> str:
    """Serialize a movie history using the frozen M4-v1 template."""
    required = {"movie_id", "title", "genres"}
    if not required.issubset(movies.columns):
        raise ValueError(
            f"movies must contain {sorted(required)}"
        )

    metadata = movies.set_index("movie_id")
    lines = ["User movie history:"]

    for position, movie_id in enumerate(history, start=1):
        try:
            row = metadata.loc[int(movie_id)]
        except KeyError as exc:
            raise ValueError(
                f"unknown movie_id: {movie_id}"
            ) from exc

        genres = ", ".join(str(row["genres"]).split("|"))
        lines.append(
            f"{position}. {row['title']} | {genres}"
        )

    lines.extend(
        [
            "",
            "Represent this user's movie preferences.",
        ]
    )

    return "\n".join(lines)