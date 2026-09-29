"""Leakage-safe interaction ordering and next-item example construction."""
from dataclasses import dataclass
from typing import Hashable
import pandas as pd

REQUIRED_COLUMNS = {"user_id", "movie_id", "timestamp"}

@dataclass(frozen=True)
class NextItemExample:
    user_id: Hashable
    history_movie_ids: tuple[Hashable, ...]
    history_timestamps: tuple[int, ...]
    target_movie_id: Hashable
    target_timestamp: int

def normalize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = interactions.copy()
    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("user_id, movie_id, and timestamp must not contain nulls")
    return frame.sort_values(["user_id", "timestamp", "movie_id"], kind="stable").reset_index(drop=True)

def build_next_item_examples(interactions: pd.DataFrame, *, min_history: int = 1) -> list[NextItemExample]:
    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    frame = normalize_interactions(interactions)
    examples = []
    for user_id, group in frame.groupby("user_id", sort=False):
        rows = list(group.itertuples(index=False))
        for index in range(min_history, len(rows)):
            target = rows[index]
            safe_history = [row for row in rows[:index] if row.timestamp < target.timestamp]
            if len(safe_history) < min_history:
                continue
            examples.append(NextItemExample(
                user_id=user_id,
                history_movie_ids=tuple(row.movie_id for row in safe_history),
                history_timestamps=tuple(int(row.timestamp) for row in safe_history),
                target_movie_id=target.movie_id,
                target_timestamp=int(target.timestamp),
            ))
    return examples
