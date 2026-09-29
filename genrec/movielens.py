"""MovieLens 1M loading, splitting, and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import urllib.request
import zipfile

import pandas as pd

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def read_ratings(path: str | Path) -> pd.DataFrame:
    """Read MovieLens 1M ratings.dat."""
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=RATING_COLUMNS,
        encoding="latin-1",
    )
    return frame.astype(
        {"user_id": "int64", "movie_id": "int64", "rating": "int64", "timestamp": "int64"}
    )


def read_movies(path: str | Path) -> pd.DataFrame:
    """Read MovieLens 1M movies.dat."""
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=MOVIE_COLUMNS,
        encoding="latin-1",
    )
    frame["movie_id"] = frame["movie_id"].astype("int64")
    return frame


def load_movielens_1m(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata from an extracted ml-1m directory."""
    directory = Path(directory)
    ratings_path = directory / "ratings.dat"
    movies_path = directory / "movies.dat"
    missing = [str(p) for p in (ratings_path, movies_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files: {missing}")
    return read_ratings(ratings_path), read_movies(movies_path)


def temporal_leave_last_split(
    interactions: pd.DataFrame,
    *,
    validation_items: int = 1,
    test_items: int = 1,
    min_train_items: int = 3,
) -> TemporalSplit:
    """Split each user using complete timestamp groups.

    Each held-out unit is one distinct timestamp group, not one interaction.
    Keeping ties together preserves strict temporal separation without
    inventing an order among interactions recorded at the same time.
    """
    if validation_items != 1 or test_items != 1:
        raise ValueError(
            "timestamp-group splitting currently requires "
            "validation_items=test_items=1"
        )
    if min_train_items < 1:
        raise ValueError("min_train_items must be >= 1")

    required = {"user_id", "movie_id", "timestamp"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = interactions.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="stable"
    ).reset_index(drop=True)

    train_parts, validation_parts, test_parts = [], [], []

    for _, group in ordered.groupby("user_id", sort=False):
        timestamps = group["timestamp"].drop_duplicates().sort_values().tolist()
        if len(timestamps) < 3:
            continue

        validation_timestamp = timestamps[-2]
        test_timestamp = timestamps[-1]

        train = group[group["timestamp"] < validation_timestamp]
        validation = group[group["timestamp"] == validation_timestamp]
        test = group[group["timestamp"] == test_timestamp]

        if len(train) < min_train_items:
            continue

        train_parts.append(train)
        validation_parts.append(validation)
        test_parts.append(test)

    if not train_parts:
        empty = ordered.iloc[0:0].copy()
        return TemporalSplit(empty, empty.copy(), empty.copy())

    return TemporalSplit(
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )

def dataset_statistics(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    split: TemporalSplit | None = None,
) -> dict[str, int | float]:
    """Return compact, JSON-serializable dataset statistics."""
    stats: dict[str, int | float] = {
        "ratings": int(len(ratings)),
        "users": int(ratings["user_id"].nunique()),
        "rated_movies": int(ratings["movie_id"].nunique()),
        "catalog_movies": int(movies["movie_id"].nunique()),
        "min_timestamp": int(ratings["timestamp"].min()),
        "max_timestamp": int(ratings["timestamp"].max()),
    }
    if split is not None:
        stats.update(
            train_interactions=int(len(split.train)),
            validation_interactions=int(len(split.validation)),
            test_interactions=int(len(split.test)),
            split_users=int(split.test["user_id"].nunique()),
        )
    return stats


def download_movielens_1m(destination: str | Path) -> Path:
    """Download and safely extract MovieLens 1M into destination."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-1m.zip"

    urllib.request.urlretrieve(MOVIELENS_1M_URL, archive)

    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe zip member: {member.filename}")
        zf.extractall(destination)

    return destination / "ml-1m"


def prepare_movielens_1m(
    source: str | Path,
    output: str | Path,
    *,
    min_train_items: int = 3,
) -> dict[str, int | float]:
    """Load, split, and write normalized CSV artifacts plus statistics."""
    ratings, movies = load_movielens_1m(source)
    split = temporal_leave_last_split(ratings, min_train_items=min_train_items)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(output / "train.csv", index=False)
    split.validation.to_csv(output / "validation.csv", index=False)
    split.test.to_csv(output / "test.csv", index=False)
    movies.to_csv(output / "movies.csv", index=False)

    stats = dataset_statistics(ratings, movies, split)
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
