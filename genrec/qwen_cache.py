"""Utilities for frozen Qwen history-representation caches."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
POOLING = "final-layer-final-query-token"
CACHE_DTYPE = np.float32


def example_id(example: dict[str, Any]) -> str:
    """Return a stable identity for one manifest example."""
    payload = {
        "user_id": int(example["user_id"]),
        "timestamp": int(example["timestamp"]),
        "history": [int(x) for x in example["history"]],
        "targets": [int(x) for x in example["targets"]],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_representations(
    representations: np.ndarray,
    *,
    expected_rows: int,
    hidden_size: int = 1024,
) -> None:
    """Validate a frozen representation matrix before persistence."""
    expected = (expected_rows, hidden_size)

    if representations.shape != expected:
        raise ValueError(
            f"expected representation shape {expected}, "
            f"got {representations.shape}"
        )

    if representations.dtype != CACHE_DTYPE:
        raise ValueError(
            f"expected dtype {CACHE_DTYPE}, "
            f"got {representations.dtype}"
        )

    if not np.isfinite(representations).all():
        raise ValueError("representations contain non-finite values")


def validation_histories(
    train,
    validation,
    *,
    min_items: int = 50,
) -> dict[int, list[int]]:
    """Build validation inference histories from train interactions only."""
    from genrec.semantic import (
        recent_complete_groups,
        timestamp_group_histories,
    )

    groups = timestamp_group_histories(train)
    users = sorted(
        validation["user_id"].astype(int).unique()
    )

    missing = set(users) - set(groups)
    if missing:
        raise ValueError(
            f"{len(missing)} validation users have no train history"
        )

    return {
        user_id: recent_complete_groups(
            groups[user_id],
            min_items=min_items,
        )
        for user_id in users
    }


def test_histories(
    train,
    validation,
    test,
    *,
    min_items: int = 50,
) -> dict[int, list[int]]:
    """Build test inference histories from train + validation only."""
    import pandas as pd

    from genrec.semantic import (
        recent_complete_groups,
        timestamp_group_histories,
    )

    observed = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    groups = timestamp_group_histories(observed)
    users = sorted(test["user_id"].astype(int).unique())

    missing = set(users) - set(groups)
    if missing:
        raise ValueError(
            f"{len(missing)} test users have no observed history"
        )

    return {
        user_id: recent_complete_groups(
            groups[user_id],
            min_items=min_items,
        )
        for user_id in users
    }
