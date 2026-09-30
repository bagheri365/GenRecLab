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
