"""Evaluate the frozen, validation-selected M4 model on test once."""

from __future__ import annotations

import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import pandas as pd

from evaluation.protocol import (
    build_user_histories,
    evaluate_full_catalog,
    held_out_groups,
)
from genrec.m4_checkpoint import (
    SELECTED_CONFIG,
    SELECTED_VALIDATION,
    load_ranking_head,
)
from genrec.qwen_cache import (
    MODEL_ID,
    POOLING,
    file_sha256,
    validate_representations,
)


def load_test_cache(path):
    """Load and validate frozen test representations."""
    with np.load(path) as cache:
        representations = cache["representations"].copy()
        user_ids = cache["user_ids"].copy()

    validate_representations(
        representations,
        expected_rows=len(user_ids),
    )
    if len(set(user_ids.tolist())) != len(user_ids):
        raise ValueError("test cache contains duplicate user_ids")
    return representations, user_ids


def validate_selection_provenance(path) -> dict:
    """Require a validation-only frozen model-selection record."""
    payload = json.loads(Path(path).read_text())
    if payload.get("selection_split") != "validation_only":
        raise ValueError(
            "M4 checkpoint was not selected on validation only"
        )
    if payload.get("test_used_for_selection") is not False:
        raise ValueError(
            "selection provenance does not exclude test selection"
        )
    if payload.get("selected_config") != SELECTED_CONFIG:
        raise ValueError("selected configuration mismatch")
    if payload.get("selected_validation") != SELECTED_VALIDATION:
        raise ValueError("selected validation metrics mismatch")
    return payload


def validate_test_cache_provenance(
    path,
    *,
    cache_path,
    train_path,
    validation_path,
    test_path,
) -> dict:
    """Verify the frozen test cache and its data provenance."""
    payload = json.loads(Path(path).read_text())
    expected = {
        "contract": "m4-frozen-qwen-v1",
        "split": "test",
        "model_id": MODEL_ID,
        "pooling": POOLING,
        "model_parameters": "frozen",
        "history_source": "train_plus_validation",
        "history_min_items": 50,
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "test_sha256": file_sha256(test_path),
        "cache_sha256": file_sha256(cache_path),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"test cache provenance mismatch for {key}"
            )
    return payload


def make_scorer(
    model,
    *,
    representations,
    user_ids,
    item_to_index,
):
    """Build the full-catalog scorer for frozen test contexts."""
    row_by_user = {
        int(user_id): row
        for row, user_id in enumerate(user_ids.tolist())
    }

    def score_user_items(user_id, movie_ids):
        movie_ids = list(movie_ids)
        row = row_by_user[int(user_id)]
        indices = mx.array(
            [[item_to_index[int(x)] for x in movie_ids]],
            dtype=mx.int32,
        )
        rep = mx.array(
            representations[row : row + 1],
            dtype=mx.float32,
        )
        scores = model(rep, indices)[0]
        mx.eval(scores)
        values = np.array(scores.astype(mx.float32)).tolist()
        return {
            movie_id: float(value)
            for movie_id, value in zip(movie_ids, values)
        }

    return score_user_items


def main() -> None:
    data_dir = Path("data/processed/ml-1m")
    qwen_dir = Path("artifacts/m4/qwen")
    checkpoint_dir = Path("results/checkpoints")

    train_path = data_dir / "train.csv"
    validation_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    movies_path = data_dir / "movies.csv"
    cache_path = qwen_dir / "test_full.npz"
    cache_provenance_path = (
        qwen_dir / "test_full_provenance.json"
    )
    checkpoint_path = checkpoint_dir / "m4_selected.safetensors"
    selection_path = (
        checkpoint_dir / "m4_selected_provenance.json"
    )

    validate_selection_provenance(selection_path)
    cache_provenance = validate_test_cache_provenance(
        cache_provenance_path,
        cache_path=cache_path,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
    )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    movies = pd.read_csv(movies_path)
    representations, user_ids = load_test_cache(cache_path)

    expected_users = sorted(
        test["user_id"].astype(int).unique().tolist()
    )
    cached_users = user_ids.astype(int).tolist()
    if cached_users != expected_users:
        raise ValueError(
            "test cache user_ids do not match test users"
        )
    if cache_provenance.get("contexts") != len(user_ids):
        raise ValueError("test cache context count mismatch")

    catalog = tuple(
        movies["movie_id"].astype(int).tolist()
    )
    item_to_index = {
        movie_id: index
        for index, movie_id in enumerate(catalog)
    }

    model = load_ranking_head(
        checkpoint_path,
        hidden_size=representations.shape[1],
        num_items=len(catalog),
        dims=SELECTED_CONFIG["dims"],
    )
    scorer = make_scorer(
        model,
        representations=representations,
        user_ids=user_ids,
        item_to_index=item_to_index,
    )

    observed = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    started = time.perf_counter()
    test_metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=build_user_histories(observed),
        relevant_by_user=held_out_groups(test),
        score_user_items=scorer,
        k=10,
    )
    elapsed = time.perf_counter() - started

    result = {
        "experiment": "m4_frozen_qwen_ranking_head",
        "evaluation": "final_test",
        "selection_split": "validation_only",
        "test_used_for_selection": False,
        "model": {
            "encoder": MODEL_ID,
            "encoder_parameters": "frozen",
            "pooling": POOLING,
            "head": "linear_projection_dot_product",
            **SELECTED_CONFIG,
        },
        "selected_validation": SELECTED_VALIDATION,
        "protocol": {
            "split": "timestamp_group_v0.2",
            "candidates": "full_catalog_excluding_seen",
            "relevance": "multi_positive_timestamp_group",
            "evaluation_split": "test_only",
            "history_source": "train_plus_validation",
            "k": 10,
        },
        "test": test_metrics,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "test_cache": str(cache_path),
            "test_cache_sha256": file_sha256(cache_path),
            "selection_provenance": str(selection_path),
            "test_cache_provenance": str(
                cache_provenance_path
            ),
        },
        "runtime": {
            "evaluation_seconds": elapsed,
        },
    }

    output = Path("results/raw/m4_frozen_qwen_test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
