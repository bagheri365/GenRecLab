#!/usr/bin/env python
"""Run M3 frozen semantic recommendation on MovieLens 1M."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import pandas as pd

from evaluation.protocol import (
    build_user_histories,
    evaluate_full_catalog,
    held_out_groups,
)
from genrec.semantic import (
    FrozenSemanticRecommender,
    recent_complete_groups,
    timestamp_group_histories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3 frozen semantic baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/ml-1m"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "data/embeddings/qwen3_embedding_0.6b/"
            "embeddings.npz"
        ),
    )
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/m3_semantic_validation.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    train = pd.read_csv(args.data / "train.csv")
    validation = pd.read_csv(
        args.data / "validation.csv"
    )
    test = pd.read_csv(args.data / "test.csv")
    movies = pd.read_csv(args.data / "movies.csv")

    if args.split == "validation":
        history_frame = train
        evaluation = validation
        history_source = "train_only"
    else:
        history_frame = pd.concat(
            [train, validation],
            ignore_index=True,
        )
        evaluation = test
        history_source = "train_plus_validation"

    model = FrozenSemanticRecommender.from_npz(
        str(args.embeddings),
        max_history=args.max_history,
    )

    grouped_history = timestamp_group_histories(
        history_frame
    )
    filter_history = build_user_histories(history_frame)
    relevant = held_out_groups(evaluation)
    catalog = movies["movie_id"].tolist()

    def score_user_items(user_id, candidates):
        history = recent_complete_groups(
            grouped_history.get(int(user_id), []),
            min_items=args.max_history,
        )
        return model.score_user_items(
            history,
            candidates,
            truncate=False,
        )

    metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=filter_history,
        relevant_by_user=relevant,
        score_user_items=score_user_items,
        k=args.k,
    )

    elapsed = time.perf_counter() - started

    result = {
        "experiment": "m3_frozen_semantic",
        "model": {
            "encoder": "qwen3-embedding:0.6b",
            "embedding_dim": int(
                model.embeddings.shape[1]
            ),
            "trainable_parameters": 0,
            "pooling": "uniform_mean",
            "history_selection": (
                "recent_complete_timestamp_groups"
            ),
            "minimum_history_items": args.max_history,
            "short_history_policy": "use_all_available",
            "timestamp_tie_policy": "never_split_group",
            "scoring": "cosine_similarity",
        },
        "dataset": {
            "name": "MovieLens 1M",
            "train_interactions": int(len(train)),
            "validation_interactions": int(
                len(validation)
            ),
            "test_interactions": int(len(test)),
            "catalog_movies": int(len(movies)),
        },
        "protocol": {
            "split": "timestamp_group_v0.2",
            "candidates": "full_catalog_excluding_seen",
            "relevance": "multi_positive_timestamp_group",
            "evaluation_split": (
                "validation_only"
                if args.split == "validation"
                else "test_only"
            ),
            "history_source": history_source,
            "k": args.k,
        },
        args.split: metrics,
        "runtime": {
            "seconds": elapsed,
            "python": platform.python_version(),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("M3 Frozen Semantic — MovieLens 1M")
    print(json.dumps(metrics, indent=2))
    print(f"\nRuntime: {elapsed:.3f}s")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
