#!/usr/bin/env python
"""Run the M0 popularity baseline on prepared MovieLens 1M."""

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
from genrec.popularity import PopularityRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M0 popularity baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/ml-1m"),
        help="Prepared MovieLens directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/m0_popularity.json"
        ),
        help="JSON result path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    train = pd.read_csv(
        args.data / "train.csv"
    )
    validation = pd.read_csv(
        args.data / "validation.csv"
    )
    test = pd.read_csv(
        args.data / "test.csv"
    )
    movies = pd.read_csv(
        args.data / "movies.csv"
    )

    # Critical invariant:
    # model fitting sees training interactions only.
    model = PopularityRecommender().fit(train)

    catalog = movies["movie_id"].tolist()

    validation_metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=build_user_histories(
            train
        ),
        relevant_by_user=held_out_groups(
            validation
        ),
        score_items=model.scores,
        k=args.k,
    )

    # Validation precedes test, so it becomes observed
    # history when evaluating the test timestamp group.
    test_metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=build_user_histories(
            train,
            validation,
        ),
        relevant_by_user=held_out_groups(
            test
        ),
        score_items=model.scores,
        k=args.k,
    )

    elapsed = time.perf_counter() - started

    result = {
        "experiment": "m0_popularity",
        "model": {
            "name": "popularity",
            "training_source": "train_only",
            "scoring": "training_interaction_count",
        },
        "dataset": {
            "name": "MovieLens 1M",
            "train_interactions": int(
                len(train)
            ),
            "validation_interactions": int(
                len(validation)
            ),
            "test_interactions": int(
                len(test)
            ),
            "catalog_movies": int(
                len(movies)
            ),
        },
        "protocol": {
            "split": "timestamp_group_v0.2",
            "candidates": (
                "full_catalog_excluding_seen"
            ),
            "relevance": (
                "multi_positive_timestamp_group"
            ),
            "k": args.k,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "runtime": {
            "seconds": elapsed,
            "python": platform.python_version(),
            "generated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print("M0 Popularity — MovieLens 1M")
    print(
        "Protocol: full catalog / "
        "temporal / multi-positive"
    )

    print("\nValidation")
    print(
        json.dumps(
            validation_metrics,
            indent=2,
        )
    )

    print("\nTest")
    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    print(
        f"\nRuntime: {elapsed:.3f}s"
    )
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
