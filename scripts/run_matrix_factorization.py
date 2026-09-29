"""Run M1 BPR matrix factorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from evaluation.protocol import (
    build_user_histories,
    evaluate_full_catalog,
    held_out_groups,
)
from genrec.matrix_factorization import BPRMatrixFactorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test split after model selection.",
    )
    args = parser.parse_args()

    data = Path("data/processed/ml-1m")

    train = pd.read_csv(data / "train.csv")
    validation = pd.read_csv(data / "validation.csv")
    test = pd.read_csv(data / "test.csv")
    movies = pd.read_csv(data / "movies.csv")

    model = BPRMatrixFactorization(
        factors=args.factors,
        epochs=args.epochs,
        seed=args.seed,
    )

    start = time.perf_counter()
    model.fit(train)
    training_seconds = time.perf_counter() - start

    catalog = movies["movie_id"].tolist()

    validation_metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=build_user_histories(train),
        relevant_by_user=held_out_groups(validation),
        score_user_items=model.scores,
        k=args.k,
    )

    test_metrics = None

    if args.evaluate_test:
        test_metrics = evaluate_full_catalog(
            catalog_items=catalog,
            history_by_user=build_user_histories(
                train,
                validation,
            ),
            relevant_by_user=held_out_groups(test),
            score_user_items=model.scores,
            k=args.k,
        )

    result = {
        "experiment": "m1_matrix_factorization",
        "model": {
            "name": "bpr_matrix_factorization",
            "factors": args.factors,
            "epochs": args.epochs,
            "seed": args.seed,
            "training_source": "train_only",
        },
        "protocol": {
            "split": "timestamp_group_v0.2",
            "candidates": "full_catalog_excluding_seen",
            "relevance": "multi_positive_timestamp_group",
            "k": args.k,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "runtime": {
            "training_seconds": training_seconds,
        },
    }

    output = Path("results/raw/m1_matrix_factorization.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print("M1 BPR Matrix Factorization")
    print(
        f"Factors: {args.factors} | "
        f"Epochs: {args.epochs} | "
        f"Seed: {args.seed}"
    )
    print(f"Training: {training_seconds:.3f}s")

    print("\nValidation")
    print(json.dumps(validation_metrics, indent=2))

    if test_metrics is not None:
        print("\nTest")
        print(json.dumps(test_metrics, indent=2))
    else:
        print("\nTest: not evaluated")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
