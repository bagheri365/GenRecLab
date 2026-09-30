"""Train and validate the M4 frozen-Qwen ranking head."""

from __future__ import annotations

import argparse
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
from genrec.m4_ranking import (
    FrozenQwenRankingHead,
    load_training_targets,
)
from genrec.m4_training import RankingHeadTrainer
from genrec.qwen_cache import (
    example_id,
    validate_representations,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
    )
    return parser.parse_args()


def load_cache(path, *, id_key):
    with np.load(path) as cache:
        representations = cache[
            "representations"
        ].copy()
        ids = cache[id_key].copy()

    validate_representations(
        representations,
        expected_rows=len(ids),
    )
    return representations, ids


def make_validation_scorer(
    model,
    *,
    representations,
    user_ids,
    item_to_index,
):
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

        values = np.array(
            scores.astype(mx.float32)
        ).tolist()

        return {
            movie_id: float(value)
            for movie_id, value in zip(
                movie_ids,
                values,
            )
        }

    return score_user_items


def main() -> None:
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("--train-limit must be >= 1")

    data_dir = Path("data/processed/ml-1m")
    artifact_dir = Path("artifacts/m4")
    qwen_dir = artifact_dir / "qwen"

    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(
        data_dir / "validation.csv"
    )
    movies = pd.read_csv(data_dir / "movies.csv")

    manifest = json.loads(
        (artifact_dir / "train_manifest.json").read_text()
    )

    train_reps, train_ids = load_cache(
        qwen_dir / "train_full.npz",
        id_key="example_ids",
    )
    validation_reps, validation_users = load_cache(
        qwen_dir / "validation_full.npz",
        id_key="user_ids",
    )

    targets = load_training_targets(
        manifest,
        train_ids,
        example_id_fn=example_id,
    )

    if args.train_limit is not None:
        train_reps = train_reps[: args.train_limit]
        targets = targets[: args.train_limit]

    catalog = tuple(
        movies["movie_id"].astype(int).tolist()
    )
    item_to_index = {
        movie_id: index
        for index, movie_id in enumerate(catalog)
    }

    mx.random.seed(args.seed)

    model = FrozenQwenRankingHead(
        hidden_size=train_reps.shape[1],
        num_items=len(catalog),
        dims=args.dims,
    )
    trainer = RankingHeadTrainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    started = time.perf_counter()
    trajectory = []

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        loss = trainer.train_epoch(
            train_reps,
            targets,
            item_to_index,
            catalog_items=catalog,
            batch_size=args.batch_size,
            num_negatives=args.negatives,
            seed=args.seed + epoch,
        )
        seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "seconds": seconds,
        }
        trajectory.append(record)
        print(json.dumps(record, indent=2))

    scorer = make_validation_scorer(
        model,
        representations=validation_reps,
        user_ids=validation_users,
        item_to_index=item_to_index,
    )

    validation_metrics = evaluate_full_catalog(
        catalog_items=catalog,
        history_by_user=build_user_histories(train),
        relevant_by_user=held_out_groups(validation),
        score_user_items=scorer,
        k=args.k,
    )

    elapsed = time.perf_counter() - started

    result = {
        "experiment": "m4_frozen_qwen_ranking_head",
        "model": {
            "encoder": "Qwen/Qwen3-0.6B-Base",
            "encoder_parameters": "frozen",
            "pooling": "final-layer-final-query-token",
            "head": "linear_projection_dot_product",
            "dims": args.dims,
            "batch_size": args.batch_size,
            "negatives": args.negatives,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "training": {
            "examples": len(train_reps),
            "trajectory": trajectory,
        },
        "protocol": {
            "split": "timestamp_group_v0.2",
            "candidates": "full_catalog_excluding_seen",
            "relevance": "multi_positive_timestamp_group",
            "evaluation_split": "validation_only",
            "k": args.k,
        },
        "validation": validation_metrics,
        "runtime": {
            "seconds": elapsed,
        },
    }

    output = Path("results/raw/m4_frozen_qwen_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
