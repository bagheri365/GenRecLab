"""Train and evaluate the M2 SASRec baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd

from evaluation.protocol import (
    build_user_histories,
    evaluate_full_catalog,
    held_out_groups,
)
from genrec.sasrec import SASRec
from genrec.sasrec_training import SASRecTrainer
from genrec.sequential import (
    ItemVocabulary,
    build_sequence_examples,
    encode_history,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--mlp-dims", type=int, default=64)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
    )

    return parser.parse_args()


def make_scorer(
    model,
    vocabulary,
    histories,
    *,
    max_history,
):
    def score_user_items(user_id, movie_ids):
        movie_ids = list(movie_ids)

        encoded = encode_history(
            histories[user_id],
            vocabulary,
            max_length=max_history,
        )

        history_array = mx.array(
            [encoded],
            dtype=mx.int32,
        )

        known_positions = []
        known_indices = []

        scores = {
            movie_id: 0.0
            for movie_id in movie_ids
        }

        for position, movie_id in enumerate(movie_ids):
            if movie_id in vocabulary.item_to_index:
                known_positions.append(position)
                known_indices.append(
                    vocabulary.encode(movie_id)
                )

        if known_indices:
            candidate_array = mx.array(
                known_indices,
                dtype=mx.int32,
            )

            values = model.score_items(
                history_array,
                candidate_array,
            )[0]

            mx.eval(values)

            for position, value in zip(
                known_positions,
                values.tolist(),
            ):
                scores[movie_ids[position]] = float(value)

        return scores

    return score_user_items


def main():
    args = parse_args()

    if args.evaluate_test and args.checkpoint is None:
        raise ValueError(
            "--evaluate-test requires --checkpoint"
        )

    data_dir = Path("data/processed/ml-1m")

    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(
        data_dir / "validation.csv"
    )
    test = pd.read_csv(data_dir / "test.csv")
    movies = pd.read_csv(data_dir / "movies.csv")

    catalog = set(movies["movie_id"].tolist())

    vocabulary = ItemVocabulary.from_items(catalog)

    examples = build_sequence_examples(
        train,
        max_history=args.max_history,
    )

    mx.random.seed(args.seed)

    model = SASRec(
        num_items=len(vocabulary.item_to_index),
        max_length=args.max_history,
        dims=args.dims,
        num_heads=args.heads,
        num_layers=args.layers,
        mlp_dims=args.mlp_dims,
        dropout=0.0,
    )

    if args.checkpoint is not None:
        if not args.evaluate_test:
            raise ValueError(
                "--checkpoint requires --evaluate-test"
            )

        model.load_weights(
            str(args.checkpoint)
        )
        mx.eval(model.parameters())

        test_history_frame = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        test_histories = (
            test_history_frame.sort_values(
                ["user_id", "timestamp", "movie_id"]
            )
            .groupby("user_id")["movie_id"]
            .apply(lambda values: tuple(values.tolist()))
            .to_dict()
        )

        test_scorer = make_scorer(
            model,
            vocabulary,
            test_histories,
            max_history=args.max_history,
        )

        started = time.perf_counter()

        test_metrics = evaluate_full_catalog(
            catalog_items=catalog,
            history_by_user=build_user_histories(
                train,
                validation,
            ),
            relevant_by_user=held_out_groups(test),
            score_user_items=test_scorer,
            k=args.k,
        )

        evaluation_seconds = (
            time.perf_counter() - started
        )

        result = {
            "experiment": "m2_sasrec_test",
            "model": {
                "name": "sasrec_mlx",
                "dims": args.dims,
                "heads": args.heads,
                "layers": args.layers,
                "mlp_dims": args.mlp_dims,
                "max_history": args.max_history,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
            },
            "protocol": {
                "split": "timestamp_group_v0.2",
                "candidates": "full_catalog_excluding_seen",
                "relevance": "multi_positive_timestamp_group",
                "k": args.k,
                "selection": "validation_ndcg_at_10",
            },
            "training": None,
            "validation": None,
            "test": test_metrics,
            "runtime": {
                "evaluation_seconds": evaluation_seconds,
            },
        }

        output = Path(
            "results/raw/m2_sasrec_test.json"
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        print(json.dumps(result, indent=2))
        return

    validation_histories = (
        train.sort_values(
            ["user_id", "timestamp", "movie_id"]
        )
        .groupby("user_id")["movie_id"]
        .apply(lambda values: tuple(values.tolist()))
        .to_dict()
    )

    validation_scorer = make_scorer(
        model,
        vocabulary,
        validation_histories,
        max_history=args.max_history,
    )

    validation_history_sets = build_user_histories(
        train
    )
    validation_relevant = held_out_groups(
        validation
    )

    trainer = SASRecTrainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    best_checkpoint = (
        checkpoint_dir / "m2_sasrec_best.safetensors"
    )

    started = time.perf_counter()
    trajectory = []
    best_epoch = None
    best_ndcg = float("-inf")

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()

        loss = trainer.train_epoch(
            examples,
            vocabulary,
            max_length=args.max_history,
            batch_size=args.batch_size,
            num_negatives=args.negatives,
            seed=args.seed + epoch,
        )

        validation_metrics = evaluate_full_catalog(
            catalog_items=catalog,
            history_by_user=validation_history_sets,
            relevant_by_user=validation_relevant,
            score_user_items=validation_scorer,
            k=args.k,
        )

        epoch_seconds = (
            time.perf_counter() - epoch_started
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "validation": validation_metrics,
            "seconds": epoch_seconds,
        }
        trajectory.append(record)

        ndcg = validation_metrics[
            f"ndcg@{args.k}"
        ]

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch + 1
            model.save_weights(
                str(best_checkpoint)
            )

        print(json.dumps(record, indent=2))

    training_seconds = time.perf_counter() - started

    model.load_weights(
        str(best_checkpoint)
    )
    mx.eval(model.parameters())

    validation_metrics = trajectory[
        best_epoch - 1
    ]["validation"]

    test_metrics = None

    result = {
        "experiment": "m2_sasrec",
        "model": {
            "name": "sasrec_mlx",
            "dims": args.dims,
            "heads": args.heads,
            "layers": args.layers,
            "mlp_dims": args.mlp_dims,
            "max_history": args.max_history,
            "batch_size": args.batch_size,
            "negatives": args.negatives,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
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
        "training": {
            "trajectory": trajectory,
            "best_epoch": best_epoch,
            "selection_metric": f"ndcg@{args.k}",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "runtime": {
            "training_seconds": training_seconds,
        },
    }

    output = Path("results/raw/m2_sasrec.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
