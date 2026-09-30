"""Freeze the validation-selected M4 ranking head."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

from genrec.m4_checkpoint import (
    SELECTED_CONFIG,
    SELECTED_VALIDATION,
    save_ranking_head,
    write_selection_provenance,
)
from genrec.m4_ranking import (
    FrozenQwenRankingHead,
    load_training_targets,
)
from genrec.m4_training import RankingHeadTrainer
from genrec.qwen_cache import example_id


def main() -> None:
    data_dir = Path("data/processed/ml-1m")
    artifact_dir = Path("artifacts/m4")
    qwen_dir = artifact_dir / "qwen"
    checkpoint_dir = Path("results/checkpoints")

    movies = pd.read_csv(data_dir / "movies.csv")
    manifest = json.loads(
        (artifact_dir / "train_manifest.json").read_text()
    )

    with np.load(qwen_dir / "train_full.npz") as cache:
        representations = cache["representations"].copy()
        example_ids = cache["example_ids"].copy()

    targets = load_training_targets(
        manifest,
        example_ids,
        example_id_fn=example_id,
    )

    catalog = tuple(
        movies["movie_id"].astype(int).tolist()
    )
    item_to_index = {
        movie_id: index
        for index, movie_id in enumerate(catalog)
    }

    config = SELECTED_CONFIG
    mx.random.seed(config["seed"])

    model = FrozenQwenRankingHead(
        hidden_size=representations.shape[1],
        num_items=len(catalog),
        dims=config["dims"],
    )
    trainer = RankingHeadTrainer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    trajectory = []
    for epoch in range(config["epochs"]):
        loss = trainer.train_epoch(
            representations,
            targets,
            item_to_index,
            catalog_items=catalog,
            batch_size=config["batch_size"],
            num_negatives=config["negatives"],
            seed=config["seed"] + epoch,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": loss,
        }
        trajectory.append(record)
        print(json.dumps(record))

    checkpoint_path = checkpoint_dir / "m4_selected.safetensors"
    provenance_path = (
        checkpoint_dir / "m4_selected_provenance.json"
    )

    save_ranking_head(model, checkpoint_path)
    write_selection_provenance(
        provenance_path,
        checkpoint_path=checkpoint_path,
        extra={
            "training_examples": len(representations),
            "training_trajectory": trajectory,
            "catalog_movies": len(catalog),
            "hidden_size": int(representations.shape[1]),
            "selected_validation_source": (
                "results/raw/m4_lr0.001_e20.json"
            ),
        },
    )

    print("selected validation:")
    print(json.dumps(SELECTED_VALIDATION, indent=2))
    print(f"checkpoint: {checkpoint_path}")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()
