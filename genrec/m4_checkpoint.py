"""Checkpoint helpers for the selected M4 ranking head."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from genrec.m4_ranking import FrozenQwenRankingHead


SELECTED_CONFIG = {
    "dims": 32,
    "batch_size": 256,
    "negatives": 20,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 20,
    "seed": 42,
}

SELECTED_VALIDATION = {
    "users": 6039,
    "recall@10": 0.045616774441911055,
    "mrr@10": 0.029000846350573126,
    "ndcg@10": 0.026150149520975174,
}


def save_ranking_head(model, path) -> None:
    """Persist the trainable M4 ranking-head parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def load_ranking_head(
    path,
    *,
    hidden_size: int,
    num_items: int,
    dims: int,
):
    """Load a persisted M4 ranking head."""
    model = FrozenQwenRankingHead(
        hidden_size=hidden_size,
        num_items=num_items,
        dims=dims,
    )
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


def write_selection_provenance(
    path,
    *,
    checkpoint_path,
    extra=None,
) -> None:
    """Record the validation-only M4 model-selection decision."""
    payload = {
        "contract": "m4-frozen-qwen-v1",
        "selection_split": "validation_only",
        "test_used_for_selection": False,
        "selected_config": SELECTED_CONFIG,
        "selected_validation": SELECTED_VALIDATION,
        "checkpoint": str(checkpoint_path),
        "reproducibility_note": (
            "Repeated MLX GPU training produced identical Recall@10 "
            "and only approximately 1e-6 differences in MRR/NDCG; "
            "the original sweep result is retained as the selected "
            "validation measurement."
        ),
    }
    if extra:
        payload.update(extra)

    Path(path).write_text(
        json.dumps(payload, indent=2) + "\n"
    )
