"""Tests for M4 checkpoint and selection provenance."""

import json

import mlx.core as mx

from genrec.m4_checkpoint import (
    SELECTED_CONFIG,
    SELECTED_VALIDATION,
    load_ranking_head,
    save_ranking_head,
    write_selection_provenance,
)
from genrec.m4_ranking import FrozenQwenRankingHead


def test_selected_configuration_is_frozen():
    assert SELECTED_CONFIG == {
        "dims": 32,
        "batch_size": 256,
        "negatives": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 20,
        "seed": 42,
    }
    assert SELECTED_VALIDATION["recall@10"] == (
        0.045616774441911055
    )


def test_checkpoint_round_trip(tmp_path):
    mx.random.seed(42)
    model = FrozenQwenRankingHead(
        hidden_size=4,
        num_items=5,
        dims=3,
    )
    reps = mx.array([[1.0, 2.0, 3.0, 4.0]])
    items = mx.array([[0, 2, 4]], dtype=mx.int32)

    before = model(reps, items)
    mx.eval(before)

    path = tmp_path / "head.safetensors"
    save_ranking_head(model, path)

    restored = load_ranking_head(
        path,
        hidden_size=4,
        num_items=5,
        dims=3,
    )
    after = restored(reps, items)
    mx.eval(after)

    assert mx.array_equal(before, after).item()


def test_selection_provenance_marks_test_unused(tmp_path):
    path = tmp_path / "selection.json"
    write_selection_provenance(
        path,
        checkpoint_path="head.safetensors",
    )

    payload = json.loads(path.read_text())

    assert payload["selection_split"] == "validation_only"
    assert payload["test_used_for_selection"] is False
    assert payload["selected_config"] == SELECTED_CONFIG
    assert payload["selected_validation"] == (
        SELECTED_VALIDATION
    )
