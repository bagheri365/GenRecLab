"""Tests for the M4 frozen-Qwen ranking head."""

import mlx.core as mx
import numpy as np

from genrec.m4_ranking import (
    FrozenQwenRankingHead,
    load_training_targets,
)
from genrec.m4_training import sample_training_batch
from genrec.qwen_cache import example_id


def test_ranking_head_scores_candidates():
    model = FrozenQwenRankingHead(
        hidden_size=4,
        num_items=7,
        dims=3,
    )
    reps = mx.zeros((2, 4))
    items = mx.array(
        [[0, 1, 2], [3, 4, 5]],
        dtype=mx.int32,
    )

    scores = model(reps, items)
    mx.eval(scores)

    assert scores.shape == (2, 3)


def test_load_training_targets_aligns_cache_order():
    a = {
        "user_id": 1,
        "timestamp": 10,
        "history": [1],
        "targets": [2, 3],
    }
    b = {
        "user_id": 2,
        "timestamp": 20,
        "history": [4],
        "targets": [5],
    }

    targets = load_training_targets(
        {"examples": [a, b]},
        np.asarray(
            [example_id(b), example_id(a)]
        ),
        example_id_fn=example_id,
    )

    assert targets == [(5,), (2, 3)]


def test_negative_sampling_is_deterministic_and_excludes_group():
    targets = [(2, 3)]
    item_to_index = {
        item: index
        for index, item in enumerate(range(1, 8))
    }

    a = sample_training_batch(
        targets,
        item_to_index,
        catalog_items=range(1, 8),
        batch_indices=[0],
        num_negatives=3,
        seed=42,
    )
    b = sample_training_batch(
        targets,
        item_to_index,
        catalog_items=range(1, 8),
        batch_indices=[0],
        num_negatives=3,
        seed=42,
    )

    assert np.array_equal(a, b)

    decoded = {
        index: item
        for item, index in item_to_index.items()
    }
    negatives = [
        decoded[int(x)]
        for x in a[0, 1:]
    ]
    assert not ({2, 3} & set(negatives))
