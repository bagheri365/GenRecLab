import pandas as pd
import pytest

from evaluation.protocol import (
    build_user_histories,
    evaluate_full_catalog,
    held_out_groups,
)


def test_validation_history_is_added_for_test():
    train = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10},
            {"user_id": 1, "movie_id": 11},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 12},
        ]
    )

    assert build_user_histories(train) == {
        1: {10, 11}
    }

    assert build_user_histories(
        train,
        validation,
    ) == {
        1: {10, 11, 12}
    }


def test_held_out_group_supports_multiple_positives():
    held_out = pd.DataFrame(
        [
            {
                "user_id": 1,
                "movie_id": 20,
                "timestamp": 500,
            },
            {
                "user_id": 1,
                "movie_id": 21,
                "timestamp": 500,
            },
        ]
    )

    assert held_out_groups(held_out) == {
        1: {20, 21}
    }


def test_held_out_group_rejects_multiple_timestamps():
    held_out = pd.DataFrame(
        [
            {
                "user_id": 1,
                "movie_id": 20,
                "timestamp": 500,
            },
            {
                "user_id": 1,
                "movie_id": 21,
                "timestamp": 600,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="expected exactly one",
    ):
        held_out_groups(held_out)


def test_full_catalog_multi_positive_evaluation():
    metrics = evaluate_full_catalog(
        catalog_items=[1, 2, 3, 4, 5],
        history_by_user={
            7: {1},
        },
        relevant_by_user={
            7: {2, 3},
        },
        score_items=lambda items: {
            item: {
                2: 10.0,
                4: 9.0,
                3: 8.0,
            }.get(item, 0.0)
            for item in items
        },
        k=2,
    )

    assert metrics["users"] == 1
    assert metrics["recall@2"] == 0.5
    assert metrics["mrr@2"] == 1.0
    assert 0.0 < metrics["ndcg@2"] < 1.0


def test_evaluator_requires_complete_scores():
    with pytest.raises(
        ValueError,
        match="exactly one score",
    ):
        evaluate_full_catalog(
            catalog_items=[1, 2, 3],
            history_by_user={
                1: set(),
            },
            relevant_by_user={
                1: {2},
            },
            score_items=lambda items: {
                2: 1.0,
            },
            k=2,
        )


def test_evaluator_supports_user_dependent_scores():
    metrics = evaluate_full_catalog(
        catalog_items=[1, 2, 3],
        history_by_user={7: {1}},
        relevant_by_user={7: {3}},
        score_user_items=lambda user_id, items: {
            item: (10.0 if user_id == 7 and item == 3 else 0.0)
            for item in items
        },
        k=1,
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0


def test_evaluator_rejects_two_scoring_functions():
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_full_catalog(
            catalog_items=[1, 2],
            history_by_user={1: set()},
            relevant_by_user={1: {2}},
            score_items=lambda items: {
                item: 1.0 for item in items
            },
            score_user_items=lambda user_id, items: {
                item: 1.0 for item in items
            },
            k=1,
        )
