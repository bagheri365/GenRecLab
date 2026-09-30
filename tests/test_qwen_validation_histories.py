"""Tests for M4 validation-history construction."""

import pandas as pd
import pytest

from genrec.qwen_cache import validation_histories


def test_validation_histories_use_train_only():
    train = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 1},
            {"user_id": 1, "movie_id": 11, "timestamp": 1},
            {"user_id": 1, "movie_id": 20, "timestamp": 2},
            {"user_id": 1, "movie_id": 21, "timestamp": 2},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 99, "timestamp": 3},
        ]
    )

    histories = validation_histories(
        train,
        validation,
        min_items=3,
    )

    assert histories == {
        1: [10, 11, 20, 21],
    }
    assert 99 not in histories[1]


def test_validation_histories_keep_short_users():
    train = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 1},
            {"user_id": 2, "movie_id": 20, "timestamp": 1},
            {"user_id": 2, "movie_id": 21, "timestamp": 2},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 30, "timestamp": 2},
            {"user_id": 2, "movie_id": 31, "timestamp": 3},
        ]
    )

    histories = validation_histories(
        train,
        validation,
        min_items=50,
    )

    assert histories[1] == [10]
    assert histories[2] == [20, 21]


def test_validation_histories_reject_missing_user():
    train = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 1},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": 2, "movie_id": 20, "timestamp": 2},
        ]
    )

    with pytest.raises(
        ValueError,
        match="validation users have no train history",
    ):
        validation_histories(train, validation)
