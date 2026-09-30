"""Tests for leakage-safe M4 test histories."""

import pandas as pd

from genrec.qwen_cache import test_histories as build_test_histories


def frame(rows):
    return pd.DataFrame(
        rows,
        columns=["user_id", "movie_id", "timestamp"],
    )


def test_test_history_uses_validation_but_not_test():
    train = frame([
        (1, 10, 1),
        (1, 11, 2),
    ])
    validation = frame([
        (1, 20, 3),
        (1, 21, 3),
    ])
    test = frame([
        (1, 99, 4),
    ])

    histories = build_test_histories(
        train,
        validation,
        test,
        min_items=3,
    )

    assert histories[1] == [11, 20, 21]
    assert 99 not in histories[1]


def test_test_history_keeps_complete_timestamp_groups():
    train = frame([
        (1, 10, 1),
        (1, 11, 2),
        (1, 12, 2),
    ])
    validation = frame([
        (1, 20, 3),
        (1, 21, 3),
    ])
    test = frame([
        (1, 99, 4),
    ])

    histories = build_test_histories(
        train,
        validation,
        test,
        min_items=3,
    )

    assert histories[1] == [11, 12, 20, 21]


def test_test_histories_cover_only_test_users():
    train = frame([
        (1, 10, 1),
        (2, 20, 1),
    ])
    validation = frame([
        (1, 11, 2),
        (2, 21, 2),
    ])
    test = frame([
        (2, 99, 3),
    ])

    histories = build_test_histories(
        train,
        validation,
        test,
        min_items=1,
    )

    assert list(histories) == [2]
