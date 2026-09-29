import pandas as pd
import pytest
from genrec.data import build_next_item_examples, normalize_interactions

def test_normalize_orders_interactions():
    frame = pd.DataFrame([
        {"user_id": 2, "movie_id": 9, "timestamp": 30},
        {"user_id": 1, "movie_id": 3, "timestamp": 20},
        {"user_id": 1, "movie_id": 2, "timestamp": 10},
    ])
    ordered = normalize_interactions(frame)
    assert ordered[["user_id", "movie_id"]].values.tolist() == [[1, 2], [1, 3], [2, 9]]

def test_history_never_contains_future_or_equal_timestamp():
    frame = pd.DataFrame([
        {"user_id": 1, "movie_id": 10, "timestamp": 100},
        {"user_id": 1, "movie_id": 11, "timestamp": 200},
        {"user_id": 1, "movie_id": 12, "timestamp": 200},
        {"user_id": 1, "movie_id": 13, "timestamp": 300},
    ])
    examples = build_next_item_examples(frame)
    assert examples
    for example in examples:
        assert all(ts < example.target_timestamp for ts in example.history_timestamps)

def test_min_history_applies_after_safety_filter():
    frame = pd.DataFrame([
        {"user_id": 1, "movie_id": 10, "timestamp": 100},
        {"user_id": 1, "movie_id": 11, "timestamp": 100},
        {"user_id": 1, "movie_id": 12, "timestamp": 200},
    ])
    assert build_next_item_examples(frame, min_history=3) == []

def test_missing_column_fails_fast():
    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_interactions(pd.DataFrame([{"user_id": 1, "movie_id": 2}]))
