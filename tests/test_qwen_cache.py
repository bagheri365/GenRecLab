import numpy as np
import pytest

from genrec.qwen_cache import (
    example_id,
    validate_representations,
)


def test_example_id_is_stable():
    example = {
        "user_id": 7,
        "timestamp": 123,
        "history": [1, 2, 3],
        "targets": [4, 5],
    }

    assert example_id(example) == example_id(dict(example))


def test_example_id_changes_with_example():
    first = {
        "user_id": 7,
        "timestamp": 123,
        "history": [1, 2, 3],
        "targets": [4],
    }
    second = {
        **first,
        "targets": [5],
    }

    assert example_id(first) != example_id(second)


def test_validate_representations_accepts_float32():
    values = np.zeros((8, 1024), dtype=np.float32)

    validate_representations(
        values,
        expected_rows=8,
    )


def test_validate_representations_rejects_shape():
    values = np.zeros((7, 1024), dtype=np.float32)

    with pytest.raises(ValueError):
        validate_representations(
            values,
            expected_rows=8,
        )


def test_validate_representations_rejects_dtype():
    values = np.zeros((8, 1024), dtype=np.float64)

    with pytest.raises(ValueError):
        validate_representations(
            values,
            expected_rows=8,
        )


def test_validate_representations_rejects_nonfinite():
    values = np.zeros((8, 1024), dtype=np.float32)
    values[0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_representations(
            values,
            expected_rows=8,
        )
