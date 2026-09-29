import pandas as pd
import pytest

from genrec.matrix_factorization import BPRMatrixFactorization


def tiny_data():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [10, 11, 11, 12, 10, 12],
        }
    )


def test_requires_user_and_movie_columns():
    model = BPRMatrixFactorization()

    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"movie_id": [1]}))


def test_scoring_before_fit_fails():
    model = BPRMatrixFactorization()

    with pytest.raises(RuntimeError):
        model.score(1, 10)


def test_fit_creates_expected_embeddings():
    model = BPRMatrixFactorization(
        factors=4,
        epochs=1,
        seed=7,
    ).fit(tiny_data())

    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (3, 4)


def test_training_is_deterministic():
    first = BPRMatrixFactorization(
        factors=4,
        epochs=2,
        seed=7,
    ).fit(tiny_data())

    second = BPRMatrixFactorization(
        factors=4,
        epochs=2,
        seed=7,
    ).fit(tiny_data())

    assert first.score(1, 10) == pytest.approx(
        second.score(1, 10)
    )


def test_unknown_user_or_item_scores_zero():
    model = BPRMatrixFactorization(
        factors=4,
        epochs=1,
    ).fit(tiny_data())

    assert model.score(999, 10) == 0.0
    assert model.score(1, 999) == 0.0


def test_batch_scores_match_scalar_scores():
    model = BPRMatrixFactorization(
        factors=4,
        epochs=1,
        seed=7,
    ).fit(tiny_data())

    movie_ids = [10, 11, 12, 999]
    batch = model.scores(1, movie_ids)

    for movie_id in movie_ids:
        assert batch[movie_id] == pytest.approx(
            model.score(1, movie_id)
        )
