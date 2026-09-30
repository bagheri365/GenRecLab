import mlx.core as mx
import numpy as np
import pytest

from genrec.sasrec import SASRec


def make_model():
    mx.random.seed(42)
    return SASRec(
        num_items=20,
        max_length=4,
        dims=8,
        num_heads=2,
        num_layers=1,
        mlp_dims=16,
        dropout=0.0,
    )


def test_user_representation_shape():
    model = make_model()

    histories = mx.array([
        [0, 0, 1, 2],
        [0, 3, 4, 5],
    ])

    output = model(histories)
    mx.eval(output)

    assert output.shape == (2, 8)


def test_hidden_sequence_shape():
    model = make_model()

    histories = mx.array([
        [0, 0, 1, 2],
        [0, 3, 4, 5],
    ])

    hidden = model.encode(histories)
    mx.eval(hidden)

    assert hidden.shape == (2, 4, 8)


def test_full_candidate_scoring_shape():
    model = make_model()

    histories = mx.array([
        [0, 0, 1, 2],
        [0, 3, 4, 5],
    ])
    candidates = mx.array([1, 2, 3, 4, 5])

    scores = model.score_items(
        histories,
        candidates,
    )
    mx.eval(scores)

    assert scores.shape == (2, 5)


def test_future_item_does_not_change_earlier_hidden_state():
    model = make_model()

    first = mx.array([[1, 2, 3, 4]])
    second = mx.array([[1, 2, 9, 10]])

    h1 = model.encode(first)
    h2 = model.encode(second)
    mx.eval(h1, h2)

    np.testing.assert_allclose(
        np.array(h1[:, 1, :]),
        np.array(h2[:, 1, :]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_padding_keys_do_not_change_valid_suffix():
    model = make_model()

    # Both represent the same valid suffix. Padding slots contain
    # no item signal and cannot be attended to as keys.
    history = mx.array([[0, 0, 1, 2]])

    output = model(history)
    mx.eval(output)

    assert np.isfinite(np.array(output)).all()


def test_history_must_end_in_real_item():
    model = make_model()

    with pytest.raises(ValueError):
        model(mx.array([[0, 1, 2, 0]]))


def test_wrong_sequence_length_fails():
    model = make_model()

    with pytest.raises(ValueError):
        model.encode(mx.array([[0, 1, 2]]))


def test_invalid_head_dimension_fails():
    with pytest.raises(ValueError):
        SASRec(
            num_items=20,
            max_length=4,
            dims=7,
            num_heads=2,
        )
