import numpy as np
import pytest

from genrec.semantic import (
    mean_history_embedding,
    movie_text,
    normalize_embeddings,
)


def test_movie_text():
    assert movie_text(
        "Toy Story (1995)",
        "Animation|Children's|Comedy",
    ) == (
        "Title: Toy Story (1995)\n"
        "Genres: Animation, Children's, Comedy"
    )


def test_normalize_embeddings():
    values = np.array(
        [[3.0, 4.0], [0.0, 2.0]],
        dtype=np.float32,
    )

    result = normalize_embeddings(values)

    np.testing.assert_allclose(
        np.linalg.norm(result, axis=1),
        [1.0, 1.0],
        rtol=1e-6,
    )


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        normalize_embeddings(
            np.array([[0.0, 0.0]])
        )


def test_mean_history_uses_recent_items():
    embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([1.0, 1.0]),
    }

    result = mean_history_embedding(
        [1, 2, 3],
        embeddings,
        max_history=2,
    )

    expected = np.array([0.5, 1.0])
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
    )


def test_mean_history_skips_unknown_items():
    embeddings = {
        2: np.array([0.0, 1.0]),
    }

    result = mean_history_embedding(
        [1, 2],
        embeddings,
    )

    np.testing.assert_allclose(
        result,
        [0.0, 1.0],
    )


def test_mean_history_requires_known_item():
    with pytest.raises(ValueError):
        mean_history_embedding(
            [1, 2],
            {},
        )


def test_mean_history_validates_max_history():
    with pytest.raises(ValueError):
        mean_history_embedding(
            [1],
            {1: np.array([1.0])},
            max_history=0,
        )


from genrec.semantic import FrozenSemanticRecommender


def test_frozen_semantic_scores_cosine_similarity():
    model = FrozenSemanticRecommender(
        np.array([1, 2, 3]),
        np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]),
    )

    scores = model.score_user_items([1], [1, 2, 3])

    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)
    assert scores[3] == pytest.approx(2 ** -0.5)


def test_frozen_semantic_respects_recent_history():
    model = FrozenSemanticRecommender(
        np.array([1, 2]),
        np.eye(2),
        max_history=1,
    )

    user = model.user_embedding([1, 2])

    np.testing.assert_allclose(user, [0.0, 1.0])


def test_frozen_semantic_unknown_candidate_scores_zero():
    model = FrozenSemanticRecommender(
        np.array([1]),
        np.array([[1.0, 0.0]]),
    )

    assert model.score_user_items(
        [1],
        [999],
    )[999] == 0.0


def test_frozen_semantic_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        FrozenSemanticRecommender(
            np.array([1, 1]),
            np.eye(2),
        )


def test_frozen_semantic_npz_round_trip(tmp_path):
    path = tmp_path / "embeddings.npz"

    np.savez_compressed(
        path,
        movie_ids=np.array([10, 20]),
        embeddings=np.eye(2, dtype=np.float32),
    )

    model = FrozenSemanticRecommender.from_npz(
        str(path)
    )

    assert model.movie_ids.tolist() == [10, 20]
    np.testing.assert_allclose(
        model.embeddings,
        np.eye(2),
    )



def test_timestamp_group_histories_preserve_ties():
    import pandas as pd

    from genrec.semantic import timestamp_group_histories

    frame = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "movie_id": [30, 20, 10, 40],
        "timestamp": [200, 100, 100, 50],
    })

    histories = timestamp_group_histories(frame)

    assert histories == {
        1: [[10, 20], [30]],
        2: [[40]],
    }


def test_recent_complete_groups_crosses_boundary():
    from genrec.semantic import recent_complete_groups

    groups = [[1, 2], [3, 4, 5], [6]]

    assert recent_complete_groups(
        groups,
        min_items=3,
    ) == [3, 4, 5, 6]


def test_timestamp_group_histories_requires_columns():
    import pandas as pd

    from genrec.semantic import timestamp_group_histories

    with pytest.raises(ValueError):
        timestamp_group_histories(
            pd.DataFrame({"user_id": [1]})
        )
