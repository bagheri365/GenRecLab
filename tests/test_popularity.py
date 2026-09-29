import pandas as pd

from genrec.popularity import PopularityRecommender


def test_popularity_counts_training_interactions():
    train = pd.DataFrame(
        {"movie_id": [1, 1, 1, 2, 2, 3]}
    )

    model = PopularityRecommender().fit(train)

    assert model.score(1) == 3.0
    assert model.score(2) == 2.0
    assert model.score(3) == 1.0
    assert model.score(999) == 0.0


def test_popularity_fit_replaces_previous_counts():
    model = PopularityRecommender().fit(
        pd.DataFrame({"movie_id": [1, 1, 1]})
    )

    model.fit(
        pd.DataFrame({"movie_id": [2]})
    )

    assert model.score(1) == 0.0
    assert model.score(2) == 1.0


def test_held_out_data_does_not_change_model():
    train = pd.DataFrame(
        {"movie_id": [1, 1, 2]}
    )

    model = PopularityRecommender().fit(train)

    # Held-out interactions are intentionally never passed to fit().
    validation = pd.DataFrame({"movie_id": [9] * 100})
    test = pd.DataFrame({"movie_id": [9] * 100})

    assert model.counts == {1: 2, 2: 1}
    assert model.score(9) == 0.0
    assert len(validation) == 100
    assert len(test) == 100
