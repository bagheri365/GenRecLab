from pathlib import Path

import pandas as pd

from genrec.movielens import (
    dataset_statistics,
    load_movielens_1m,
    prepare_movielens_1m,
    temporal_leave_last_split,
)


def _write_fixture(directory: Path) -> None:
    directory.mkdir()
    (directory / "ratings.dat").write_text(
        "\n".join(
            [
                "1::10::5::100",
                "1::11::4::200",
                "1::12::3::300",
                "1::13::5::400",
                "1::14::4::500",
                "2::10::4::110",
                "2::12::5::210",
                "2::13::3::310",
                "2::14::4::410",
                "2::15::5::510",
            ]
        )
        + "\n"
    )
    (directory / "movies.dat").write_text(
        "\n".join(
            [
                "10::Toy Story (1995)::Animation|Comedy",
                "11::Heat (1995)::Action|Crime",
                "12::GoldenEye (1995)::Action|Adventure",
                "13::Casino (1995)::Drama",
                "14::Sense and Sensibility (1995)::Drama|Romance",
                "15::Apollo 13 (1995)::Drama",
            ]
        )
        + "\n"
    )


def test_load_movielens_parses_original_double_colon_format(tmp_path):
    source = tmp_path / "ml-1m"
    _write_fixture(source)

    ratings, movies = load_movielens_1m(source)

    assert ratings.columns.tolist() == ["user_id", "movie_id", "rating", "timestamp"]
    assert movies.columns.tolist() == ["movie_id", "title", "genres"]
    assert len(ratings) == 10
    assert movies.loc[movies.movie_id == 10, "title"].item() == "Toy Story (1995)"


def test_temporal_leave_last_split_is_per_user_and_leakage_safe():
    frame = pd.DataFrame(
        [
            {"user_id": user, "movie_id": movie, "rating": 5, "timestamp": ts}
            for user in (1, 2)
            for movie, ts in zip(range(10, 15), range(100, 600, 100))
        ]
    )

    split = temporal_leave_last_split(frame, min_train_items=3)

    assert len(split.train) == 6
    assert len(split.validation) == 2
    assert len(split.test) == 2

    for user_id in (1, 2):
        train = split.train[split.train.user_id == user_id]
        validation = split.validation[split.validation.user_id == user_id]
        test = split.test[split.test.user_id == user_id]
        assert train.timestamp.max() < validation.timestamp.min()
        assert validation.timestamp.max() < test.timestamp.min()


def test_equal_timestamp_groups_are_kept_together():
    frame = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
            {"user_id": 1, "movie_id": 11, "timestamp": 200},
            {"user_id": 1, "movie_id": 12, "timestamp": 300},
            {"user_id": 1, "movie_id": 13, "timestamp": 400},
            {"user_id": 1, "movie_id": 14, "timestamp": 400},
            {"user_id": 1, "movie_id": 15, "timestamp": 500},
            {"user_id": 1, "movie_id": 16, "timestamp": 500},
        ]
    )

    split = temporal_leave_last_split(frame, min_train_items=3)

    assert split.train.movie_id.tolist() == [10, 11, 12]
    assert split.validation.movie_id.tolist() == [13, 14]
    assert split.test.movie_id.tolist() == [15, 16]
    assert split.train.timestamp.max() < split.validation.timestamp.min()
    assert split.validation.timestamp.max() < split.test.timestamp.min()


def test_short_users_are_excluded():
    frame = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": movie, "timestamp": ts}
            for movie, ts in [(10, 100), (11, 200), (12, 300), (13, 400)]
        ]
    )

    split = temporal_leave_last_split(frame, min_train_items=3)

    assert split.train.empty


def test_prepare_writes_expected_artifacts(tmp_path):
    source = tmp_path / "ml-1m"
    _write_fixture(source)
    output = tmp_path / "processed"

    stats = prepare_movielens_1m(source, output, min_train_items=3)

    assert set(p.name for p in output.iterdir()) == {
        "train.csv",
        "validation.csv",
        "test.csv",
        "movies.csv",
        "stats.json",
    }
    assert stats["ratings"] == 10
    assert stats["split_users"] == 2


def test_statistics_are_json_friendly():
    ratings = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 100},
            {"user_id": 1, "movie_id": 11, "rating": 4, "timestamp": 200},
        ]
    )
    movies = pd.DataFrame(
        [
            {"movie_id": 10, "title": "A", "genres": "Drama"},
            {"movie_id": 11, "title": "B", "genres": "Comedy"},
        ]
    )

    stats = dataset_statistics(ratings, movies)

    assert stats == {
        "ratings": 2,
        "users": 1,
        "rated_movies": 2,
        "catalog_movies": 2,
        "min_timestamp": 100,
        "max_timestamp": 200,
    }
