import pandas as pd
import pytest

from genrec.genrec import (
    GenRecExample,
    build_genrec_examples,
    history_text,
    sample_user_balanced,
)


def test_build_examples_preserves_complete_timestamp_groups():
    interactions = pd.DataFrame(
        [
            (1, 10, 1),
            (1, 11, 1),
            (1, 20, 2),
            (1, 21, 2),
            (1, 30, 3),
        ],
        columns=["user_id", "movie_id", "timestamp"],
    )

    examples = build_genrec_examples(
        interactions,
        min_history=3,
    )

    assert len(examples) == 1
    assert examples[0] == GenRecExample(
        user_id=1,
        history=(10, 11, 20, 21),
        targets=(30,),
        timestamp=3,
    )


def test_build_examples_uses_only_recent_complete_groups():
    interactions = pd.DataFrame(
        [
            (1, 10, 1),
            (1, 11, 1),
            (1, 20, 2),
            (1, 30, 3),
            (1, 31, 3),
            (1, 40, 4),
        ],
        columns=["user_id", "movie_id", "timestamp"],
    )

    examples = build_genrec_examples(
        interactions,
        min_history=3,
    )

    assert examples[-1].history == (20, 30, 31)
    assert examples[-1].targets == (40,)


def test_same_timestamp_targets_never_enter_history():
    interactions = pd.DataFrame(
        [
            (1, 10, 1),
            (1, 20, 2),
            (1, 21, 2),
        ],
        columns=["user_id", "movie_id", "timestamp"],
    )

    examples = build_genrec_examples(
        interactions,
        min_history=1,
    )

    assert examples[-1].history == (10,)
    assert examples[-1].targets == (20, 21)


def test_balanced_sampling_is_deterministic():
    examples = [
        GenRecExample(user, (1,), (i,), i)
        for user in range(1, 4)
        for i in range(10)
    ]

    first = sample_user_balanced(
        examples,
        n=7,
        seed=42,
    )
    second = sample_user_balanced(
        examples,
        n=7,
        seed=42,
    )

    assert first == second
    assert len(first) == 7


def test_balanced_first_round_contains_each_user():
    examples = [
        GenRecExample(user, (1,), (i,), i)
        for user in range(1, 5)
        for i in range(3)
    ]

    sampled = sample_user_balanced(
        examples,
        n=4,
        seed=42,
    )

    assert {example.user_id for example in sampled} == {
        1, 2, 3, 4
    }


def test_sample_rejects_oversized_request():
    example = GenRecExample(1, (1,), (2,), 1)

    with pytest.raises(ValueError):
        sample_user_balanced([example], n=2)


def test_history_text_contract():
    movies = pd.DataFrame(
        [
            (
                1,
                "Toy Story (1995)",
                "Animation|Children's|Comedy",
            ),
            (
                2,
                "Jumanji (1995)",
                "Adventure|Children's|Fantasy",
            ),
        ],
        columns=["movie_id", "title", "genres"],
    )

    text = history_text((1, 2), movies)

    assert text == (
        "User movie history:\n"
        "1. Toy Story (1995) | "
        "Animation, Children's, Comedy\n"
        "2. Jumanji (1995) | "
        "Adventure, Children's, Fantasy\n"
        "\n"
        "Represent this user's movie preferences."
    )