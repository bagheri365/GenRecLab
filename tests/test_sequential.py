import pandas as pd
import pytest

from genrec.sequential import build_sequence_examples


def test_sequence_examples_use_only_earlier_timestamps():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
            {"user_id": 1, "movie_id": 20, "timestamp": 200},
            {"user_id": 1, "movie_id": 30, "timestamp": 300},
        ]
    )

    examples = build_sequence_examples(interactions)

    assert len(examples) == 2

    assert examples[0].history == (10,)
    assert examples[0].targets == (20,)

    assert examples[1].history == (10, 20)
    assert examples[1].targets == (30,)


def test_timestamp_ties_are_one_target_group():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
            {"user_id": 1, "movie_id": 20, "timestamp": 200},
            {"user_id": 1, "movie_id": 21, "timestamp": 200},
            {"user_id": 1, "movie_id": 30, "timestamp": 300},
        ]
    )

    examples = build_sequence_examples(interactions)

    assert examples[0].history == (10,)
    assert examples[0].targets == (20, 21)

    assert examples[1].history == (10, 20, 21)
    assert examples[1].targets == (30,)


def test_target_items_never_appear_in_same_timestamp_history():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 1, "timestamp": 100},
            {"user_id": 1, "movie_id": 2, "timestamp": 200},
            {"user_id": 1, "movie_id": 3, "timestamp": 200},
        ]
    )

    example = build_sequence_examples(interactions)[0]

    assert set(example.history).isdisjoint(example.targets)


def test_max_history_keeps_most_recent_items():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
            {"user_id": 1, "movie_id": 20, "timestamp": 200},
            {"user_id": 1, "movie_id": 30, "timestamp": 300},
            {"user_id": 1, "movie_id": 40, "timestamp": 400},
        ]
    )

    examples = build_sequence_examples(
        interactions,
        max_history=2,
    )

    assert examples[-1].history == (20, 30)
    assert examples[-1].targets == (40,)


def test_min_history_filters_short_contexts():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
            {"user_id": 1, "movie_id": 20, "timestamp": 200},
            {"user_id": 1, "movie_id": 30, "timestamp": 300},
        ]
    )

    examples = build_sequence_examples(
        interactions,
        min_history=2,
    )

    assert len(examples) == 1
    assert examples[0].history == (10, 20)


def test_invalid_history_settings_fail():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "timestamp": 100},
        ]
    )

    with pytest.raises(ValueError):
        build_sequence_examples(
            interactions,
            min_history=0,
        )

    with pytest.raises(ValueError):
        build_sequence_examples(
            interactions,
            max_history=0,
        )


from genrec.sequential import (
    ItemVocabulary,
    encode_history,
)


def test_item_vocabulary_reserves_zero_for_padding():
    vocabulary = ItemVocabulary.from_items(
        [30, 10, 20]
    )

    assert vocabulary.size == 4
    assert vocabulary.encode(10) == 1
    assert vocabulary.encode(20) == 2
    assert vocabulary.encode(30) == 3

    assert vocabulary.decode(1) == 10
    assert vocabulary.decode(3) == 30


def test_item_vocabulary_rejects_unknown_items():
    vocabulary = ItemVocabulary.from_items(
        [10, 20]
    )

    with pytest.raises(ValueError):
        vocabulary.encode(999)

    with pytest.raises(ValueError):
        vocabulary.decode(999)

    with pytest.raises(ValueError):
        vocabulary.decode(0)


def test_encode_history_left_pads():
    vocabulary = ItemVocabulary.from_items(
        [10, 20, 30]
    )

    encoded = encode_history(
        (10, 20),
        vocabulary,
        max_length=4,
    )

    assert encoded == (0, 0, 1, 2)


def test_encode_history_keeps_recent_context():
    vocabulary = ItemVocabulary.from_items(
        [10, 20, 30, 40]
    )

    encoded = encode_history(
        (10, 20, 30, 40),
        vocabulary,
        max_length=2,
    )

    assert encoded == (3, 4)


def test_encode_history_rejects_invalid_length():
    vocabulary = ItemVocabulary.from_items(
        [10]
    )

    with pytest.raises(ValueError):
        encode_history(
            (10,),
            vocabulary,
            max_length=0,
        )
