import pytest

from genrec.sasrec_training import sample_training_rows
from genrec.sequential import (
    ItemVocabulary,
    SequenceExample,
)


def example():
    return SequenceExample(
        user_id=1,
        history=(10, 20),
        targets=(30, 40),
        timestamp=200,
    )


def vocabulary():
    return ItemVocabulary.from_items(
        [10, 20, 30, 40, 50, 60, 70, 80]
    )


def test_target_group_expands_without_inventing_order():
    rows = sample_training_rows(
        [example()],
        vocabulary(),
        max_length=4,
        num_negatives=2,
        seed=42,
    )

    assert len(rows) == 2

    vocab = vocabulary()

    assert rows[0].history == (0, 0, 1, 2)
    assert rows[1].history == (0, 0, 1, 2)

    assert rows[0].positive == vocab.encode(30)
    assert rows[1].positive == vocab.encode(40)


def test_negatives_exclude_history_and_all_targets():
    vocab = vocabulary()

    rows = sample_training_rows(
        [example()],
        vocab,
        max_length=4,
        num_negatives=3,
        seed=42,
    )

    forbidden = {
        vocab.encode(10),
        vocab.encode(20),
        vocab.encode(30),
        vocab.encode(40),
    }

    for row in rows:
        assert forbidden.isdisjoint(row.negatives)


def test_negative_sampling_is_deterministic():
    first = sample_training_rows(
        [example()],
        vocabulary(),
        max_length=4,
        num_negatives=3,
        seed=42,
    )

    second = sample_training_rows(
        [example()],
        vocabulary(),
        max_length=4,
        num_negatives=3,
        seed=42,
    )

    assert first == second


def test_negatives_are_unique_per_row():
    rows = sample_training_rows(
        [example()],
        vocabulary(),
        max_length=4,
        num_negatives=3,
        seed=42,
    )

    for row in rows:
        assert len(row.negatives) == len(
            set(row.negatives)
        )


def test_requires_negative_sample():
    with pytest.raises(ValueError):
        sample_training_rows(
            [example()],
            vocabulary(),
            max_length=4,
            num_negatives=0,
            seed=42,
        )


def test_fails_when_negative_pool_is_too_small():
    vocab = ItemVocabulary.from_items(
        [10, 20, 30, 40, 50]
    )

    with pytest.raises(ValueError):
        sample_training_rows(
            [example()],
            vocab,
            max_length=4,
            num_negatives=2,
            seed=42,
        )

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from genrec.sasrec import SASRec
from genrec.sasrec_training import (
    TrainingRow,
    batch_arrays,
    sampled_softmax_loss,
)


def small_rows():
    return [
        TrainingRow(
            history=(0, 0, 1, 2),
            positive=3,
            negatives=(5, 6, 7),
        ),
        TrainingRow(
            history=(0, 1, 2, 3),
            positive=4,
            negatives=(5, 7, 8),
        ),
    ]


def small_model():
    mx.random.seed(42)
    return SASRec(
        num_items=8,
        max_length=4,
        dims=8,
        num_heads=2,
        num_layers=1,
        mlp_dims=16,
        dropout=0.0,
    )


def test_batch_arrays_shapes():
    histories, candidates = batch_arrays(
        small_rows()
    )

    assert histories.shape == (2, 4)
    assert candidates.shape == (2, 4)


def test_sampled_softmax_loss_is_finite():
    model = small_model()
    histories, candidates = batch_arrays(
        small_rows()
    )

    loss = sampled_softmax_loss(
        model,
        histories,
        candidates,
    )
    mx.eval(loss)

    assert loss.ndim == 0
    assert np.isfinite(loss.item())


def test_loss_produces_gradients():
    model = small_model()
    histories, candidates = batch_arrays(
        small_rows()
    )

    loss_and_grad = nn.value_and_grad(
        model,
        sampled_softmax_loss,
    )

    loss, gradients = loss_and_grad(
        model,
        histories,
        candidates,
    )
    mx.eval(loss, gradients)

    assert np.isfinite(loss.item())
    assert gradients


def test_optimizer_step_changes_item_embeddings():
    model = small_model()
    optimizer = optim.AdamW(
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    histories, candidates = batch_arrays(
        small_rows()
    )

    before = np.array(
        model.item_embedding.weight
    ).copy()

    loss_and_grad = nn.value_and_grad(
        model,
        sampled_softmax_loss,
    )

    loss, gradients = loss_and_grad(
        model,
        histories,
        candidates,
    )

    optimizer.update(model, gradients)
    mx.eval(
        loss,
        model.parameters(),
        optimizer.state,
    )

    after = np.array(
        model.item_embedding.weight
    )

    assert not np.allclose(before, after)


def test_batch_arrays_reject_empty_rows():
    with pytest.raises(ValueError):
        batch_arrays([])


def training_examples():
    return [
        SequenceExample(
            user_id=1,
            history=(10, 20),
            targets=(30,),
            timestamp=200,
        ),
        SequenceExample(
            user_id=1,
            history=(10, 20, 30),
            targets=(40,),
            timestamp=300,
        ),
        SequenceExample(
            user_id=2,
            history=(20, 30),
            targets=(50,),
            timestamp=200,
        ),
    ]


def test_train_sasrec_returns_epoch_losses():
    from genrec.sasrec_training import train_sasrec

    vocab = ItemVocabulary.from_items(
        [10, 20, 30, 40, 50, 60, 70, 80]
    )

    mx.random.seed(42)
    model = SASRec(
        num_items=len(vocab.item_to_index),
        max_length=4,
        dims=8,
        num_heads=2,
        num_layers=1,
        mlp_dims=16,
        dropout=0.0,
    )

    losses = train_sasrec(
        model,
        training_examples(),
        vocab,
        max_length=4,
        epochs=2,
        batch_size=2,
        num_negatives=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=42,
    )

    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)


def test_train_sasrec_rejects_invalid_epochs():
    from genrec.sasrec_training import train_sasrec

    vocab = ItemVocabulary.from_items(
        [10, 20, 30, 40, 50, 60]
    )

    model = SASRec(
        num_items=len(vocab.item_to_index),
        max_length=4,
        dims=8,
        num_heads=2,
        num_layers=1,
        mlp_dims=16,
    )

    with pytest.raises(ValueError):
        train_sasrec(
            model,
            training_examples(),
            vocab,
            max_length=4,
            epochs=0,
            batch_size=2,
            num_negatives=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=42,
        )


def test_streaming_batches_cover_all_positives():
    from genrec.sasrec_training import iter_training_batches

    vocab = vocabulary()

    batches = list(
        iter_training_batches(
            [example()],
            vocab,
            max_length=4,
            batch_size=1,
            num_negatives=2,
            seed=42,
        )
    )

    assert len(batches) == 2
    assert sum(
        histories.shape[0]
        for histories, _ in batches
    ) == 2


def test_streaming_batches_respect_batch_size():
    from genrec.sasrec_training import iter_training_batches

    vocab = vocabulary()

    examples = [
        example(),
        SequenceExample(
            user_id=2,
            history=(10,),
            targets=(20, 30, 40),
            timestamp=300,
        ),
    ]

    batches = list(
        iter_training_batches(
            examples,
            vocab,
            max_length=4,
            batch_size=3,
            num_negatives=2,
            seed=42,
        )
    )

    sizes = [
        histories.shape[0]
        for histories, _ in batches
    ]

    assert sizes == [3, 2]


def test_streaming_sampling_is_deterministic():
    from genrec.sasrec_training import iter_training_batches

    vocab = vocabulary()

    first = list(
        iter_training_batches(
            [example()],
            vocab,
            max_length=4,
            batch_size=2,
            num_negatives=3,
            seed=42,
        )
    )

    second = list(
        iter_training_batches(
            [example()],
            vocab,
            max_length=4,
            batch_size=2,
            num_negatives=3,
            seed=42,
        )
    )

    assert np.array_equal(
        np.array(first[0][0]),
        np.array(second[0][0]),
    )
    assert np.array_equal(
        np.array(first[0][1]),
        np.array(second[0][1]),
    )


def test_persistent_trainer_runs_multiple_epochs():
    from genrec.sasrec_training import SASRecTrainer

    vocab = ItemVocabulary.from_items(
        [10, 20, 30, 40, 50, 60, 70, 80]
    )

    mx.random.seed(42)
    model = SASRec(
        num_items=len(vocab.item_to_index),
        max_length=4,
        dims=8,
        num_heads=2,
        num_layers=1,
        mlp_dims=16,
        dropout=0.0,
    )

    trainer = SASRecTrainer(
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    first = trainer.train_epoch(
        training_examples(),
        vocab,
        max_length=4,
        batch_size=2,
        num_negatives=2,
        seed=42,
    )

    second = trainer.train_epoch(
        training_examples(),
        vocab,
        max_length=4,
        batch_size=2,
        num_negatives=2,
        seed=43,
    )

    assert np.isfinite(first)
    assert np.isfinite(second)
    assert trainer.optimizer.state
