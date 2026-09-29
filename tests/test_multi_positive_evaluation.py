import math

import pytest

from evaluation.candidates import full_catalog_candidates
from evaluation.ranking import (
    mrr_at_k,
    ndcg_at_k,
    rank_by_score,
    recall_at_k,
)


def test_multi_positive_recall():
    assert recall_at_k([4, 2, 8, 3], {2, 3}, 3) == 0.5


def test_multi_positive_recall_all_found():
    assert recall_at_k([2, 3, 9], {2, 3}, 2) == 1.0


def test_multi_positive_mrr_uses_first_relevant():
    assert mrr_at_k([9, 2, 3], {2, 3}, 3) == 0.5


def test_multi_positive_mrr_zero_when_no_hit():
    assert mrr_at_k([9, 8, 7], {2, 3}, 3) == 0.0


def test_multi_positive_ndcg_is_one_for_ideal_ranking():
    assert ndcg_at_k([2, 3, 9], {2, 3}, 2) == pytest.approx(1.0)


def test_multi_positive_ndcg_discount():
    expected = (1 / math.log2(3)) / (
        1 / math.log2(2) + 1 / math.log2(3)
    )
    assert ndcg_at_k([9, 2, 8], {2, 3}, 2) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_metrics_reject_empty_relevance(metric):
    with pytest.raises(ValueError, match="relevant_items"):
        metric([1, 2, 3], set(), 2)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_metrics_reject_invalid_k(metric):
    with pytest.raises(ValueError, match="k must be"):
        metric([1], {1}, 0)


def test_candidates_remove_seen_items_and_keep_positive():
    candidates = full_catalog_candidates(
        catalog_items={1, 2, 3, 4, 5},
        seen_items={1, 2},
        relevant_items={2, 4},
    )
    assert set(candidates) == {2, 3, 4, 5}


def test_candidates_require_positives_in_catalog():
    with pytest.raises(ValueError, match="missing from catalog"):
        full_catalog_candidates({1, 2}, {1}, {3})


def test_score_ranking_is_deterministic_for_ties():
    scores = {3: 0.5, 1: 0.9, 2: 0.5}
    assert rank_by_score(scores) == [1, 2, 3]
