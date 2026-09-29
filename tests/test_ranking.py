import math
import pytest
from evaluation.ranking import mrr_at_k, ndcg_at_k, recall_at_k

def test_rank_one_scores_one():
    ranked = [42, 7, 9]
    assert recall_at_k(ranked, 42, 3) == 1.0
    assert mrr_at_k(ranked, 42, 3) == 1.0
    assert ndcg_at_k(ranked, 42, 3) == 1.0

def test_rank_three():
    ranked = [7, 9, 42]
    assert recall_at_k(ranked, 42, 3) == 1.0
    assert mrr_at_k(ranked, 42, 3) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, 42, 3) == pytest.approx(1 / math.log2(4))

def test_outside_cutoff_scores_zero():
    ranked = [1, 2, 3, 42]
    assert recall_at_k(ranked, 42, 3) == 0.0
    assert mrr_at_k(ranked, 42, 3) == 0.0
    assert ndcg_at_k(ranked, 42, 3) == 0.0

@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_invalid_k(metric):
    with pytest.raises(ValueError):
        metric([1], 1, 0)
