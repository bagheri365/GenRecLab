"""Per-example ranking metrics."""
import math

def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")

def _rank_of_target(ranked_items, target):
    try:
        return list(ranked_items).index(target) + 1
    except ValueError:
        return None

def recall_at_k(ranked_items, target, k: int = 10) -> float:
    _validate_k(k)
    return 1.0 if _rank_of_target(ranked_items[:k], target) is not None else 0.0

def mrr_at_k(ranked_items, target, k: int = 10) -> float:
    _validate_k(k)
    rank = _rank_of_target(ranked_items[:k], target)
    return 0.0 if rank is None else 1.0 / rank

def ndcg_at_k(ranked_items, target, k: int = 10) -> float:
    _validate_k(k)
    rank = _rank_of_target(ranked_items[:k], target)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)
