# Evaluation Contract v0.1

Changes affecting headline comparability must increment this contract version.

## Task

Next-item ranking. Each example receives only interactions strictly before the target timestamp.

## Temporal integrity

```text
max(history.timestamp) < target.timestamp
```

Equal timestamps are not treated as safely ordered.

## Candidate protocols

- **Full catalog:** all eligible unseen items.
- **Sampled:** one positive plus a deterministic, versioned negative set.

Results from different candidate protocols are not directly comparable.

## Primary metrics

At `k=10`: NDCG, MRR, and Recall.

## Invalidation

Broken runs are preserved and marked invalid. Leakage, candidate mismatches, or accidental contract changes invalidate comparisons.

## Multi-positive temporal evaluation (v0.2)

MovieLens can contain several interactions for the same user at exactly the
same timestamp. GenRecLab does not invent an ordering inside such timestamp
groups. Validation and test therefore support multiple relevant items.

For a held-out timestamp group:

- all movies in the group are relevant positives;
- observed history items are excluded from the candidate catalog;
- held-out relevant items are always retained as candidates;
- Recall@K is the fraction of held-out positives retrieved;
- MRR@K is the reciprocal rank of the first retrieved positive;
- NDCG@K uses binary relevance and the ideal ranking for the number of
  available positives;
- score ties are resolved deterministically.

Headline full-catalog evaluation must use this contract. Sampled-candidate
experiments, if added later, must be reported separately and document their
negative-sampling procedure.
\n