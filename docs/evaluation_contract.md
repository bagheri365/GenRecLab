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
