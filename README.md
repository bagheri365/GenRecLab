# GenRecLab

GenRecLab is a reproducible research lab for testing **LLM-native recommendation** against conventional recommender-system baselines.

> **Research question:** When does language-model pretraining help recommendation, what part of the gain comes from pretraining versus adaptation, and what does that gain cost?

The project is inspired by Netflix's GenRec direction, but is a controlled public-data study rather than a production reproduction.

## Model ladder

1. Popularity
2. Matrix factorization
3. Small sequential recommender
4. Frozen semantic embedding baseline
5. Frozen Qwen hidden-state representation + ranking head
6. Qwen + LoRA + ranking head

The primary dataset is **MovieLens 1M**. Evaluation is temporal: histories contain only interactions occurring strictly before the target.

## Local setup

```bash
python -m pip install -e ".[dev]"
pytest
```

See `docs/research_plan.md` and `docs/evaluation_contract.md`.

## Prepare MovieLens 1M

```bash
python scripts/prepare_movielens.py
```

This downloads MovieLens 1M into `data/raw/` and writes deterministic temporal split artifacts under `data/processed/ml-1m/`. Both directories are ignored by Git.

## M0: Popularity baseline

Run with: `python scripts/run_popularity.py`

M0 ranks candidate movies by interaction frequency in the training split only. Validation and test interactions never contribute to popularity counts. Evaluation uses the shared full-catalog temporal protocol with seen-item filtering, timestamp-group multi-positive relevance, and Recall@K, MRR@K, and NDCG@K.

Results are written to `results/raw/m0_popularity.json`.

## M1: Matrix factorization

M1 uses implicit-feedback BPR matrix factorization. Hyperparameter selection uses validation only; test evaluation requires the explicit `--evaluate-test` flag. With 32 factors and seed 42, epochs {1, 5, 10, 20, 30} were evaluated on validation, and 20 epochs was selected by validation NDCG@10. Run the frozen configuration with: `python scripts/run_matrix_factorization.py --factors 32 --epochs 20 --seed 42 --evaluate-test`.

## M2: SASRec-style sequential recommender

M2 is a compact SASRec-style sequential recommender implemented with MLX.
It uses strictly earlier timestamp groups as history and treats every item in
the held-out timestamp group as relevant. Same-timestamp target items never
predict one another.

The frozen configuration uses a maximum history of 50 items, 32 hidden
dimensions, 2 attention heads, 1 Transformer layer, a 64-dimensional MLP,
20 sampled negatives, batch size 256, learning rate 1e-3, weight decay 1e-4,
and seed 42.

Epochs 1 through 10 were evaluated using validation NDCG@10. Epoch 10 was
selected under a predetermined 10-epoch cap, with validation Recall@10
0.23052, MRR@10 0.15443, and NDCG@10 0.14730. The selected checkpoint was
then evaluated once on the held-out test split, obtaining Recall@10 0.21118,
MRR@10 0.13746, and NDCG@10 0.13256.

Historical interactions sharing a timestamp are deterministically ordered by
movie ID when flattened into the positional sequence. This ordering is a
representation convention rather than an inferred within-timestamp event
order; held-out timestamp groups remain unordered multi-positive targets.

Selection evidence is preserved in
`results/raw/m2_sasrec_epochs10_seed42.json`, and final test evidence in
`results/raw/m2_sasrec_test_epoch10_seed42.json`.
