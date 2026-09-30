# GenRecLab

When does language-model pretraining actually help recommendation?

GenRecLab is a controlled, reproducible research project for comparing conventional recommender systems with language-model-based recommendation under a shared temporal evaluation protocol.

> **Research question:** When does language-model pretraining help recommendation, what part of the gain comes from pretraining versus adaptation, and what does that gain cost?

The project is inspired by Netflix's GenRec direction, but is a controlled public-data study rather than a production reproduction.

## At a Glance

- **Dataset:** MovieLens 1M
- **Evaluation:** temporal splits, full-catalog ranking, seen-item filtering, and multi-positive timestamp-group relevance
- **Completed final test evaluations:** M0–M4
- **Strongest completed model:** M2 SASRec under the current evaluation protocol
- **Key negative result:** the frozen semantic baseline is not competitive with the conventional baselines in the current setup
- **Next experiment:** M5 — Qwen + LoRA + ranking head
- **Canonical results:** `results/m0_m4_test_summary.json`

## Why This Project Exists

Modern recommendation systems can draw signal from interaction frequency, collaborative latent factors, sequential behavior, pretrained semantic representations, recommendation-specific adaptation, or combinations of these mechanisms.

When an LLM-based recommender improves, it can therefore be difficult to determine what produced the gain.

GenRecLab builds an experimental ladder that progressively changes the recommendation mechanism while preserving a shared evaluation contract. The goal is not for every milestone to outperform the previous one. The goal is to understand where useful recommendation capability enters the system and to preserve both positive and negative evidence.

## Experimental Ladder

| Milestone | Model | Main question |
|---|---|---|
| **M0** | Popularity | How strong is a non-personalized frequency baseline? |
| **M1** | Matrix factorization | How much does collaborative personalization add? |
| **M2** | SASRec-style sequential recommender | How does explicit sequence modeling perform? |
| **M3** | Frozen semantic embeddings | Are pretrained semantic representations useful without recommendation-specific training? |
| **M4** | Frozen Qwen hidden states + ranking head | How well does a learned recommender perform over frozen LM representations? |
| **M5** | Qwen + LoRA + ranking head | Does recommendation-specific LM adaptation change the result? |

M0–M4 have completed final test evaluation. M5 is the next planned milestone.

## Key Results

### Final M0–M4 Test Comparison

All reported test results use the shared MovieLens 1M temporal full-catalog evaluation contract.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| M0 — Popularity | 0.03940 | 0.02474 | 0.02310 |
| M1 — Matrix factorization | 0.06631 | 0.04147 | 0.03878 |
| **M2 — SASRec** | **0.21118** | **0.13746** | **0.13256** |
| M3 — Frozen semantic | 0.02174 | 0.01471 | 0.01256 |
| M4 — Frozen Qwen + ranking head | 0.04008 | 0.02595 | 0.02348 |

The exact values and source result files are frozen in `results/m0_m4_test_summary.json`.

### M2 Is the Strongest Completed Model

Under the current evaluation protocol, M2 is substantially ahead of the other completed milestones:

- Recall@10: **0.21118**
- MRR@10: **0.13746**
- NDCG@10: **0.13256**

This establishes the empirical ordering for the completed experiments. It does not by itself establish which architectural property causes the difference.

### Frozen Semantic Similarity Is Not Competitive Here

M3 uses frozen `qwen3-embedding:0.6b` representations with cosine similarity and no trainable recommendation parameters.

Final test performance:

- Recall@10: **0.02174**
- MRR@10: **0.01471**
- NDCG@10: **0.01256**

Under this setup, M3 is below M0 popularity on all three metrics. This negative result is retained as part of the research record rather than discarded.

### M4 Improves over M3 but Remains Below M1 and M2

M4 uses frozen hidden-state representations from `Qwen/Qwen3-0.6B-Base` together with a trainable recommendation ranking head.

Final test performance:

- Recall@10: **0.04008**
- MRR@10: **0.02595**
- NDCG@10: **0.02348**

M4 outperforms M3 under the shared evaluation protocol and slightly exceeds M0 on the reported metrics, but remains below M1 matrix factorization and far below M2 SASRec.

M3 and M4 differ in representation construction and scoring design, so their gap should not be interpreted as a clean one-variable ablation of the ranking head.

## Evaluation Protocol

GenRecLab uses a shared temporal protocol across the model ladder.

### Temporal Splitting

Interactions are ordered by time. Recommendation histories contain only information available before the held-out target timestamp.

Interactions sharing a timestamp form a complete group. Held-out items from the same timestamp are treated as simultaneous multi-positive targets rather than being allowed to predict one another.

### Full-Catalog Ranking

Evaluation ranks against the movie catalog rather than a small sampled candidate set. Previously seen items are excluded from the candidate set.

### Metrics

The primary ranking metrics are:

- Recall@10
- MRR@10
- NDCG@10

### Model Selection

Hyperparameters and checkpoints are selected using validation data. The held-out test split is reserved for final evaluation of the selected configuration.

For M4, the selected configuration was determined from validation results before final test evaluation.

See `docs/evaluation_contract.md` for the evaluation contract.

## Benchmark

The current benchmark is **MovieLens 1M**.

The processed dataset uses deterministic temporal splits under the `timestamp_group_v0.2` contract. The experiment catalog contains 3,883 movies, and the final validation and test evaluations each cover 6,039 users.

Prepare the dataset with:

```bash
python scripts/prepare_movielens.py
```

This downloads MovieLens 1M into `data/raw/` and writes deterministic processed artifacts under `data/processed/ml-1m/`. Both directories are ignored by Git.

See `docs/dataset.md` for dataset documentation.

## Research Evolution

> popularity → collaborative filtering → sequence modeling → frozen semantics → frozen LM + ranking head → LM adaptation

| Milestone | Question | Final test Recall@10 | Status |
|---|---|---:|---|
| M0 | How strong is popularity? | 0.03940 | Complete |
| M1 | How does collaborative personalization perform? | 0.06631 | Complete |
| M2 | How does sequential modeling perform? | 0.21118 | Complete |
| M3 | Are frozen semantic embeddings sufficient? | 0.02174 | Complete |
| M4 | How does a learned head over frozen Qwen states perform? | 0.04008 | Complete |
| M5 | Does LoRA adaptation change the result? | — | Planned |

## M0 — Popularity

M0 ranks candidate movies by interaction frequency in the training split only. Validation and test interactions never contribute to popularity counts.

Run:

```bash
python scripts/run_popularity.py
```

Final test:

- Recall@10: 0.03940
- MRR@10: 0.02474
- NDCG@10: 0.02310

Evidence: `results/raw/m0_popularity.json`

## M1 — Matrix Factorization

M1 uses implicit-feedback BPR matrix factorization. Hyperparameter selection uses validation only; test evaluation requires the explicit `--evaluate-test` flag.

The frozen configuration uses 32 factors, seed 42, and 20 epochs.

Run:

```bash
python scripts/run_matrix_factorization.py \
  --factors 32 \
  --epochs 20 \
  --seed 42 \
  --evaluate-test
```

Final test:

- Recall@10: 0.06631
- MRR@10: 0.04147
- NDCG@10: 0.03878

Evidence: `results/raw/m1_matrix_factorization.json`

## M2 — SASRec-Style Sequential Recommender

M2 is a compact SASRec-style sequential recommender implemented with MLX.

It uses strictly earlier timestamp groups as history and treats every item in the held-out timestamp group as relevant. Same-timestamp target items never predict one another.

The frozen configuration uses:

- maximum history: 50 items
- hidden dimensions: 32
- attention heads: 2
- Transformer layers: 1
- MLP dimension: 64
- sampled negatives: 20
- batch size: 256
- learning rate: `1e-3`
- weight decay: `1e-4`
- seed: 42

Epochs 1–10 were evaluated using validation NDCG@10. Epoch 10 was selected under a predetermined 10-epoch cap.

Selected validation:

- Recall@10: 0.23052
- MRR@10: 0.15443
- NDCG@10: 0.14730

Final test:

- Recall@10: 0.21118
- MRR@10: 0.13746
- NDCG@10: 0.13256

Historical interactions sharing a timestamp are deterministically ordered by movie ID when flattened into the positional sequence. This is a representation convention rather than an inferred within-timestamp event order; held-out timestamp groups remain unordered multi-positive targets.

Evidence:

- `results/raw/m2_sasrec_epochs10_seed42.json`
- `results/raw/m2_sasrec_test_epoch10_seed42.json`

## M3 — Frozen Semantic Baseline

M3 tests frozen semantic representations without recommendation-specific training.

Configuration:

- encoder: `qwen3-embedding:0.6b`
- embedding dimension: 1024
- trainable parameters: 0
- pooling: uniform mean
- history selection: recent complete timestamp groups
- minimum history items: 50
- short-history policy: use all available
- timestamp-tie policy: never split group
- scoring: cosine similarity

Validation:

- Recall@10: 0.02562
- MRR@10: 0.01681
- NDCG@10: 0.01495

Final test:

- Recall@10: 0.02174
- MRR@10: 0.01471
- NDCG@10: 0.01256

Evidence:

- `results/raw/m3_semantic_validation.json`
- `results/raw/m3_semantic_test.json`

## M4 — Frozen Qwen + Ranking Head

M4 tests a recommendation-specific ranking head over frozen Qwen history representations.

Histories are serialized as text and encoded using `Qwen/Qwen3-0.6B-Base`.

The frozen representation contract is:

- encoder parameters: frozen
- pooling: final-layer final-query-token
- hidden size: 1024
- cache dtype: float32

The trainable ranking head uses a linear projection and item dot-product scoring.

The selected configuration is:

- projection dimensions: 32
- batch size: 256
- sampled negatives: 20
- learning rate: `1e-3`
- weight decay: `1e-4`
- epochs: 20
- seed: 42

Selected validation:

- Recall@10: 0.04562
- MRR@10: 0.02900
- NDCG@10: 0.02615

Final test:

- Recall@10: 0.04008
- MRR@10: 0.02595
- NDCG@10: 0.02348

The configuration was selected using validation only. Test data was not used for model selection.

Evidence:

- `results/raw/m4_frozen_qwen_validation.json`
- `results/raw/m4_frozen_qwen_test.json`
- `results/checkpoints/m4_selected_provenance.json`

## Negative Results Matter

GenRecLab intentionally preserves experiments that do not improve the headline metric.

### Frozen Semantics Underperform the Conventional Baselines

M3 performs below popularity and matrix factorization under the current protocol. This is evidence against treating the tested frozen semantic-similarity setup as a competitive recommender on this benchmark.

It does not establish that all pretrained semantic representations or all semantic recommendation methods will behave the same way.

### Frozen Qwen Does Not Match the Sequential Baseline

M4 performs better than M3 and approximately reaches the popularity regime, but it remains below matrix factorization and substantially below SASRec.

This does not establish that language-model pretraining is ineffective for recommendation. M4 keeps the encoder frozen. Recommendation-specific LM adaptation remains untested in the current ladder.

## Reproducibility

The repository preserves experimental evidence rather than only final metric tables.

The project includes:

- deterministic temporal data contracts
- shared evaluation code
- frozen M4 training manifests
- frozen Qwen representation caches
- SHA-256 artifact hashes
- validation-only model selection
- persisted checkpoints
- provenance JSON files
- raw evaluation outputs
- regression tests for data, ranking, and representation contracts

For M4, expensive Qwen representations are cached separately from ranking-head training.

Selected M4 checkpoint:

`results/checkpoints/m4_selected.safetensors`

Selection provenance:

`results/checkpoints/m4_selected_provenance.json`

Canonical cross-model test summary:

`results/m0_m4_test_summary.json`

## Repository Structure

```text
GenRecLab/
├── artifacts/          # Generated experiment artifacts and M4 caches
├── data/               # Raw, processed, and embedding data
├── docs/               # Dataset, research-plan, and evaluation documentation
├── evaluation/         # Shared candidate, protocol, and ranking evaluation
├── genrec/             # Core recommender, semantic, and M4 model code
├── results/
│   ├── checkpoints/    # Selected model checkpoints and provenance
│   ├── raw/            # Raw experiment outputs
│   └── m0_m4_test_summary.json
├── scripts/            # Preparation, caching, training, and evaluation entry points
├── tests/              # Contract, model, and evaluation regression tests
├── LICENSE
├── README.md
└── pyproject.toml
```

Generated environments, caches, processed data, and other local artifacts are not part of the conceptual repository structure above.

## Local Setup

Install the project in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

See also:

- `docs/research_plan.md`
- `docs/evaluation_contract.md`
- `docs/dataset.md`

## Experimental Discipline

Several rules are intentionally enforced across milestones.

**Temporal integrity.** Future interactions cannot enter a user's recommendation history.

**Timestamp integrity.** Items sharing the same timestamp are held out as one multi-positive target group.

**Shared evaluation.** Models are compared using the same full-catalog candidate protocol and ranking metrics.

**Validation-only selection.** Test metrics are not used to choose hyperparameters or checkpoints.

**Frozen evidence.** Important configurations, outputs, checkpoints, hashes, and provenance are preserved.

**Negative-result preservation.** A milestone remains part of the research record even when it performs worse than a simpler baseline.

These constraints are part of the experiment, not implementation details.

## Limitations

The current evidence should be interpreted within the scope of the experiment.

- Results currently come from MovieLens 1M.
- M3 and M4 use specific representation, history-construction, and scoring choices.
- M3 and M4 are not a clean one-variable ablation of the ranking head.
- M4 keeps Qwen frozen and therefore does not test recommendation-specific encoder adaptation.
- The current headline comparison focuses on ranking quality rather than a complete quality-versus-compute tradeoff.
- Additional datasets and ablations would be needed to determine how broadly the observed ordering generalizes.

## What the Experiments Suggest

The M0–M4 ladder currently supports several observations.

First, the **SASRec-style sequential model is the strongest completed system under this protocol**. Its final test metrics are substantially above the other completed milestones.

Second, **the tested frozen semantic baseline is not competitive**. M3 falls below popularity and matrix factorization.

Third, **M4 performs substantially better than M3 but does not close the gap to conventional recommendation models**. Because the two experiments differ in both representation construction and scoring, the comparison does not isolate a single cause for that improvement.

The next open question is whether recommendation-specific adaptation of Qwen changes the observed ordering.

## Next: M5

M5 will investigate Qwen with recommendation-specific LoRA adaptation together with a learned ranking head.

The experimental ladder becomes:

```text
M3: frozen semantic similarity
        ↓
M4: frozen Qwen + learned ranking head
        ↓
M5: LoRA-adapted Qwen + learned ranking head
```

The intended question is not merely whether an LLM can recommend movies. It is whether recommendation-specific adaptation changes the performance of the language-model approach enough to alter the conclusions from M3 and M4.

The exact strength of that comparison will depend on keeping the M5 data and evaluation contracts aligned with the existing ladder.

## Current Conclusion

Under GenRecLab's MovieLens 1M temporal full-catalog protocol, the SASRec-style sequential recommender is substantially stronger than the completed popularity, matrix-factorization, frozen-semantic, and frozen-Qwen-ranking-head systems.

Frozen semantic representations are not competitive in the current setup. M4 improves substantially over M3 and reaches approximately the popularity baseline, but remains below matrix factorization and far below SASRec. Because M3 and M4 differ in representation and scoring design, that comparison does not by itself isolate the cause of the improvement.

Whether recommendation-specific adaptation of Qwen changes this result remains an open M5 question.
