# Research Plan

## Thesis

GenRecLab tests whether pretrained language-model representations provide measurable recommendation value beyond conventional collaborative and sequential models.

```text
interaction history -> textual context -> Qwen Base -> hidden states
                    -> pooled user representation -> ranking head -> item scores
```

The LLM represents preference state; it does not generate movie titles.

## Research questions

1. Does a pretrained LM representation improve next-item ranking?
2. How much of any gain comes from pretraining versus LoRA adaptation?
3. Does pretraining improve data efficiency?
4. Which context strategies give the best ranking-quality/token-cost tradeoff?
5. How do models behave under temporal shift?

## Dataset and protocol

Start with MovieLens 1M using user ID, movie ID, rating, timestamp, title, and genres.

For target interaction `t`:

```text
history_t = interactions strictly before t
target_t  = interaction at t
```

Primary metrics: NDCG@10, MRR@10, Recall@10. Headline comparisons should use paired examples and bootstrap confidence intervals.

## Model ladder

- M0: popularity
- M1: matrix factorization
- M2: small SASRec-style model
- M3: frozen local semantic embedding baseline
- M4: frozen Qwen + ranking head
- M5: Qwen + LoRA + ranking head

## Milestones

1. Freeze research/evaluation contract.
2. Build MovieLens temporal pipeline and metric tests.
3. Add popularity, matrix-factorization, and small sequential baselines.
4. Add cached local semantic/Ollama baseline.
5. Add frozen Qwen hidden-state GenRec.
6. Add Qwen LoRA trained with ranking loss.
7. Run pretraining/adaptation ablations.
8. Run data-efficiency study.
9. Run context-engineering study.
10. Run temporal-shift study and final report.
