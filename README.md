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
