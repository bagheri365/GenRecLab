# MovieLens 1M Dataset

GenRecLab starts with MovieLens 1M.

## Preparation

Download and prepare in one command:

```bash
python scripts/prepare_movielens.py
```

Or prepare an already extracted copy:

```bash
python scripts/prepare_movielens.py --source /path/to/ml-1m
```

Generated files are written under `data/processed/ml-1m/` and are ignored by Git.

## Split v0.2

The protocol reserves complete timestamp groups per user:

- latest distinct timestamp group -> test;
- preceding distinct timestamp group -> validation;
- all strictly earlier interactions -> train;
- require at least 3 train interactions.

MovieLens users can rate several movies at exactly the same timestamp. Those
interactions have no defensible internal temporal order, so GenRecLab keeps a
tied timestamp group together rather than discarding the user or breaking the
tie by movie ID.

This guarantees strict train < validation < test temporal separation.
Validation and test may contain more than one relevant item per user.
