"""Cache frozen Qwen representations for M4 histories."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load

from genrec.genrec import history_text
from genrec.qwen_cache import (
    MODEL_ID,
    POOLING,
    example_id,
    file_sha256,
    validate_representations,
    validation_histories,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _inputs(args, data_dir, manifest_path):
    if args.split == "train":
        manifest = json.loads(manifest_path.read_text())
        examples = manifest["examples"]
        return [
            {
                "id": example_id(example),
                "history": example["history"],
            }
            for example in examples
        ], {
            "source_manifest_sha256": file_sha256(
                manifest_path
            ),
        }

    train_path = data_dir / "train.csv"
    validation_path = data_dir / "validation.csv"
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    histories = validation_histories(
        train,
        validation,
        min_items=50,
    )

    return [
        {
            "id": int(user_id),
            "history": history,
        }
        for user_id, history in histories.items()
    ], {
        "history_source": "train_only",
        "history_min_items": 50,
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
    }


def main() -> None:
    args = parse_args()

    data_dir = Path("data/processed/ml-1m")
    manifest_path = Path("artifacts/m4/train_manifest.json")
    movies_path = data_dir / "movies.csv"
    output_dir = Path("artifacts/m4/qwen")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_provenance = _inputs(
        args,
        data_dir,
        manifest_path,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        rows = rows[:args.limit]

    movies = pd.read_csv(movies_path)
    model, tokenizer = load(MODEL_ID)

    vectors = []
    token_counts = []
    ids = []

    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        text = history_text(row["history"], movies)
        tokens = tokenizer.encode(text)

        inputs = mx.array([tokens])
        hidden = model.model(inputs)
        vector = hidden[0, -1]
        mx.eval(vector)

        vector = vector.astype(mx.float32)
        mx.eval(vector)
        vectors.append(np.array(vector))
        token_counts.append(len(tokens))
        ids.append(row["id"])

        if index % 100 == 0:
            print(f"cached {index}/{len(rows)}")

    elapsed = time.perf_counter() - started

    representations = np.stack(vectors)
    validate_representations(
        representations,
        expected_rows=len(rows),
    )

    suffix = (
        f"limit{args.limit}"
        if args.limit is not None
        else "full"
    )

    cache_path = (
        output_dir / f"{args.split}_{suffix}.npz"
    )

    id_key = (
        "example_ids"
        if args.split == "train"
        else "user_ids"
    )

    np.savez(
        cache_path,
        representations=representations,
        **{id_key: np.asarray(ids)},
        token_counts=np.asarray(
            token_counts,
            dtype=np.int32,
        ),
    )

    provenance = {
        "contract": "m4-frozen-qwen-v1",
        "split": args.split,
        "model_id": MODEL_ID,
        "pooling": POOLING,
        "model_parameters": "frozen",
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        **source_provenance,
        "contexts": len(rows),
        "hidden_size": int(representations.shape[1]),
        "cache_dtype": str(representations.dtype),
        "token_count": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "total": sum(token_counts),
        },
        "elapsed_seconds": elapsed,
        "seconds_per_context": elapsed / len(rows),
        "cache_sha256": file_sha256(cache_path),
    }

    provenance_path = (
        output_dir
        / f"{args.split}_{suffix}_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(json.dumps(provenance, indent=2))
    print(f"cache: {cache_path}")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()
