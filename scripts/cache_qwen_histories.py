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
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path("artifacts/m4/train_manifest.json")
    movies_path = Path("data/processed/ml-1m/movies.csv")
    output_dir = Path("artifacts/m4/qwen")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    examples = manifest["examples"]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        examples = examples[:args.limit]

    movies = pd.read_csv(movies_path)
    model, tokenizer = load(MODEL_ID)

    vectors = []
    token_counts = []
    ids = []

    started = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        text = history_text(example["history"], movies)
        tokens = tokenizer.encode(text)

        inputs = mx.array([tokens])
        hidden = model.model(inputs)
        vector = hidden[0, -1]
        mx.eval(vector)

        vector = vector.astype(mx.float32)
        mx.eval(vector)
        vectors.append(np.array(vector))
        token_counts.append(len(tokens))
        ids.append(example_id(example))

        if index % 100 == 0:
            print(f"cached {index}/{len(examples)}")

    elapsed = time.perf_counter() - started

    representations = np.stack(vectors)
    validate_representations(
        representations,
        expected_rows=len(examples),
    )

    suffix = (
        f"limit{args.limit}"
        if args.limit is not None
        else "full"
    )

    cache_path = output_dir / f"train_{suffix}.npz"

    np.savez(
        cache_path,
        representations=representations,
        example_ids=np.asarray(ids),
        token_counts=np.asarray(
            token_counts,
            dtype=np.int32,
        ),
    )

    provenance = {
        "contract": "m4-frozen-qwen-v1",
        "model_id": MODEL_ID,
        "pooling": POOLING,
        "model_parameters": "frozen",
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "source_manifest_sha256": file_sha256(
            manifest_path
        ),
        "examples": len(examples),
        "hidden_size": int(representations.shape[1]),
        "cache_dtype": str(representations.dtype),
        "token_count": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "total": sum(token_counts),
        },
        "elapsed_seconds": elapsed,
        "seconds_per_context": elapsed / len(examples),
        "cache_sha256": file_sha256(cache_path),
    }

    provenance_path = (
        output_dir / f"train_{suffix}_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print(json.dumps(provenance, indent=2))
    print(f"cache: {cache_path}")
    print(f"provenance: {provenance_path}")


if __name__ == "__main__":
    main()
