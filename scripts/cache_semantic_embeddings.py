"""Cache frozen MovieLens semantic embeddings from Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from genrec.semantic import movie_text, normalize_embeddings


TEMPLATE_VERSION = "movielens-title-genres-v1"


def embed_batch(
    texts: list[str],
    *,
    model: str,
    url: str,
) -> list[list[float]]:
    payload = json.dumps({
        "model": model,
        "input": texts,
    }).encode()

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.load(response)

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("Ollama response missing embeddings")

    if len(embeddings) != len(texts):
        raise ValueError("embedding count does not match input count")

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/processed/ml-1m/movies.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/embeddings/qwen3_embedding_0.6b"
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen3-embedding:0.6b",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--url",
        default="http://localhost:11434/api/embed",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    movies = pd.read_csv(args.movies)
    required = {"movie_id", "title", "genres"}
    if not required.issubset(movies.columns):
        raise ValueError(f"movies must contain {sorted(required)}")

    movies = movies.sort_values("movie_id").reset_index(drop=True)
    movie_ids = movies["movie_id"].to_numpy(dtype=np.int64)

    texts = [
        movie_text(row.title, row.genres)
        for row in movies.itertuples(index=False)
    ]

    started = time.perf_counter()
    batches: list[np.ndarray] = []
    expected_dim: int | None = None

    for start in range(0, len(texts), args.batch_size):
        stop = min(start + args.batch_size, len(texts))
        raw = embed_batch(
            texts[start:stop],
            model=args.model,
            url=args.url,
        )
        batch = np.asarray(raw, dtype=np.float32)

        if batch.ndim != 2:
            raise ValueError("Ollama embeddings must be 2D")

        if expected_dim is None:
            expected_dim = batch.shape[1]
        elif batch.shape[1] != expected_dim:
            raise ValueError("embedding dimension changed")

        batches.append(batch)
        print(f"embedded {stop}/{len(texts)}", flush=True)

    matrix = normalize_embeddings(
        np.concatenate(batches, axis=0)
    )

    if matrix.shape[0] != len(movie_ids):
        raise ValueError("final embedding count mismatch")

    args.output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output / "embeddings.npz",
        movie_ids=movie_ids,
        embeddings=matrix,
    )

    text_sha256 = hashlib.sha256(
        "\n\0\n".join(texts).encode("utf-8")
    ).hexdigest()

    metadata = {
        "text_sha256": text_sha256,
        "model": args.model,
        "template_version": TEMPLATE_VERSION,
        "text_fields": ["title", "genres"],
        "normalization": "l2",
        "movies": len(movie_ids),
        "embedding_dim": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "batch_size": args.batch_size,
        "source": str(args.movies),
        "seconds": time.perf_counter() - started,
    }

    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
