"""Prepare the deterministic M4-v1 training manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from genrec.genrec import build_genrec_examples, sample_user_balanced

SEED = 42
SAMPLE_SIZE = 25_000
MIN_HISTORY = 50
CONTRACT = "m4-frozen-qwen-v1"


def main() -> None:
    data_dir = Path("data/processed/ml-1m")
    output_dir = Path("artifacts/m4")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    examples = build_genrec_examples(train, min_history=MIN_HISTORY)
    sampled = sample_user_balanced(
        examples,
        n=SAMPLE_SIZE,
        seed=SEED,
    )

    payload = {
        "contract": CONTRACT,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "min_history": MIN_HISTORY,
        "context_policy": "recent-complete-timestamp-groups",
        "examples": [
            {
                "user_id": x.user_id,
                "timestamp": x.timestamp,
                "history": list(x.history),
                "targets": list(x.targets),
            }
            for x in sampled
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(encoded).hexdigest()

    manifest = output_dir / "train_manifest.json"
    manifest.write_bytes(encoded)

    summary = {
        "contract": CONTRACT,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "min_history": MIN_HISTORY,
        "eligible_examples": len(examples),
        "eligible_users": len({x.user_id for x in examples}),
        "sampled_examples": len(sampled),
        "sampled_users": len({x.user_id for x in sampled}),
        "manifest_sha256": digest,
    }

    summary_path = output_dir / "train_manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print(f"manifest: {manifest}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
