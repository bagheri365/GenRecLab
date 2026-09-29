#!/usr/bin/env python
"""Download (optionally) and prepare MovieLens 1M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genrec.movielens import download_movielens_1m, prepare_movielens_1m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Path to extracted ml-1m directory.")
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ml-1m"))
    parser.add_argument("--min-train-items", type=int, default=3)
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = download_movielens_1m(args.download_dir)

    stats = prepare_movielens_1m(
        source,
        args.output,
        min_train_items=args.min_train_items,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
