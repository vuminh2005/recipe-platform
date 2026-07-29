"""Validate the built-in dataset and emit reproducible metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.data import load_and_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_and_split_dataset(args.random_seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            splits.summary(random_seed=args.random_seed),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Validated built-in dataset: {output_path}", flush=True)


if __name__ == "__main__":
    main()
