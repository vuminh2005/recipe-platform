from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from trainer.data import collect_image_paths, prepare_dataset
from trainer.observability import (
    CATS_DOGS_DATASET_ID,
    DATASET_SOURCE_TYPE,
    build_dataset_metadata,
    safe_uri_metadata,
)

LOGGER = logging.getLogger("cats_dogs_validate")
EXPECTED_CLASSES = ("cats", "dogs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Cats & Dogs dataset before final training."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="KFP output-parameter file that receives validation JSON.",
    )
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def validate_split(root: Path, split: str) -> dict[str, object]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_root}")

    missing_classes = [
        class_name
        for class_name in EXPECTED_CLASSES
        if not (split_root / class_name).is_dir()
    ]
    if missing_classes:
        raise ValueError(
            f"Split {split!r} is missing class directories: {missing_classes}"
        )

    frame = collect_image_paths(split_root)
    if frame.empty:
        raise ValueError(f"Split {split!r} contains no supported images")

    counts = {
        class_name: int((frame["label_name"] == class_name[:-1]).sum())
        for class_name in EXPECTED_CLASSES
    }
    for class_name, count in counts.items():
        if count <= 0:
            raise ValueError(
                f"Split {split!r} contains no images for class {class_name!r}"
            )

    duplicate_count = int(frame["filepath"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Split {split!r} contains {duplicate_count} duplicate paths"
        )

    return {
        "image_count": int(len(frame)),
        "class_counts": counts,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    dataset_uri = required_env("DATASET_URI")
    source = safe_uri_metadata(
        dataset_uri,
        source_type=DATASET_SOURCE_TYPE,
        identifier=CATS_DOGS_DATASET_ID,
    )
    LOGGER.info(
        "Dataset source configured | type=%s | scheme=%s | id=%s",
        source["source_type"],
        source["scheme"],
        source["identifier"],
    )

    dataset_root = prepare_dataset(
        dataset_uri,
        work_dir="/tmp/cats-dogs-validation",
    )

    train_summary = validate_split(dataset_root, "train")
    test_summary = validate_split(dataset_root, "test")
    dataset_metadata = build_dataset_metadata(dataset_root)

    result = {
        "valid": True,
        **dataset_metadata,
        "dataset_fingerprint_sha256": dataset_metadata["dataset_checksum"],
        "train": train_summary,
        "test": test_summary,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # KFP OutputPath(str) expects the parameter value as text, so write one JSON string.
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    LOGGER.info(
        "Dataset valid | train=%s | test=%s | sha256=%s",
        train_summary["image_count"],
        test_summary["image_count"],
        dataset_metadata["dataset_checksum"],
    )


if __name__ == "__main__":
    main()
