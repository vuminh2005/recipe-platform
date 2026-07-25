from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from trainer.data import collect_image_paths, prepare_dataset

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


def dataset_fingerprint(root: Path) -> str:
    """Build a deterministic SHA-256 over relative paths, sizes and bytes."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())

    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    return digest.hexdigest()


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

    dataset_root = prepare_dataset(
        dataset_uri,
        work_dir="/tmp/cats-dogs-validation",
    )

    train_summary = validate_split(dataset_root, "train")
    test_summary = validate_split(dataset_root, "test")
    fingerprint = dataset_fingerprint(dataset_root)

    result = {
        "valid": True,
        "dataset_uri": dataset_uri,
        "dataset_fingerprint_sha256": fingerprint,
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
        fingerprint,
    )


if __name__ == "__main__":
    main()
