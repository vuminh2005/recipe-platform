"""Offline dataset loading, validation, and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


DATASET_NAME = "sklearn.datasets.load_breast_cancer"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass(frozen=True)
class DatasetSplits:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    test_features: pd.DataFrame
    train_target: pd.Series
    validation_target: pd.Series
    test_target: pd.Series
    feature_names: tuple[str, ...]
    class_mapping: dict[int, str]

    def summary(self, *, random_seed: int) -> dict[str, Any]:
        return {
            "dataset_name": DATASET_NAME,
            "sample_count": int(
                len(self.train_features)
                + len(self.validation_features)
                + len(self.test_features)
            ),
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "class_mapping": {
                str(key): value for key, value in self.class_mapping.items()
            },
            "split_ratios": {
                "train": TRAIN_RATIO,
                "validation": VALIDATION_RATIO,
                "test": TEST_RATIO,
            },
            "split_sizes": {
                "train": len(self.train_features),
                "validation": len(self.validation_features),
                "test": len(self.test_features),
            },
            "class_distributions": {
                "train": _class_distribution(self.train_target),
                "validation": _class_distribution(
                    self.validation_target
                ),
                "test": _class_distribution(self.test_target),
            },
            "random_seed": random_seed,
        }


def _class_distribution(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _validate_split(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    expected_feature_names: tuple[str, ...],
) -> None:
    if features.empty or target.empty:
        raise ValueError(f"{name} split must not be empty")
    if len(features) != len(target):
        raise ValueError(f"{name} feature and target counts differ")
    if tuple(features.columns) != expected_feature_names:
        raise ValueError(f"{name} split feature names changed")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} split contains non-finite feature values")
    classes = set(int(value) for value in target.unique())
    if classes != {0, 1}:
        raise ValueError(
            f"{name} split must contain both binary classes; found {classes}"
        )


def load_and_split_dataset(random_seed: int = 42) -> DatasetSplits:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.copy()
    feature_names = tuple(str(name) for name in dataset.feature_names)

    if tuple(features.columns) != feature_names:
        raise ValueError("Dataset feature names do not match feature columns")
    if dataset.target.name in features.columns:
        raise ValueError("Target leakage detected in feature columns")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite feature values")
    if set(int(value) for value in target.unique()) != {0, 1}:
        raise ValueError("Dataset target must contain exactly two classes")

    train_features, remainder_features, train_target, remainder_target = (
        train_test_split(
            features,
            target,
            test_size=VALIDATION_RATIO + TEST_RATIO,
            random_state=random_seed,
            stratify=target,
        )
    )
    (
        validation_features,
        test_features,
        validation_target,
        test_target,
    ) = train_test_split(
        remainder_features,
        remainder_target,
        test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
        random_state=random_seed,
        stratify=remainder_target,
    )

    splits = DatasetSplits(
        train_features=train_features.reset_index(drop=True),
        validation_features=validation_features.reset_index(drop=True),
        test_features=test_features.reset_index(drop=True),
        train_target=train_target.reset_index(drop=True),
        validation_target=validation_target.reset_index(drop=True),
        test_target=test_target.reset_index(drop=True),
        feature_names=feature_names,
        class_mapping={
            index: str(name)
            for index, name in enumerate(dataset.target_names)
        },
    )
    _validate_split(
        "training",
        splits.train_features,
        splits.train_target,
        feature_names,
    )
    _validate_split(
        "validation",
        splits.validation_features,
        splits.validation_target,
        feature_names,
    )
    _validate_split(
        "test",
        splits.test_features,
        splits.test_target,
        feature_names,
    )
    return splits
