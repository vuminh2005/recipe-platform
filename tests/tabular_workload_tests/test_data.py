from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKLOAD_ROOT = (
    Path(__file__).resolve().parents[2]
    / "workloads"
    / "tabular-random-forest"
)
sys.path.insert(0, str(WORKLOAD_ROOT))

from trainer.data import load_and_split_dataset  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_builtin_dataset_loads_offline_and_is_valid(self) -> None:
        splits = load_and_split_dataset(42)

        self.assertEqual(len(splits.feature_names), 30)
        self.assertEqual(
            len(splits.train_features)
            + len(splits.validation_features)
            + len(splits.test_features),
            569,
        )
        self.assertEqual(
            splits.class_mapping,
            {0: "malignant", 1: "benign"},
        )
        for features, target in (
            (splits.train_features, splits.train_target),
            (splits.validation_features, splits.validation_target),
            (splits.test_features, splits.test_target),
        ):
            self.assertTrue(np.isfinite(features.to_numpy()).all())
            self.assertEqual(set(target.unique()), {0, 1})
            self.assertEqual(tuple(features.columns), splits.feature_names)

    def test_split_is_deterministic_for_the_same_seed(self) -> None:
        first = load_and_split_dataset(7)
        second = load_and_split_dataset(7)

        pd.testing.assert_frame_equal(
            first.train_features,
            second.train_features,
        )
        pd.testing.assert_series_equal(
            first.validation_target,
            second.validation_target,
        )
        pd.testing.assert_frame_equal(
            first.test_features,
            second.test_features,
        )

    def test_stratification_preserves_class_proportions(self) -> None:
        splits = load_and_split_dataset(42)
        overall_positive_ratio = (
            sum(
                int(target.sum())
                for target in (
                    splits.train_target,
                    splits.validation_target,
                    splits.test_target,
                )
            )
            / 569
        )
        for target in (
            splits.train_target,
            splits.validation_target,
            splits.test_target,
        ):
            self.assertAlmostEqual(
                float(target.mean()),
                overall_positive_ratio,
                delta=0.02,
            )


if __name__ == "__main__":
    unittest.main()
