from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


WORKLOAD_ROOT = (
    Path(__file__).resolve().parents[2]
    / "workloads"
    / "tabular-random-forest"
)
sys.path.insert(0, str(WORKLOAD_ROOT))

from trainer.data import load_and_split_dataset  # noqa: E402
from trainer.metrics import evaluate_classifier  # noqa: E402
from trainer.model import build_model  # noqa: E402


class ModelAndMetricsTests(unittest.TestCase):
    def test_default_model_parameters_are_recipe_owned(self) -> None:
        model = build_model()

        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_depth, 8)
        self.assertEqual(model.min_samples_split, 2)
        self.assertEqual(model.max_features, "sqrt")
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, 1)

    def test_training_is_deterministic_for_fixed_seed(self) -> None:
        splits = load_and_split_dataset(42)
        first = build_model(n_estimators=25, random_seed=42)
        second = build_model(n_estimators=25, random_seed=42)
        first.fit(splits.train_features, splits.train_target)
        second.fit(splits.train_features, splits.train_target)

        np.testing.assert_array_equal(
            first.predict(splits.test_features),
            second.predict(splits.test_features),
        )

    def test_metrics_include_probability_based_roc_auc(self) -> None:
        model = Mock()
        model.predict.return_value = np.array([0, 0, 1, 1])
        model.predict_proba.return_value = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
        target = np.array([0, 0, 1, 1])

        metrics, matrix = evaluate_classifier(
            model,
            np.zeros((4, 2)),
            target,
            prefix="test",
        )

        self.assertEqual(metrics["test_accuracy"], 1.0)
        self.assertEqual(metrics["test_f1"], 1.0)
        self.assertEqual(metrics["test_roc_auc"], 1.0)
        self.assertEqual(matrix, [[2, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()
