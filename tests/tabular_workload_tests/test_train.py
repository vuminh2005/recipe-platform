from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlflow import MlflowClient


WORKLOAD_ROOT = (
    Path(__file__).resolve().parents[2]
    / "workloads"
    / "tabular-random-forest"
)
sys.path.insert(0, str(WORKLOAD_ROOT))

from trainer.train import run  # noqa: E402


def arguments(*, mode: str, result_path: str | None = None):
    return argparse.Namespace(
        mode=mode,
        n_estimators=20,
        max_depth=6,
        min_samples_split=2,
        max_features="sqrt",
        random_seed=42,
        recipe_id="tabular-random-forest",
        recipe_version="1.0",
        mlflow_experiment_name="tabular-workload-test",
        result_path=result_path,
    )


class TrainingTests(unittest.TestCase):
    def test_trial_prints_exact_katib_metrics_and_tags_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = Path(directory).resolve().as_uri()
            environment = {
                "MLFLOW_TRACKING_URI": tracking_uri,
                "PLATFORM_JOB_ID": "job-trial",
                "MLFLOW_PARENT_RUN_ID": "parent-run",
                "KATIB_EXPERIMENT_NAME": "tabular-hpo",
                "KATIB_TRIAL_NAME": "trial-1",
            }
            output = io.StringIO()
            with patch.dict(os.environ, environment, clear=False):
                with contextlib.redirect_stdout(output):
                    result = run(arguments(mode="trial"))

            lines = output.getvalue().splitlines()
            for metric in (
                "val_f1",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_roc_auc",
            ):
                self.assertTrue(
                    any(line.startswith(f"{metric}=") for line in lines)
                )

            mlflow_run = MlflowClient(
                tracking_uri=tracking_uri
            ).get_run(result["mlflow_run_id"])
            self.assertEqual(
                mlflow_run.data.tags["platform.recipe_id"],
                "tabular-random-forest",
            )
            self.assertEqual(
                mlflow_run.data.tags["platform.katib_experiment_id"],
                "tabular-hpo",
            )

    def test_final_logs_model_metrics_and_recipe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = Path(directory).resolve().as_uri()
            result_path = Path(directory) / "result.json"
            environment = {
                "MLFLOW_TRACKING_URI": tracking_uri,
                "PLATFORM_JOB_ID": "job-final",
                "MLFLOW_PARENT_RUN_ID": "parent-run",
            }
            with patch.dict(os.environ, environment, clear=False):
                result = run(
                    arguments(
                        mode="final",
                        result_path=str(result_path),
                    )
                )

            self.assertTrue(result_path.is_file())
            for metric in (
                "val_f1",
                "val_roc_auc",
                "test_f1",
                "test_roc_auc",
            ):
                self.assertIn(metric, result)
            self.assertEqual(
                result["probability_output"],
                "predict_proba_class_1_benign",
            )
            self.assertEqual(result["feature_count"], 30)

            mlflow_run = MlflowClient(
                tracking_uri=tracking_uri
            ).get_run(result["mlflow_run_id"])
            self.assertEqual(
                mlflow_run.data.tags["platform.result"],
                "final_model_logged",
            )
            serialized_tags = str(mlflow_run.data.tags).lower()
            for forbidden in (
                "mobilenet",
                "prob_dog",
                "image_size",
                "final_threshold",
            ):
                self.assertNotIn(forbidden, serialized_tags)


if __name__ == "__main__":
    unittest.main()
