from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from agent.tabular_random_forest_executor import (
    _attach_kfp_lineage,
    execute_tabular_random_forest_job,
)
from agent.tabular_random_forest_katib import TabularKatibResult


FINAL_METRICS = {
    "val_accuracy": 0.95,
    "val_precision": 0.96,
    "val_recall": 0.94,
    "val_f1": 0.95,
    "val_roc_auc": 0.98,
    "test_accuracy": 0.94,
    "test_precision": 0.95,
    "test_recall": 0.93,
    "test_f1": 0.94,
    "test_roc_auc": 0.98,
}
FINAL_RUN = {
    "run_id": "final-run",
    "tags": {
        "platform.result": "final_model_registered",
        "platform.model_uri": "runs:/final-run/model",
        "platform.registered_model_name": (
            "tabular_random_forest_classifier"
        ),
        "platform.registered_model_version": "4",
    },
    "metrics": FINAL_METRICS,
}


class TabularExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            mlflow_tracking_uri="http://mlflow",
            agent_id="test-agent",
            kfp_endpoint="http://kfp",
            katib_namespace="ml-platform",
            tabular_random_forest=SimpleNamespace(
                pipeline_path=Path("/tmp/tabular-rf.yaml"),
                mlflow_experiment_name="tabular-experiment",
                registered_model_name="actual-tabular-model-name",
            ),
        )
        self.backend = Mock()
        self.base_job = {
            "id": "12345678-abcd",
            "recipe": {
                "name": "tabular-job",
                "recipe_id": "tabular-random-forest",
                "recipe_version": "1.0",
                "configuration": {
                    "training": {"random_seed": 42},
                    "automl": {
                        "enabled": True,
                        "max_trials": 4,
                        "parallel_trials": 2,
                        "algorithm": "random",
                        "search_space": {
                            "n_estimators": {"min": 60, "max": 250},
                            "max_depth": {"min": 3, "max": 15},
                            "min_samples_split": {"min": 2, "max": 8},
                        },
                    },
                    "effective_final_parameters": None,
                },
            },
        }

    @staticmethod
    def configure_integrations(mlflow_class, kfp_class):
        mlflow = mlflow_class.return_value
        mlflow.get_or_create_experiment.return_value = "experiment-1"
        mlflow.ensure_parent_run.return_value = "parent-run"
        mlflow.find_latest_run.side_effect = [None, FINAL_RUN]

        kfp = kfp_class.return_value
        kfp.submit_pipeline.return_value = "kfp-run"
        kfp.get_status.return_value = "SUCCEEDED"
        kfp.is_success.return_value = True
        return mlflow, kfp

    @patch("agent.tabular_random_forest_executor.KfpRunner")
    @patch("agent.tabular_random_forest_executor.KatibRunner")
    @patch("agent.tabular_random_forest_executor.MlflowRestClient")
    @patch("agent.tabular_random_forest_executor.parse_experiment_result")
    @patch("agent.tabular_random_forest_executor.build_experiment_manifest")
    def test_automl_enabled_uses_typed_katib_values_and_sequential_patches(
        self,
        build_manifest,
        parse_result,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        mlflow, kfp = self.configure_integrations(
            mlflow_class,
            kfp_class,
        )
        build_manifest.return_value = {
            "metadata": {"name": "tabular-rf-hpo-12345678"}
        }
        katib_class.return_value.wait_for_success.return_value = {}
        parse_result.return_value = TabularKatibResult(
            experiment_name="tabular-rf-hpo-12345678",
            best_trial_name="best-trial",
            best_params={
                "n_estimators": 187,
                "max_depth": 9,
                "min_samples_split": 4,
            },
            best_metric=0.9521,
            metrics={"val_f1": 0.9521},
        )

        execute_tabular_random_forest_job(
            self.base_job,
            settings=self.settings,
            backend=self.backend,
        )

        arguments = kfp.submit_pipeline.call_args.kwargs["arguments"]
        self.assertEqual(arguments["n_estimators"], 187)
        self.assertEqual(arguments["max_depth"], 9)
        self.assertEqual(arguments["min_samples_split"], 4)
        self.assertEqual(arguments["max_features"], "sqrt")
        self.assertEqual(
            arguments["registered_model_name"],
            "actual-tabular-model-name",
        )
        self.assertEqual(
            arguments["katib_experiment_id"],
            "tabular-rf-hpo-12345678",
        )
        self.backend.patch_job.assert_has_calls(
            [
                call(
                    "12345678-abcd",
                    result_patch={
                        "external_ids": {
                            "katib_experiment_id": (
                                "tabular-rf-hpo-12345678"
                            )
                        }
                    },
                ),
                call(
                    "12345678-abcd",
                    status="TRAINING",
                    result_patch={
                        "objective": {"value": 0.9521},
                        "best_params": {
                            "n_estimators": 187,
                            "max_depth": 9,
                            "min_samples_split": 4,
                        },
                    },
                ),
                call(
                    "12345678-abcd",
                    result_patch={
                        "external_ids": {"kfp_run_id": "kfp-run"}
                    },
                ),
            ],
            any_order=False,
        )
        success_call = [
            item
            for item in self.backend.patch_job.call_args_list
            if item.kwargs.get("status") == "SUCCEEDED"
        ][0]
        self.assertEqual(
            success_call.kwargs["result_patch"]["model"][
                "registered_name"
            ],
            "tabular_random_forest_classifier",
        )
        mlflow.set_run_tag.assert_called_with(
            run_id="final-run",
            key="platform.kfp_run_id",
            value="kfp-run",
        )

    @patch("agent.tabular_random_forest_executor.KfpRunner")
    @patch("agent.tabular_random_forest_executor.KatibRunner")
    @patch("agent.tabular_random_forest_executor.MlflowRestClient")
    @patch("agent.tabular_random_forest_executor.build_experiment_manifest")
    def test_automl_disabled_skips_katib_and_uses_snapshot_parameters(
        self,
        build_manifest,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        self.configure_integrations(mlflow_class, kfp_class)
        job = deepcopy(self.base_job)
        job["recipe"]["configuration"]["automl"]["enabled"] = False
        job["recipe"]["configuration"]["effective_final_parameters"] = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "random_seed": 42,
        }

        execute_tabular_random_forest_job(
            job,
            settings=self.settings,
            backend=self.backend,
        )

        katib_class.assert_not_called()
        build_manifest.assert_not_called()
        arguments = kfp_class.return_value.submit_pipeline.call_args.kwargs[
            "arguments"
        ]
        self.assertEqual(arguments["n_estimators"], 200)
        self.assertNotIn("katib_experiment_id", arguments)
        patches = [
            item.kwargs["result_patch"]
            for item in self.backend.patch_job.call_args_list
            if "result_patch" in item.kwargs
        ]
        self.assertFalse(any("objective" in patch for patch in patches))
        self.assertFalse(any("best_params" in patch for patch in patches))
        self.assertFalse(
            any(
                "katib_experiment_id"
                in patch.get("external_ids", {})
                for patch in patches
            )
        )

    @patch("agent.tabular_random_forest_executor.KfpRunner")
    @patch("agent.tabular_random_forest_executor.MlflowRestClient")
    def test_invalid_persisted_config_fails_before_external_clients(
        self,
        mlflow_class,
        kfp_class,
    ) -> None:
        self.base_job["recipe"]["configuration"]["automl"][
            "algorithm"
        ] = "tpe"

        with self.assertRaisesRegex(ValueError, "Unsupported automl.algorithm"):
            execute_tabular_random_forest_job(
                self.base_job,
                settings=self.settings,
                backend=self.backend,
            )

        mlflow_class.assert_not_called()
        kfp_class.assert_not_called()

    @patch("agent.tabular_random_forest_executor.time.sleep")
    def test_lineage_failure_is_bounded_and_non_fatal(self, sleep) -> None:
        mlflow = Mock()
        mlflow.set_run_tag.side_effect = RuntimeError("metadata unavailable")

        attached = _attach_kfp_lineage(
            mlflow,
            final_run=FINAL_RUN,
            kfp_run_id="kfp-run",
        )

        self.assertFalse(attached)
        self.assertEqual(mlflow.set_run_tag.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        mlflow.set_model_version_tag.assert_not_called()


if __name__ == "__main__":
    unittest.main()
