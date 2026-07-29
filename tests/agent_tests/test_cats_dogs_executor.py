from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from agent.cats_dogs_katib import CatsDogsKatibResult
from agent.cats_dogs_executor import (
    _attach_kfp_lineage,
    execute_cats_dogs_job,
)


FINAL_RUN = {
    "run_id": "final-run",
    "tags": {
        "platform.result": "final_model_registered",
        "platform.model_uri": "runs:/final-run/model",
        "platform.registered_model_name": "actual-cats-model-name",
        "platform.registered_model_version": "7",
    },
    "metrics": {"val_auc": 0.91, "test_f1": 0.88},
}


class CatsDogsExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            mlflow_tracking_uri="http://mlflow",
            agent_id="test-agent",
            kfp_endpoint="http://kfp",
            katib_namespace="ml-platform",
            cats_dogs=SimpleNamespace(
                pipeline_path=Path("/tmp/cats-dogs.yaml"),
                mlflow_experiment_name="cats-dogs",
            ),
        )
        self.backend = Mock()
        self.base_job = {
            "id": "12345678-abcd",
            "recipe": {
                "name": "cats-dogs-recipe",
                "workload": "cats-dogs",
                "training": {
                    "model": "mobilenet_v2",
                    "trial_epochs": 2,
                    "final_epochs": 5,
                    "batch_size": 8,
                    "dense_units": 128,
                    "image_size": 224,
                    "trainable_backbone": True,
                },
                "automl": {
                    "enabled": True,
                    "max_trials": 4,
                    "parallel_trials": 2,
                    "algorithm": "random",
                    "search_space": {
                        "learning_rate": {"min": 0.0001, "max": 0.001},
                        "dropout_rate": {"min": 0.2, "max": 0.4},
                    },
                },
            },
        }

    @staticmethod
    def configure_integrations(mlflow_class, kfp_class) -> tuple[Mock, Mock]:
        mlflow = mlflow_class.return_value
        mlflow.get_or_create_experiment.return_value = "experiment-1"
        mlflow.ensure_parent_run.return_value = "parent-run"
        mlflow.find_latest_run.side_effect = [None, FINAL_RUN]

        kfp = kfp_class.return_value
        kfp.submit_pipeline.return_value = "kfp-run"
        kfp.get_status.return_value = "SUCCEEDED"
        kfp.is_success.return_value = True
        return mlflow, kfp

    @patch("agent.cats_dogs_executor.KfpRunner")
    @patch("agent.cats_dogs_executor.KatibRunner")
    @patch("agent.cats_dogs_executor.MlflowRestClient")
    @patch("agent.cats_dogs_executor.parse_experiment_result")
    @patch("agent.cats_dogs_executor.build_experiment_manifest")
    def test_enabled_automl_runs_katib_then_maps_best_params_to_kfp(
        self,
        build_manifest,
        parse_result,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        mlflow, kfp = self.configure_integrations(mlflow_class, kfp_class)
        katib = katib_class.return_value
        build_manifest.return_value = {
            "metadata": {"name": "cats-dogs-hpo-12345678"}
        }
        katib.wait_for_success.return_value = {
            "metadata": {"name": "cats-dogs-hpo-12345678"}
        }
        parse_result.return_value = CatsDogsKatibResult(
            experiment_name="cats-dogs-hpo-12345678",
            best_trial_name="best-trial",
            best_params={"learning_rate": 0.0007, "dropout_rate": 0.3},
            best_metric=0.93,
            metrics={"val_auc": 0.93},
        )

        execute_cats_dogs_job(
            self.base_job,
            settings=self.settings,
            backend=self.backend,
        )

        build_manifest.assert_called_once_with(
            namespace="ml-platform",
            job_id="12345678-abcd",
            parent_run_id="parent-run",
            trial_epochs=2,
            batch_size=8,
            dense_units=128,
            image_size=224,
            max_trial_count=4,
            parallel_trial_count=2,
            algorithm_name="random",
            learning_rate_min=0.0001,
            learning_rate_max=0.001,
            dropout_rate_min=0.2,
            dropout_rate_max=0.4,
            trainable_backbone=True,
        )
        arguments = kfp.submit_pipeline.call_args.kwargs["arguments"]
        self.assertEqual(arguments["learning_rate"], 0.0007)
        self.assertEqual(arguments["dropout_rate"], 0.3)
        self.assertEqual(arguments["recipe_id"], "cats-dogs")
        self.assertEqual(arguments["recipe_version"], "1.0")
        self.assertTrue(arguments["trainable_backbone"])
        self.assertEqual(
            arguments["katib_experiment_name"],
            "cats-dogs-hpo-12345678",
        )
        self.backend.patch_job.assert_has_calls(
            [
                call("12345678-abcd", status="TUNING", error_message=""),
                call(
                    "12345678-abcd",
                    result_patch={
                        "external_ids": {
                            "katib_experiment_id": (
                                "cats-dogs-hpo-12345678"
                            ),
                        }
                    },
                ),
                call(
                    "12345678-abcd",
                    status="TRAINING",
                    result_patch={
                        "objective": {"value": 0.93},
                        "best_params": {
                            "learning_rate": 0.0007,
                            "dropout_rate": 0.3,
                        },
                    },
                ),
            ]
        )
        success_patch = [
            item
            for item in self.backend.patch_job.call_args_list
            if item.kwargs.get("status") == "SUCCEEDED"
        ][0]
        self.assertEqual(
            success_patch.kwargs["result_patch"]["model"][
                "registered_name"
            ],
            "actual-cats-model-name",
        )
        mlflow.terminate_run.assert_called_with("parent-run", status="FINISHED")
        mlflow.set_run_tag.assert_called_with(
            run_id="final-run",
            key="platform.kfp_run_id",
            value="kfp-run",
        )

    @patch("agent.cats_dogs_executor.KfpRunner")
    @patch("agent.cats_dogs_executor.KatibRunner")
    @patch("agent.cats_dogs_executor.MlflowRestClient")
    @patch("agent.cats_dogs_executor.build_experiment_manifest")
    def test_disabled_automl_skips_katib_and_uses_recipe_defaults(
        self,
        build_manifest,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        self.configure_integrations(mlflow_class, kfp_class)
        self.base_job["recipe"]["automl"]["enabled"] = False

        execute_cats_dogs_job(
            self.base_job,
            settings=self.settings,
            backend=self.backend,
        )

        katib_class.assert_not_called()
        build_manifest.assert_not_called()
        arguments = kfp_class.return_value.submit_pipeline.call_args.kwargs[
            "arguments"
        ]
        self.assertEqual(arguments["learning_rate"], 0.0003)
        self.assertEqual(arguments["dropout_rate"], 0.25)
        self.assertEqual(arguments["katib_experiment_name"], "")
        self.assertTrue(arguments["trainable_backbone"])
        statuses = [
            item.kwargs.get("status")
            for item in self.backend.patch_job.call_args_list
        ]
        self.assertNotIn("TUNING", statuses)
        self.assertIn("TRAINING", statuses)
        self.assertIn("SUCCEEDED", statuses)
        result_patches = [
            item.kwargs["result_patch"]
            for item in self.backend.patch_job.call_args_list
            if "result_patch" in item.kwargs
        ]
        self.assertFalse(
            any("best_params" in patch for patch in result_patches)
        )
        self.assertFalse(
            any("objective" in patch for patch in result_patches)
        )
        self.assertFalse(
            any(
                patch.get("external_ids", {}).get(
                    "katib_experiment_id"
                )
                for patch in result_patches
            )
        )

    @patch("agent.cats_dogs_executor.KfpRunner")
    @patch("agent.cats_dogs_executor.KatibRunner")
    @patch("agent.cats_dogs_executor.MlflowRestClient")
    def test_disabled_automl_uses_normalized_effective_parameters(
        self,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        self.configure_integrations(mlflow_class, kfp_class)
        job = deepcopy(self.base_job)
        job["recipe"]["configuration"] = {
            "training": job["recipe"]["training"],
            "automl": {
                **job["recipe"]["automl"],
                "enabled": False,
            },
            "effective_final_parameters": {
                "learning_rate": 0.0004,
                "dropout_rate": 0.35,
            },
        }

        execute_cats_dogs_job(
            job,
            settings=self.settings,
            backend=self.backend,
        )

        katib_class.assert_not_called()
        arguments = kfp_class.return_value.submit_pipeline.call_args.kwargs[
            "arguments"
        ]
        self.assertEqual(arguments["learning_rate"], 0.0004)
        self.assertEqual(arguments["dropout_rate"], 0.35)
        result_patches = [
            item.kwargs["result_patch"]
            for item in self.backend.patch_job.call_args_list
            if "result_patch" in item.kwargs
        ]
        self.assertFalse(
            any("best_params" in patch for patch in result_patches)
        )

    @patch("agent.cats_dogs_executor.KfpRunner")
    @patch("agent.cats_dogs_executor.KatibRunner")
    @patch("agent.cats_dogs_executor.MlflowRestClient")
    @patch("agent.cats_dogs_executor.build_experiment_manifest")
    def test_invalid_algorithm_fails_before_any_integration_is_created(
        self,
        build_manifest,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        self.base_job["recipe"]["automl"]["algorithm"] = "tpe"

        with self.assertRaisesRegex(ValueError, "Unsupported automl.algorithm"):
            execute_cats_dogs_job(
                self.base_job,
                settings=self.settings,
                backend=self.backend,
            )

        self.backend.patch_job.assert_called_once_with(
            "12345678-abcd",
            status="FAILED",
            error_message=(
                "Unsupported automl.algorithm 'tpe'; supported: random"
            ),
        )
        mlflow_class.assert_not_called()
        katib_class.assert_not_called()
        kfp_class.assert_not_called()
        build_manifest.assert_not_called()

    @patch("agent.cats_dogs_executor.KfpRunner")
    @patch("agent.cats_dogs_executor.KatibRunner")
    @patch("agent.cats_dogs_executor.MlflowRestClient")
    def test_explicit_unsupported_version_fails_before_integrations(
        self,
        mlflow_class,
        katib_class,
        kfp_class,
    ) -> None:
        self.base_job["recipe"]["recipe_version"] = "2.0"

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Cats & Dogs recipe version",
        ):
            execute_cats_dogs_job(
                self.base_job,
                settings=self.settings,
                backend=self.backend,
            )

        mlflow_class.assert_not_called()
        katib_class.assert_not_called()
        kfp_class.assert_not_called()

    @patch("agent.cats_dogs_executor.time.sleep")
    def test_kfp_lineage_failure_is_bounded_and_non_fatal(
        self,
        sleep,
    ) -> None:
        mlflow = Mock()
        mlflow.set_run_tag.side_effect = RuntimeError("MLflow unavailable")

        with self.assertLogs("cats_dogs_executor", level="ERROR"):
            attached = _attach_kfp_lineage(
                mlflow,
                final_run=FINAL_RUN,
                kfp_run_id="kfp-run",
            )

        self.assertFalse(attached)
        self.assertEqual(mlflow.set_run_tag.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
