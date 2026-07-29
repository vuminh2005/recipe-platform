from __future__ import annotations

import inspect
import unittest

import agent.katib_runner as generic_katib
from agent.tabular_random_forest_katib import (
    ADDITIONAL_METRICS,
    OBJECTIVE_METRIC,
    TRAINER_IMAGE,
    build_experiment_manifest,
    parse_experiment_result,
)


class TabularKatibTests(unittest.TestCase):
    def manifest(self, **overrides):
        values = {
            "namespace": "ml-platform",
            "job_id": "12345678-abcd",
            "parent_run_id": "parent-run",
            "mlflow_experiment_name": "tabular-experiment",
            "recipe_version": "1.0",
            "random_seed": 42,
            "max_trial_count": 3,
            "parallel_trial_count": 1,
            "algorithm_name": "random",
            "n_estimators_min": 50,
            "n_estimators_max": 300,
            "max_depth_min": 2,
            "max_depth_max": 20,
            "min_samples_split_min": 2,
            "min_samples_split_max": 10,
        }
        values.update(overrides)
        return build_experiment_manifest(**values)

    def test_manifest_owns_objective_integer_spaces_image_and_command(
        self,
    ) -> None:
        manifest = self.manifest()
        spec = manifest["spec"]
        container = spec["trialTemplate"]["trialSpec"]["spec"]["template"][
            "spec"
        ]["containers"][0]

        self.assertEqual(spec["objective"]["objectiveMetricName"], OBJECTIVE_METRIC)
        self.assertEqual(
            spec["objective"]["additionalMetricNames"],
            list(ADDITIONAL_METRICS),
        )
        self.assertEqual(
            [parameter["parameterType"] for parameter in spec["parameters"]],
            ["int", "int", "int"],
        )
        self.assertEqual(container["image"], TRAINER_IMAGE)
        self.assertEqual(
            container["command"],
            ["python", "-m", "trainer.train"],
        )
        self.assertIn("--max-features=sqrt", container["args"])

    def test_secret_injection_is_individual_and_generic_only(self) -> None:
        manifest = self.manifest()
        container = manifest["spec"]["trialTemplate"]["trialSpec"]["spec"][
            "template"
        ]["spec"]["containers"][0]

        self.assertNotIn("envFrom", container)
        secret_names = {
            item["name"]
            for item in container["env"]
            if "valueFrom" in item
        }
        self.assertEqual(
            secret_names,
            {
                "MLFLOW_TRACKING_URI",
                "MLFLOW_S3_ENDPOINT_URL",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
            },
        )
        for forbidden in (
            "DATASET_URI",
            "MLFLOW_REGISTERED_MODEL_NAME",
            "MLFLOW_EXPERIMENT_NAME",
        ):
            self.assertNotIn(forbidden, secret_names)

    def test_parser_converts_each_assignment_to_int(self) -> None:
        result = parse_experiment_result(
            {
                "metadata": {"name": "tabular-rf-hpo-12345678"},
                "status": {
                    "currentOptimalTrial": {
                        "bestTrialName": "best-trial",
                        "parameterAssignments": [
                            {"name": "n_estimators", "value": "187"},
                            {"name": "max_depth", "value": "9"},
                            {"name": "min_samples_split", "value": "4"},
                        ],
                        "observation": {
                            "metrics": [
                                {"name": "val_f1", "max": "0.9521"},
                                {"name": "val_accuracy", "max": "0.947"},
                            ]
                        },
                    }
                },
            }
        )

        self.assertEqual(
            result.best_params,
            {
                "n_estimators": 187,
                "max_depth": 9,
                "min_samples_split": 4,
            },
        )
        self.assertTrue(
            all(type(value) is int for value in result.best_params.values())
        )
        self.assertEqual(result.best_metric, 0.9521)

    def test_parser_rejects_non_integer_and_missing_metrics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid integer"):
            parse_experiment_result(
                {
                    "metadata": {"name": "bad"},
                    "status": {
                        "currentOptimalTrial": {
                            "parameterAssignments": [
                                {"name": "n_estimators", "value": "100.5"},
                                {"name": "max_depth", "value": "8"},
                                {
                                    "name": "min_samples_split",
                                    "value": "2",
                                },
                            ],
                            "observation": {
                                "metrics": [
                                    {"name": "val_f1", "max": "0.9"}
                                ]
                            },
                        }
                    },
                }
            )

    def test_parser_preserves_zero_objective(self) -> None:
        result = parse_experiment_result(
            {
                "metadata": {"name": "zero-objective"},
                "status": {
                    "currentOptimalTrial": {
                        "parameterAssignments": [
                            {"name": "n_estimators", "value": "100"},
                            {"name": "max_depth", "value": "8"},
                            {"name": "min_samples_split", "value": "2"},
                        ],
                        "observation": {
                            "metrics": [{"name": "val_f1", "max": 0.0}]
                        },
                    }
                },
            }
        )
        self.assertEqual(result.best_metric, 0.0)

    def test_parser_rejects_missing_objective(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without val_f1"):
            parse_experiment_result(
                {
                    "metadata": {"name": "missing-objective"},
                    "status": {
                        "currentOptimalTrial": {
                            "parameterAssignments": [
                                {"name": "n_estimators", "value": "100"},
                                {"name": "max_depth", "value": "8"},
                                {
                                    "name": "min_samples_split",
                                    "value": "2",
                                },
                            ],
                            "observation": {"metrics": []},
                        }
                    },
                }
            )

    def test_parser_rejects_unexpected_and_missing_parameters(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected parameter"):
            parse_experiment_result(
                {
                    "metadata": {"name": "unexpected"},
                    "status": {
                        "currentOptimalTrial": {
                            "parameterAssignments": [
                                {"name": "unexpected", "value": "1"}
                            ]
                        }
                    },
                }
            )
        with self.assertRaisesRegex(RuntimeError, "min_samples_split"):
            parse_experiment_result(
                {
                    "metadata": {"name": "missing"},
                    "status": {
                        "currentOptimalTrial": {
                            "parameterAssignments": [
                                {"name": "n_estimators", "value": "100"},
                                {"name": "max_depth", "value": "8"},
                            ]
                        }
                    },
                }
            )

    def test_generic_katib_runner_has_no_tabular_assumptions(self) -> None:
        source = inspect.getsource(generic_katib)
        for forbidden in (
            "tabular",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "val_f1",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
