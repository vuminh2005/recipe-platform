from __future__ import annotations

import unittest
import json
from datetime import datetime, timezone
from unittest.mock import Mock

from pydantic import ValidationError
from fastapi import HTTPException

from backend.app.job_contracts import JobResponse, RecipeCreate
from backend.app.main import create_job
from backend.app.recipe_normalization import (
    RecipeNormalizationError,
    normalize_recipe_request,
)


class RecipeContractTests(unittest.TestCase):
    def test_explicit_recipe_id_is_accepted_and_persisted(self) -> None:
        db = Mock()
        job = create_job(
            RecipeCreate(
                name="explicit-cats-dogs",
                recipe_id="cats-dogs",
                workload="hello",
            ),
            db=db,
        )

        self.assertEqual(job.recipe["recipe_id"], "cats-dogs")
        self.assertEqual(job.recipe["recipe_version"], "1.0")
        self.assertEqual(job.recipe["workload"], "cats-dogs")

    def test_legacy_request_is_normalized_with_compatibility_mirrors(self) -> None:
        db = Mock()
        job = create_job(
            RecipeCreate(
                name="legacy-cats-dogs",
                workload="cats-dogs",
            ),
            db=db,
        )

        self.assertEqual(job.recipe["recipe_id"], "cats-dogs")
        self.assertEqual(job.recipe["recipe_version"], "1.0")
        self.assertEqual(job.recipe["workload"], "cats-dogs")
        self.assertEqual(
            job.recipe["training"],
            job.recipe["configuration"]["training"],
        )
        self.assertEqual(
            job.recipe["automl"],
            job.recipe["configuration"]["automl"],
        )

    def test_old_cats_dogs_request_gets_current_defaults(self) -> None:
        recipe = normalize_recipe_request(
            RecipeCreate(
                name="cats-dogs-recipe",
                workload="cats-dogs",
            )
        )

        configuration = recipe["configuration"]
        self.assertEqual(configuration["training"]["model"], "mobilenet_v2")
        self.assertEqual(configuration["training"]["trial_epochs"], 2)
        self.assertEqual(configuration["training"]["final_epochs"], 5)
        self.assertEqual(
            configuration["automl"]["search_space"],
            {
                "learning_rate": {"min": 0.00005, "max": 0.0005},
                "dropout_rate": {"min": 0.15, "max": 0.45},
            },
        )

    def test_custom_search_space_is_normalized(self) -> None:
        request = RecipeCreate.model_validate(
            {
                "name": "cats-dogs-custom",
                "recipe_id": "cats-dogs",
                "configuration": {
                    "training": {"batch_size": 16},
                    "automl": {
                        "max_trials": 4,
                        "parallel_trials": 2,
                        "algorithm": "random",
                        "search_space": {
                            "learning_rate": {
                                "min": "0.0001",
                                "max": "0.001",
                            },
                            "dropout_rate": {"min": "0.2", "max": "0.4"},
                        },
                    },
                },
            }
        )
        recipe = normalize_recipe_request(request)

        configuration = recipe["configuration"]
        self.assertEqual(configuration["training"]["batch_size"], 16)
        self.assertEqual(configuration["automl"]["parallel_trials"], 2)
        self.assertEqual(
            configuration["automl"]["search_space"]["learning_rate"]["min"],
            0.0001,
        )
        self.assertEqual(
            configuration["automl"]["search_space"]["dropout_rate"]["max"],
            0.4,
        )

    def test_parallel_trials_cannot_exceed_max_trials(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "parallel_trials must be less than or equal to max_trials",
        ):
            RecipeCreate.model_validate(
                {
                    "name": "cats-dogs-invalid",
                    "workload": "cats-dogs",
                    "automl": {"max_trials": 1, "parallel_trials": 2},
                }
            )

    def test_unsupported_algorithm_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RecipeCreate.model_validate(
                {
                    "name": "cats-dogs-invalid",
                    "workload": "cats-dogs",
                    "automl": {"algorithm": "tpe"},
                }
            )

    def test_invalid_search_space_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "learning_rate.min must be less than learning_rate.max",
        ):
            RecipeCreate.model_validate(
                {
                    "name": "cats-dogs-invalid",
                    "workload": "cats-dogs",
                    "automl": {
                        "search_space": {
                            "learning_rate": {"min": 0.001, "max": 0.0001}
                        }
                    },
                }
            )

    def test_legacy_tiny_cnn_is_normalized_with_warning(self) -> None:
        with self.assertLogs("recipe_platform.contracts", level="WARNING"):
            recipe = normalize_recipe_request(
                RecipeCreate.model_validate(
                    {
                        "name": "legacy-cats-dogs",
                        "workload": "cats-dogs",
                        "training": {"model": "tiny_cnn"},
                    }
                )
            )

        self.assertEqual(
            recipe["configuration"]["training"]["model"],
            "mobilenet_v2",
        )
        snapshot_json = json.dumps(recipe["recipe_snapshot"])
        self.assertNotIn("tiny_cnn", snapshot_json)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RecipeCreate.model_validate(
                {
                    "name": "cats-dogs-invalid",
                    "workload": "cats-dogs",
                    "training": {"model": "resnet50"},
                }
            )

    def test_legacy_epochs_is_retained_but_warned_as_ignored(self) -> None:
        with self.assertLogs("recipe_platform.contracts", level="WARNING"):
            recipe = normalize_recipe_request(
                RecipeCreate.model_validate(
                    {
                        "name": "legacy-cats-dogs",
                        "workload": "cats-dogs",
                        "training": {"epochs": 3},
                    }
                )
            )

        training = recipe["configuration"]["training"]
        self.assertNotIn("epochs", training)
        self.assertEqual(training["trial_epochs"], 2)
        self.assertEqual(training["final_epochs"], 5)
        self.assertNotIn(
            "epochs",
            recipe["recipe_snapshot"]["configuration"]["training"],
        )

    def test_legacy_aliases_remain_supported(self) -> None:
        recipe = normalize_recipe_request(
            RecipeCreate.model_validate(
                {
                    "name": "legacy-aliases",
                    "workload": "cats-dogs",
                    "training_config": {"batch_size": 12},
                    "automl_config": {"enabled": False},
                }
            )
        )

        configuration = recipe["configuration"]
        self.assertEqual(configuration["training"]["batch_size"], 12)
        self.assertFalse(configuration["automl"]["enabled"])
        self.assertEqual(
            configuration["effective_final_parameters"],
            {"learning_rate": 0.0003, "dropout_rate": 0.25},
        )

    def test_legacy_hello_remains_supported_without_fake_ml_config(self) -> None:
        with self.assertLogs("recipe_platform.contracts", level="WARNING"):
            recipe = normalize_recipe_request(
                RecipeCreate.model_validate(
                    {
                        "name": "hello-smoke",
                        "workload": "hello",
                        "training": {"epochs": 1},
                        "automl": {"enabled": False},
                    }
                )
            )

        self.assertEqual(recipe["recipe_id"], "hello")
        self.assertEqual(recipe["configuration"], {})
        self.assertNotIn("training", recipe)
        self.assertNotIn("automl", recipe)
        self.assertIsNone(recipe["recipe_snapshot"]["objective"])
        self.assertIsNone(recipe["recipe_snapshot"]["model"])

    def test_explicit_hello_rejects_training_fields(self) -> None:
        with self.assertLogs("recipe_platform.contracts", level="WARNING"):
            request = RecipeCreate.model_validate(
                {
                    "name": "hello-smoke",
                    "recipe_id": "hello",
                    "training": {"epochs": 1},
                }
            )

        with self.assertRaisesRegex(
            RecipeNormalizationError,
            "does not accept training or AutoML",
        ):
            normalize_recipe_request(request)

    def test_unknown_recipe_and_version_fail_clearly(self) -> None:
        with self.assertRaisesRegex(
            RecipeNormalizationError,
            "Unknown recipe identifier",
        ):
            normalize_recipe_request(
                RecipeCreate(name="unknown-recipe", recipe_id="unknown")
            )

        with self.assertRaisesRegex(
            RecipeNormalizationError,
            "Unsupported version",
        ):
            normalize_recipe_request(
                RecipeCreate(
                    name="wrong-version",
                    recipe_id="cats-dogs",
                    recipe_version="2.0",
                )
            )

    def test_snapshot_is_json_serializable_and_contains_effective_values(
        self,
    ) -> None:
        recipe = normalize_recipe_request(
            RecipeCreate(
                name="snapshot-cats",
                recipe_id="cats-dogs",
                configuration={
                    "training": {"batch_size": 16},
                    "automl": {
                        "enabled": False,
                        "search_space": {
                            "learning_rate": {
                                "min": 0.0001,
                                "max": 0.001,
                            }
                        },
                    },
                },
            )
        )

        snapshot = recipe["recipe_snapshot"]
        json.dumps(snapshot)
        self.assertEqual(snapshot["model"], "mobilenet_v2")
        self.assertEqual(
            snapshot["objective"],
            {"name": "val_auc", "direction": "maximize"},
        )
        self.assertEqual(
            snapshot["configuration"]["automl"]["search_space"][
                "learning_rate"
            ],
            {"min": 0.0001, "max": 0.001},
        )
        self.assertEqual(
            snapshot["configuration"]["effective_final_parameters"],
            {"learning_rate": 0.0003, "dropout_rate": 0.25},
        )

    def test_existing_job_without_recipe_id_remains_readable(self) -> None:
        now = datetime.now(timezone.utc)
        response = JobResponse.model_validate(
            {
                "id": "legacy-job",
                "status": "SUCCEEDED",
                "recipe": {
                    "name": "legacy-cats",
                    "workload": "cats-dogs",
                    "automl": {"enabled": True},
                },
                "best_metric": 0.9,
                "created_at": now,
                "updated_at": now,
            }
        )

        self.assertNotIn("recipe_id", response.recipe)
        self.assertEqual(response.result.objective.value, 0.9)

    def test_canonical_validation_error_preserves_nested_location(self) -> None:
        request = RecipeCreate(
            name="invalid-tabular",
            recipe_id="tabular-random-forest",
            configuration={
                "automl": {
                    "max_trials": 1,
                    "parallel_trials": 2,
                }
            },
        )

        with self.assertRaises(HTTPException) as raised:
            create_job(request, db=Mock())

        self.assertEqual(raised.exception.status_code, 422)
        detail = raised.exception.detail
        self.assertIsInstance(detail, list)
        self.assertEqual(
            detail[0]["loc"],
            [
                "body",
                "configuration",
                "automl",
                "parallel_trials",
            ],
        )
        self.assertIn("parallel_trials", detail[0]["msg"])

    def test_non_pydantic_normalization_error_remains_readable_string(
        self,
    ) -> None:
        request = RecipeCreate(
            name="unknown-recipe",
            recipe_id="unknown",
        )
        with self.assertRaises(HTTPException) as raised:
            create_job(request, db=Mock())
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIsInstance(raised.exception.detail, str)


if __name__ == "__main__":
    unittest.main()
