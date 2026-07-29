from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from backend.app.job_contracts import AgentUpdate
from backend.app.main import update_job
from backend.app.job_results import (
    LEGACY_RESULT_FIELDS,
    JobResultPatch,
    apply_result_patch,
    build_job_result,
)


def stored_job(
    *,
    recipe: dict,
    **values,
) -> SimpleNamespace:
    fields = {field: None for field in LEGACY_RESULT_FIELDS}
    fields.update(values)
    return SimpleNamespace(recipe=recipe, **fields)


def cats_recipe(*, automl_enabled: bool) -> dict:
    configuration = {
        "training": {
            "model": "mobilenet_v2",
            "batch_size": 8,
        },
        "automl": {
            "enabled": automl_enabled,
            "search_space": {
                "learning_rate": {"min": 0.00005, "max": 0.0005},
                "dropout_rate": {"min": 0.15, "max": 0.45},
            },
        },
        "effective_final_parameters": (
            {"learning_rate": 0.0003, "dropout_rate": 0.25}
            if not automl_enabled
            else None
        ),
    }
    return {
        "recipe_id": "cats-dogs",
        "recipe_version": "1.0",
        "configuration": configuration,
        "recipe_snapshot": {
            "objective": {
                "name": "val_auc",
                "direction": "maximize",
            },
            "configuration": configuration,
        },
    }


class JobResultTests(unittest.TestCase):
    def test_sequential_result_patches_merge_without_erasing_values(self) -> None:
        job = stored_job(recipe=cats_recipe(automl_enabled=True))

        patches = [
            {
                "external_ids": {
                    "katib_experiment_id": "cats-hpo-123",
                }
            },
            {
                "objective": {"value": 0.93},
                "best_params": {
                    "learning_rate": 0.0002,
                    "dropout_rate": 0.3,
                    "recipe_specific": {"nested": True},
                },
            },
            {"external_ids": {"kfp_run_id": "kfp-456"}},
            {
                "external_ids": {
                    "mlflow_parent_run_id": "parent-1",
                    "mlflow_run_id": "run-789",
                },
                "model": {
                    "uri": "runs:/run-789/model",
                    "registered_name": "actual-cats-model-name",
                    "version": "7",
                },
                "final_metrics": {
                    "any_metric": 0.94,
                    "nested_metric_context": {"split": "test"},
                },
            },
        ]
        db = Mock()
        db.get.return_value = job
        for payload in patches:
            update_job(
                "job-1",
                AgentUpdate.model_validate({"result_patch": payload}),
                db=db,
            )

        result = build_job_result(job)
        self.assertEqual(result.objective.value, 0.93)
        self.assertEqual(
            result.best_params["recipe_specific"],
            {"nested": True},
        )
        self.assertEqual(
            result.external_ids.katib_experiment_id,
            "cats-hpo-123",
        )
        self.assertEqual(result.external_ids.kfp_run_id, "kfp-456")
        self.assertEqual(
            result.external_ids.mlflow_parent_run_id,
            "parent-1",
        )
        self.assertEqual(result.external_ids.mlflow_run_id, "run-789")
        self.assertEqual(
            result.model.registered_name,
            "actual-cats-model-name",
        )
        self.assertEqual(result.final_metrics["any_metric"], 0.94)
        self.assertEqual(db.commit.call_count, 4)

    def test_partial_nested_updates_preserve_existing_model_and_ids(self) -> None:
        job = stored_job(
            recipe=cats_recipe(automl_enabled=True),
            katib_experiment_name="katib-existing",
            model_uri="runs:/existing/model",
            registered_model_name="existing-name",
        )

        apply_result_patch(
            job,
            JobResultPatch.model_validate(
                {
                    "external_ids": {"kfp_run_id": "new-kfp"},
                    "model": {"version": "9"},
                }
            ),
        )

        self.assertEqual(job.katib_experiment_name, "katib-existing")
        self.assertEqual(job.kfp_run_id, "new-kfp")
        self.assertEqual(job.model_uri, "runs:/existing/model")
        self.assertEqual(job.registered_model_name, "existing-name")
        self.assertEqual(job.registered_model_version, "9")

    def test_explicit_null_cannot_clear_a_result_value(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "does not clear values",
        ):
            JobResultPatch.model_validate(
                {"external_ids": {"kfp_run_id": None}}
            )

    def test_canonical_and_legacy_fields_cannot_share_a_patch(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "cannot be combined with legacy result fields",
        ):
            AgentUpdate.model_validate(
                {
                    "result_patch": {
                        "external_ids": {"kfp_run_id": "canonical"}
                    },
                    "kfp_run_id": "legacy",
                }
            )

    def test_automl_disabled_has_no_objective_value_or_best_params(self) -> None:
        job = stored_job(
            recipe=cats_recipe(automl_enabled=False),
            kfp_run_id="kfp-direct",
            best_metric=0.99,
            best_params={"learning_rate": 0.0003},
            final_metrics={"val_auc": 0.91},
        )

        result = build_job_result(job)

        self.assertEqual(result.objective.name, "val_auc")
        self.assertEqual(result.objective.direction, "maximize")
        self.assertIsNone(result.objective.value)
        self.assertIsNone(result.best_params)
        self.assertEqual(result.final_metrics["val_auc"], 0.91)
        self.assertEqual(
            job.recipe["configuration"]["effective_final_parameters"],
            {"learning_rate": 0.0003, "dropout_rate": 0.25},
        )

    def test_configured_values_do_not_become_katib_best_params(self) -> None:
        recipe = cats_recipe(automl_enabled=False)
        recipe["configuration"]["training"]["batch_size"] = 16
        recipe["configuration"]["automl"]["search_space"]["learning_rate"] = {
            "min": 0.0001,
            "max": 0.001,
        }
        job = stored_job(recipe=recipe, kfp_run_id="kfp-direct")

        result = build_job_result(job)

        self.assertIsNone(result.best_params)
        self.assertEqual(
            recipe["configuration"]["training"]["batch_size"],
            16,
        )
        self.assertEqual(
            recipe["configuration"]["automl"]["search_space"][
                "learning_rate"
            ],
            {"min": 0.0001, "max": 0.001},
        )

    def test_hello_result_contains_no_fake_ml_integrations(self) -> None:
        job = stored_job(
            recipe={
                "recipe_id": "hello",
                "configuration": {},
                "recipe_snapshot": {"objective": None},
            },
            kfp_run_id="hello-kfp",
            katib_experiment_name="legacy-fake-katib",
            mlflow_final_run_id="legacy-fake-mlflow",
            best_params={"legacy": "fake"},
            final_metrics={"legacy_metric": 1},
            registered_model_name="legacy-fake-model",
        )

        result = build_job_result(job)

        self.assertIsNone(result.objective)
        self.assertIsNone(result.best_params)
        self.assertIsNone(result.final_metrics)
        self.assertIsNone(result.external_ids.katib_experiment_id)
        self.assertEqual(result.external_ids.kfp_run_id, "hello-kfp")
        self.assertIsNone(result.external_ids.mlflow_run_id)
        self.assertIsNone(result.model)

    def test_hello_canonical_patch_rejects_fake_ml_fields(self) -> None:
        job = stored_job(
            recipe={
                "recipe_id": "hello",
                "configuration": {},
                "recipe_snapshot": {"objective": None},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "may contain only external_ids.kfp_run_id",
        ):
            apply_result_patch(
                job,
                JobResultPatch.model_validate(
                    {
                        "external_ids": {
                            "kfp_run_id": "hello-kfp",
                            "mlflow_run_id": "fake-run",
                        }
                    }
                ),
            )

    def test_existing_legacy_job_derives_a_canonical_result(self) -> None:
        job = stored_job(
            recipe={
                "workload": "cats-dogs",
                "automl": {"enabled": True},
            },
            katib_experiment_name="legacy-katib",
            best_metric=0.88,
            best_params={"arbitrary": ["json", 1, True]},
        )

        result = build_job_result(job)

        self.assertEqual(result.objective.value, 0.88)
        self.assertEqual(
            result.best_params,
            {"arbitrary": ["json", 1, True]},
        )
        self.assertEqual(
            result.external_ids.katib_experiment_id,
            "legacy-katib",
        )


if __name__ == "__main__":
    unittest.main()
