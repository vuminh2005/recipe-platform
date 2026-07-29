from __future__ import annotations

import json
import unittest

from backend.app.job_contracts import RecipeCreate
from backend.app.recipe_normalization import (
    RecipeNormalizationError,
    normalize_recipe_request,
)


class TabularRecipeContractTests(unittest.TestCase):
    def normalize(self, configuration=None, **request_values):
        values = {
            "name": "tabular-random-forest-job",
            "recipe_id": "tabular-random-forest",
            **request_values,
        }
        if configuration is not None:
            values["configuration"] = configuration
        return normalize_recipe_request(RecipeCreate.model_validate(values))

    def test_default_request_is_typed_and_versioned(self) -> None:
        recipe = self.normalize()

        self.assertEqual(recipe["recipe_id"], "tabular-random-forest")
        self.assertEqual(recipe["recipe_version"], "1.0")
        self.assertEqual(recipe["workload"], "tabular-random-forest")
        self.assertEqual(
            recipe["configuration"]["training"],
            {"random_seed": 42},
        )
        self.assertTrue(recipe["configuration"]["automl"]["enabled"])
        self.assertIsNone(
            recipe["configuration"]["effective_final_parameters"]
        )

    def test_custom_seed_and_disabled_automl_store_effective_parameters(
        self,
    ) -> None:
        recipe = self.normalize(
            {
                "training": {"random_seed": 7},
                "automl": {"enabled": False},
            }
        )

        self.assertEqual(
            recipe["configuration"]["effective_final_parameters"],
            {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 2,
                "max_features": "sqrt",
                "random_seed": 7,
            },
        )

    def test_custom_automl_and_search_space_are_normalized(self) -> None:
        recipe = self.normalize(
            {
                "automl": {
                    "enabled": True,
                    "max_trials": 6,
                    "parallel_trials": 2,
                    "algorithm": "random",
                    "search_space": {
                        "n_estimators": {"min": "80", "max": "240"},
                        "max_depth": {"min": 3, "max": 12},
                        "min_samples_split": {"min": 3, "max": 8},
                    },
                }
            }
        )
        automl = recipe["configuration"]["automl"]

        self.assertEqual(automl["max_trials"], 6)
        self.assertEqual(automl["parallel_trials"], 2)
        self.assertEqual(
            automl["search_space"]["n_estimators"],
            {"min": 80, "max": 240},
        )

    def test_invalid_algorithm_parallelism_and_ranges_fail(self) -> None:
        invalid_configurations = (
            {"automl": {"algorithm": "tpe"}},
            {"automl": {"max_trials": 1, "parallel_trials": 2}},
            {
                "automl": {
                    "search_space": {
                        "n_estimators": {"min": 300, "max": 50}
                    }
                }
            },
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration):
                with self.assertRaises(RecipeNormalizationError):
                    self.normalize(configuration)

    def test_unknown_fields_and_cats_legacy_fields_are_rejected(self) -> None:
        with self.assertRaises(RecipeNormalizationError):
            self.normalize({"training": {"image_size": 224}})

        request = RecipeCreate.model_validate(
            {
                "name": "tabular-legacy-fields",
                "recipe_id": "tabular-random-forest",
                "training": {"batch_size": 8},
            }
        )
        with self.assertRaisesRegex(
            RecipeNormalizationError,
            "recipe-scoped configuration",
        ):
            normalize_recipe_request(request)

    def test_legacy_workload_discriminator_is_supported(self) -> None:
        recipe = self.normalize(
            workload="tabular-random-forest",
            recipe_id=None,
        )
        self.assertEqual(recipe["recipe_id"], "tabular-random-forest")

    def test_snapshot_is_complete_serializable_and_secret_free(self) -> None:
        recipe = self.normalize(
            {
                "training": {"random_seed": 11},
                "automl": {"enabled": False},
            }
        )
        snapshot = recipe["recipe_snapshot"]
        serialized = json.dumps(snapshot)

        self.assertEqual(snapshot["model"], "RandomForestClassifier")
        self.assertEqual(snapshot["framework"], "scikit_learn")
        self.assertEqual(
            snapshot["objective"],
            {"name": "val_f1", "direction": "maximize"},
        )
        self.assertEqual(
            snapshot["recipe_defaults"]["automl_disabled_parameters"][
                "n_estimators"
            ],
            200,
        )
        self.assertEqual(
            snapshot["configuration"]["effective_final_parameters"][
                "random_seed"
            ],
            11,
        )
        for forbidden in (
            "secret",
            "credential",
            "access_key",
            "pipeline_path",
            "trainer_image",
            "endpoint_url",
        ):
            self.assertNotIn(forbidden, serialized.lower())


if __name__ == "__main__":
    unittest.main()
