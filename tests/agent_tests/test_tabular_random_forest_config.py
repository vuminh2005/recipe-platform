from __future__ import annotations

import unittest

from agent.tabular_random_forest_config import validate_execution_config


class TabularExecutionConfigTests(unittest.TestCase):
    def test_defaults_are_recipe_owned(self) -> None:
        config = validate_execution_config({})

        self.assertTrue(config.automl_enabled)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(
            config.effective_final_parameters,
            {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 2,
                "max_features": "sqrt",
                "random_seed": 42,
            },
        )

    def test_invalid_persisted_config_is_rejected(self) -> None:
        invalid_values = (
            {"automl": {"algorithm": "tpe"}},
            {"automl": {"max_trials": 1, "parallel_trials": 2}},
            {
                "automl": {
                    "search_space": {
                        "max_depth": {"min": 20, "max": 2}
                    }
                }
            },
            {
                "training": {"random_seed": 7},
                "automl": {"enabled": False},
                "effective_final_parameters": {"random_seed": 8},
            },
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_execution_config(value)


if __name__ == "__main__":
    unittest.main()
