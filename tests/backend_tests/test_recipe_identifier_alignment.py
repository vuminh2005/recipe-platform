from __future__ import annotations

import subprocess
import sys
import unittest

from agent.recipe_ids import BUILTIN_RECIPE_IDS
from backend.app.recipe_catalog import CATALOG_RECIPE_IDS
from backend.app.recipe_catalog import (
    TABULAR_DEFAULT_MAX_DEPTH,
    TABULAR_DEFAULT_MAX_FEATURES,
    TABULAR_DEFAULT_MIN_SAMPLES_SPLIT,
    TABULAR_DEFAULT_N_ESTIMATORS,
    TABULAR_DEFAULT_RANDOM_SEED,
)
from agent.tabular_random_forest_config import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_SEED,
)


class RecipeIdentifierAlignmentTests(unittest.TestCase):
    def test_backend_catalog_and_agent_registry_identifiers_align(self) -> None:
        self.assertEqual(CATALOG_RECIPE_IDS, BUILTIN_RECIPE_IDS)

    def test_lightweight_agent_identifier_import_has_no_execution_dependencies(
        self,
    ) -> None:
        script = """
import sys
from agent.recipe_ids import BUILTIN_RECIPE_IDS
forbidden = (
    "agent.cats_dogs_executor",
    "agent.hello_executor",
    "agent.katib_runner",
    "agent.kfp_runner",
    "agent.mlflow_rest",
    "kubernetes",
    "kfp",
    "mlflow",
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"unexpected heavy imports: {loaded}")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_tabular_defaults_align_without_runtime_coupling(self) -> None:
        self.assertEqual(TABULAR_DEFAULT_N_ESTIMATORS, DEFAULT_N_ESTIMATORS)
        self.assertEqual(TABULAR_DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH)
        self.assertEqual(
            TABULAR_DEFAULT_MIN_SAMPLES_SPLIT,
            DEFAULT_MIN_SAMPLES_SPLIT,
        )
        self.assertEqual(TABULAR_DEFAULT_MAX_FEATURES, DEFAULT_MAX_FEATURES)
        self.assertEqual(TABULAR_DEFAULT_RANDOM_SEED, DEFAULT_RANDOM_SEED)


if __name__ == "__main__":
    unittest.main()
