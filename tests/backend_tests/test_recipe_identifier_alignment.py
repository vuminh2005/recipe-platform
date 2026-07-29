from __future__ import annotations

import subprocess
import sys
import unittest

from agent.recipe_ids import BUILTIN_RECIPE_IDS
from backend.app.recipe_catalog import CATALOG_RECIPE_IDS


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


if __name__ == "__main__":
    unittest.main()
