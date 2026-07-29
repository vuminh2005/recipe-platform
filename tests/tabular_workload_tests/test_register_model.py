from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "workloads"
    / "tabular-random-forest"
    / "trainer"
    / "register_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tabular_register_model_under_test",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
REGISTER_MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTER_MODEL)


class RegistrationTests(unittest.TestCase):
    def test_reuses_versions_by_job_run_or_source(self) -> None:
        cases = (
            SimpleNamespace(
                version="1",
                tags={"platform.job_id": "job-1"},
                run_id="other",
                source="other",
            ),
            SimpleNamespace(
                version="2",
                tags={},
                run_id="run-1",
                source="other",
            ),
            SimpleNamespace(
                version="3",
                tags={},
                run_id="other",
                source="runs:/run-1/model",
            ),
        )
        for existing in cases:
            with self.subTest(version=existing.version):
                client = Mock()
                client.search_model_versions.return_value = [existing]
                with patch.object(
                    REGISTER_MODEL.mlflow,
                    "register_model",
                ) as register:
                    result = REGISTER_MODEL.get_or_register_model_version(
                        client,
                        model_name="actual-tabular-name",
                        model_uri="runs:/run-1/model",
                        platform_job_id="job-1",
                        run_id="run-1",
                    )
                self.assertIs(result, existing)
                register.assert_not_called()

    def test_registers_actual_recipe_owned_name_once(self) -> None:
        client = Mock()
        client.search_model_versions.return_value = []
        created = SimpleNamespace(version="4")

        with patch.object(
            REGISTER_MODEL.mlflow,
            "register_model",
            return_value=created,
        ) as register:
            result = REGISTER_MODEL.get_or_register_model_version(
                client,
                model_name="tabular_random_forest_classifier",
                model_uri="runs:/run-1/model",
                platform_job_id="job-1",
                run_id="run-1",
            )

        self.assertIs(result, created)
        register.assert_called_once_with(
            model_uri="runs:/run-1/model",
            name="tabular_random_forest_classifier",
            await_registration_for=300,
        )


if __name__ == "__main__":
    unittest.main()
