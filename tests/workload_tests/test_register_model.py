from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "workloads"
    / "cats-dogs"
    / "trainer"
    / "register_model.py"
)
SPEC = importlib.util.spec_from_file_location(
    "cats_dogs_register_model_under_test",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
REGISTER_MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTER_MODEL)


class RegisterModelIdempotencyTests(unittest.TestCase):
    def test_reuses_version_with_matching_platform_job_id(self) -> None:
        existing = SimpleNamespace(
            version="4",
            tags={"platform.job_id": "job-1"},
            run_id="different-run",
            source="different-source",
        )
        client = Mock()
        client.search_model_versions.return_value = [existing]

        with patch.object(REGISTER_MODEL.mlflow, "register_model") as register:
            result = REGISTER_MODEL.get_or_register_model_version(
                client=client,
                model_name="cats-dogs",
                model_uri="runs:/run-1/model",
                platform_job_id="job-1",
                run_id="run-1",
            )

        self.assertIs(result, existing)
        register.assert_not_called()

    def test_reuses_version_with_matching_run(self) -> None:
        existing = SimpleNamespace(
            version="5",
            tags={},
            run_id="run-1",
            source="different-source",
        )
        client = Mock()
        client.search_model_versions.return_value = [existing]

        with patch.object(REGISTER_MODEL.mlflow, "register_model") as register:
            result = REGISTER_MODEL.get_or_register_model_version(
                client=client,
                model_name="cats-dogs",
                model_uri="runs:/run-1/model",
                platform_job_id="job-1",
                run_id="run-1",
            )

        self.assertIs(result, existing)
        register.assert_not_called()

    def test_reuses_version_with_matching_source(self) -> None:
        existing = SimpleNamespace(
            version="5",
            tags={},
            run_id="different-run",
            source="runs:/run-1/model",
        )
        client = Mock()
        client.search_model_versions.return_value = [existing]

        with patch.object(REGISTER_MODEL.mlflow, "register_model") as register:
            result = REGISTER_MODEL.get_or_register_model_version(
                client=client,
                model_name="cats-dogs",
                model_uri="runs:/run-1/model",
                platform_job_id="job-1",
                run_id="run-1",
            )

        self.assertIs(result, existing)
        register.assert_not_called()

    def test_registers_once_when_no_version_matches(self) -> None:
        created = SimpleNamespace(version="6")
        client = Mock()
        client.search_model_versions.return_value = []

        with patch.object(
            REGISTER_MODEL.mlflow,
            "register_model",
            return_value=created,
        ) as register:
            result = REGISTER_MODEL.get_or_register_model_version(
                client=client,
                model_name="cats-dogs",
                model_uri="runs:/run-1/model",
                platform_job_id="job-1",
                run_id="run-1",
            )

        self.assertIs(result, created)
        register.assert_called_once_with(
            model_uri="runs:/run-1/model",
            name="cats-dogs",
            await_registration_for=300,
        )


if __name__ == "__main__":
    unittest.main()
