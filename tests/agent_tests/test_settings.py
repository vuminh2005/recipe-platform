from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.settings import Settings


BASE_ENV = {
    "BACKEND_URL": "http://backend",
    "MLFLOW_TRACKING_URI": "http://mlflow",
}


class AgentTokenSettingsTests(unittest.TestCase):
    def settings_from(self, values: dict[str, str]) -> Settings:
        with patch.dict(os.environ, {**BASE_ENV, **values}, clear=True):
            return Settings.from_env()

    def test_missing_or_empty_token_fails(self) -> None:
        for token_values in ({}, {"AGENT_TOKEN": ""}):
            with self.subTest(token_values=token_values):
                with self.assertRaisesRegex(RuntimeError, "AGENT_TOKEN"):
                    self.settings_from(token_values)

    def test_development_token_requires_explicit_escape_hatch(self) -> None:
        for allow_value in ("", "1", "yes", "true-value"):
            with self.subTest(allow_value=allow_value):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "insecure development token",
                ):
                    self.settings_from(
                        {
                            "AGENT_TOKEN": "development-token",
                            "ALLOW_INSECURE_DEVELOPMENT_TOKEN": allow_value,
                        }
                    )

    def test_development_token_is_allowed_only_with_case_insensitive_true(
        self,
    ) -> None:
        for allow_value in ("true", "TRUE", " True "):
            with self.subTest(allow_value=allow_value):
                settings = self.settings_from(
                    {
                        "AGENT_TOKEN": "development-token",
                        "ALLOW_INSECURE_DEVELOPMENT_TOKEN": allow_value,
                    }
                )
                self.assertEqual(settings.agent_token, "development-token")

    def test_non_default_token_is_accepted(self) -> None:
        settings = self.settings_from(
            {"AGENT_TOKEN": "safe-agent-test-token"}
        )
        self.assertEqual(settings.agent_token, "safe-agent-test-token")


if __name__ == "__main__":
    unittest.main()
