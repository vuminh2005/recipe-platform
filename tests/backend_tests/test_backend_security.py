from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.app import main


class BackendAgentTokenTests(unittest.TestCase):
    def validate(self, token: str, allow_value: str = "") -> None:
        with (
            patch.object(main, "AGENT_TOKEN", token),
            patch.dict(
                os.environ,
                {"ALLOW_INSECURE_DEVELOPMENT_TOKEN": allow_value},
                clear=False,
            ),
        ):
            main.validate_agent_token_configuration()

    def test_missing_or_empty_token_fails_startup_validation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "AGENT_TOKEN"):
            self.validate("")

    def test_development_token_requires_explicit_escape_hatch(self) -> None:
        for allow_value in ("", "1", "yes", "true-value"):
            with self.subTest(allow_value=allow_value):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "insecure development token",
                ):
                    self.validate("development-token", allow_value)

    def test_development_token_is_allowed_only_with_case_insensitive_true(
        self,
    ) -> None:
        for allow_value in ("true", "TRUE", " True "):
            with self.subTest(allow_value=allow_value):
                self.validate("development-token", allow_value)

    def test_non_default_token_is_accepted(self) -> None:
        self.validate("safe-backend-test-token")

    def test_lifespan_validates_token_before_database_initialization(
        self,
    ) -> None:
        with (
            patch.object(main, "AGENT_TOKEN", ""),
            patch.object(main.Base.metadata, "create_all") as create_all,
        ):
            with self.assertRaisesRegex(RuntimeError, "AGENT_TOKEN"):
                self.run_lifespan_startup()
        create_all.assert_not_called()

    @staticmethod
    def run_lifespan_startup() -> None:
        import asyncio

        async def enter() -> None:
            async with main.lifespan(main.app):
                pass

        asyncio.run(enter())


if __name__ == "__main__":
    unittest.main()
