from __future__ import annotations

import asyncio
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import HTTPException

from backend.app import main
from backend.app.job_contracts import RecipeCreate


class FakeDatabase:
    def scalar(self, _statement):
        return None

    def add(self, job) -> None:
        now = datetime.now(timezone.utc)
        job.id = str(uuid.uuid4())
        job.created_at = now
        job.updated_at = now

    def commit(self) -> None:
        pass

    def refresh(self, _job) -> None:
        pass


class JobSubmissionSecurityTests(unittest.TestCase):
    @staticmethod
    def payload() -> dict[str, object]:
        return {
            "name": "submission-token-contract",
            "recipe_id": "hello",
            "recipe_version": "1.0",
            "configuration": {},
        }

    @staticmethod
    def route_dependencies(path: str, method: str) -> set[object]:
        route = next(
            route
            for route in main.app.routes
            if getattr(route, "path", None) == path
            and method in getattr(route, "methods", set())
        )
        return {dependency.call for dependency in route.dependant.dependencies}

    def test_post_without_submission_token_returns_401(self) -> None:
        with patch.object(main, "JOB_SUBMISSION_TOKEN", "configured-secret"):
            with self.assertRaises(HTTPException) as raised:
                main.verify_job_submission_token(None)

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(
            raised.exception.detail,
            "Job submission token is required",
        )

    def test_post_with_incorrect_submission_token_returns_403(self) -> None:
        configured_token = "configured-secret-not-for-output"
        supplied_token = "incorrect-secret-not-for-output"
        with (
            patch.object(main, "JOB_SUBMISSION_TOKEN", configured_token),
            patch.object(
                main.secrets,
                "compare_digest",
                wraps=main.secrets.compare_digest,
            ) as compare_digest,
        ):
            with self.assertRaises(HTTPException) as raised:
                main.verify_job_submission_token(supplied_token)

        self.assertEqual(raised.exception.status_code, 403)
        error_text = str(raised.exception.detail)
        self.assertNotIn(configured_token, error_text)
        self.assertNotIn(supplied_token, error_text)
        compare_digest.assert_called_once_with(
            supplied_token,
            configured_token,
        )

    def test_post_with_correct_submission_token_succeeds(self) -> None:
        with patch.object(main, "JOB_SUBMISSION_TOKEN", "correct-token"):
            main.verify_job_submission_token("correct-token")

        job = main.create_job(
            RecipeCreate.model_validate(self.payload()),
            db=FakeDatabase(),
        )
        self.assertEqual(job.recipe["recipe_id"], "hello")

    def test_post_route_requires_submission_token_dependency(self) -> None:
        self.assertIn(
            main.verify_job_submission_token,
            self.route_dependencies("/api/jobs", "POST"),
        )

    def test_openapi_declares_submission_security_only_for_post(self) -> None:
        schema = main.app.openapi()
        security_scheme = schema["components"]["securitySchemes"][
            "JobSubmissionToken"
        ]

        self.assertEqual(security_scheme["type"], "apiKey")
        self.assertEqual(security_scheme["in"], "header")
        self.assertEqual(
            security_scheme["name"],
            "X-Job-Submission-Token",
        )
        self.assertIn(
            {"JobSubmissionToken": []},
            schema["paths"]["/api/jobs"]["post"]["security"],
        )
        self.assertNotIn(
            "security",
            schema["paths"]["/api/jobs"]["get"],
        )

    def test_public_get_routes_do_not_require_submission_token(self) -> None:
        for path in ("/health", "/api/jobs"):
            self.assertNotIn(
                main.verify_job_submission_token,
                self.route_dependencies(path, "GET"),
            )

    def test_agent_routes_continue_to_use_agent_token(self) -> None:
        for path, method in (
            ("/api/agent/jobs/claim", "POST"),
            ("/api/agent/jobs/{job_id}", "PATCH"),
        ):
            dependencies = self.route_dependencies(path, method)
            self.assertIn(main.verify_agent_token, dependencies)
            self.assertNotIn(main.verify_job_submission_token, dependencies)

        with (
            patch.object(main, "AGENT_TOKEN", "agent-only-token"),
            patch.object(main, "JOB_SUBMISSION_TOKEN", "submission-only-token"),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.verify_agent_token("submission-only-token")
            main.verify_agent_token("agent-only-token")

        self.assertEqual(raised.exception.status_code, 401)


class JobSubmissionStartupTests(unittest.TestCase):
    def test_missing_or_blank_submission_token_fails_startup(self) -> None:
        for token in ("", "   "):
            with self.subTest(token_is_blank=bool(token)):
                with (
                    patch.object(main, "AGENT_TOKEN", "safe-agent-token"),
                    patch.object(main, "JOB_SUBMISSION_TOKEN", token.strip()),
                    patch.object(main.Base.metadata, "create_all") as create_all,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "JOB_SUBMISSION_TOKEN",
                    ):
                        self.run_lifespan_startup()
                create_all.assert_not_called()

    @staticmethod
    def run_lifespan_startup() -> None:
        async def enter() -> None:
            async with main.lifespan(main.app):
                pass

        asyncio.run(enter())


class DatabaseUrlNormalizationTests(unittest.TestCase):
    def test_plain_postgresql_url_uses_psycopg_driver(self) -> None:
        self.assertEqual(
            main.normalize_database_url(
                "postgresql://user:password@database.example/app"
            ),
            "postgresql+psycopg://user:password@database.example/app",
        )

    def test_explicit_psycopg_url_is_unchanged(self) -> None:
        database_url = (
            "postgresql+psycopg://user:password@database.example/app"
        )
        self.assertEqual(
            main.normalize_database_url(database_url),
            database_url,
        )

    def test_sqlite_url_is_unchanged(self) -> None:
        database_url = "sqlite:///./recipe_platform.db"
        self.assertEqual(
            main.normalize_database_url(database_url),
            database_url,
        )


if __name__ == "__main__":
    unittest.main()
