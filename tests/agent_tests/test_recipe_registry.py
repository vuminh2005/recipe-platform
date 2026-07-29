from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from agent.recipe_ids import BUILTIN_RECIPE_IDS
from agent.recipe_registry import HANDLERS, execute_job, resolve_recipe_id


class RecipeRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Mock()
        self.backend = Mock()
        self.backend.get_job.return_value = {"status": "CLAIMED"}

    def test_handlers_cover_declared_builtin_recipe_ids(self) -> None:
        self.assertEqual(frozenset(HANDLERS), BUILTIN_RECIPE_IDS)

    def execute_with_handlers(self, job, *, cats_dogs, hello) -> None:
        with patch.dict(
            HANDLERS,
            {"cats-dogs": cats_dogs, "hello": hello},
            clear=True,
        ):
            execute_job(
                job,
                settings=self.settings,
                backend=self.backend,
            )

    def test_recipe_id_dispatches_to_cats_dogs(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {"id": "job-1", "recipe": {"recipe_id": "cats-dogs"}}

        self.execute_with_handlers(job, cats_dogs=cats_dogs, hello=hello)

        cats_dogs.assert_called_once_with(
            job,
            settings=self.settings,
            backend=self.backend,
        )
        hello.assert_not_called()

    def test_recipe_id_dispatches_to_hello(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {"id": "job-1", "recipe": {"recipe_id": "hello"}}

        self.execute_with_handlers(job, cats_dogs=cats_dogs, hello=hello)

        hello.assert_called_once_with(
            job,
            settings=self.settings,
            backend=self.backend,
        )
        cats_dogs.assert_not_called()

    def test_legacy_cats_dogs_workload_remains_supported(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {"id": "job-1", "recipe": {"workload": "cats-dogs"}}

        self.execute_with_handlers(job, cats_dogs=cats_dogs, hello=hello)

        cats_dogs.assert_called_once()
        hello.assert_not_called()

    def test_legacy_hello_workload_remains_supported(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {"id": "job-1", "recipe": {"workload": "hello"}}

        self.execute_with_handlers(job, cats_dogs=cats_dogs, hello=hello)

        hello.assert_called_once()
        cats_dogs.assert_not_called()

    def test_recipe_id_takes_precedence_over_legacy_workload(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {
            "id": "job-1",
            "recipe": {
                "recipe_id": "hello",
                "workload": "cats-dogs",
            },
        }

        self.assertEqual(resolve_recipe_id(job), "hello")
        self.execute_with_handlers(job, cats_dogs=cats_dogs, hello=hello)
        hello.assert_called_once()
        cats_dogs.assert_not_called()

    def test_unknown_recipe_fails_and_never_calls_cats_dogs(self) -> None:
        cats_dogs = Mock()
        hello = Mock()
        job = {"id": "job-1", "recipe": {"recipe_id": "unknown"}}

        with patch.dict(
            HANDLERS,
            {"cats-dogs": cats_dogs, "hello": hello},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported recipe identifier 'unknown'",
            ):
                execute_job(
                    job,
                    settings=self.settings,
                    backend=self.backend,
                )

        cats_dogs.assert_not_called()
        hello.assert_not_called()
        self.backend.patch_job.assert_called_once_with(
            "job-1",
            status="FAILED",
            error_message=(
                "Unsupported recipe identifier 'unknown'; "
                "supported built-ins: cats-dogs, hello"
            ),
        )

    def test_missing_recipe_identifier_fails_clearly(self) -> None:
        job = {"id": "job-1", "recipe": {"name": "missing-id"}}

        with self.assertRaisesRegex(
            ValueError,
            "missing recipe_id and legacy workload",
        ):
            execute_job(
                job,
                settings=self.settings,
                backend=self.backend,
            )


if __name__ == "__main__":
    unittest.main()
