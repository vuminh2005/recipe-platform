from __future__ import annotations

import logging
from typing import Any, Protocol

from agent.backend_client import BackendClient
from agent.cats_dogs_executor import execute_cats_dogs_job
from agent.hello_executor import execute_hello_job
from agent.recipe_ids import CATS_DOGS_RECIPE_ID, HELLO_RECIPE_ID
from agent.settings import Settings


LOGGER = logging.getLogger("recipe_registry")


class RecipeHandler(Protocol):
    def __call__(
        self,
        job: dict[str, Any],
        *,
        settings: Settings,
        backend: BackendClient,
    ) -> None: ...


HANDLERS: dict[str, RecipeHandler] = {
    CATS_DOGS_RECIPE_ID: execute_cats_dogs_job,
    HELLO_RECIPE_ID: execute_hello_job,
}


def resolve_recipe_id(job: dict[str, Any]) -> str:
    recipe = job.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Job recipe must be an object")

    if "recipe_id" in recipe and recipe["recipe_id"] is not None:
        value = recipe["recipe_id"]
    else:
        value = recipe.get("workload")

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Job recipe is missing recipe_id and legacy workload"
        )
    return value.strip()


def get_handler(recipe_id: str) -> RecipeHandler:
    handler = HANDLERS.get(recipe_id)
    if handler is None:
        supported = ", ".join(sorted(HANDLERS))
        raise ValueError(
            f"Unsupported recipe identifier {recipe_id!r}; "
            f"supported built-ins: {supported}"
        )
    return handler


def _report_failure(
    job: dict[str, Any],
    *,
    backend: BackendClient,
    error: Exception,
) -> None:
    job_id = str(job.get("id", ""))
    if not job_id:
        LOGGER.error("Cannot report failed job without an ID: %s", error)
        return

    try:
        current = backend.get_job(job_id)
        if current.get("status") == "FAILED":
            return
    except Exception:
        LOGGER.exception(
            "Could not inspect platform job %s before reporting failure",
            job_id,
        )

    try:
        backend.patch_job(
            job_id,
            status="FAILED",
            error_message=str(error)[:2000],
        )
    except Exception:
        LOGGER.exception("Could not report FAILED status for job %s", job_id)


def execute_job(
    job: dict[str, Any],
    *,
    settings: Settings,
    backend: BackendClient,
) -> None:
    try:
        recipe_id = resolve_recipe_id(job)
        handler = get_handler(recipe_id)
        handler(job, settings=settings, backend=backend)
    except Exception as exc:
        _report_failure(job, backend=backend, error=exc)
        raise
