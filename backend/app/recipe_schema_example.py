"""Example using the active recipe request contracts."""

from __future__ import annotations

from .job_contracts import RecipeCreate
from .recipe_normalization import normalize_recipe_request


EXAMPLE_RECIPE = RecipeCreate(
    name="cats-dogs-recipe",
    recipe_id="cats-dogs",
    recipe_version="1.0",
    configuration={
        "training": {},
        "automl": {},
    },
)

NORMALIZED_EXAMPLE = normalize_recipe_request(EXAMPLE_RECIPE)
