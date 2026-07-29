"""Recipe-scoped request normalization and reproducible snapshot creation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .job_contracts import (
    AutoMLConfig,
    CatsDogsConfiguration,
    HelloConfiguration,
    RecipeCreate,
    TabularRandomForestConfiguration,
    TrainingConfig,
)
from .recipe_catalog import (
    CATS_DOGS_DEFAULT_DROPOUT_RATE,
    CATS_DOGS_DEFAULT_LEARNING_RATE,
    CATS_DOGS_RECIPE_ID,
    HELLO_RECIPE_ID,
    TABULAR_DEFAULT_MAX_DEPTH,
    TABULAR_DEFAULT_MAX_FEATURES,
    TABULAR_DEFAULT_MIN_SAMPLES_SPLIT,
    TABULAR_DEFAULT_N_ESTIMATORS,
    TABULAR_RANDOM_FOREST_RECIPE_ID,
    RecipeDefinition,
    get_recipe_definition,
)


class RecipeNormalizationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        detail: str | list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = detail if detail is not None else message


def _structured_validation_error(
    error: ValidationError,
) -> RecipeNormalizationError:
    detail = []
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        detail.append(
            {
                "type": item["type"],
                "loc": [
                    "body",
                    "configuration",
                    *item.get("loc", ()),
                ],
                "msg": item["msg"],
            }
        )
    return RecipeNormalizationError(str(error), detail=detail)


def resolve_request_recipe_id(request: RecipeCreate) -> str:
    if request.recipe_id is not None:
        recipe_id = request.recipe_id.strip()
        if not recipe_id:
            raise RecipeNormalizationError("recipe_id must not be empty")
        return recipe_id

    if request.workload is not None:
        workload = request.workload.strip()
        if not workload:
            raise RecipeNormalizationError("workload must not be empty")
        return workload

    # Preserve the original API's implicit internal Hello request.
    return HELLO_RECIPE_ID


def _definition_for_request(request: RecipeCreate) -> RecipeDefinition:
    recipe_id = resolve_request_recipe_id(request)
    definition = get_recipe_definition(recipe_id)
    if definition is None:
        raise RecipeNormalizationError(
            f"Unknown recipe identifier {recipe_id!r}"
        )

    if (
        request.recipe_version is not None
        and request.recipe_version != definition.version
    ):
        raise RecipeNormalizationError(
            f"Unsupported version {request.recipe_version!r} for recipe "
            f"{recipe_id!r}; supported: {definition.version}"
        )
    return definition


def _snapshot(
    definition: RecipeDefinition,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recipe_id": definition.recipe_id,
        "recipe_version": definition.version,
        "display_name": definition.display_name,
        "description": definition.description,
        "visibility": definition.visibility,
        "task_type": definition.task_type,
        "framework": definition.framework,
        "model": definition.model,
        "supports_automl": definition.supports_automl,
        "supported_algorithms": list(definition.supported_algorithms),
        "objective": (
            definition.objective.model_dump(mode="json")
            if definition.objective is not None
            else None
        ),
        "configuration": deepcopy(configuration),
        "recipe_defaults": (
            {
                "automl_disabled_parameters": deepcopy(
                    definition.default_configuration.get(
                        "effective_final_parameters"
                    )
                )
            }
            if "effective_final_parameters" in definition.default_configuration
            else {}
        ),
    }


def _cats_dogs_configuration(request: RecipeCreate) -> dict[str, Any]:
    legacy_fields = {
        "training",
        "automl",
        "training_config",
        "automl_config",
    }
    supplied_legacy_fields = legacy_fields & request.model_fields_set
    try:
        if request.configuration is not None:
            if supplied_legacy_fields:
                raise RecipeNormalizationError(
                    "configuration cannot be combined with legacy configuration "
                    f"fields: {sorted(supplied_legacy_fields)}"
                )
            parsed = CatsDogsConfiguration.model_validate(
                request.configuration
            )
        else:
            training = (
                request.training
                or request.training_config
                or TrainingConfig()
            )
            automl = (
                request.automl
                or request.automl_config
                or AutoMLConfig()
            )
            parsed = CatsDogsConfiguration(
                training=training,
                automl=automl,
            )
    except ValidationError as exc:
        raise _structured_validation_error(exc) from exc

    training_values = parsed.training.model_dump(mode="json")
    # Accepted for compatibility, but it does not affect execution or snapshots.
    training_values.pop("epochs", None)
    automl_values = parsed.automl.model_dump(mode="json")
    effective_final_parameters = (
        {
            "learning_rate": CATS_DOGS_DEFAULT_LEARNING_RATE,
            "dropout_rate": CATS_DOGS_DEFAULT_DROPOUT_RATE,
        }
        if not parsed.automl.enabled
        else None
    )
    return {
        "training": training_values,
        "automl": automl_values,
        "effective_final_parameters": effective_final_parameters,
    }


def _hello_configuration(request: RecipeCreate) -> dict[str, Any]:
    legacy_fields = {
        "training",
        "automl",
        "training_config",
        "automl_config",
    }
    supplied_legacy_fields = legacy_fields & request.model_fields_set

    if request.configuration is not None:
        if supplied_legacy_fields:
            raise RecipeNormalizationError(
                "configuration cannot be combined with legacy configuration "
                f"fields: {sorted(supplied_legacy_fields)}"
            )
        try:
            HelloConfiguration.model_validate(request.configuration)
        except ValidationError as exc:
            raise _structured_validation_error(exc) from exc
    elif request.recipe_id is not None and supplied_legacy_fields:
        raise RecipeNormalizationError(
            "The Hello smoke recipe does not accept training or AutoML fields"
        )
    # Legacy workload-only Hello requests may carry the old shared fields. They
    # are intentionally ignored and never copied into normalized configuration.
    return {}


def _tabular_random_forest_configuration(
    request: RecipeCreate,
) -> dict[str, Any]:
    legacy_fields = {
        "training",
        "automl",
        "training_config",
        "automl_config",
    }
    supplied_legacy_fields = legacy_fields & request.model_fields_set
    if supplied_legacy_fields:
        raise RecipeNormalizationError(
            "The Tabular Random Forest recipe accepts configuration only "
            "through the recipe-scoped configuration field; unsupported "
            f"legacy fields: {sorted(supplied_legacy_fields)}"
        )

    try:
        parsed = TabularRandomForestConfiguration.model_validate(
            request.configuration or {}
        )
    except ValidationError as exc:
        raise _structured_validation_error(exc) from exc

    training_values = parsed.training.model_dump(mode="json")
    automl_values = parsed.automl.model_dump(mode="json")
    effective_final_parameters = (
        {
            "n_estimators": TABULAR_DEFAULT_N_ESTIMATORS,
            "max_depth": TABULAR_DEFAULT_MAX_DEPTH,
            "min_samples_split": TABULAR_DEFAULT_MIN_SAMPLES_SPLIT,
            "max_features": TABULAR_DEFAULT_MAX_FEATURES,
            "random_seed": parsed.training.random_seed,
        }
        if not parsed.automl.enabled
        else None
    )
    return {
        "training": training_values,
        "automl": automl_values,
        "effective_final_parameters": effective_final_parameters,
    }


def normalize_recipe_request(request: RecipeCreate) -> dict[str, Any]:
    definition = _definition_for_request(request)
    if definition.recipe_id == CATS_DOGS_RECIPE_ID:
        configuration = _cats_dogs_configuration(request)
    elif definition.recipe_id == HELLO_RECIPE_ID:
        configuration = _hello_configuration(request)
    elif definition.recipe_id == TABULAR_RANDOM_FOREST_RECIPE_ID:
        configuration = _tabular_random_forest_configuration(request)
    else:  # pragma: no cover - guarded by the explicit catalog
        raise RecipeNormalizationError(
            f"No normalizer for recipe {definition.recipe_id!r}"
        )

    stored_recipe: dict[str, Any] = {
        "name": request.name,
        "recipe_id": definition.recipe_id,
        "recipe_version": definition.version,
        # Compatibility mirror for the current dashboard and older agents.
        "workload": definition.recipe_id,
        "configuration": deepcopy(configuration),
        "recipe_snapshot": _snapshot(definition, configuration),
    }

    if definition.recipe_id == CATS_DOGS_RECIPE_ID:
        # Compatibility mirrors are generated only from canonical configuration.
        stored_recipe["training"] = deepcopy(configuration["training"])
        stored_recipe["automl"] = deepcopy(configuration["automl"])

    return stored_recipe
