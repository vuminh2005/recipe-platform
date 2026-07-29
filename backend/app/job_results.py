"""Canonical job-result models and the legacy storage compatibility adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .recipe_catalog import HELLO_RECIPE_ID, get_recipe_definition


ObjectiveDirection = Literal["maximize", "minimize"]

LEGACY_RESULT_FIELDS = frozenset(
    {
        "best_metric",
        "best_params",
        "katib_experiment_name",
        "kfp_run_id",
        "mlflow_parent_run_id",
        "mlflow_final_run_id",
        "model_uri",
        "registered_model_name",
        "registered_model_version",
        "final_metrics",
    }
)


class ResultObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: ObjectiveDirection
    value: float | None = None


class ResultExternalIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    katib_experiment_id: str | None = Field(default=None, max_length=255)
    kfp_run_id: str | None = Field(default=None, max_length=128)
    mlflow_parent_run_id: str | None = Field(default=None, max_length=128)
    mlflow_run_id: str | None = Field(default=None, max_length=128)


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str | None = Field(default=None, max_length=500)
    registered_name: str | None = Field(default=None, max_length=255)
    version: str | None = Field(default=None, max_length=64)


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: ResultObjective | None = None
    best_params: dict[str, JsonValue] | None = None
    final_metrics: dict[str, JsonValue] | None = None
    external_ids: ResultExternalIds
    model: ResultModel | None = None


class _NonClearingPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            null_fields = [key for key, item in value.items() if item is None]
            if null_fields:
                raise ValueError(
                    "result_patch does not clear values; omit fields instead "
                    f"of sending null: {sorted(null_fields)}"
                )
        return value

    @model_validator(mode="after")
    def reject_empty_patch(self) -> "_NonClearingPatch":
        if not self.model_fields_set:
            raise ValueError("result_patch sections must contain at least one field")
        return self


class ResultObjectivePatch(_NonClearingPatch):
    value: float | None = None


class ResultExternalIdsPatch(_NonClearingPatch):
    katib_experiment_id: str | None = Field(default=None, max_length=255)
    kfp_run_id: str | None = Field(default=None, max_length=128)
    mlflow_parent_run_id: str | None = Field(default=None, max_length=128)
    mlflow_run_id: str | None = Field(default=None, max_length=128)


class ResultModelPatch(_NonClearingPatch):
    uri: str | None = Field(default=None, max_length=500)
    registered_name: str | None = Field(default=None, max_length=255)
    version: str | None = Field(default=None, max_length=64)


class JobResultPatch(_NonClearingPatch):
    objective: ResultObjectivePatch | None = None
    best_params: dict[str, JsonValue] | None = None
    final_metrics: dict[str, JsonValue] | None = None
    external_ids: ResultExternalIdsPatch | None = None
    model: ResultModelPatch | None = None


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _recipe_identity(recipe: Mapping[str, Any]) -> str | None:
    recipe_id = recipe.get("recipe_id")
    if isinstance(recipe_id, str) and recipe_id:
        return recipe_id
    workload = recipe.get("workload")
    if isinstance(workload, str) and workload:
        return workload
    return None


def _objective_definition(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    snapshot = recipe.get("recipe_snapshot")
    if isinstance(snapshot, Mapping):
        objective = snapshot.get("objective")
        if isinstance(objective, Mapping):
            return dict(objective)
        if objective is None and "objective" in snapshot:
            return None

    recipe_id = _recipe_identity(recipe)
    definition = get_recipe_definition(recipe_id) if recipe_id else None
    if definition is None or definition.objective is None:
        return None
    return definition.objective.model_dump(mode="json")


def _automl_enabled(recipe: Mapping[str, Any]) -> bool | None:
    configuration = recipe.get("configuration")
    if isinstance(configuration, Mapping):
        automl = configuration.get("automl")
    else:
        automl = recipe.get("automl") or recipe.get("automl_config")
    if not isinstance(automl, Mapping) or "enabled" not in automl:
        return None
    return bool(automl["enabled"])


def build_job_result(source: Any) -> JobResult | None:
    result_values = {
        field_name: _value(source, field_name)
        for field_name in LEGACY_RESULT_FIELDS
    }
    if not any(value is not None for value in result_values.values()):
        return None

    recipe_value = _value(source, "recipe")
    recipe = recipe_value if isinstance(recipe_value, Mapping) else {}
    is_hello = _recipe_identity(recipe) == HELLO_RECIPE_ID
    objective_definition = _objective_definition(recipe)
    objective: ResultObjective | None = None
    if objective_definition is not None:
        objective_value = result_values["best_metric"]
        if _automl_enabled(recipe) is False:
            objective_value = None
        objective = ResultObjective(
            name=str(objective_definition["name"]),
            direction=objective_definition["direction"],
            value=objective_value,
        )

    best_params = result_values["best_params"]
    if is_hello or _automl_enabled(recipe) is False:
        best_params = None

    model_values = {
        "uri": result_values["model_uri"],
        "registered_name": result_values["registered_model_name"],
        "version": result_values["registered_model_version"],
    }
    model = (
        ResultModel(**model_values)
        if not is_hello
        and any(value is not None for value in model_values.values())
        else None
    )

    return JobResult(
        objective=objective,
        best_params=best_params,
        final_metrics=(
            None if is_hello else result_values["final_metrics"]
        ),
        external_ids=ResultExternalIds(
            katib_experiment_id=(
                None
                if is_hello
                else result_values["katib_experiment_name"]
            ),
            kfp_run_id=result_values["kfp_run_id"],
            mlflow_parent_run_id=(
                None
                if is_hello
                else result_values["mlflow_parent_run_id"]
            ),
            mlflow_run_id=(
                None
                if is_hello
                else result_values["mlflow_final_run_id"]
            ),
        ),
        model=model,
    )


def apply_result_patch(target: Any, patch: JobResultPatch) -> None:
    recipe_value = _value(target, "recipe")
    recipe = recipe_value if isinstance(recipe_value, Mapping) else {}
    if _recipe_identity(recipe) == HELLO_RECIPE_ID:
        unsupported_sections = patch.model_fields_set - {"external_ids"}
        external_fields = (
            patch.external_ids.model_fields_set
            if patch.external_ids is not None
            else set()
        )
        if unsupported_sections or external_fields - {"kfp_run_id"}:
            raise ValueError(
                "The Hello smoke recipe result may contain only "
                "external_ids.kfp_run_id"
            )

    if "objective" in patch.model_fields_set:
        objective = patch.objective
        if objective is not None and "value" in objective.model_fields_set:
            target.best_metric = objective.value

    if "best_params" in patch.model_fields_set:
        target.best_params = patch.best_params

    if "final_metrics" in patch.model_fields_set:
        target.final_metrics = patch.final_metrics

    if "external_ids" in patch.model_fields_set:
        external_ids = patch.external_ids
        if external_ids is not None:
            external_mapping = {
                "katib_experiment_id": "katib_experiment_name",
                "kfp_run_id": "kfp_run_id",
                "mlflow_parent_run_id": "mlflow_parent_run_id",
                "mlflow_run_id": "mlflow_final_run_id",
            }
            for canonical_name, storage_name in external_mapping.items():
                if canonical_name in external_ids.model_fields_set:
                    setattr(target, storage_name, getattr(external_ids, canonical_name))

    if "model" in patch.model_fields_set:
        model = patch.model
        if model is not None:
            model_mapping = {
                "uri": "model_uri",
                "registered_name": "registered_model_name",
                "version": "registered_model_version",
            }
            for canonical_name, storage_name in model_mapping.items():
                if canonical_name in model.model_fields_set:
                    setattr(target, storage_name, getattr(model, canonical_name))
