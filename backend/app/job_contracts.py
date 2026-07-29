"""Pydantic contracts used by the Recipe Platform API."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .job_results import (
    LEGACY_RESULT_FIELDS,
    JobResult,
    JobResultPatch,
    build_job_result,
)


LOGGER = logging.getLogger("recipe_platform.contracts")
BuiltinRecipeId = Literal["hello", "cats-dogs"]


class JobStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"

    # Giữ lại để workload Hello cũ vẫn hoạt động.
    RUNNING = "RUNNING"

    # Trạng thái Cats & Dogs end-to-end.
    TUNING = "TUNING"
    TRAINING = "TRAINING"
    REGISTERING = "REGISTERING"

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["mobilenet_v2"] = "mobilenet_v2"

    @field_validator("model", mode="before")
    @classmethod
    def normalize_legacy_model(cls, value: Any) -> Any:
        if value == "tiny_cnn":
            LOGGER.warning(
                "Legacy training.model='tiny_cnn' is not implemented; "
                "normalizing it to 'mobilenet_v2'."
            )
            return "mobilenet_v2"
        return value

    image_size: int = Field(
        default=224,
        ge=32,
        le=512,
    )

    trial_epochs: int = Field(
        default=2,
        ge=1,
        le=5,
    )

    final_epochs: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    batch_size: int = Field(
        default=8,
        ge=1,
        le=32,
    )

    dense_units: int = Field(
        default=128,
        ge=32,
        le=512,
    )

    trainable_backbone: bool = False

    epochs: int | None = Field(
        default=None,
        ge=1,
        le=20,
        deprecated=True,
        description=(
            "Legacy compatibility field. It is accepted but ignored; "
            "use trial_epochs and final_epochs."
        ),
    )

    @field_validator("epochs", mode="before")
    @classmethod
    def warn_about_legacy_epochs(cls, value: Any) -> Any:
        if value is not None:
            LOGGER.warning(
                "Legacy training.epochs is ignored; use training.trial_epochs "
                "and training.final_epochs."
            )
        return value


class LearningRateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float = Field(default=0.00005, gt=0)
    max: float = Field(default=0.0005, gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "LearningRateRange":
        if self.min >= self.max:
            raise ValueError("learning_rate.min must be less than learning_rate.max")
        return self


class DropoutRateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float = Field(default=0.15, ge=0, lt=1)
    max: float = Field(default=0.45, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_order(self) -> "DropoutRateRange":
        if self.min >= self.max:
            raise ValueError("dropout_rate.min must be less than dropout_rate.max")
        return self


class CatsDogsSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: LearningRateRange = Field(
        default_factory=LearningRateRange,
    )
    dropout_rate: DropoutRateRange = Field(
        default_factory=DropoutRateRange,
    )


class AutoMLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_trials: int = Field(default=3, ge=1, le=20)
    parallel_trials: int = Field(default=1, ge=1, le=4)
    algorithm: Literal["random"] = "random"
    search_space: CatsDogsSearchSpace = Field(
        default_factory=CatsDogsSearchSpace,
    )

    @model_validator(mode="after")
    def validate_parallel_trials(self) -> "AutoMLConfig":
        if self.parallel_trials > self.max_trials:
            raise ValueError("parallel_trials must be less than or equal to max_trials")
        return self


class CatsDogsConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    automl: AutoMLConfig = Field(default_factory=AutoMLConfig)


class HelloConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecipeCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=3,
        max_length=100,
    )

    recipe_id: str | None = None
    recipe_version: str | None = None
    workload: str | None = None
    configuration: dict[str, Any] | None = None

    training: TrainingConfig | None = None
    automl: AutoMLConfig | None = None
    training_config: TrainingConfig | None = None
    automl_config: AutoMLConfig | None = None


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Optional để Agent có thể PATCH riêng từng nhóm dữ liệu.
    status: JobStatus | None = None
    result_patch: JobResultPatch | None = None

    kfp_run_id: str | None = Field(
        default=None,
        max_length=128,
    )

    katib_experiment_name: str | None = Field(
        default=None,
        max_length=255,
    )

    mlflow_parent_run_id: str | None = Field(
        default=None,
        max_length=128,
    )

    best_params: dict[str, Any] | None = None
    best_metric: float | None = None

    mlflow_final_run_id: str | None = Field(
        default=None,
        max_length=128,
    )

    model_uri: str | None = Field(
        default=None,
        max_length=500,
    )

    registered_model_name: str | None = Field(
        default=None,
        max_length=255,
    )

    registered_model_version: str | None = Field(
        default=None,
        max_length=64,
    )

    final_metrics: dict[str, Any] | None = None

    error_message: str | None = Field(
        default=None,
        max_length=2000,
    )

    @model_validator(mode="after")
    def prevent_mixed_result_contracts(self) -> "AgentUpdate":
        if "result_patch" not in self.model_fields_set:
            return self
        if self.result_patch is None:
            raise ValueError("result_patch cannot be null")
        mixed_fields = sorted(LEGACY_RESULT_FIELDS & self.model_fields_set)
        if mixed_fields:
            raise ValueError(
                "result_patch cannot be combined with legacy result fields: "
                f"{mixed_fields}"
            )
        return self


class JobResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    id: str
    status: JobStatus
    recipe: dict[str, Any]

    agent_id: str | None = None

    kfp_run_id: str | None = None
    katib_experiment_name: str | None = None

    mlflow_parent_run_id: str | None = None
    mlflow_final_run_id: str | None = None

    best_params: dict[str, Any] | None = None
    best_metric: float | None = None

    model_uri: str | None = None

    registered_model_name: str | None = None
    registered_model_version: str | None = None

    final_metrics: dict[str, Any] | None = None
    result: JobResult | None = None

    error_message: str | None = None

    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def reconstruct_canonical_result(self) -> "JobResponse":
        if self.result is None:
            self.result = build_job_result(self)
        return self
