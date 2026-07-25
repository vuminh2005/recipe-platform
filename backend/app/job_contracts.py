"""Pydantic contracts used by the Recipe Platform API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    # Giữ tiny_cnn tạm thời để frontend/Hello Job cũ không bị 422.
    model: Literal[
        "tiny_cnn",
        "mobilenet_v2",
    ] = "mobilenet_v2"

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

    # Trường legacy do frontend cũ đang gửi.
    # Hello workload có thể tiếp tục dùng trường này.
    epochs: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Optional để Agent có thể PATCH riêng từng nhóm dữ liệu.
    status: JobStatus | None = None

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

    error_message: str | None = None

    created_at: datetime
    updated_at: datetime
