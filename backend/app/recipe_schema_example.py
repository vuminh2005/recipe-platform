"""Example showing where TrainingConfig belongs in your existing recipe schema."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .job_contracts import TrainingConfig


class AutoMLConfig(BaseModel):
    enabled: bool = True
    max_trials: int = Field(default=3, ge=1, le=20)


class RecipeCreate(BaseModel):
    name: str = "cats-dogs-recipe"
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    automl: AutoMLConfig = Field(default_factory=AutoMLConfig)
