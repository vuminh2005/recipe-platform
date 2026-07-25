"""Copy these fields into the existing PlatformJob SQLAlchemy model."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Inside class PlatformJob(Base):
#
# best_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
# best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
# mlflow_final_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
# model_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
# registered_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
# registered_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
# final_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


def apply_agent_update(job: Any, update_payload: Any) -> None:
    """Use this in PATCH /api/agent/jobs/{job_id}; do not keep an old whitelist."""
    for field_name, value in update_payload.model_dump(
        exclude_unset=True
    ).items():
        setattr(job, field_name, value)
