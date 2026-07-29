import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
    Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from .job_contracts import (
    AgentUpdate,
    JobResponse,
    RecipeCreate,
)
from .job_results import apply_result_patch
from .recipe_catalog import router as recipe_catalog_router
from .recipe_normalization import (
    RecipeNormalizationError,
    normalize_recipe_request,
)


def normalize_database_url(database_url: str) -> str:
    """Select psycopg 3 for Render-style PostgreSQL URLs."""
    prefix = "postgresql://"
    if database_url.startswith(prefix):
        return f"postgresql+psycopg://{database_url[len(prefix):]}"
    return database_url


DATABASE_URL = normalize_database_url(
    os.getenv(
        "DATABASE_URL",
        "sqlite:///./recipe_platform.db",
    )
)

AGENT_TOKEN = os.getenv("AGENT_TOKEN", "").strip()
JOB_SUBMISSION_TOKEN = os.getenv("JOB_SUBMISSION_TOKEN", "").strip()
JOB_SUBMISSION_HEADER = APIKeyHeader(
    name="X-Job-Submission-Token",
    scheme_name="JobSubmissionToken",
    description="Runtime token required only when creating a job.",
    auto_error=False,
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]


connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "platform_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="PENDING",
        index=True,
    )

    recipe: Mapped[dict[str, Any]] = mapped_column(JSON)

    agent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    kfp_run_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    katib_experiment_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    mlflow_parent_run_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    best_params: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    best_metric: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    mlflow_final_run_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    model_uri: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    registered_model_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    registered_model_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    final_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AgentClaim(BaseModel):
    agent_id: str


def get_db():
    with Session(engine) as session:
        yield session


def verify_agent_token(
    x_agent_token: str = Header(alias="X-Agent-Token"),
) -> None:
    if x_agent_token != AGENT_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid agent token",
        )


def verify_job_submission_token(
    x_job_submission_token: str | None = Security(JOB_SUBMISSION_HEADER),
) -> None:
    if x_job_submission_token is None:
        raise HTTPException(
            status_code=401,
            detail="Job submission token is required",
        )
    if not secrets.compare_digest(
        x_job_submission_token,
        JOB_SUBMISSION_TOKEN,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid job submission token",
        )


def validate_agent_token_configuration() -> None:
    if not AGENT_TOKEN:
        raise RuntimeError(
            "AGENT_TOKEN must be configured before the Backend starts"
        )
    allow_insecure = (
        os.getenv("ALLOW_INSECURE_DEVELOPMENT_TOKEN", "").strip().lower()
        == "true"
    )
    if AGENT_TOKEN == "development-token" and not allow_insecure:
        raise RuntimeError(
            "AGENT_TOKEN must not use the insecure development token; "
            "set a secure token or explicitly set "
            "ALLOW_INSECURE_DEVELOPMENT_TOKEN=true for local development"
        )


def validate_job_submission_token_configuration() -> None:
    if not JOB_SUBMISSION_TOKEN:
        raise RuntimeError(
            "JOB_SUBMISSION_TOKEN must be configured before the Backend starts"
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_agent_token_configuration()
    validate_job_submission_token_configuration()
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Recipe Platform API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(recipe_catalog_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/api/jobs",
    response_model=JobResponse,
    status_code=201,
    dependencies=[Depends(verify_job_submission_token)],
)
def create_job(
    recipe: RecipeCreate,
    db: Session = Depends(get_db),
):
    try:
        recipe_payload = normalize_recipe_request(recipe)
    except RecipeNormalizationError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    job = Job(
        recipe=recipe_payload,
        status="PENDING",
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return job


@app.get(
    "/api/jobs",
    response_model=list[JobResponse],
)
def list_jobs(
    db: Session = Depends(get_db),
):
    statement = (
        select(Job)
        .order_by(Job.created_at.desc())
        .limit(100)
    )

    return list(db.scalars(statement))


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobResponse,
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job


@app.post(
    "/api/agent/jobs/claim",
    response_model=JobResponse,
    dependencies=[Depends(verify_agent_token)],
)
def claim_next_job(
    payload: AgentClaim,
    db: Session = Depends(get_db),
):
    statement = (
        select(Job)
        .where(Job.status == "PENDING")
        .order_by(Job.created_at.asc())
        .with_for_update(skip_locked=True)
	.limit(1)
    )

    job = db.scalar(statement)

    if job is None:
        return Response(status_code=204)

    job.status = "CLAIMED"
    job.agent_id = payload.agent_id
    job.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)

    return job

@app.patch(
    "/api/agent/jobs/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(verify_agent_token)],
)
def update_job(
    job_id: str,
    payload: AgentUpdate,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    # mode="json" chuyển JobStatus enum thành chuỗi,
    # ví dụ JobStatus.TUNING → "TUNING".
    updates = payload.model_dump(
        exclude_unset=True,
        mode="json",
        exclude={"result_patch"},
    )

    for field_name, value in updates.items():
        if not hasattr(job, field_name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported job field: "
                    f"{field_name}"
                ),
            )

        setattr(job, field_name, value)

    if payload.result_patch is not None:
        try:
            apply_result_patch(job, payload.result_patch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    job.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)

    return job
