import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./recipe_platform.db",
)

AGENT_TOKEN = os.getenv("AGENT_TOKEN", "development-token")

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


class AutoMLConfig(BaseModel):
    enabled: bool = True
    max_trials: int = Field(default=3, ge=1, le=20)
    parallel_trials: int = Field(default=1, ge=1, le=4)
    algorithm: str = "random"


class TrainingConfig(BaseModel):
    model: str = "tiny_cnn"
    epochs: int = Field(default=1, ge=1, le=20)
    batch_size: int = Field(default=8, ge=1, le=128)


class RecipeCreate(BaseModel):
    name: str = Field(min_length=3, max_length=100)
    workload: Literal["hello", "cats-dogs"] = "hello"
    automl: AutoMLConfig = AutoMLConfig()
    training: TrainingConfig = TrainingConfig()


class AgentClaim(BaseModel):
    agent_id: str


class AgentUpdate(BaseModel):
    status: Literal[
        "CLAIMED",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
    ]

    kfp_run_id: str | None = None
    katib_experiment_name: str | None = None
    mlflow_parent_run_id: str | None = None
    error_message: str | None = None


class JobResponse(BaseModel):
    id: str
    status: str
    recipe: dict[str, Any]
    agent_id: str | None
    kfp_run_id: str | None
    katib_experiment_name: str | None
    mlflow_parent_run_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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


@asynccontextmanager
async def lifespan(_: FastAPI):
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/api/jobs",
    response_model=JobResponse,
    status_code=201,
)
def create_job(
    recipe: RecipeCreate,
    db: Session = Depends(get_db),
):
    job = Job(
        recipe=recipe.model_dump(),
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

    job.status = payload.status

    if payload.kfp_run_id is not None:
        job.kfp_run_id = payload.kfp_run_id

    if payload.katib_experiment_name is not None:
        job.katib_experiment_name = (
            payload.katib_experiment_name
        )

    if payload.mlflow_parent_run_id is not None:
        job.mlflow_parent_run_id = (
            payload.mlflow_parent_run_id
        )

    if payload.error_message is not None:
        job.error_message = payload.error_message

    job.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)

    return job
