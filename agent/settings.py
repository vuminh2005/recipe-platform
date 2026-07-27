from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    backend_url: str
    agent_id: str
    agent_token: str
    kfp_endpoint: str
    poll_interval_seconds: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    katib_namespace: str
    cats_dogs_pipeline_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend_url=required_env("BACKEND_URL").rstrip("/"),
            agent_id=os.getenv("AGENT_ID", "laptop-k3s-agent"),
            agent_token=required_env("AGENT_TOKEN"),
            kfp_endpoint=os.getenv("KFP_ENDPOINT", "http://127.0.0.1:8080"),
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "10")),
            mlflow_tracking_uri=required_env("MLFLOW_TRACKING_URI").rstrip("/"),
            mlflow_experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "cats_dogs_recipe_demo"
            ),
            katib_namespace=os.getenv("KATIB_NAMESPACE", "ml-platform"),
            cats_dogs_pipeline_path=Path(
                os.getenv(
                    "CATS_DOGS_PIPELINE_PATH",
                    "pipelines/compiled/cats_dogs_final_pipeline.yaml",
                )
            ).resolve(),
        )
