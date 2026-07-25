from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import requests
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger("cats_dogs_register")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an MLflow model logged by the preceding KFP task."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to the JSON file produced by final train-and-evaluate.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="KFP output-parameter file for the enriched registration JSON.",
    )
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def wait_for_mlflow(tracking_uri: str) -> None:
    health_url = f"{tracking_uri.rstrip('/')}/health"
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            response = requests.get(health_url, timeout=(10, 180))
            if response.status_code < 500:
                return
            last_error = RuntimeError(
                f"MLflow returned HTTP {response.status_code}: {response.text[:200]}"
            )
        except requests.RequestException as exc:
            last_error = exc
        LOGGER.warning("MLflow not ready, attempt %d/8: %s", attempt, last_error)
        time.sleep(10)
    raise RuntimeError("MLflow tracking server did not become ready") from last_error


def find_existing_version(
    client: MlflowClient,
    *,
    model_name: str,
    platform_job_id: str,
    run_id: str,
    model_uri: str,
):
    """Make registration idempotent when a KFP task is retried."""
    for version in client.search_model_versions(f"name='{model_name}'"):
        tags = dict(getattr(version, "tags", {}) or {})
        if tags.get("platform.job_id") == platform_job_id:
            return version
        if str(getattr(version, "run_id", "") or "") == run_id:
            return version
        if str(getattr(version, "source", "") or "") == model_uri:
            return version
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    input_path = Path(args.input_json)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training result JSON does not exist: {input_path}"
        )

    raw_result = input_path.read_text(encoding="utf-8").strip()

    if not raw_result:
        raise RuntimeError(
            f"Training result JSON is empty: {input_path}"
        )

    LOGGER.info(
        "Reading training result from %s (%d bytes)",
        input_path,
        len(raw_result.encode("utf-8")),
    )

    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Training result is not valid JSON: {input_path}. "
            f"Content starts with: {raw_result[:200]!r}"
        ) from exc
    run_id = str(result["mlflow_run_id"])
    model_uri = str(result["model_uri"])
    platform_job_id = str(
        result.get("platform_job_id")
        or os.getenv("PLATFORM_JOB_ID")
        or "standalone"
    )

    tracking_uri = required_env("MLFLOW_TRACKING_URI")
    model_name = required_env("MLFLOW_REGISTERED_MODEL_NAME")
    wait_for_mlflow(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    existing = find_existing_version(
        client,
        model_name=model_name,
        platform_job_id=platform_job_id,
        run_id=run_id,
        model_uri=model_uri,
    )
    if existing is None:
        version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            await_registration_for=300,
        )
    else:
        version = existing
        LOGGER.info(
            "Reusing registered model version %s for platform job %s",
            version.version,
            platform_job_id,
        )

    version_number = str(version.version)
    version_tags = {
        "platform.job_id": platform_job_id,
        "mlflow.run_id": run_id,
        "platform.model_uri": model_uri,
    }
    for metric_name in (
        "final_threshold",
        "test_accuracy",
        "test_f1",
        "test_auc",
        "val_auc",
    ):
        value = result.get(metric_name)
        if value is not None:
            version_tags[f"metric.{metric_name}"] = str(value)

    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version_number, key, value)

    run_tags = {
        "platform.model_uri": model_uri,
        "platform.registered_model_name": model_name,
        "platform.registered_model_version": version_number,
        "platform.result": "final_model_registered",
    }
    for key, value in run_tags.items():
        client.set_tag(run_id, key, value)

    result.update(
        {
            "registered_model_name": model_name,
            "registered_model_version": version_number,
        }
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    LOGGER.info(
        "Registered %s version %s from run %s",
        model_name,
        version_number,
        run_id,
    )


if __name__ == "__main__":
    main()
