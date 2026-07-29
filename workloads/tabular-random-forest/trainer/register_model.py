"""Idempotently register the final Tabular Random Forest MLflow model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import requests
from mlflow import MlflowClient


LOGGER = logging.getLogger("tabular_random_forest_register_model")
CANDIDATE_ALIAS = "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--registered-model-name", required=True)
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def wait_for_mlflow(tracking_uri: str) -> None:
    health_url = f"{tracking_uri.rstrip('/')}/health"
    for attempt in range(60):
        try:
            response = requests.get(health_url, timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass
        if attempt < 59:
            time.sleep(2)
    raise RuntimeError(f"MLflow did not become ready at {health_url}")


def read_result(path_value: str) -> dict[str, Any]:
    input_path = Path(path_value)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training result file does not exist: {input_path}"
        )
    try:
        result = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid training result JSON in {input_path}: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("Training result must be a JSON object")
    return result


def require_result_field(result: dict[str, Any], name: str) -> Any:
    value = result.get(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Training result is missing required field: {name}"
        )
    return value


def tag_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(value)


def find_existing_version(
    client: MlflowClient,
    model_name: str,
    platform_job_id: str,
    run_id: str,
    model_uri: str,
) -> Any | None:
    for version in client.search_model_versions(f"name = '{model_name}'"):
        tags = getattr(version, "tags", {}) or {}
        if tags.get("platform.job_id") == platform_job_id:
            return version
        if str(getattr(version, "run_id", "") or "") == run_id:
            return version
        if str(getattr(version, "source", "") or "") == model_uri:
            return version
    return None


def get_or_register_model_version(
    client: MlflowClient,
    *,
    model_name: str,
    model_uri: str,
    platform_job_id: str,
    run_id: str,
) -> Any:
    existing = find_existing_version(
        client,
        model_name,
        platform_job_id,
        run_id,
        model_uri,
    )
    if existing is not None:
        LOGGER.info(
            "Reusing registered model version %s for platform job %s",
            existing.version,
            platform_job_id,
        )
        return existing
    return mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        await_registration_for=300,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    args = parse_args()
    result = read_result(args.input_json)
    tracking_uri = required_env("MLFLOW_TRACKING_URI")
    platform_job_id = str(
        os.getenv("PLATFORM_JOB_ID")
        or require_result_field(result, "platform_job_id")
    )
    model_uri = str(require_result_field(result, "model_uri"))
    run_id = str(require_result_field(result, "mlflow_run_id"))
    model_name = args.registered_model_name.strip()
    if not model_name:
        raise ValueError("--registered-model-name must not be empty")

    wait_for_mlflow(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = get_or_register_model_version(
        client,
        model_name=model_name,
        model_uri=model_uri,
        platform_job_id=platform_job_id,
        run_id=run_id,
    )
    version = str(model_version.version)

    client.set_registered_model_tag(
        name=model_name,
        key="task",
        value="binary_tabular_classification",
    )
    client.set_registered_model_tag(
        name=model_name,
        key="framework",
        value="scikit_learn",
    )
    client.set_registered_model_tag(
        name=model_name,
        key="estimator",
        value="RandomForestClassifier",
    )

    version_tags = {
        "platform.job_id": platform_job_id,
        "platform.run_id": run_id,
        "platform.recipe_id": result.get(
            "recipe_id",
            "tabular-random-forest",
        ),
        "platform.recipe_version": result.get("recipe_version", "1.0"),
        "platform.lifecycle": "candidate",
        "model.framework": "scikit_learn",
        "model.task_type": "binary_tabular_classification",
        "model.architecture": "RandomForestClassifier",
        "model.feature_count": result.get("feature_count"),
        "model.feature_names": result.get("feature_names"),
        "model.class_mapping": result.get("class_mapping"),
        "model.probability_output": result.get("probability_output"),
    }
    for key, value in version_tags.items():
        if value is not None:
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=tag_value(value),
            )

    client.set_registered_model_alias(
        name=model_name,
        alias=CANDIDATE_ALIAS,
        version=version,
    )
    client.set_tag(
        run_id=run_id,
        key="platform.result",
        value="final_model_registered",
    )
    client.set_tag(
        run_id=run_id,
        key="platform.registered_model_name",
        value=model_name,
    )
    client.set_tag(
        run_id=run_id,
        key="platform.registered_model_version",
        value=version,
    )

    output_result = {
        **result,
        "registered_model_name": model_name,
        "registered_model_version": version,
        "registered_model_alias": CANDIDATE_ALIAS,
        "registered_model_uri": f"models:/{model_name}/{version}",
        "candidate_model_uri": f"models:/{model_name}@{CANDIDATE_ALIAS}",
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"registered_model_name={model_name}", flush=True)
    print(f"registered_model_version={version}", flush=True)


if __name__ == "__main__":
    main()
