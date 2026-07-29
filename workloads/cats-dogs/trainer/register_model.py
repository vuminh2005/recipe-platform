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


LOGGER = logging.getLogger("cats_dogs_register_model")

CANDIDATE_ALIAS = "candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register a logged Cats & Dogs model, "
            "assign the candidate alias, and attach inference metadata."
        )
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to the final training result JSON file.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the registration result JSON is written.",
    )

    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

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

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training result file does not exist: {input_path}"
        )

    raw_result = input_path.read_text(
        encoding="utf-8"
    ).strip()

    if not raw_result:
        raise RuntimeError(
            f"Training result file is empty: {input_path}"
        )

    LOGGER.info(
        "Reading training result from %s (%d bytes)",
        input_path,
        len(raw_result.encode("utf-8")),
    )

    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid training result JSON in {input_path}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Training result must be a JSON object"
        )

    return parsed


def require_result_field(
    result: dict[str, Any],
    field_name: str,
) -> Any:
    value = result.get(field_name)

    if value is None or value == "":
        raise RuntimeError(
            f"Training result is missing required field: {field_name}"
        )

    return value


def tag_value(value: Any) -> str:
    """Convert metadata to a safe MLflow tag string."""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return str(value)


def set_version_tag_if_present(
    client: MlflowClient,
    *,
    model_name: str,
    version: str,
    key: str,
    value: Any,
) -> None:
    if value is None:
        return

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key=key,
        value=tag_value(value),
    )


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
    model_name: str,
    model_uri: str,
    platform_job_id: str,
    run_id: str,
) -> Any:
    existing = find_existing_version(
        client=client,
        model_name=model_name,
        platform_job_id=platform_job_id,
        run_id=run_id,
        model_uri=model_uri,
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


def build_version_tags(
    *,
    result: dict[str, Any],
    platform_job_id: str,
    mlflow_run_id: str,
) -> dict[str, Any]:
    return {
        "platform.job_id": platform_job_id,
        "platform.run_id": mlflow_run_id,
        "platform.recipe_id": result.get("recipe_id", "cats-dogs"),
        "platform.recipe_version": result.get("recipe_version", "1.0"),
        "platform.katib_experiment_id": result.get(
            "katib_experiment_id"
        ),
        "platform.lifecycle": "candidate",
        "model.framework": "tensorflow_keras",
        "model.task_type": "binary_image_classification",
        "model.architecture": result.get(
            "model_architecture",
            "MobileNetV2",
        ),
        "inference.image_size": result.get("image_size"),
        "inference.num_channels": result.get("num_channels", 3),
        "inference.threshold": result.get("final_threshold"),
        "inference.output": result.get("output_semantics", "prob_dog"),
        "inference.preprocessing": result.get(
            "preprocessing",
            "embedded_mobilenet_v2",
        ),
        "metric.final_threshold": result.get("final_threshold"),
        "metric.val_loss": result.get("val_loss"),
        "metric.val_accuracy": result.get("val_accuracy"),
        "metric.val_auc": result.get("val_auc"),
        "metric.test_accuracy": result.get("test_accuracy"),
        "metric.test_f1": result.get("test_f1"),
        "metric.test_auc": result.get("test_auc"),
        "metric.best_epoch": result.get("best_epoch"),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()
    result = read_result(args.input_json)

    tracking_uri = required_env(
        "MLFLOW_TRACKING_URI"
    )

    registered_model_name = required_env(
        "MLFLOW_REGISTERED_MODEL_NAME"
    )

    model_uri = str(
        require_result_field(result, "model_uri")
    )

    mlflow_run_id = str(
        require_result_field(result, "mlflow_run_id")
    )

    platform_job_id = (
        os.getenv("PLATFORM_JOB_ID")
        or result.get("platform_job_id")
        or "standalone"
    )

    wait_for_mlflow(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    LOGGER.info(
        "Registering model | name=%s | source=%s",
        registered_model_name,
        model_uri,
    )

    model_version = get_or_register_model_version(
        client=client,
        model_name=registered_model_name,
        model_uri=model_uri,
        platform_job_id=str(platform_job_id),
        run_id=mlflow_run_id,
    )

    version_number = str(model_version.version)

    LOGGER.info(
        "Registered model version | name=%s | version=%s",
        registered_model_name,
        version_number,
    )

    # ------------------------------------------------------------------
    # Registered model-level metadata
    # ------------------------------------------------------------------

    client.set_registered_model_tag(
        name=registered_model_name,
        key="task",
        value="binary_image_classification",
    )

    client.set_registered_model_tag(
        name=registered_model_name,
        key="framework",
        value="tensorflow_keras",
    )

    client.set_registered_model_tag(
        name=registered_model_name,
        key="labels",
        value="cat,dog",
    )

    # ------------------------------------------------------------------
    # Model version-level metadata
    # ------------------------------------------------------------------

    version_tags = build_version_tags(
        result=result,
        platform_job_id=str(platform_job_id),
        mlflow_run_id=mlflow_run_id,
    )

    for key, value in version_tags.items():
        set_version_tag_if_present(
            client,
            model_name=registered_model_name,
            version=version_number,
            key=key,
            value=value,
        )

    # ------------------------------------------------------------------
    # Candidate alias
    # ------------------------------------------------------------------
    #
    # This moves the candidate alias to the newly registered version.
    # It intentionally does not modify the champion alias.
    # ------------------------------------------------------------------

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CANDIDATE_ALIAS,
        version=version_number,
    )

    # Mark the source MLflow run as fully registered.
    client.set_tag(
        run_id=mlflow_run_id,
        key="platform.result",
        value="final_model_registered",
    )

    client.set_tag(
        run_id=mlflow_run_id,
        key="platform.registered_model_name",
        value=registered_model_name,
    )

    client.set_tag(
        run_id=mlflow_run_id,
        key="platform.registered_model_version",
        value=version_number,
    )

    output_result = {
        **result,

        "platform_job_id": platform_job_id,

        "registered_model_name": (
            registered_model_name
        ),

        "registered_model_version": (
            version_number
        ),

        "registered_model_alias": (
            CANDIDATE_ALIAS
        ),

        "registered_model_uri": (
            f"models:/{registered_model_name}/"
            f"{version_number}"
        ),

        "candidate_model_uri": (
            f"models:/{registered_model_name}"
            f"@{CANDIDATE_ALIAS}"
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output_result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Candidate alias assigned | "
        "model=%s | alias=%s | version=%s",
        registered_model_name,
        CANDIDATE_ALIAS,
        version_number,
    )

    LOGGER.info(
        "Registration output written to %s",
        output_path,
    )

    print(
        f"registered_model_name="
        f"{registered_model_name}",
        flush=True,
    )

    print(
        f"registered_model_version="
        f"{version_number}",
        flush=True,
    )

    print(
        f"registered_model_alias="
        f"{CANDIDATE_ALIAS}",
        flush=True,
    )

    print(
        f"candidate_model_uri="
        f"models:/{registered_model_name}"
        f"@{CANDIDATE_ALIAS}",
        flush=True,
    )

    print(
        f"output_path={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
