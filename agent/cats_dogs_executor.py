from __future__ import annotations

import logging
import time
from typing import Any

from agent.backend_client import BackendClient
from agent.cats_dogs_katib import (
    SUPPORTED_ALGORITHMS,
    build_experiment_manifest,
    parse_experiment_result,
)
from agent.katib_runner import KatibRunner
from agent.kfp_runner import KfpRunner
from agent.mlflow_rest import MlflowRestClient
from agent.settings import Settings

LOGGER = logging.getLogger("cats_dogs_executor")

DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_DROPOUT_RATE = 0.25
DEFAULT_LEARNING_RATE_MIN = 0.00005
DEFAULT_LEARNING_RATE_MAX = 0.0005
DEFAULT_DROPOUT_RATE_MIN = 0.15
DEFAULT_DROPOUT_RATE_MAX = 0.45
FINAL_RUN_ROLE = "final_training"
FINAL_LOGGED_RESULT = "final_model_logged"
FINAL_REGISTERED_RESULT = "final_model_registered"


def _recipe_sections(
    job: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recipe = job.get("recipe") or {}
    configuration = recipe.get("configuration")
    if not isinstance(configuration, dict):
        configuration = {}
    training = (
        configuration.get("training")
        or recipe.get("training")
        or recipe.get("training_config")
        or {}
    )
    automl = (
        configuration.get("automl")
        or recipe.get("automl")
        or recipe.get("automl_config")
        or {}
    )
    return training, automl, configuration


def _final_result_patch(final_run: dict[str, Any]) -> dict[str, Any]:
    tags = final_run["tags"]
    metrics = final_run["metrics"]
    required_tags = (
        "platform.model_uri",
        "platform.registered_model_name",
        "platform.registered_model_version",
    )
    missing = [key for key in required_tags if not tags.get(key)]
    if missing:
        raise RuntimeError(
            f"Final MLflow run is missing registration tags: {missing}; tags={tags}"
        )

    final_metrics = {
        name: float(metrics[name])
        for name in (
            "val_loss",
            "val_accuracy",
            "val_auc",
            "final_threshold",
            "test_accuracy",
            "test_f1",
            "test_auc",
            "best_epoch",
        )
        if name in metrics
    }
    return {
        "final_metrics": final_metrics,
        "external_ids": {
            "mlflow_run_id": str(final_run["run_id"]),
        },
        "model": {
            "uri": str(tags["platform.model_uri"]),
            "registered_name": str(tags["platform.registered_model_name"]),
            "version": str(tags["platform.registered_model_version"]),
        },
    }


def _find_final_run(
    mlflow: MlflowRestClient,
    *,
    experiment_id: str,
    platform_job_id: str,
) -> dict[str, Any] | None:
    return mlflow.find_latest_run(
        experiment_id=experiment_id,
        platform_job_id=platform_job_id,
        run_role=FINAL_RUN_ROLE,
        max_results=5,
    )


def _wait_for_registered_final_run(
    mlflow: MlflowRestClient,
    *,
    experiment_id: str,
    platform_job_id: str,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_run: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_run = _find_final_run(
            mlflow,
            experiment_id=experiment_id,
            platform_job_id=platform_job_id,
        )
        if last_run:
            tags = last_run["tags"]
            if (
                tags.get("platform.result") == FINAL_REGISTERED_RESULT
                and tags.get("platform.registered_model_version")
            ):
                return last_run
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Final registered MLflow run not found for job {platform_job_id}; "
        f"last_run={last_run}"
    )


def _validate_execution_config(
    training: dict[str, Any],
    automl: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    model = str(training.get("model", "mobilenet_v2"))
    if model == "tiny_cnn":
        LOGGER.warning(
            "Legacy training.model='tiny_cnn' is not implemented; "
            "normalizing it to 'mobilenet_v2'."
        )
        model = "mobilenet_v2"
    if model != "mobilenet_v2":
        raise ValueError(
            f"Unsupported Cats & Dogs training model: {model!r}; "
            "supported: 'mobilenet_v2'"
        )

    if training.get("epochs") is not None:
        LOGGER.warning(
            "Legacy training.epochs is ignored; use training.trial_epochs "
            "and training.final_epochs."
        )

    max_trial_count = int(automl.get("max_trials", 3))
    parallel_trial_count = int(automl.get("parallel_trials", 1))
    if max_trial_count < 1:
        raise ValueError("automl.max_trials must be at least 1")
    if parallel_trial_count < 1:
        raise ValueError("automl.parallel_trials must be at least 1")
    if parallel_trial_count > max_trial_count:
        raise ValueError(
            "automl.parallel_trials must be less than or equal to "
            "automl.max_trials"
        )

    algorithm_name = str(automl.get("algorithm", "random"))
    if algorithm_name not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(
            f"Unsupported automl.algorithm {algorithm_name!r}; supported: {supported}"
        )

    search_space = automl.get("search_space") or {}
    learning_rate_range = search_space.get("learning_rate") or {}
    dropout_rate_range = search_space.get("dropout_rate") or {}
    learning_rate_min = float(
        learning_rate_range.get("min", DEFAULT_LEARNING_RATE_MIN)
    )
    learning_rate_max = float(
        learning_rate_range.get("max", DEFAULT_LEARNING_RATE_MAX)
    )
    dropout_rate_min = float(
        dropout_rate_range.get("min", DEFAULT_DROPOUT_RATE_MIN)
    )
    dropout_rate_max = float(
        dropout_rate_range.get("max", DEFAULT_DROPOUT_RATE_MAX)
    )

    if learning_rate_min <= 0 or learning_rate_min >= learning_rate_max:
        raise ValueError(
            "automl.search_space.learning_rate must satisfy 0 < min < max"
        )
    if not 0 <= dropout_rate_min < dropout_rate_max < 1:
        raise ValueError(
            "automl.search_space.dropout_rate must satisfy 0 <= min < max < 1"
        )

    effective_final_parameters = configuration.get(
        "effective_final_parameters"
    )
    if not isinstance(effective_final_parameters, dict):
        effective_final_parameters = {}
    final_learning_rate = float(
        effective_final_parameters.get(
            "learning_rate",
            DEFAULT_LEARNING_RATE,
        )
    )
    final_dropout_rate = float(
        effective_final_parameters.get(
            "dropout_rate",
            DEFAULT_DROPOUT_RATE,
        )
    )
    if final_learning_rate <= 0:
        raise ValueError(
            "effective_final_parameters.learning_rate must be positive"
        )
    if not 0 <= final_dropout_rate < 1:
        raise ValueError(
            "effective_final_parameters.dropout_rate must satisfy 0 <= value < 1"
        )

    return {
        "automl_enabled": bool(automl.get("enabled", True)),
        "max_trial_count": max_trial_count,
        "parallel_trial_count": parallel_trial_count,
        "algorithm_name": algorithm_name,
        "learning_rate_min": learning_rate_min,
        "learning_rate_max": learning_rate_max,
        "dropout_rate_min": dropout_rate_min,
        "dropout_rate_max": dropout_rate_max,
        "final_learning_rate": final_learning_rate,
        "final_dropout_rate": final_dropout_rate,
    }


def execute_cats_dogs_job(
    job: dict[str, Any],
    *,
    settings: Settings,
    backend: BackendClient,
) -> None:
    job_id = str(job["id"])
    training, automl, configuration = _recipe_sections(job)
    try:
        execution_config = _validate_execution_config(
            training,
            automl,
            configuration,
        )
    except Exception as exc:
        backend.patch_job(job_id, status="FAILED", error_message=str(exc))
        raise

    trial_epochs = int(training.get("trial_epochs", 2))
    final_epochs = int(training.get("final_epochs", 5))
    batch_size = int(training.get("batch_size", 8))
    dense_units = int(training.get("dense_units", 128))
    image_size = int(training.get("image_size", 224))
    trainable_backbone = bool(training.get("trainable_backbone", False))

    mlflow = MlflowRestClient(settings.mlflow_tracking_uri)
    kfp = KfpRunner(settings.kfp_endpoint)

    parent_run_id = job.get("mlflow_parent_run_id")

    try:
        # 1-2. Wake Render, resolve experiment and create/reuse parent run.
        mlflow.warm()
        experiment_id = mlflow.get_or_create_experiment(
            settings.cats_dogs.mlflow_experiment_name
        )
        if not parent_run_id:
            parent_run_id = mlflow.ensure_parent_run(
                experiment_id=experiment_id,
                platform_job_id=job_id,
                agent_id=settings.agent_id,
            )
            backend.patch_job(
                job_id,
                result_patch={
                    "external_ids": {
                        "mlflow_parent_run_id": parent_run_id,
                    }
                },
            )

        # A retry can finish immediately if the final registered run already exists.
        existing_final = _find_final_run(
            mlflow,
            experiment_id=experiment_id,
            platform_job_id=job_id,
        )
        if (
            existing_final
            and existing_final["tags"].get("platform.result")
            == FINAL_REGISTERED_RESULT
        ):
            backend.patch_job(
                job_id,
                status="SUCCEEDED",
                error_message="",
                result_patch=_final_result_patch(existing_final),
            )
            mlflow.terminate_run(str(parent_run_id), status="FINISHED")
            return

        katib_name = ""
        learning_rate = execution_config["final_learning_rate"]
        dropout_rate = execution_config["final_dropout_rate"]

        if execution_config["automl_enabled"]:
            # 3-6. Create/reuse deterministic Katib Experiment and wait for it.
            backend.patch_job(job_id, status="TUNING", error_message="")
            katib = KatibRunner(settings.katib_namespace)
            katib_manifest = build_experiment_manifest(
                namespace=settings.katib_namespace,
                job_id=job_id,
                parent_run_id=str(parent_run_id),
                trial_epochs=trial_epochs,
                batch_size=batch_size,
                dense_units=dense_units,
                image_size=image_size,
                max_trial_count=execution_config["max_trial_count"],
                parallel_trial_count=execution_config["parallel_trial_count"],
                algorithm_name=execution_config["algorithm_name"],
                learning_rate_min=execution_config["learning_rate_min"],
                learning_rate_max=execution_config["learning_rate_max"],
                dropout_rate_min=execution_config["dropout_rate_min"],
                dropout_rate_max=execution_config["dropout_rate_max"],
                trainable_backbone=trainable_backbone,
            )
            katib_name = str(katib_manifest["metadata"]["name"])
            katib.ensure_experiment(katib_manifest)
            backend.patch_job(
                job_id,
                result_patch={
                    "external_ids": {
                        "katib_experiment_id": katib_name,
                    }
                },
            )

            katib_experiment = katib.wait_for_success(katib_name)
            katib_result = parse_experiment_result(katib_experiment)
            learning_rate = katib_result.best_params["learning_rate"]
            dropout_rate = katib_result.best_params["dropout_rate"]
            best_params = {
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "best_trial_name": katib_result.best_trial_name,
                "katib_metrics": katib_result.metrics,
            }
            backend.patch_job(
                job_id,
                status="TRAINING",
                result_patch={
                    "objective": {
                        "value": katib_result.best_metric,
                    },
                    "best_params": best_params,
                },
            )
        else:
            backend.patch_job(job_id, status="TRAINING", error_message="")

        # 7-10. Submit or resume the compiled final KFP run.
        kfp_run_id = job.get("kfp_run_id")
        if not kfp_run_id:
            kfp_run_id = kfp.submit_pipeline(
                pipeline_path=settings.cats_dogs.pipeline_path,
                run_name=f"cats-dogs-final-{job_id[:8]}",
                arguments={
                    "platform_job_id": job_id,
                    "mlflow_parent_run_id": str(parent_run_id),
                    "katib_experiment_name": katib_name,
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "dense_units": dense_units,
                    "batch_size": batch_size,
                    "final_epochs": final_epochs,
                    "image_size": image_size,
                    "trainable_backbone": trainable_backbone,
                },
            )
            backend.patch_job(
                job_id,
                result_patch={
                    "external_ids": {
                        "kfp_run_id": kfp_run_id,
                    }
                },
            )

        registering_reported = False
        while True:
            status = kfp.get_status(str(kfp_run_id))
            if kfp.is_success(status):
                break
            if kfp.is_failure(status):
                raise RuntimeError(
                    f"KFP run {kfp_run_id} finished with status {status}"
                )

            # The training task sets final_model_logged before the registration
            # component starts. This gives the UI a real REGISTERING transition.
            final_run = _find_final_run(
                mlflow,
                experiment_id=experiment_id,
                platform_job_id=job_id,
            )
            if (
                not registering_reported
                and final_run
                and final_run["tags"].get("platform.result")
                == FINAL_LOGGED_RESULT
            ):
                backend.patch_job(job_id, status="REGISTERING")
                registering_reported = True

            time.sleep(5)

        if not registering_reported:
            backend.patch_job(job_id, status="REGISTERING")

        # 11-15. Resolve the registered final run and persist the complete result.
        final_run = _wait_for_registered_final_run(
            mlflow,
            experiment_id=experiment_id,
            platform_job_id=job_id,
            timeout_seconds=300,
        )
        backend.patch_job(
            job_id,
            status="SUCCEEDED",
            error_message="",
            result_patch=_final_result_patch(final_run),
        )
        mlflow.terminate_run(str(parent_run_id), status="FINISHED")
        LOGGER.info("Platform job %s succeeded", job_id)

    except Exception as exc:
        LOGGER.exception("Platform job %s failed", job_id)
        try:
            backend.patch_job(
                job_id,
                status="FAILED",
                error_message=str(exc)[:4000],
            )
        finally:
            if parent_run_id:
                try:
                    mlflow.terminate_run(str(parent_run_id), status="FAILED")
                except Exception:
                    LOGGER.exception("Could not terminate parent MLflow run")
        raise
