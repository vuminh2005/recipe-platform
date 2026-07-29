"""Recipe-specific orchestration for Tabular Random Forest classification."""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.backend_client import BackendClient
from agent.katib_runner import KatibRunner
from agent.kfp_runner import KfpRunner
from agent.mlflow_rest import MlflowRestClient
from agent.recipe_ids import TABULAR_RANDOM_FOREST_RECIPE_ID
from agent.settings import Settings
from agent.tabular_random_forest_config import (
    RECIPE_VERSION,
    validate_execution_config,
)
from agent.tabular_random_forest_katib import (
    build_experiment_manifest,
    parse_experiment_result,
)


LOGGER = logging.getLogger("tabular_random_forest_executor")
FINAL_RUN_ROLE = "final_training"
FINAL_LOGGED_RESULT = "final_model_logged"
FINAL_REGISTERED_RESULT = "final_model_registered"
LINEAGE_UPDATE_ATTEMPTS = 3
LINEAGE_UPDATE_RETRY_SECONDS = 1


def _configuration(job: dict[str, Any]) -> dict[str, Any]:
    recipe = job.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Job recipe must be an object")
    configuration = recipe.get("configuration")
    if configuration is None:
        return {}
    if not isinstance(configuration, dict):
        raise ValueError("recipe.configuration must be an object")
    return configuration


def _recipe_version(job: dict[str, Any]) -> str:
    recipe = job.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Job recipe must be an object")
    value = recipe.get("recipe_version", RECIPE_VERSION)
    if value != RECIPE_VERSION:
        raise ValueError(
            f"Unsupported Tabular Random Forest recipe version {value!r}; "
            f"supported: {RECIPE_VERSION}"
        )
    return str(value)


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
            f"Final MLflow run is missing registration tags: {missing}; "
            f"tags={tags}"
        )

    expected_metrics = (
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    )
    final_metrics = {
        name: float(metrics[name])
        for name in expected_metrics
        if name in metrics
    }
    missing_metrics = sorted(set(expected_metrics) - set(final_metrics))
    if missing_metrics:
        raise RuntimeError(
            "Final MLflow run is missing required Tabular metrics: "
            f"{missing_metrics}"
        )

    return {
        "final_metrics": final_metrics,
        "external_ids": {
            "mlflow_run_id": str(final_run["run_id"]),
        },
        "model": {
            "uri": str(tags["platform.model_uri"]),
            "registered_name": str(
                tags["platform.registered_model_name"]
            ),
            "version": str(tags["platform.registered_model_version"]),
        },
    }


def _attach_kfp_lineage(
    mlflow: MlflowRestClient,
    *,
    final_run: dict[str, Any],
    kfp_run_id: str,
) -> bool:
    tags = final_run["tags"]
    run_id = str(final_run["run_id"])
    model_name = str(tags["platform.registered_model_name"])
    model_version = str(tags["platform.registered_model_version"])
    last_error: Exception | None = None

    for attempt in range(1, LINEAGE_UPDATE_ATTEMPTS + 1):
        try:
            mlflow.set_run_tag(
                run_id=run_id,
                key="platform.kfp_run_id",
                value=kfp_run_id,
            )
            mlflow.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="platform.kfp_run_id",
                value=kfp_run_id,
            )
            LOGGER.info(
                "Attached KFP run %s to MLflow run %s and model %s/%s",
                kfp_run_id,
                run_id,
                model_name,
                model_version,
            )
            return True
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Could not attach KFP lineage on attempt %d/%d: %s",
                attempt,
                LINEAGE_UPDATE_ATTEMPTS,
                exc,
            )
            if attempt < LINEAGE_UPDATE_ATTEMPTS:
                time.sleep(LINEAGE_UPDATE_RETRY_SECONDS)

    LOGGER.error(
        "Training and registration succeeded, but KFP run %s could not be "
        "attached to MLflow metadata after %d attempts. The backend KFP ID "
        "remains canonical. Last error: %s",
        kfp_run_id,
        LINEAGE_UPDATE_ATTEMPTS,
        last_error,
    )
    return False


def execute_tabular_random_forest_job(
    job: dict[str, Any],
    *,
    settings: Settings,
    backend: BackendClient,
) -> None:
    job_id = str(job["id"])
    try:
        recipe_version = _recipe_version(job)
        execution_config = validate_execution_config(_configuration(job))
    except Exception as exc:
        backend.patch_job(job_id, status="FAILED", error_message=str(exc))
        raise

    mlflow = MlflowRestClient(settings.mlflow_tracking_uri)
    kfp = KfpRunner(settings.kfp_endpoint)
    parent_run_id = job.get("mlflow_parent_run_id")

    try:
        mlflow.warm()
        experiment_id = mlflow.get_or_create_experiment(
            settings.tabular_random_forest.mlflow_experiment_name
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
                        "mlflow_parent_run_id": str(parent_run_id),
                    }
                },
            )

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
            existing_kfp_run_id = job.get("kfp_run_id")
            if existing_kfp_run_id:
                _attach_kfp_lineage(
                    mlflow,
                    final_run=existing_final,
                    kfp_run_id=str(existing_kfp_run_id),
                )
            backend.patch_job(
                job_id,
                status="SUCCEEDED",
                error_message="",
                result_patch=_final_result_patch(existing_final),
            )
            mlflow.terminate_run(str(parent_run_id), status="FINISHED")
            return

        final_parameters = execution_config.effective_final_parameters
        katib_name: str | None = None

        if execution_config.automl_enabled:
            backend.patch_job(job_id, status="TUNING", error_message="")
            katib_manifest = build_experiment_manifest(
                namespace=settings.katib_namespace,
                job_id=job_id,
                parent_run_id=str(parent_run_id),
                mlflow_experiment_name=(
                    settings.tabular_random_forest.mlflow_experiment_name
                ),
                recipe_version=recipe_version,
                random_seed=execution_config.random_seed,
                max_trial_count=execution_config.max_trial_count,
                parallel_trial_count=execution_config.parallel_trial_count,
                algorithm_name=execution_config.algorithm_name,
                n_estimators_min=execution_config.n_estimators_min,
                n_estimators_max=execution_config.n_estimators_max,
                max_depth_min=execution_config.max_depth_min,
                max_depth_max=execution_config.max_depth_max,
                min_samples_split_min=(
                    execution_config.min_samples_split_min
                ),
                min_samples_split_max=(
                    execution_config.min_samples_split_max
                ),
            )
            katib_name = str(katib_manifest["metadata"]["name"])
            katib = KatibRunner(settings.katib_namespace)
            katib.ensure_experiment(katib_manifest)
            backend.patch_job(
                job_id,
                result_patch={
                    "external_ids": {
                        "katib_experiment_id": katib_name,
                    }
                },
            )

            katib_result = parse_experiment_result(
                katib.wait_for_success(katib_name)
            )
            final_parameters = {
                **katib_result.best_params,
                "max_features": execution_config.final_max_features,
                "random_seed": execution_config.random_seed,
            }
            backend.patch_job(
                job_id,
                status="TRAINING",
                result_patch={
                    "objective": {"value": katib_result.best_metric},
                    "best_params": dict(katib_result.best_params),
                },
            )
        else:
            backend.patch_job(job_id, status="TRAINING", error_message="")

        kfp_run_id = job.get("kfp_run_id")
        if not kfp_run_id:
            arguments: dict[str, Any] = {
                "platform_job_id": job_id,
                "recipe_id": TABULAR_RANDOM_FOREST_RECIPE_ID,
                "recipe_version": recipe_version,
                "random_seed": final_parameters["random_seed"],
                "n_estimators": final_parameters["n_estimators"],
                "max_depth": final_parameters["max_depth"],
                "min_samples_split": final_parameters[
                    "min_samples_split"
                ],
                "max_features": final_parameters["max_features"],
                "mlflow_parent_run_id": str(parent_run_id),
                "mlflow_experiment_name": (
                    settings.tabular_random_forest.mlflow_experiment_name
                ),
                "registered_model_name": (
                    settings.tabular_random_forest.registered_model_name
                ),
            }
            if katib_name is not None:
                arguments["katib_experiment_id"] = katib_name

            kfp_run_id = kfp.submit_pipeline(
                pipeline_path=(
                    settings.tabular_random_forest.pipeline_path
                ),
                run_name=f"tabular-rf-final-{job_id[:8]}",
                arguments=arguments,
            )
            backend.patch_job(
                job_id,
                result_patch={
                    "external_ids": {
                        "kfp_run_id": str(kfp_run_id),
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

        final_run = _wait_for_registered_final_run(
            mlflow,
            experiment_id=experiment_id,
            platform_job_id=job_id,
        )
        _attach_kfp_lineage(
            mlflow,
            final_run=final_run,
            kfp_run_id=str(kfp_run_id),
        )
        backend.patch_job(
            job_id,
            status="SUCCEEDED",
            error_message="",
            result_patch=_final_result_patch(final_run),
        )
        mlflow.terminate_run(str(parent_run_id), status="FINISHED")
        LOGGER.info("Tabular platform job %s succeeded", job_id)

    except Exception as exc:
        LOGGER.exception("Tabular platform job %s failed", job_id)
        try:
            backend.patch_job(
                job_id,
                status="FAILED",
                error_message=str(exc)[:2000],
            )
        finally:
            if parent_run_id:
                try:
                    mlflow.terminate_run(
                        str(parent_run_id),
                        status="FAILED",
                    )
                except Exception:
                    LOGGER.exception("Could not terminate parent MLflow run")
        raise
