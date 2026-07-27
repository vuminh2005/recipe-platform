from __future__ import annotations

import logging
import time
from typing import Any

from agent.backend_client import BackendClient
from agent.katib_runner import KatibRunner
from agent.kfp_runner import KfpRunner
from agent.mlflow_rest import MlflowRestClient
from agent.settings import Settings

LOGGER = logging.getLogger("cats_dogs_executor")


def _recipe_sections(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = job.get("recipe") or {}
    training = recipe.get("training") or recipe.get("training_config") or {}
    automl = recipe.get("automl") or recipe.get("automl_config") or {}
    return training, automl


def _final_fields(final_run: dict[str, Any]) -> dict[str, Any]:
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
        "mlflow_final_run_id": str(final_run["run_id"]),
        "model_uri": str(tags["platform.model_uri"]),
        "registered_model_name": str(tags["platform.registered_model_name"]),
        "registered_model_version": str(
            tags["platform.registered_model_version"]
        ),
        "final_metrics": final_metrics,
    }


def execute_cats_dogs_job(
    job: dict[str, Any],
    *,
    settings: Settings,
    backend: BackendClient,
) -> None:
    job_id = str(job["id"])
    training, automl = _recipe_sections(job)

    trial_epochs = int(training.get("trial_epochs", 2))
    final_epochs = int(training.get("final_epochs", 5))
    batch_size = int(training.get("batch_size", 8))
    dense_units = int(training.get("dense_units", 128))
    image_size = int(training.get("image_size", 224))
    trainable_backbone = bool(training.get("trainable_backbone", False))
    max_trial_count = int(automl.get("max_trials", 3))

    mlflow = MlflowRestClient(settings.mlflow_tracking_uri)
    katib = KatibRunner(settings.katib_namespace)
    kfp = KfpRunner(settings.kfp_endpoint)

    parent_run_id = job.get("mlflow_parent_run_id")

    try:
        # 1-2. Wake Render, resolve experiment and create/reuse parent run.
        mlflow.warm()
        experiment_id = mlflow.get_or_create_experiment(
            settings.mlflow_experiment_name
        )
        if not parent_run_id:
            parent_run_id = mlflow.ensure_parent_run(
                experiment_id=experiment_id,
                platform_job_id=job_id,
                agent_id=settings.agent_id,
            )
            backend.patch_job(job_id, mlflow_parent_run_id=parent_run_id)

        # A retry can finish immediately if the final registered run already exists.
        existing_final = mlflow.find_final_run(
            experiment_id=experiment_id,
            platform_job_id=job_id,
        )
        if (
            existing_final
            and existing_final["tags"].get("platform.result")
            == "final_model_registered"
        ):
            backend.patch_job(
                job_id,
                status="SUCCEEDED",
                error_message="",
                **_final_fields(existing_final),
            )
            mlflow.terminate_run(str(parent_run_id), status="FINISHED")
            return

        # 3-6. Create/reuse deterministic Katib Experiment and wait for it.
        backend.patch_job(job_id, status="TUNING", error_message="")
        katib_manifest = katib.build_manifest(
            job_id=job_id,
            parent_run_id=str(parent_run_id),
            trial_epochs=trial_epochs,
            batch_size=batch_size,
            dense_units=dense_units,
            image_size=image_size,
            max_trial_count=max_trial_count,
        )
        katib_name = str(katib_manifest["metadata"]["name"])
        katib.ensure_experiment(katib_manifest)
        backend.patch_job(job_id, katib_experiment_name=katib_name)

        katib_result = katib.wait_for_success(katib_name)
        best_params = {
            "learning_rate": katib_result.best_params["learning_rate"],
            "dropout_rate": katib_result.best_params["dropout_rate"],
            "best_trial_name": katib_result.best_trial_name,
            "katib_metrics": katib_result.metrics,
        }
        backend.patch_job(
            job_id,
            status="TRAINING",
            best_params=best_params,
            best_metric=katib_result.best_metric,
        )

        # 7-10. Submit or resume the compiled final KFP run.
        kfp_run_id = job.get("kfp_run_id")
        if not kfp_run_id:
            kfp_run_id = kfp.submit_final_pipeline(
                pipeline_path=settings.cats_dogs_pipeline_path,
                run_name=f"cats-dogs-final-{job_id[:8]}",
                arguments={
                    "platform_job_id": job_id,
                    "mlflow_parent_run_id": str(parent_run_id),
                    "katib_experiment_name": katib_name,
                    "learning_rate": katib_result.best_params[
                        "learning_rate"
                    ],
                    "dropout_rate": katib_result.best_params[
                        "dropout_rate"
                    ],
                    "dense_units": dense_units,
                    "batch_size": batch_size,
                    "final_epochs": final_epochs,
                    "image_size": image_size,
                    "trainable_backbone": trainable_backbone,
                },
            )
            backend.patch_job(job_id, kfp_run_id=kfp_run_id)

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
            final_run = mlflow.find_final_run(
                experiment_id=experiment_id,
                platform_job_id=job_id,
            )
            if (
                not registering_reported
                and final_run
                and final_run["tags"].get("platform.result")
                == "final_model_logged"
            ):
                backend.patch_job(job_id, status="REGISTERING")
                registering_reported = True

            time.sleep(5)

        if not registering_reported:
            backend.patch_job(job_id, status="REGISTERING")

        # 11-15. Resolve the registered final run and persist the complete result.
        final_run = mlflow.wait_for_registered_final_run(
            experiment_id=experiment_id,
            platform_job_id=job_id,
            timeout_seconds=300,
        )
        backend.patch_job(
            job_id,
            status="SUCCEEDED",
            error_message="",
            **_final_fields(final_run),
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
