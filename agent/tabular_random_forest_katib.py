"""Tabular Random Forest Katib manifest construction and result parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.tabular_random_forest_config import SUPPORTED_ALGORITHMS


TRAINER_IMAGE = (
    "docker.io/library/tabular-random-forest-trainer:1.0"
)
PLATFORM_SECRET = "cats-dogs-platform-secrets"
OBJECTIVE_METRIC = "val_f1"
ADDITIONAL_METRICS = (
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_roc_auc",
)


@dataclass(frozen=True)
class TabularKatibResult:
    experiment_name: str
    best_trial_name: str
    best_params: dict[str, int]
    best_metric: float
    metrics: dict[str, float]


def experiment_name(job_id: str) -> str:
    return f"tabular-rf-hpo-{job_id[:8]}".lower()


def _secret_env(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {
            "secretKeyRef": {
                "name": PLATFORM_SECRET,
                "key": name,
            }
        },
    }


def build_experiment_manifest(
    *,
    namespace: str,
    job_id: str,
    parent_run_id: str,
    mlflow_experiment_name: str,
    recipe_version: str,
    random_seed: int,
    max_trial_count: int,
    parallel_trial_count: int,
    algorithm_name: str,
    n_estimators_min: int,
    n_estimators_max: int,
    max_depth_min: int,
    max_depth_max: int,
    min_samples_split_min: int,
    min_samples_split_max: int,
) -> dict[str, Any]:
    if max_trial_count < 1:
        raise ValueError("max_trial_count must be at least 1")
    if parallel_trial_count < 1:
        raise ValueError("parallel_trial_count must be at least 1")
    if parallel_trial_count > max_trial_count:
        raise ValueError(
            "parallel_trial_count must be less than or equal to max_trial_count"
        )
    if algorithm_name not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(
            f"Unsupported Katib algorithm {algorithm_name!r}; supported: "
            f"{supported}"
        )

    ranges = {
        "n_estimators": (n_estimators_min, n_estimators_max, 1),
        "max_depth": (max_depth_min, max_depth_max, 1),
        "min_samples_split": (
            min_samples_split_min,
            min_samples_split_max,
            2,
        ),
    }
    for parameter, (minimum, maximum, lower_bound) in ranges.items():
        if minimum < lower_bound or minimum >= maximum:
            raise ValueError(
                f"{parameter} must satisfy {lower_bound} <= min < max"
            )

    name = experiment_name(job_id)
    parameter_specs = [
        {
            "name": parameter,
            "parameterType": "int",
            "feasibleSpace": {
                "min": str(minimum),
                "max": str(maximum),
            },
        }
        for parameter, (minimum, maximum, _) in ranges.items()
    ]
    metric_strategies = [
        {"name": OBJECTIVE_METRIC, "value": "max"},
        *[
            {"name": metric_name, "value": "max"}
            for metric_name in ADDITIONAL_METRICS
        ],
    ]
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "maxTrialCount": max_trial_count,
            "parallelTrialCount": parallel_trial_count,
            "maxFailedTrialCount": 1,
            "objective": {
                "type": "maximize",
                "objectiveMetricName": OBJECTIVE_METRIC,
                "additionalMetricNames": list(ADDITIONAL_METRICS),
                "metricStrategies": metric_strategies,
            },
            "algorithm": {"algorithmName": algorithm_name},
            "parameters": parameter_specs,
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "trialTemplate": {
                "retain": True,
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "nEstimators",
                        "reference": "n_estimators",
                    },
                    {
                        "name": "maxDepth",
                        "reference": "max_depth",
                    },
                    {
                        "name": "minSamplesSplit",
                        "reference": "min_samples_split",
                    },
                    {
                        "name": "trialName",
                        "reference": "${trialSpec.Name}",
                    },
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "backoffLimit": 0,
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "tabular-random-forest-trainer",
                                    "platform.run-role": "katib-trial",
                                }
                            },
                            "spec": {
                                "restartPolicy": "Never",
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": TRAINER_IMAGE,
                                        "imagePullPolicy": "Never",
                                        "command": [
                                            "python",
                                            "-m",
                                            "trainer.train",
                                        ],
                                        "args": [
                                            "--mode=trial",
                                            (
                                                "--n-estimators="
                                                "${trialParameters.nEstimators}"
                                            ),
                                            (
                                                "--max-depth="
                                                "${trialParameters.maxDepth}"
                                            ),
                                            (
                                                "--min-samples-split="
                                                "${trialParameters.minSamplesSplit}"
                                            ),
                                            "--max-features=sqrt",
                                            f"--random-seed={random_seed}",
                                            (
                                                "--mlflow-experiment-name="
                                                f"{mlflow_experiment_name}"
                                            ),
                                            f"--recipe-version={recipe_version}",
                                        ],
                                        "env": [
                                            _secret_env("MLFLOW_TRACKING_URI"),
                                            _secret_env(
                                                "MLFLOW_S3_ENDPOINT_URL"
                                            ),
                                            _secret_env("AWS_ACCESS_KEY_ID"),
                                            _secret_env(
                                                "AWS_SECRET_ACCESS_KEY"
                                            ),
                                            _secret_env("AWS_DEFAULT_REGION"),
                                            {
                                                "name": "PLATFORM_JOB_ID",
                                                "value": job_id,
                                            },
                                            {
                                                "name": "MLFLOW_PARENT_RUN_ID",
                                                "value": parent_run_id,
                                            },
                                            {
                                                "name": (
                                                    "KATIB_EXPERIMENT_NAME"
                                                ),
                                                "value": name,
                                            },
                                            {
                                                "name": "KATIB_TRIAL_NAME",
                                                "value": (
                                                    "${trialParameters."
                                                    "trialName}"
                                                ),
                                            },
                                        ],
                                        "resources": {
                                            "requests": {
                                                "cpu": "250m",
                                                "memory": "512Mi",
                                            },
                                            "limits": {
                                                "cpu": "2",
                                                "memory": "2Gi",
                                            },
                                        },
                                    }
                                ],
                            },
                        },
                    },
                },
            },
        },
    }


def parse_experiment_result(
    experiment: dict[str, Any],
) -> TabularKatibResult:
    name = str(experiment["metadata"]["name"])
    optimal = experiment.get("status", {}).get("currentOptimalTrial") or {}
    assignments = optimal.get("parameterAssignments") or []
    converters = {
        "n_estimators": int,
        "max_depth": int,
        "min_samples_split": int,
    }
    best_params: dict[str, int] = {}
    for item in assignments:
        parameter_name = str(item["name"])
        converter = converters.get(parameter_name)
        if converter is None:
            raise RuntimeError(
                f"Katib experiment {name} returned unexpected parameter "
                f"{parameter_name!r}"
            )
        try:
            best_params[parameter_name] = converter(str(item["value"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Katib experiment {name} returned invalid integer value for "
                f"{parameter_name}: {item.get('value')!r}"
            ) from exc

    missing_params = sorted(set(converters) - set(best_params))
    if missing_params:
        raise RuntimeError(
            f"Katib experiment {name} succeeded without expected best params: "
            f"{missing_params}"
        )

    metrics: dict[str, float] = {}
    for item in optimal.get("observation", {}).get("metrics", []):
        value = None
        for field_name in ("max", "latest", "min"):
            candidate = item.get(field_name)
            if candidate is not None:
                value = candidate
                break
        if value is not None:
            metrics[str(item["name"])] = float(value)
    if OBJECTIVE_METRIC not in metrics:
        raise RuntimeError(
            f"Katib experiment {name} succeeded without {OBJECTIVE_METRIC}: "
            f"{metrics}"
        )

    return TabularKatibResult(
        experiment_name=name,
        best_trial_name=str(optimal.get("bestTrialName", "")),
        best_params=best_params,
        best_metric=metrics[OBJECTIVE_METRIC],
        metrics=metrics,
    )
