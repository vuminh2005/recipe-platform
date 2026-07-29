from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


SUPPORTED_ALGORITHMS = {"random"}
TRAINER_IMAGE = "docker.io/library/cats-dogs-trainer:0.7"


@dataclass(frozen=True)
class CatsDogsKatibResult:
    experiment_name: str
    best_trial_name: str
    best_params: dict[str, float]
    best_metric: float
    metrics: dict[str, float]


def experiment_name(job_id: str) -> str:
    return f"cats-dogs-hpo-{job_id[:8]}".lower()


def build_experiment_manifest(
    *,
    namespace: str,
    job_id: str,
    parent_run_id: str,
    trial_epochs: int,
    batch_size: int,
    dense_units: int,
    image_size: int,
    max_trial_count: int = 3,
    parallel_trial_count: int = 1,
    algorithm_name: str = "random",
    learning_rate_min: float = 0.00005,
    learning_rate_max: float = 0.0005,
    dropout_rate_min: float = 0.15,
    dropout_rate_max: float = 0.45,
    trainable_backbone: bool = False,
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
            f"Unsupported Katib algorithm {algorithm_name!r}; supported: {supported}"
        )
    if learning_rate_min <= 0 or learning_rate_min >= learning_rate_max:
        raise ValueError(
            "learning_rate_min must be positive and less than learning_rate_max"
        )
    if not 0 <= dropout_rate_min < dropout_rate_max < 1:
        raise ValueError("dropout rates must satisfy 0 <= min < max < 1")

    def format_number(value: float) -> str:
        return format(Decimal(str(value)), "f")

    name = experiment_name(job_id)
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
                "objectiveMetricName": "val_auc",
                "additionalMetricNames": ["val_accuracy", "val_loss"],
                "metricStrategies": [
                    {"name": "val_auc", "value": "max"},
                    {"name": "val_accuracy", "value": "max"},
                    {"name": "val_loss", "value": "min"},
                ],
            },
            "algorithm": {"algorithmName": algorithm_name},
            "parameters": [
                {
                    "name": "learning_rate",
                    "parameterType": "double",
                    "feasibleSpace": {
                        "min": format_number(learning_rate_min),
                        "max": format_number(learning_rate_max),
                    },
                },
                {
                    "name": "dropout_rate",
                    "parameterType": "double",
                    "feasibleSpace": {
                        "min": format_number(dropout_rate_min),
                        "max": format_number(dropout_rate_max),
                    },
                },
            ],
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "trialTemplate": {
                "retain": True,
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": "learningRate",
                        "reference": "learning_rate",
                    },
                    {
                        "name": "dropoutRate",
                        "reference": "dropout_rate",
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
                                    "app": "cats-dogs-trainer",
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
                                                "--learning-rate="
                                                "${trialParameters.learningRate}"
                                            ),
                                            (
                                                "--dropout-rate="
                                                "${trialParameters.dropoutRate}"
                                            ),
                                            f"--dense-units={dense_units}",
                                            f"--batch-size={batch_size}",
                                            f"--epochs={trial_epochs}",
                                            f"--image-size={image_size}",
                                            (
                                                "--trainable-backbone="
                                                f"{str(trainable_backbone).lower()}"
                                            ),
                                        ],
                                        "envFrom": [
                                            {
                                                "secretRef": {
                                                    "name": (
                                                        "cats-dogs-"
                                                        "platform-secrets"
                                                    )
                                                }
                                            }
                                        ],
                                        "env": [
                                            {
                                                "name": "PLATFORM_JOB_ID",
                                                "value": job_id,
                                            },
                                            {
                                                "name": "MLFLOW_PARENT_RUN_ID",
                                                "value": parent_run_id,
                                            },
                                            {
                                                "name": "KATIB_EXPERIMENT_NAME",
                                                "value": name,
                                            },
                                            {
                                                "name": "KATIB_TRIAL_NAME",
                                                "value": (
                                                    "${trialParameters.trialName}"
                                                ),
                                            },
                                        ],
                                        "resources": {
                                            "requests": {
                                                "cpu": "2",
                                                "memory": "3Gi",
                                            },
                                            "limits": {
                                                "cpu": "6",
                                                "memory": "6Gi",
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
) -> CatsDogsKatibResult:
    name = str(experiment["metadata"]["name"])
    optimal = experiment.get("status", {}).get("currentOptimalTrial") or {}
    assignments = optimal.get("parameterAssignments") or []
    converters = {
        "learning_rate": float,
        "dropout_rate": float,
    }
    best_params: dict[str, float] = {}
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
                f"Katib experiment {name} returned invalid floating-point "
                f"value for {parameter_name}: {item.get('value')!r}"
            ) from exc

    missing_params = sorted(set(converters) - set(best_params))
    if missing_params:
        raise RuntimeError(
            f"Katib experiment {name} succeeded without expected best params: "
            f"{missing_params}"
        )

    metrics_items = optimal.get("observation", {}).get("metrics", [])
    metrics: dict[str, float] = {}
    for item in metrics_items:
        value = None
        for field_name in ("max", "latest", "min"):
            candidate = item.get(field_name)
            if candidate is not None:
                value = candidate
                break
        if value is not None:
            metrics[str(item["name"])] = float(value)

    if "val_auc" not in metrics:
        raise RuntimeError(
            f"Katib experiment {name} succeeded without val_auc: {metrics}"
        )

    return CatsDogsKatibResult(
        experiment_name=name,
        best_trial_name=str(optimal.get("bestTrialName", "")),
        best_params=best_params,
        best_metric=metrics["val_auc"],
        metrics=metrics,
    )
