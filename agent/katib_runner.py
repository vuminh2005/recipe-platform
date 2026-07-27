from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

KATIB_GROUP = "kubeflow.org"
KATIB_VERSION = "v1beta1"
KATIB_PLURAL = "experiments"


@dataclass(frozen=True)
class KatibResult:
    experiment_name: str
    best_trial_name: str
    best_params: dict[str, float]
    best_metric: float
    metrics: dict[str, float]


class KatibRunner:
    def __init__(self, namespace: str) -> None:
        config.load_kube_config()
        self.namespace = namespace
        self.api = client.CustomObjectsApi()

    @staticmethod
    def experiment_name(job_id: str) -> str:
        return f"cats-dogs-hpo-{job_id[:8]}".lower()

    def build_manifest(
        self,
        *,
        job_id: str,
        parent_run_id: str,
        trial_epochs: int,
        batch_size: int,
        dense_units: int,
        image_size: int,
        max_trial_count: int = 3,
    ) -> dict[str, Any]:
        name = self.experiment_name(job_id)
        return {
            "apiVersion": "kubeflow.org/v1beta1",
            "kind": "Experiment",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": {
                "maxTrialCount": max_trial_count,
                "parallelTrialCount": 1,
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
                "algorithm": {"algorithmName": "random"},
                "parameters": [
                    {
                        "name": "learning_rate",
                        "parameterType": "double",
                        "feasibleSpace": {
                            "min": "0.00005",
                            "max": "0.0005",
                        },
                    },
                    {
                        "name": "dropout_rate",
                        "parameterType": "double",
                        "feasibleSpace": {"min": "0.15", "max": "0.45"},
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
                                            "image": (
                                                "docker.io/library/"
                                                "cats-dogs-trainer:0.3"
                                            ),
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
                                                    "name": (
                                                        "KATIB_EXPERIMENT_NAME"
                                                    ),
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

    def get_experiment(self, name: str) -> dict[str, Any] | None:
        try:
            return self.api.get_namespaced_custom_object(
                group=KATIB_GROUP,
                version=KATIB_VERSION,
                namespace=self.namespace,
                plural=KATIB_PLURAL,
                name=name,
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise

    def ensure_experiment(self, manifest: dict[str, Any]) -> dict[str, Any]:
        name = str(manifest["metadata"]["name"])
        existing = self.get_experiment(name)
        if existing is not None:
            return existing
        return self.api.create_namespaced_custom_object(
            group=KATIB_GROUP,
            version=KATIB_VERSION,
            namespace=self.namespace,
            plural=KATIB_PLURAL,
            body=manifest,
        )

    @staticmethod
    def _condition_status(
        experiment: dict[str, Any], condition_type: str
    ) -> tuple[bool, str]:
        conditions = experiment.get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition.get("type") == condition_type:
                return condition.get("status") == "True", str(
                    condition.get("message", "")
                )
        return False, ""

    @staticmethod
    def _parse_result(experiment: dict[str, Any]) -> KatibResult:
        name = str(experiment["metadata"]["name"])
        optimal = experiment.get("status", {}).get("currentOptimalTrial") or {}
        assignments = optimal.get("parameterAssignments") or []
        best_params = {
            str(item["name"]): float(item["value"]) for item in assignments
        }
        metrics_items = optimal.get("observation", {}).get("metrics", [])
        metrics: dict[str, float] = {}
        for item in metrics_items:
            value = item.get("max") or item.get("latest") or item.get("min")
            if value is not None:
                metrics[str(item["name"])] = float(value)

        if "learning_rate" not in best_params or "dropout_rate" not in best_params:
            raise RuntimeError(
                f"Katib experiment {name} succeeded without expected best params: "
                f"{best_params}"
            )
        if "val_auc" not in metrics:
            raise RuntimeError(
                f"Katib experiment {name} succeeded without val_auc: {metrics}"
            )

        return KatibResult(
            experiment_name=name,
            best_trial_name=str(optimal.get("bestTrialName", "")),
            best_params=best_params,
            best_metric=metrics["val_auc"],
            metrics=metrics,
        )

    def wait_for_success(
        self,
        name: str,
        *,
        timeout_seconds: int = 3600,
        poll_seconds: int = 5,
    ) -> KatibResult:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            experiment = self.get_experiment(name)
            if experiment is None:
                raise RuntimeError(f"Katib experiment disappeared: {name}")

            failed, failure_message = self._condition_status(experiment, "Failed")
            if failed:
                raise RuntimeError(
                    f"Katib experiment {name} failed: {failure_message}"
                )

            succeeded, _ = self._condition_status(experiment, "Succeeded")
            if succeeded:
                return self._parse_result(experiment)

            time.sleep(poll_seconds)

        raise TimeoutError(f"Timed out waiting for Katib experiment {name}")
