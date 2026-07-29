from __future__ import annotations

import time
from typing import Any

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

KATIB_GROUP = "kubeflow.org"
KATIB_VERSION = "v1beta1"
KATIB_PLURAL = "experiments"


class KatibRunner:
    def __init__(self, namespace: str) -> None:
        config.load_kube_config()
        self.namespace = namespace
        self.api = client.CustomObjectsApi()

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

    def wait_for_success(
        self,
        name: str,
        *,
        timeout_seconds: int = 3600,
        poll_seconds: int = 5,
    ) -> dict[str, Any]:
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
                return experiment

            time.sleep(poll_seconds)

        raise TimeoutError(f"Timed out waiting for Katib experiment {name}")
