from __future__ import annotations

import time
from typing import Any

import requests


class MlflowRestClient:
    """Small MLflow 2.x REST client; the Agent does not import MLflow SDK."""

    def __init__(self, tracking_uri: str) -> None:
        self.base_url = tracking_uri.rstrip("/")
        self.session = requests.Session()

    def warm(self, attempts: int = 12, sleep_seconds: int = 10) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = self.session.get(
                    f"{self.base_url}/health",
                    timeout=(10, 180),
                )
                if response.status_code < 500:
                    return
                last_error = RuntimeError(
                    f"MLflow health returned HTTP {response.status_code}"
                )
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(sleep_seconds)
        raise RuntimeError("MLflow did not become ready") from last_error

    def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=(10, 180),
        )
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=(10, 180),
        )
        response.raise_for_status()
        return response.json()

    def get_or_create_experiment(self, experiment_name: str) -> str:
        data = self._get(
            "/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
        )
        experiment = data.get("experiment")
        if experiment:
            return str(experiment["experiment_id"])

        created = self._post(
            "/api/2.0/mlflow/experiments/create",
            {"name": experiment_name},
        )
        return str(created["experiment_id"])

    @staticmethod
    def _escape_filter_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def search_runs(
        self,
        *,
        experiment_id: str,
        platform_job_id: str,
        run_role: str,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        job_value = self._escape_filter_value(platform_job_id)
        role_value = self._escape_filter_value(run_role)
        filter_string = (
            f"tags.`platform.job_id` = '{job_value}' and "
            f"tags.`platform.run_role` = '{role_value}'"
        )
        payload = {
            "experiment_ids": [experiment_id],
            "filter": filter_string,
            "max_results": max_results,
            "order_by": ["attributes.start_time DESC"],
        }
        return self._post("/api/2.0/mlflow/runs/search", payload).get(
            "runs", []
        )

    def ensure_parent_run(
        self,
        *,
        experiment_id: str,
        platform_job_id: str,
        agent_id: str,
    ) -> str:
        existing = self.search_runs(
            experiment_id=experiment_id,
            platform_job_id=platform_job_id,
            run_role="platform_parent",
            max_results=1,
        )
        if existing:
            return str(existing[0]["info"]["run_id"])

        now_ms = int(time.time() * 1000)
        tags = [
            {"key": "mlflow.runName", "value": f"platform-job-{platform_job_id[:12]}"},
            {"key": "platform.job_id", "value": platform_job_id},
            {"key": "platform.run_role", "value": "platform_parent"},
            {"key": "platform.agent_id", "value": agent_id},
        ]
        data = self._post(
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": experiment_id,
                "start_time": now_ms,
                "tags": tags,
            },
        )
        return str(data["run"]["info"]["run_id"])

    def terminate_run(self, run_id: str, *, status: str) -> None:
        self._post(
            "/api/2.0/mlflow/runs/update",
            {
                "run_id": run_id,
                "status": status,
                "end_time": int(time.time() * 1000),
            },
        )

    @staticmethod
    def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
        info = run.get("info", {})
        data = run.get("data", {})
        tags = {item["key"]: item.get("value") for item in data.get("tags", [])}
        params = {
            item["key"]: item.get("value") for item in data.get("params", [])
        }
        metrics = {
            item["key"]: item.get("value") for item in data.get("metrics", [])
        }
        return {
            "run_id": info.get("run_id"),
            "status": info.get("status"),
            "tags": tags,
            "params": params,
            "metrics": metrics,
        }

    def find_final_run(
        self,
        *,
        experiment_id: str,
        platform_job_id: str,
    ) -> dict[str, Any] | None:
        runs = self.search_runs(
            experiment_id=experiment_id,
            platform_job_id=platform_job_id,
            run_role="final_training",
            max_results=5,
        )
        if not runs:
            return None
        return self.normalize_run(runs[0])

    def wait_for_registered_final_run(
        self,
        *,
        experiment_id: str,
        platform_job_id: str,
        timeout_seconds: int = 300,
        poll_seconds: int = 5,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_run: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last_run = self.find_final_run(
                experiment_id=experiment_id,
                platform_job_id=platform_job_id,
            )
            if last_run:
                tags = last_run["tags"]
                if (
                    tags.get("platform.result") == "final_model_registered"
                    and tags.get("platform.registered_model_version")
                ):
                    return last_run
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Final registered MLflow run not found for job {platform_job_id}; "
            f"last_run={last_run}"
        )
