from __future__ import annotations

from pathlib import Path
from typing import Any

from kfp import Client


TERMINAL_SUCCESS = {"SUCCEEDED", "SUCCESS", "COMPLETED"}
TERMINAL_FAILURE = {
    "FAILED",
    "ERROR",
    "CANCELLED",
    "CANCELED",
    "SKIPPED",
}


class KfpRunner:
    def __init__(self, endpoint: str) -> None:
        self.client = Client(host=endpoint)

    @staticmethod
    def _extract_run_id(run: Any) -> str:
        for candidate in (
            getattr(run, "run_id", None),
            getattr(run, "id", None),
            getattr(getattr(run, "run", None), "id", None),
            getattr(getattr(run, "run", None), "run_id", None),
        ):
            if candidate:
                return str(candidate)
        if isinstance(run, dict):
            for key in ("run_id", "id"):
                if run.get(key):
                    return str(run[key])
        raise RuntimeError(f"Could not extract KFP run ID from {run!r}")

    @staticmethod
    def _extract_status(run: Any) -> str:
        candidates = [
            getattr(run, "state", None),
            getattr(run, "status", None),
            getattr(getattr(run, "run", None), "state", None),
            getattr(getattr(run, "run", None), "status", None),
        ]
        if isinstance(run, dict):
            candidates.extend(
                [
                    run.get("state"),
                    run.get("status"),
                    (run.get("run") or {}).get("state"),
                    (run.get("run") or {}).get("status"),
                ]
            )
        for value in candidates:
            if value:
                # Enum values may stringify as PipelineState.SUCCEEDED.
                return str(value).split(".")[-1].upper()
        return "UNKNOWN"

    def submit_final_pipeline(
        self,
        *,
        pipeline_path: Path,
        run_name: str,
        arguments: dict[str, Any],
    ) -> str:
        if not pipeline_path.is_file():
            raise FileNotFoundError(f"Compiled KFP pipeline not found: {pipeline_path}")
        run = self.client.create_run_from_pipeline_package(
            pipeline_file=str(pipeline_path),
            arguments=arguments,
            run_name=run_name,
        )
        return self._extract_run_id(run)

    def get_status(self, run_id: str) -> str:
        return self._extract_status(self.client.get_run(run_id=run_id))

    @staticmethod
    def is_success(status: str) -> bool:
        return status.upper() in TERMINAL_SUCCESS

    @staticmethod
    def is_failure(status: str) -> bool:
        return status.upper() in TERMINAL_FAILURE
