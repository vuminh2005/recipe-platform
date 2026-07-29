from __future__ import annotations

from typing import Any

from agent.backend_client import BackendClient
from agent.kfp_runner import KfpRunner
from agent.settings import Settings


HELLO_RUN_TIMEOUT_SECONDS = 900


def execute_hello_job(
    job: dict[str, Any],
    *,
    settings: Settings,
    backend: BackendClient,
) -> None:
    job_id = str(job["id"])
    recipe = job.get("recipe") or {}
    recipient = str(recipe["name"])
    kfp = KfpRunner(settings.kfp_endpoint)

    kfp_run_id = job.get("kfp_run_id")
    if not kfp_run_id:
        kfp_run_id = kfp.submit_pipeline(
            pipeline_path=settings.hello.pipeline_path,
            run_name=f"job-{job_id[:8]}",
            arguments={"recipient": recipient},
            experiment_name=settings.hello.kfp_experiment_name,
        )

    backend.patch_job(
        job_id,
        status="RUNNING",
        result_patch={
            "external_ids": {
                "kfp_run_id": str(kfp_run_id),
            }
        },
    )

    status = kfp.wait_for_completion(
        str(kfp_run_id),
        timeout_seconds=HELLO_RUN_TIMEOUT_SECONDS,
        poll_seconds=5,
    )
    if not kfp.is_success(status):
        raise RuntimeError(
            f"Hello KFP run {kfp_run_id} did not succeed; status={status}"
        )

    backend.patch_job(
        job_id,
        status="SUCCEEDED",
        error_message="",
    )
