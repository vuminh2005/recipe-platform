import os
import time
import traceback
from pathlib import Path
from typing import Any

import requests
from kfp import Client


BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")
AGENT_ID = os.getenv("AGENT_ID", "local-agent")
AGENT_TOKEN = os.environ["AGENT_TOKEN"]

KFP_ENDPOINT = os.getenv(
    "KFP_ENDPOINT",
    "http://127.0.0.1:8080",
)

POLL_INTERVAL_SECONDS = int(
    os.getenv("POLL_INTERVAL_SECONDS", "5")
)

HELLO_PIPELINE_PATH = os.getenv(
    "HELLO_PIPELINE_PATH",
    "pipelines/compiled/hello_pipeline.yaml",
)

HEADERS = {
    "X-Agent-Token": AGENT_TOKEN,
    "Content-Type": "application/json",
}


def claim_job() -> dict[str, Any] | None:
    response = requests.post(
        f"{BACKEND_URL}/api/agent/jobs/claim",
        headers=HEADERS,
        json={"agent_id": AGENT_ID},
        timeout=30,
    )

    if response.status_code == 204:
        return None

    response.raise_for_status()
    return response.json()


def update_job(
    job_id: str,
    *,
    status: str,
    kfp_run_id: str | None = None,
    error_message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
    }

    if kfp_run_id is not None:
        payload["kfp_run_id"] = kfp_run_id

    if error_message is not None:
        payload["error_message"] = error_message[:2000]

    response = requests.patch(
        f"{BACKEND_URL}/api/agent/jobs/{job_id}",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()


def execute_hello_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    recipe = job["recipe"]

    pipeline_path = Path(HELLO_PIPELINE_PATH)

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline package not found: {pipeline_path.resolve()}"
        )

    client = Client(host=KFP_ENDPOINT)

    run = client.create_run_from_pipeline_package(
        pipeline_file=str(pipeline_path),
        arguments={
            "recipient": recipe["name"],
        },
        run_name=f"job-{job_id[:8]}",
        experiment_name="recipe-platform-jobs",
    )

    update_job(
        job_id,
        status="RUNNING",
        kfp_run_id=run.run_id,
    )

    completed_run = client.wait_for_run_completion(
        run_id=run.run_id,
        timeout=900,
        sleep_duration=5,
    )

    state = str(
        getattr(completed_run, "state", "UNKNOWN")
    ).upper()

    if "SUCCEEDED" not in state:
        raise RuntimeError(
            f"KFP run did not succeed. State={state}"
        )

    update_job(
        job_id,
        status="SUCCEEDED",
        kfp_run_id=run.run_id,
    )


def execute_job(job: dict[str, Any]) -> None:
    workload = job["recipe"]["workload"]

    if workload == "hello":
        execute_hello_job(job)
        return

    raise RuntimeError(
        f"Unsupported workload: {workload}"
    )


def main() -> None:
    print(f"Agent ID: {AGENT_ID}")
    print(f"Backend: {BACKEND_URL}")
    print(f"KFP: {KFP_ENDPOINT}")
    print(f"Pipeline: {Path(HELLO_PIPELINE_PATH).resolve()}")

    while True:
        try:
            job = claim_job()

            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            print(f"Claimed job: {job['id']}")

            try:
                execute_job(job)
                print(f"Completed job: {job['id']}")

            except Exception as exc:
                traceback.print_exc()

                try:
                    update_job(
                        job["id"],
                        status="FAILED",
                        error_message=str(exc),
                    )
                except Exception:
                    print(
                        "Could not report FAILED status "
                        "to backend."
                    )
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("Agent stopped.")
            break

        except Exception:
            traceback.print_exc()
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
