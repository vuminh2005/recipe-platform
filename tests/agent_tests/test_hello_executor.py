from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from agent.hello_executor import (
    HELLO_RUN_TIMEOUT_SECONDS,
    execute_hello_job,
)


class HelloExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            kfp_endpoint="http://kfp",
            hello=SimpleNamespace(
                pipeline_path=Path("/tmp/hello-pipeline.yaml"),
                kfp_experiment_name="recipe-platform-jobs",
            ),
        )
        self.backend = Mock()
        self.job = {
            "id": "12345678-abcd",
            "recipe": {
                "name": "Hello Smoke",
                "recipe_id": "hello",
            },
        }

    @patch("agent.mlflow_rest.MlflowRestClient")
    @patch("agent.katib_runner.KatibRunner")
    @patch("agent.hello_executor.KfpRunner")
    def test_submits_existing_pipeline_without_katib_or_mlflow(
        self,
        kfp_class,
        katib_class,
        mlflow_class,
    ) -> None:
        kfp = kfp_class.return_value
        kfp.submit_pipeline.return_value = "kfp-run-1"
        kfp.wait_for_completion.return_value = "SUCCEEDED"
        kfp.is_success.return_value = True

        execute_hello_job(
            self.job,
            settings=self.settings,
            backend=self.backend,
        )

        kfp.submit_pipeline.assert_called_once_with(
            pipeline_path=Path("/tmp/hello-pipeline.yaml"),
            run_name="job-12345678",
            arguments={"recipient": "Hello Smoke"},
            experiment_name="recipe-platform-jobs",
        )
        kfp.wait_for_completion.assert_called_once_with(
            "kfp-run-1",
            timeout_seconds=HELLO_RUN_TIMEOUT_SECONDS,
            poll_seconds=5,
        )
        self.backend.patch_job.assert_has_calls(
            [
                call(
                    "12345678-abcd",
                    status="RUNNING",
                    result_patch={
                        "external_ids": {
                            "kfp_run_id": "kfp-run-1",
                        }
                    },
                ),
                call(
                    "12345678-abcd",
                    status="SUCCEEDED",
                    error_message="",
                ),
            ]
        )
        katib_class.assert_not_called()
        mlflow_class.assert_not_called()

    @patch("agent.hello_executor.KfpRunner")
    def test_resumes_existing_kfp_run_without_resubmitting(
        self,
        kfp_class,
    ) -> None:
        self.job["kfp_run_id"] = "existing-run"
        kfp = kfp_class.return_value
        kfp.wait_for_completion.return_value = "SUCCEEDED"
        kfp.is_success.return_value = True

        execute_hello_job(
            self.job,
            settings=self.settings,
            backend=self.backend,
        )

        kfp.submit_pipeline.assert_not_called()
        kfp.wait_for_completion.assert_called_once_with(
            "existing-run",
            timeout_seconds=HELLO_RUN_TIMEOUT_SECONDS,
            poll_seconds=5,
        )

    @patch("agent.hello_executor.KfpRunner")
    def test_actual_kfp_failure_is_not_reported_as_success(
        self,
        kfp_class,
    ) -> None:
        kfp = kfp_class.return_value
        kfp.submit_pipeline.return_value = "failed-run"
        kfp.wait_for_completion.return_value = "FAILED"
        kfp.is_success.return_value = False

        with self.assertRaisesRegex(
            RuntimeError,
            "status=FAILED",
        ):
            execute_hello_job(
                self.job,
                settings=self.settings,
                backend=self.backend,
            )

        statuses = [
            item.kwargs.get("status")
            for item in self.backend.patch_job.call_args_list
        ]
        self.assertEqual(statuses, ["RUNNING"])


if __name__ == "__main__":
    unittest.main()
