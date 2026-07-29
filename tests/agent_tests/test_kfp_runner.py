from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.kfp_runner import KfpRunner


class KfpRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = KfpRunner.__new__(KfpRunner)
        self.runner.client = Mock()

    def test_submit_pipeline_forwards_only_generic_inputs(self) -> None:
        self.runner.client.create_run_from_pipeline_package.return_value = (
            SimpleNamespace(run_id="run-1")
        )
        with TemporaryDirectory() as directory:
            package = Path(directory) / "pipeline.yaml"
            package.touch()

            run_id = self.runner.submit_pipeline(
                pipeline_path=package,
                run_name="smoke-run",
                arguments={"recipient": "Recipe Platform"},
                experiment_name="smoke-experiment",
            )

        self.assertEqual(run_id, "run-1")
        self.runner.client.create_run_from_pipeline_package.assert_called_once_with(
            pipeline_file=str(package),
            arguments={"recipient": "Recipe Platform"},
            run_name="smoke-run",
            experiment_name="smoke-experiment",
        )

    @patch("agent.kfp_runner.time.sleep")
    def test_wait_for_completion_returns_actual_terminal_status(
        self,
        sleep,
    ) -> None:
        self.runner.get_status = Mock(side_effect=["RUNNING", "SUCCEEDED"])

        status = self.runner.wait_for_completion(
            "run-1",
            timeout_seconds=30,
            poll_seconds=1,
        )

        self.assertEqual(status, "SUCCEEDED")
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
