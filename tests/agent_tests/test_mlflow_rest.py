from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock

import agent.mlflow_rest as mlflow_module
from agent.mlflow_rest import MlflowRestClient


class GenericMlflowRestTests(unittest.TestCase):
    def test_shared_client_contains_no_cats_dogs_completion_contract(self) -> None:
        source = inspect.getsource(mlflow_module)
        for forbidden in (
            "final_training",
            "final_model_logged",
            "final_model_registered",
            "final_threshold",
            "val_auc",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_find_latest_run_accepts_the_recipe_owned_role(self) -> None:
        client = MlflowRestClient.__new__(MlflowRestClient)
        run = {
            "info": {"run_id": "run-1", "status": "FINISHED"},
            "data": {"tags": [], "params": [], "metrics": []},
        }
        client.search_runs = Mock(return_value=[run])

        result = client.find_latest_run(
            experiment_id="experiment-1",
            platform_job_id="job-1",
            run_role="recipe-final",
            max_results=3,
        )

        self.assertEqual(result["run_id"], "run-1")
        client.search_runs.assert_called_once_with(
            experiment_id="experiment-1",
            platform_job_id="job-1",
            run_role="recipe-final",
            max_results=3,
        )


if __name__ == "__main__":
    unittest.main()
