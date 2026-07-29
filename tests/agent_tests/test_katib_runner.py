from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock

import agent.katib_runner as katib_module
from agent.katib_runner import KatibRunner


class GenericKatibRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = KatibRunner.__new__(KatibRunner)
        self.runner.namespace = "ml-platform"
        self.runner.api = Mock()

    def test_generic_module_contains_no_cats_dogs_manifest_assumptions(self) -> None:
        source = inspect.getsource(katib_module).lower()
        for forbidden in (
            "cats",
            "dogs",
            "mobilenet",
            "learning_rate",
            "dropout_rate",
            "val_auc",
            "image_size",
            "trainer image",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        self.assertFalse(hasattr(KatibRunner, "build_manifest"))
        self.assertFalse(hasattr(KatibRunner, "_parse_result"))

    def test_wait_for_success_returns_raw_experiment(self) -> None:
        experiment = {
            "metadata": {"name": "any-experiment"},
            "status": {
                "conditions": [
                    {"type": "Succeeded", "status": "True", "message": ""}
                ]
            },
        }
        self.runner.get_experiment = Mock(return_value=experiment)

        result = self.runner.wait_for_success(
            "any-experiment",
            timeout_seconds=1,
            poll_seconds=0,
        )

        self.assertIs(result, experiment)

    def test_ensure_experiment_reuses_existing_resource(self) -> None:
        manifest = {
            "metadata": {
                "name": "any-experiment",
                "namespace": "ml-platform",
            }
        }
        existing = {"metadata": {"name": "any-experiment"}}
        self.runner.get_experiment = Mock(return_value=existing)

        result = self.runner.ensure_experiment(manifest)

        self.assertIs(result, existing)
        self.runner.api.create_namespaced_custom_object.assert_not_called()


if __name__ == "__main__":
    unittest.main()
