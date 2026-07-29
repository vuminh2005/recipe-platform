from __future__ import annotations

import hashlib
import json
import unittest

from agent.cats_dogs_katib import (
    TRAINER_IMAGE,
    build_experiment_manifest,
    parse_experiment_result,
)


class KatibManifestTests(unittest.TestCase):
    def build_manifest(self, **overrides):
        arguments = {
            "namespace": "ml-platform",
            "job_id": "12345678-abcd",
            "parent_run_id": "parent-run",
            "trial_epochs": 2,
            "batch_size": 8,
            "dense_units": 128,
            "image_size": 224,
        }
        arguments.update(overrides)
        return build_experiment_manifest(**arguments)

    def test_current_defaults_are_preserved(self) -> None:
        manifest = self.build_manifest()
        spec = manifest["spec"]
        container = spec["trialTemplate"]["trialSpec"]["spec"]["template"][
            "spec"
        ]["containers"][0]

        self.assertEqual(spec["maxTrialCount"], 3)
        self.assertEqual(spec["parallelTrialCount"], 1)
        self.assertEqual(spec["algorithm"], {"algorithmName": "random"})
        self.assertEqual(
            spec["parameters"][0]["feasibleSpace"],
            {"min": "0.00005", "max": "0.0005"},
        )
        self.assertEqual(
            spec["parameters"][1]["feasibleSpace"],
            {"min": "0.15", "max": "0.45"},
        )
        self.assertEqual(container["image"], TRAINER_IMAGE)
        self.assertIn("--trainable-backbone=false", container["args"])
        canonical = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "8350a17445a1b36258d12410c81bd37742ff260825298600ddc06c393b42b8b4",
        )

    def test_configurable_fields_reach_the_manifest(self) -> None:
        manifest = self.build_manifest(
            max_trial_count=5,
            parallel_trial_count=2,
            algorithm_name="random",
            learning_rate_min=0.0001,
            learning_rate_max=0.001,
            dropout_rate_min=0.2,
            dropout_rate_max=0.4,
            trainable_backbone=True,
        )
        spec = manifest["spec"]
        container_args = spec["trialTemplate"]["trialSpec"]["spec"]["template"][
            "spec"
        ]["containers"][0]["args"]

        self.assertEqual(spec["maxTrialCount"], 5)
        self.assertEqual(spec["parallelTrialCount"], 2)
        self.assertEqual(
            spec["parameters"][0]["feasibleSpace"],
            {"min": "0.0001", "max": "0.001"},
        )
        self.assertEqual(
            spec["parameters"][1]["feasibleSpace"],
            {"min": "0.2", "max": "0.4"},
        )
        self.assertIn("--trainable-backbone=true", container_args)

    def test_unsupported_algorithm_is_rejected_before_manifest_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Katib algorithm"):
            self.build_manifest(algorithm_name="tpe")

    def test_result_parser_owns_cats_dogs_assignments_and_objective(self) -> None:
        result = parse_experiment_result(
            {
                "metadata": {"name": "cats-dogs-hpo-12345678"},
                "status": {
                    "currentOptimalTrial": {
                        "bestTrialName": "best-trial",
                        "parameterAssignments": [
                            {"name": "learning_rate", "value": "0.0003"},
                            {"name": "dropout_rate", "value": "0.25"},
                        ],
                        "observation": {
                            "metrics": [
                                {"name": "val_auc", "max": "0.91"},
                                {"name": "val_loss", "min": "0.22"},
                            ]
                        },
                    }
                },
            }
        )

        self.assertEqual(result.best_params["learning_rate"], 0.0003)
        self.assertEqual(result.best_params["dropout_rate"], 0.25)
        self.assertEqual(result.best_metric, 0.91)


if __name__ == "__main__":
    unittest.main()
