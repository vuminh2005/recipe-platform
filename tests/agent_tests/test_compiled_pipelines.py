from __future__ import annotations

import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from kfp import compiler

from pipelines.cats_dogs_final_pipeline import cats_dogs_final_pipeline
from pipelines.hello_pipeline import hello_pipeline
from pipelines.tabular_random_forest_pipeline import (
    tabular_random_forest_pipeline,
)


ROOT = Path(__file__).resolve().parents[2]
PIPELINES = (
    (
        hello_pipeline,
        ROOT / "pipelines" / "compiled" / "hello_pipeline.yaml",
    ),
    (
        cats_dogs_final_pipeline,
        ROOT / "pipelines" / "compiled" / "cats_dogs_final_pipeline.yaml",
    ),
    (
        tabular_random_forest_pipeline,
        ROOT
        / "pipelines"
        / "compiled"
        / "tabular_random_forest_pipeline.yaml",
    ),
)


class CompiledPipelineFreshnessTests(unittest.TestCase):
    def test_tracked_packages_match_kfp_2_17_compilation(self) -> None:
        self.assertEqual(version("kfp"), "2.17.0")
        with tempfile.TemporaryDirectory() as directory:
            for pipeline_func, tracked_path in PIPELINES:
                with self.subTest(package=tracked_path.name):
                    self.assertTrue(
                        tracked_path.is_file(),
                        f"Missing tracked pipeline package: {tracked_path}",
                    )
                    generated_path = Path(directory) / tracked_path.name
                    compiler.Compiler().compile(
                        pipeline_func=pipeline_func,
                        package_path=str(generated_path),
                    )
                    self.assertEqual(
                        generated_path.read_bytes(),
                        tracked_path.read_bytes(),
                        f"Stale compiled pipeline package: {tracked_path}",
                    )


if __name__ == "__main__":
    unittest.main()
