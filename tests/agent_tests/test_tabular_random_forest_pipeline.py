from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kfp import compiler

from pipelines.tabular_random_forest_pipeline import (
    GENERIC_SECRET_ENV,
    TRAINER_IMAGE,
    tabular_random_forest_pipeline,
)


class TabularPipelineTests(unittest.TestCase):
    def test_pipeline_compiles_with_recipe_image_and_no_cats_parameters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pipeline.yaml"
            compiler.Compiler().compile(
                pipeline_func=tabular_random_forest_pipeline,
                package_path=str(output),
            )
            package = output.read_text(encoding="utf-8")

        self.assertIn(TRAINER_IMAGE, package)
        self.assertIn("n_estimators", package)
        self.assertIn("min_samples_split", package)
        for forbidden in (
            "learning_rate",
            "dropout_rate",
            "image_size",
            "trainable_backbone",
        ):
            self.assertNotIn(forbidden, package)

    def test_pipeline_secret_mapping_is_generic_and_individual(self) -> None:
        self.assertEqual(
            set(GENERIC_SECRET_ENV),
            {
                "MLFLOW_TRACKING_URI",
                "MLFLOW_S3_ENDPOINT_URL",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
            },
        )

    def test_repository_package_matches_current_pipeline_source(self) -> None:
        repository_package = (
            Path(__file__).resolve().parents[2]
            / "pipelines"
            / "compiled"
            / "tabular_random_forest_pipeline.yaml"
        )
        self.assertTrue(repository_package.is_file())
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "pipeline.yaml"
            compiler.Compiler().compile(
                pipeline_func=tabular_random_forest_pipeline,
                package_path=str(generated),
            )
            self.assertEqual(
                repository_package.read_bytes(),
                generated.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
