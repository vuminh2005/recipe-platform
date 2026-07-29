from __future__ import annotations

import ast
import io
import importlib.util
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = ROOT / "workloads" / "cats-dogs" / "trainer"
OBSERVABILITY_PATH = TRAINER_ROOT / "observability.py"
SPEC = importlib.util.spec_from_file_location(
    "cats_dogs_observability_under_test",
    OBSERVABILITY_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {OBSERVABILITY_PATH}")
OBSERVABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBSERVABILITY)


class SafeUriMetadataTests(unittest.TestCase):
    def test_credentials_query_and_complete_object_path_are_not_returned(
        self,
    ) -> None:
        raw_uri = (
            "s3://access-key:secret@example.invalid/private/datasets/"
            "cats-dogs.zip?X-Amz-Credential=credential"
            "&X-Amz-Signature=presigned-secret"
        )

        metadata = OBSERVABILITY.safe_uri_metadata(
            raw_uri,
            source_type="object_storage",
            identifier=OBSERVABILITY.CATS_DOGS_DATASET_ID,
        )
        rendered = repr(metadata)

        self.assertEqual(metadata["scheme"], "s3")
        self.assertEqual(metadata["source_type"], "object_storage")
        self.assertEqual(
            metadata["identifier"],
            OBSERVABILITY.CATS_DOGS_DATASET_ID,
        )
        for sensitive_fragment in (
            "access-key",
            "secret",
            "example.invalid",
            "private/datasets",
            "cats-dogs.zip",
            "X-Amz-Credential",
            "X-Amz-Signature",
            "?",
        ):
            self.assertNotIn(sensitive_fragment, rendered)

    def test_mlflow_metadata_exposes_only_safe_connection_state(self) -> None:
        raw_uri = (
            "https://user:password@mlflow.invalid/private"
            "?token=presigned-secret"
        )

        metadata = OBSERVABILITY.safe_uri_metadata(
            raw_uri,
            source_type="tracking_service",
        )

        self.assertEqual(
            metadata,
            {
                "configured": True,
                "source_type": "tracking_service",
                "scheme": "https",
            },
        )
        self.assertNotIn("mlflow.invalid", repr(metadata))


class ManagedMlflowRunTests(unittest.TestCase):
    class FakeMlflow:
        def __init__(self, tracking_uri: str) -> None:
            self.tracking_uri = tracking_uri
            self.end_statuses: list[str] = []
            self.start_kwargs: dict[str, object] | None = None

        def start_run(self, **kwargs: object) -> SimpleNamespace:
            self.start_kwargs = kwargs
            return SimpleNamespace(
                info=SimpleNamespace(run_id="run-1"),
            )

        def end_run(self, *, status: str) -> None:
            self.end_statuses.append(status)
            print(f"View run at {self.tracking_uri}/runs/run-1")
            print(
                f"View experiment at {self.tracking_uri}/experiments/1",
                file=sys.stderr,
            )

    def test_only_termination_output_is_suppressed(self) -> None:
        raw_tracking_uri = (
            "https://user:password@mlflow.invalid/private"
            "?token=presigned-secret"
        )
        fake_mlflow = self.FakeMlflow(raw_tracking_uri)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            with OBSERVABILITY.managed_mlflow_run(
                fake_mlflow,
                run_name="final-training",
                tags={"platform.run_role": "final_training"},
            ) as run:
                print("training output remains visible")
                print("TensorFlow output remains visible", file=sys.stderr)
                self.assertEqual(run.info.run_id, "run-1")
            print("val_auc=0.750000", flush=True)

        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(raw_tracking_uri, rendered)
        self.assertNotIn("presigned-secret", rendered)
        self.assertIn("training output remains visible", stdout.getvalue())
        self.assertIn("TensorFlow output remains visible", stderr.getvalue())
        self.assertIn("val_auc=0.750000", stdout.getvalue())
        self.assertEqual(fake_mlflow.end_statuses, ["FINISHED"])
        self.assertEqual(
            fake_mlflow.start_kwargs,
            {
                "run_name": "final-training",
                "tags": {"platform.run_role": "final_training"},
            },
        )

    def test_workload_exception_propagates_and_marks_run_failed(self) -> None:
        raw_tracking_uri = "https://mlflow.invalid/?token=secret"
        fake_mlflow = self.FakeMlflow(raw_tracking_uri)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaisesRegex(RuntimeError, "training failed"),
        ):
            with OBSERVABILITY.managed_mlflow_run(fake_mlflow):
                print("output before failure remains visible")
                raise RuntimeError("training failed")

        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(raw_tracking_uri, rendered)
        self.assertIn("output before failure remains visible", stdout.getvalue())
        self.assertEqual(fake_mlflow.end_statuses, ["FAILED"])


class DatasetMetadataTests(unittest.TestCase):
    def test_safe_identifier_checksum_and_counts_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, content in (
                ("train/cats/cat.jpg", b"cat"),
                ("train/dogs/dog.png", b"dog"),
                ("test/cats/cat.jpeg", b"test-cat"),
                ("notes.txt", b"not a sample"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            first = OBSERVABILITY.build_dataset_metadata(root)
            second = OBSERVABILITY.build_dataset_metadata(root)

        self.assertEqual(first, second)
        self.assertEqual(first["dataset_source_type"], "object_storage")
        self.assertEqual(first["dataset_id"], "cats-dogs-v1")
        self.assertEqual(first["dataset_sample_count"], 3)
        self.assertEqual(first["dataset_class_count"], 2)
        self.assertEqual(len(first["dataset_checksum"]), 64)


class SourceSecurityTests(unittest.TestCase):
    def test_active_sources_do_not_log_or_persist_raw_service_uris(self) -> None:
        train_source = (TRAINER_ROOT / "train.py").read_text(encoding="utf-8")
        validate_source = (TRAINER_ROOT / "validate.py").read_text(
            encoding="utf-8"
        )
        register_source = (TRAINER_ROOT / "register_model.py").read_text(
            encoding="utf-8"
        )
        observability_source = OBSERVABILITY_PATH.read_text(encoding="utf-8")

        self.assertNotIn('"dataset_uri":', train_source)
        self.assertNotIn('"dataset_uri":', validate_source)
        self.assertIn('"dataset_fingerprint_sha256":', validate_source)
        self.assertNotIn("Dataset URI:", train_source)
        self.assertNotIn("MLflow tracking URI:", train_source)
        self.assertNotIn(
            "Cannot connect to MLflow Tracking Server:",
            train_source,
        )
        self.assertNotIn(
            "MLflow did not become ready at",
            register_source,
        )
        self.assertIn('"dataset_checksum":', observability_source)
        self.assertIn('"dataset_id":', observability_source)
        self.assertIn("managed_mlflow_run", train_source)
        self.assertNotIn("with mlflow.start_run(", train_source)

        tree = ast.parse(train_source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            names = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
            constants = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            }
            if function_name in {
                "debug",
                "error",
                "exception",
                "info",
                "print",
                "warning",
            }:
                self.assertNotIn("dataset_uri", names)
                self.assertNotIn("tracking_uri", names)
            if function_name in {"log_param", "log_params"}:
                self.assertNotIn("dataset_uri", names)
                self.assertNotIn("dataset_uri", constants)

    def test_registration_does_not_manipulate_legacy_or_champion_state(
        self,
    ) -> None:
        source = (TRAINER_ROOT / "register_model.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('CANDIDATE_ALIAS = "candidate"', source)
        self.assertNotIn('"champion"', source)
        self.assertNotIn("'champion'", source)
        self.assertNotIn("cats_dogs_recipe_candidate", source)


if __name__ == "__main__":
    unittest.main()
