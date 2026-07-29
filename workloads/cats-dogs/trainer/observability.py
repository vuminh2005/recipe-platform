from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit


CATS_DOGS_DATASET_ID = "cats-dogs-v1"
DATASET_SOURCE_TYPE = "object_storage"
EXPECTED_CLASS_COUNT = 2
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
_SAFE_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*$")


class _DiscardTextOutput:
    """A text sink that discards writes without retaining sensitive output."""

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def _end_mlflow_run_without_url_output(
    mlflow_module: Any,
    *,
    status: str,
) -> None:
    sink = _DiscardTextOutput()
    with redirect_stdout(sink), redirect_stderr(sink):
        mlflow_module.end_run(status=status)


@contextmanager
def managed_mlflow_run(
    mlflow_module: Any,
    **start_run_kwargs: Any,
) -> Iterator[Any]:
    """Manage one MLflow run while suppressing only termination URL output."""

    run = mlflow_module.start_run(**start_run_kwargs)
    try:
        yield run
    except BaseException:
        _end_mlflow_run_without_url_output(
            mlflow_module,
            status="FAILED",
        )
        raise
    else:
        _end_mlflow_run_without_url_output(
            mlflow_module,
            status="FINISHED",
        )


def safe_uri_metadata(
    uri: str,
    *,
    source_type: str,
    identifier: str | None = None,
) -> dict[str, str | bool]:
    """Describe a configured URI without returning its authority or path."""

    try:
        raw_scheme = urlsplit(uri.strip()).scheme.lower()
    except ValueError:
        raw_scheme = ""
    scheme = raw_scheme if _SAFE_SCHEME.fullmatch(raw_scheme) else "unknown"
    metadata: dict[str, str | bool] = {
        "configured": bool(uri.strip()),
        "source_type": source_type,
        "scheme": scheme,
    }
    if identifier is not None:
        metadata["identifier"] = identifier
    return metadata


def build_dataset_metadata(dataset_root: Path) -> dict[str, str | int]:
    """Build stable, non-sensitive metadata from extracted dataset content."""

    digest = hashlib.sha256()
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())

    for path in files:
        relative = path.relative_to(dataset_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    sample_count = sum(
        path.suffix.lower() in IMAGE_EXTENSIONS for path in files
    )
    return {
        "dataset_source_type": DATASET_SOURCE_TYPE,
        "dataset_id": CATS_DOGS_DATASET_ID,
        "dataset_checksum": digest.hexdigest(),
        "dataset_sample_count": sample_count,
        "dataset_class_count": EXPECTED_CLASS_COUNT,
    }
