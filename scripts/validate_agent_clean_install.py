"""Validate the Agent from only its authoritative runtime requirements."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="recipe-platform-agent-clean-"
    ) as directory:
        environment_path = Path(directory) / "venv"
        run(sys.executable, "-m", "venv", str(environment_path))
        python = environment_path / "bin" / "python"
        pip = environment_path / "bin" / "pip"
        run(
            str(pip),
            "install",
            "--disable-pip-version-check",
            "-r",
            "agent/requirements.txt",
        )

        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "BACKEND_URL": "http://127.0.0.1:8000",
                "AGENT_TOKEN": "clean-install-validation-token",
                "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
            }
        )
        environment.pop("ALLOW_INSECURE_DEVELOPMENT_TOKEN", None)
        validation = """
from importlib.metadata import version
from pathlib import Path

import agent.main
import agent.recipe_registry
from agent.settings import Settings

assert version("kfp") == "2.17.0"
settings = Settings.from_env()
for path in (
    settings.cats_dogs.pipeline_path,
    settings.hello.pipeline_path,
    settings.tabular_random_forest.pipeline_path,
):
    assert Path(path).is_file(), f"Missing default pipeline package: {path}"
print("Clean Agent imports, settings, KFP version, and package paths verified.")
"""
        run(str(python), "-c", validation, env=environment)
        run(str(pip), "check")


if __name__ == "__main__":
    main()
