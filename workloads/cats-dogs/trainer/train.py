from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

# Configure timeouts before importing MLflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tensorflow import keras

from trainer.data import (
    collect_image_paths,
    get_class_weights,
    make_dataset,
    prepare_dataset,
    stratified_split,
)
from trainer.metrics import optimize_threshold
from trainer.model import build_model


LOGGER = logging.getLogger("cats_dogs_trainer")
SEED = 42
NUM_CHANNELS = 3


class MLflowEpochLogger(keras.callbacks.Callback):
    """Log Keras metrics after every epoch into the active MLflow run."""

    def __init__(self, metric_prefix: str = "epoch") -> None:
        super().__init__()
        self.metric_prefix = metric_prefix

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        metrics: dict[str, float] = {}

        for name, value in logs.items():
            if value is None or not np.isscalar(value):
                continue
            metrics[f"{self.metric_prefix}_{name}"] = float(value)

        if not metrics:
            return

        try:
            mlflow.log_metrics(metrics, step=epoch + 1)
        except Exception as exc:  # Training should continue if remote tracking is briefly unavailable.
            LOGGER.warning("Could not log epoch %d to MLflow: %s", epoch + 1, exc)


def parse_bool(value: str | bool) -> bool:
    """Parse explicit booleans while still supporting --flag without a value."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got: {value!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Cats & Dogs MobileNetV2 for Katib trials or final training."
    )
    parser.add_argument("--mode", choices=("trial", "final"), required=True)
    parser.add_argument("--recipe-id", default="cats-dogs")
    parser.add_argument("--recipe-version", default="1.0")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dropout-rate", type=float, required=True)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--result-path",
        type=str,
        default="/tmp/cats-dogs-result.json",
        help="JSON output path used by final mode.",
    )
    parser.add_argument(
        "--trainable-backbone",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help=(
            "Fine-tune MobileNetV2 backbone. Accepts either "
            "--trainable-backbone or --trainable-backbone=true/false."
        ),
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help=(
            "Log the final model and evaluation to MLflow, but leave Model "
            "Registry registration to the next KFP component."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.recipe_id != "cats-dogs":
        raise ValueError(
            f"Unsupported recipe ID {args.recipe_id!r}; expected 'cats-dogs'"
        )
    if args.recipe_version != "1.0":
        raise ValueError(
            f"Unsupported Cats & Dogs recipe version {args.recipe_version!r}; "
            "expected '1.0'"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if not 0 <= args.dropout_rate < 1:
        raise ValueError("--dropout-rate must be in [0, 1)")
    if args.dense_units <= 0:
        raise ValueError("--dense-units must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def wait_for_mlflow_server(
    tracking_uri: str,
    max_attempts: int = 8,
    sleep_seconds: int = 10,
) -> None:
    """Wake a sleeping Render service and wait until MLflow responds."""
    health_url = f"{tracking_uri.rstrip('/')}/health"
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(health_url, timeout=(10, 180))
            if response.status_code < 500:
                LOGGER.info(
                    "MLflow ready | HTTP %d | attempt %d/%d",
                    response.status_code,
                    attempt,
                    max_attempts,
                )
                return

            last_error = RuntimeError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        except requests.RequestException as exc:
            last_error = exc

        LOGGER.warning(
            "MLflow not ready (attempt %d/%d): %s",
            attempt,
            max_attempts,
            last_error,
        )
        time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Cannot connect to MLflow Tracking Server: {tracking_uri}"
    ) from last_error


def configure_mlflow() -> dict[str, str | None]:
    tracking_uri = required_env("MLFLOW_TRACKING_URI")
    experiment_name = required_env("MLFLOW_EXPERIMENT_NAME")

    wait_for_mlflow_server(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    config: dict[str, str | None] = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "registered_model_name": os.getenv("MLFLOW_REGISTERED_MODEL_NAME"),
        "parent_run_id": os.getenv("MLFLOW_PARENT_RUN_ID"),
        "platform_job_id": os.getenv("PLATFORM_JOB_ID", "standalone"),
        "katib_experiment_name": os.getenv(
            "KATIB_EXPERIMENT_NAME", "standalone"
        ),
        "katib_trial_name": os.getenv("KATIB_TRIAL_NAME")
        or os.getenv("HOSTNAME", "standalone"),
    }

    LOGGER.info("MLflow version: %s", mlflow.__version__)
    LOGGER.info("MLflow tracking URI: %s", tracking_uri)
    LOGGER.info(
        "MLflow experiment: %s | experiment_id=%s",
        experiment_name,
        experiment.experiment_id,
    )

    return config


def build_run_tags(
    *,
    mode: str,
    mlflow_config: dict[str, str | None],
    recipe_id: str,
    recipe_version: str,
) -> dict[str, str]:
    tags = {
        "platform.job_id": str(mlflow_config["platform_job_id"]),
        "platform.recipe_id": recipe_id,
        "platform.recipe_version": recipe_version,
        "platform.run_role": "katib_trial" if mode == "trial" else "final_training",
        "model.framework": "tensorflow_keras",
        "model.task_type": "binary_image_classification",
        "model.architecture": "MobileNetV2",
    }

    parent_run_id = mlflow_config.get("parent_run_id")
    if parent_run_id:
        tags["mlflow.parentRunId"] = str(parent_run_id)

    if mode == "trial":
        tags["katib.experiment"] = str(
            mlflow_config["katib_experiment_name"]
        )
        tags["katib.trial"] = str(mlflow_config["katib_trial_name"])

    katib_experiment_name = mlflow_config.get("katib_experiment_name")
    if katib_experiment_name and katib_experiment_name != "standalone":
        tags["platform.katib_experiment_id"] = str(katib_experiment_name)

    return tags


def create_callbacks(
    *,
    checkpoint_path: Path,
    patience: int,
) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        MLflowEpochLogger(),
    ]


def log_common_params(
    *,
    args: argparse.Namespace,
    dataset_uri: str,
    train_count: int,
    validation_count: int,
    test_count: int | None,
    validation_ratio: float,
) -> None:
    params: dict[str, object] = {
        "mode": args.mode,
        "model_name": "MobileNetV2",
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "dense_units": args.dense_units,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "trainable_backbone": args.trainable_backbone,
        "validation_ratio": validation_ratio,
        "train_image_count": train_count,
        "validation_image_count": validation_count,
        "dataset_uri": dataset_uri,
        "seed": SEED,
    }

    if test_count is not None:
        params["test_image_count"] = test_count

    mlflow.log_params(params)


def train_model(
    *,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    patience: int,
) -> tuple[keras.Model, keras.callbacks.History, dict[str, float]]:
    train_ds = make_dataset(
        train_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=True,
    )
    validation_ds = make_dataset(
        validation_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
    )

    keras.backend.clear_session()
    model = build_model(
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        dense_units=args.dense_units,
        trainable_backbone=args.trainable_backbone,
    )

    checkpoint_path = output_dir / "best_model.keras"
    class_weights = get_class_weights(train_df)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=create_callbacks(
            checkpoint_path=checkpoint_path,
            patience=patience,
        ),
        verbose=2,
    )

    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)

    evaluation = model.evaluate(
        validation_ds,
        verbose=0,
        return_dict=True,
    )

    validation_metrics = {
        "val_loss": float(evaluation["loss"]),
        "val_accuracy": float(evaluation["accuracy"]),
        "val_auc": float(evaluation["auc"]),
    }

    return model, history, validation_metrics


def run_trial(
    *,
    args: argparse.Namespace,
    dataset_uri: str,
    dataset_root: Path,
    output_dir: Path,
    mlflow_config: dict[str, str | None],
) -> None:
    train_df = collect_image_paths(dataset_root / "train")
    trial_train_df, trial_validation_df = stratified_split(
        train_df,
        validation_size=0.20,
        seed=SEED,
    )

    tags = build_run_tags(
        mode="trial",
        mlflow_config=mlflow_config,
        recipe_id=args.recipe_id,
        recipe_version=args.recipe_version,
    )
    trial_name = str(mlflow_config["katib_trial_name"])

    with mlflow.start_run(
        run_name=f"katib-trial-{trial_name}",
        tags=tags,
    ) as run:
        log_common_params(
            args=args,
            dataset_uri=dataset_uri,
            train_count=len(trial_train_df),
            validation_count=len(trial_validation_df),
            test_count=None,
            validation_ratio=0.20,
        )

        _, history, validation_metrics = train_model(
            args=args,
            train_df=trial_train_df,
            validation_df=trial_validation_df,
            output_dir=output_dir,
            patience=2,
        )

        best_epoch = int(np.argmax(history.history["val_auc"]) + 1)

        mlflow.log_metrics(validation_metrics)
        mlflow.log_metric("best_epoch", float(best_epoch))
        mlflow.set_tag("platform.result", "trial_completed")

        LOGGER.info(
            "Trial completed | run_id=%s | val_auc=%.6f | val_accuracy=%.6f",
            run.info.run_id,
            validation_metrics["val_auc"],
            validation_metrics["val_accuracy"],
        )

    # Katib StdOut metrics collector parses name=value lines.
    print(f"val_auc={validation_metrics['val_auc']:.6f}", flush=True)
    print(
        f"val_accuracy={validation_metrics['val_accuracy']:.6f}",
        flush=True,
    )
    print(f"val_loss={validation_metrics['val_loss']:.6f}", flush=True)
    print(f"mlflow_run_id={run.info.run_id}", flush=True)


def save_final_artifacts(
    *,
    output_dir: Path,
    test_df: pd.DataFrame,
    test_probabilities: np.ndarray,
    test_predictions: np.ndarray,
    threshold_results: pd.DataFrame,
    model_config: dict[str, object],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    predictions_df = test_df.copy()
    predictions_df["prob_dog"] = test_probabilities
    predictions_df["pred_label"] = np.where(
        test_predictions == 1,
        "dog",
        "cat",
    )
    predictions_df["is_correct"] = (
        predictions_df["label"].to_numpy() == test_predictions
    )

    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    paths["predictions"] = predictions_path

    report = classification_report(
        test_df["label"].to_numpy(),
        test_predictions,
        target_names=["cat", "dog"],
        output_dict=True,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)
    paths["classification_report"] = report_path

    threshold_path = output_dir / "threshold_optimization.csv"
    threshold_results.sort_values("threshold").to_csv(
        threshold_path,
        index=False,
    )
    paths["threshold_optimization"] = threshold_path

    config_path = output_dir / "model_config.json"
    config_path.write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["model_config"] = config_path

    matrix = confusion_matrix(
        test_df["label"].to_numpy(),
        test_predictions,
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["cat", "dog"],
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    display.plot(ax=axis, colorbar=False)
    axis.set_title("Cats & Dogs test confusion matrix")
    figure.tight_layout()

    matrix_path = output_dir / "confusion_matrix.png"
    figure.savefig(matrix_path, bbox_inches="tight")
    plt.close(figure)
    paths["confusion_matrix"] = matrix_path

    return paths


def run_final(
    *,
    args: argparse.Namespace,
    dataset_uri: str,
    dataset_root: Path,
    output_dir: Path,
    mlflow_config: dict[str, str | None],
) -> None:
    registered_model_name = mlflow_config.get("registered_model_name")
    if not args.skip_registration and not registered_model_name:
        raise RuntimeError(
            "Missing required environment variable in final mode when "
            "registration is enabled: MLFLOW_REGISTERED_MODEL_NAME"
        )

    train_df = collect_image_paths(dataset_root / "train")
    test_df = collect_image_paths(dataset_root / "test")
    final_train_df, final_validation_df = stratified_split(
        train_df,
        validation_size=0.10,
        seed=SEED,
    )

    tags = build_run_tags(
        mode="final",
        mlflow_config=mlflow_config,
        recipe_id=args.recipe_id,
        recipe_version=args.recipe_version,
    )
    platform_job_id = str(mlflow_config["platform_job_id"])

    with mlflow.start_run(
        run_name=f"final-training-{platform_job_id[:12]}",
        tags=tags,
    ) as run:
        log_common_params(
            args=args,
            dataset_uri=dataset_uri,
            train_count=len(final_train_df),
            validation_count=len(final_validation_df),
            test_count=len(test_df),
            validation_ratio=0.10,
        )

        model, history, validation_metrics = train_model(
            args=args,
            train_df=final_train_df,
            validation_df=final_validation_df,
            output_dir=output_dir,
            patience=4,
        )

        validation_ds = make_dataset(
            final_validation_df,
            image_size=args.image_size,
            batch_size=args.batch_size,
            training=False,
        )
        validation_probabilities = model.predict(
            validation_ds,
            verbose=0,
        ).ravel()

        final_threshold, threshold_results = optimize_threshold(
            final_validation_df["label"].to_numpy(dtype=int),
            validation_probabilities,
        )

        test_ds = make_dataset(
            test_df,
            image_size=args.image_size,
            batch_size=args.batch_size,
            training=False,
        )
        test_probabilities = model.predict(test_ds, verbose=0).ravel()
        test_predictions = (
            test_probabilities >= final_threshold
        ).astype(int)
        test_labels = test_df["label"].to_numpy(dtype=int)

        test_accuracy = float(
            accuracy_score(test_labels, test_predictions)
        )
        test_f1 = float(f1_score(test_labels, test_predictions))
        test_auc = float(roc_auc_score(test_labels, test_probabilities))
        best_epoch = int(np.argmax(history.history["val_auc"]) + 1)

        final_metrics = {
            **validation_metrics,
            "final_threshold": float(final_threshold),
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "best_epoch": float(best_epoch),
        }
        mlflow.log_metrics(final_metrics)

        model_config: dict[str, object] = {
            "model_name": "MobileNetV2",
            "img_size": [args.image_size, args.image_size],
            "num_channels": NUM_CHANNELS,
            "final_best_threshold": float(final_threshold),
            "class_to_id": {"cats": 0, "dogs": 1},
            "id_to_class": {"0": "cat", "1": "dog"},
            "probability_meaning": "prob_dog",
            "best_config": {
                "learning_rate": args.learning_rate,
                "dropout_rate": args.dropout_rate,
                "dense_units": args.dense_units,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "trainable_backbone": args.trainable_backbone,
            },
        }

        artifact_paths = save_final_artifacts(
            output_dir=output_dir,
            test_df=test_df,
            test_probabilities=test_probabilities,
            test_predictions=test_predictions,
            threshold_results=threshold_results,
            model_config=model_config,
        )

        mlflow.log_artifact(
            str(artifact_paths["predictions"]),
            artifact_path="predictions",
        )
        mlflow.log_artifact(
            str(artifact_paths["classification_report"]),
            artifact_path="evaluation",
        )
        mlflow.log_artifact(
            str(artifact_paths["confusion_matrix"]),
            artifact_path="evaluation",
        )
        mlflow.log_artifact(
            str(artifact_paths["threshold_optimization"]),
            artifact_path="evaluation",
        )
        mlflow.log_artifact(
            str(artifact_paths["model_config"]),
            artifact_path="model_config",
        )

        input_example = np.zeros(
            (1, args.image_size, args.image_size, NUM_CHANNELS),
            dtype=np.float32,
        )
        prediction_example = model.predict(input_example, verbose=0)
        signature = infer_signature(input_example, prediction_example)

        model_info = mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        model_uri = model_info.model_uri
        mlflow.set_tag("platform.model_uri", model_uri)

        model_version = None
        if args.skip_registration:
            mlflow.set_tag("platform.result", "final_model_logged")
        else:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=str(registered_model_name),
                await_registration_for=300,
            )
            mlflow.set_tag(
                "platform.registered_model_name",
                str(registered_model_name),
            )
            mlflow.set_tag(
                "platform.registered_model_version",
                str(model_version.version),
            )
            mlflow.set_tag("platform.result", "final_model_registered")

        result = {
            "platform_job_id": platform_job_id,
            "recipe_id": args.recipe_id,
            "recipe_version": args.recipe_version,
            "katib_experiment_id": (
                mlflow_config.get("katib_experiment_name") or None
            ),
            "mlflow_run_id": run.info.run_id,
            "model_uri": model_uri,
            "registered_model_name": (
                str(registered_model_name) if registered_model_name else None
            ),
            "registered_model_version": (
                str(model_version.version) if model_version is not None else None
            ),
            "final_threshold": float(final_threshold),
            "val_loss": validation_metrics["val_loss"],
            "val_accuracy": validation_metrics["val_accuracy"],
            "val_auc": validation_metrics["val_auc"],
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "best_epoch": best_epoch,
        }

        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(result_path), artifact_path="platform")

        if model_version is None:
            LOGGER.info(
                "Final training/evaluation completed | run_id=%s | model_uri=%s",
                run.info.run_id,
                model_uri,
            )
        else:
            LOGGER.info(
                "Final training completed | run_id=%s | model=%s version=%s",
                run.info.run_id,
                registered_model_name,
                model_version.version,
            )

    print(f"mlflow_run_id={result['mlflow_run_id']}", flush=True)
    print(f"model_uri={result['model_uri']}", flush=True)
    if result["registered_model_version"] is not None:
        print(
            f"registered_model_version={result['registered_model_version']}",
            flush=True,
        )
    print(f"final_threshold={result['final_threshold']:.6f}", flush=True)
    print(f"test_accuracy={result['test_accuracy']:.6f}", flush=True)
    print(f"test_f1={result['test_f1']:.6f}", flush=True)
    print(f"test_auc={result['test_auc']:.6f}", flush=True)
    print(f"result_path={args.result_path}", flush=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    args = parse_args()
    validate_args(args)
    seed_everything(SEED)

    dataset_uri = required_env("DATASET_URI")
    mlflow_config = configure_mlflow()

    run_suffix = str(mlflow_config["platform_job_id"] or "standalone")[:12]
    output_dir = Path(f"/tmp/cats-dogs-output/{args.mode}-{run_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Mode: %s", args.mode)
    LOGGER.info("Dataset URI: %s", dataset_uri)
    LOGGER.info("Output directory: %s", output_dir)

    dataset_root = prepare_dataset(
        dataset_uri,
        work_dir=f"/tmp/cats-dogs-data/{run_suffix}",
    )

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {train_dir}")
    if args.mode == "final" and not test_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {test_dir}")

    if args.mode == "trial":
        run_trial(
            args=args,
            dataset_uri=dataset_uri,
            dataset_root=dataset_root,
            output_dir=output_dir,
            mlflow_config=mlflow_config,
        )
    else:
        run_final(
            args=args,
            dataset_uri=dataset_uri,
            dataset_root=dataset_root,
            output_dir=output_dir,
            mlflow_config=mlflow_config,
        )


if __name__ == "__main__":
    main()
