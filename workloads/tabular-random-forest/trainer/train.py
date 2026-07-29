"""Train and evaluate the built-in Tabular Random Forest recipe."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from trainer.data import (
    DATASET_NAME,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    load_and_split_dataset,
)
from trainer.metrics import evaluate_classifier
from trainer.model import build_model


LOGGER = logging.getLogger("tabular_random_forest_train")
RECIPE_ID = "tabular-random-forest"
TASK_TYPE = "binary_tabular_classification"
FRAMEWORK = "scikit_learn"
ESTIMATOR = "RandomForestClassifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("trial", "final"), required=True)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--recipe-id", default=RECIPE_ID)
    parser.add_argument("--recipe-version", default="1.0")
    parser.add_argument("--mlflow-experiment-name", required=True)
    parser.add_argument("--result-path")
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _run_tags(
    *,
    args: argparse.Namespace,
    platform_job_id: str,
) -> dict[str, str]:
    tags = {
        "platform.job_id": platform_job_id,
        "platform.recipe_id": args.recipe_id,
        "platform.recipe_version": args.recipe_version,
        "platform.run_role": (
            "katib_trial" if args.mode == "trial" else "final_training"
        ),
        "model.framework": FRAMEWORK,
        "model.task_type": TASK_TYPE,
        "model.architecture": ESTIMATOR,
    }
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id
    katib_experiment_id = os.getenv("KATIB_EXPERIMENT_NAME")
    if katib_experiment_id:
        tags["platform.katib_experiment_id"] = katib_experiment_id
    katib_trial_name = os.getenv("KATIB_TRIAL_NAME")
    if katib_trial_name:
        tags["platform.katib_trial_name"] = katib_trial_name
    return tags


def _log_dataset_metadata(
    *,
    summary: dict[str, Any],
) -> None:
    mlflow.log_params(
        {
            "dataset_name": DATASET_NAME,
            "dataset_sample_count": summary["sample_count"],
            "dataset_feature_count": summary["feature_count"],
            "train_size": summary["split_sizes"]["train"],
            "validation_size": summary["split_sizes"]["validation"],
            "test_size": summary["split_sizes"]["test"],
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": summary["random_seed"],
        }
    )
    mlflow.log_dict(
        {"feature_names": summary["feature_names"]},
        "dataset/feature_names.json",
    )
    mlflow.log_dict(
        summary["class_mapping"],
        "dataset/class_mapping.json",
    )
    mlflow.log_dict(summary, "dataset/dataset_summary.json")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.recipe_id != RECIPE_ID:
        raise ValueError(
            f"Unsupported recipe ID {args.recipe_id!r}; expected {RECIPE_ID!r}"
        )
    if args.mode == "final" and not args.result_path:
        raise ValueError("--result-path is required in final mode")
    tracking_uri = required_env("MLFLOW_TRACKING_URI")
    platform_job_id = required_env("PLATFORM_JOB_ID")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(args.mlflow_experiment_name)

    splits = load_and_split_dataset(args.random_seed)
    summary = splits.summary(random_seed=args.random_seed)
    model = build_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features,
        random_seed=args.random_seed,
    )
    model_parameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "max_features": args.max_features,
        "random_seed": args.random_seed,
    }

    run_name = (
        f"katib-trial-{os.getenv('KATIB_TRIAL_NAME', platform_job_id[:12])}"
        if args.mode == "trial"
        else f"final-training-{platform_job_id[:12]}"
    )
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=_run_tags(args=args, platform_job_id=platform_job_id),
    ) as active_run:
        _log_dataset_metadata(summary=summary)
        mlflow.log_params(model_parameters)
        model.fit(splits.train_features, splits.train_target)

        validation_metrics, validation_matrix = evaluate_classifier(
            model,
            splits.validation_features,
            splits.validation_target,
            prefix="val",
        )
        mlflow.log_metrics(validation_metrics)
        mlflow.log_dict(
            {
                "labels": ["malignant", "benign"],
                "matrix": validation_matrix,
            },
            "metrics/validation_confusion_matrix.json",
        )

        result: dict[str, Any] = {
            "platform_job_id": platform_job_id,
            "recipe_id": args.recipe_id,
            "recipe_version": args.recipe_version,
            "mlflow_run_id": active_run.info.run_id,
            "parameters": model_parameters,
            **validation_metrics,
        }
        if args.mode == "trial":
            mlflow.set_tag("platform.result", "trial_completed")
        else:
            test_metrics, test_matrix = evaluate_classifier(
                model,
                splits.test_features,
                splits.test_target,
                prefix="test",
            )
            mlflow.log_metrics(test_metrics)
            mlflow.log_dict(
                {
                    "labels": ["malignant", "benign"],
                    "matrix": test_matrix,
                },
                "metrics/test_confusion_matrix.json",
            )

            input_example = splits.train_features.head(5)
            signature = infer_signature(
                input_example,
                model.predict_proba(input_example),
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )
            model_uri = f"runs:/{active_run.info.run_id}/model"
            result.update(
                {
                    **test_metrics,
                    "model_uri": model_uri,
                    "model_architecture": ESTIMATOR,
                    "framework": FRAMEWORK,
                    "task_type": TASK_TYPE,
                    "feature_names": list(splits.feature_names),
                    "feature_count": len(splits.feature_names),
                    "class_mapping": {
                        str(key): value
                        for key, value in splits.class_mapping.items()
                    },
                    "probability_output": "predict_proba_class_1_benign",
                }
            )
            mlflow.log_dict(result, "results/training_result.json")
            mlflow.set_tag("platform.model_uri", model_uri)
            mlflow.set_tag("platform.result", "final_model_logged")

    if args.mode == "trial":
        for metric_name in (
            "val_f1",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_roc_auc",
        ):
            print(f"{metric_name}={result[metric_name]:.6f}", flush=True)
        print(f"mlflow_run_id={result['mlflow_run_id']}", flush=True)
    else:
        output_path = Path(args.result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Final training result written to %s", output_path)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
