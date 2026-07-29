"""Recipe-specific KFP pipeline for Tabular Random Forest classification."""

from pathlib import Path

from kfp import compiler, dsl, kubernetes
from kfp.dsl import OutputPath


TRAINER_IMAGE = (
    "docker.io/library/tabular-random-forest-trainer:1.0"
)

# Naming debt: this existing Secret has a Cats & Dogs-oriented name. Tabular
# tasks inject only these generic MLflow/artifact keys, never envFrom or the
# Cats & Dogs dataset/model/experiment values.
PLATFORM_SECRET = "cats-dogs-platform-secrets"
GENERIC_SECRET_ENV = {
    "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
}


@dsl.container_component
def validate_dataset(
    random_seed: int,
    validation_json: OutputPath(str),
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "trainer.validate"],
        args=[
            "--random-seed",
            random_seed,
            "--output-path",
            validation_json,
        ],
    )


@dsl.container_component
def final_train_and_evaluate(
    platform_job_id: str,
    recipe_id: str,
    recipe_version: str,
    random_seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    max_features: str,
    mlflow_parent_run_id: str,
    mlflow_experiment_name: str,
    katib_experiment_id: str,
    result_json: OutputPath(str),
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["sh", "-c"],
        args=[
            """
export PLATFORM_JOB_ID="$1"
export MLFLOW_PARENT_RUN_ID="$2"
if [ -n "$3" ]; then
  export KATIB_EXPERIMENT_NAME="$3"
fi
shift 3
exec python -m trainer.train "$@"
""",
            "runner",
            platform_job_id,
            mlflow_parent_run_id,
            katib_experiment_id,
            "--mode=final",
            "--recipe-id",
            recipe_id,
            "--recipe-version",
            recipe_version,
            "--random-seed",
            random_seed,
            "--n-estimators",
            n_estimators,
            "--max-depth",
            max_depth,
            "--min-samples-split",
            min_samples_split,
            "--max-features",
            max_features,
            "--mlflow-experiment-name",
            mlflow_experiment_name,
            "--result-path",
            result_json,
        ],
    )


@dsl.container_component
def register_model(
    platform_job_id: str,
    mlflow_parent_run_id: str,
    registered_model_name: str,
    result_json: str,
    registered_result_json: OutputPath(str),
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["sh", "-c"],
        args=[
            """
export PLATFORM_JOB_ID="$1"
export MLFLOW_PARENT_RUN_ID="$2"
INPUT_FILE="/tmp/training-result.json"
printf '%s' "$3" > "$INPUT_FILE"
exec python -m trainer.register_model \
  --input-json "$INPUT_FILE" \
  --registered-model-name "$4" \
  --output-path "$5"
""",
            "runner",
            platform_job_id,
            mlflow_parent_run_id,
            result_json,
            registered_model_name,
            registered_result_json,
        ],
    )


def configure_local_task(
    task: dsl.PipelineTask,
    *,
    needs_mlflow: bool,
) -> dsl.PipelineTask:
    kubernetes.set_image_pull_policy(task, "Never")
    if needs_mlflow:
        kubernetes.use_secret_as_env(
            task=task,
            secret_name=PLATFORM_SECRET,
            secret_key_to_env=GENERIC_SECRET_ENV,
        )
    task.set_caching_options(False)
    return task


@dsl.pipeline(
    name="tabular-random-forest-pipeline",
    description=(
        "Validate the built-in dataset, train and evaluate a CPU-only Random "
        "Forest, then register the logged model in MLflow."
    ),
)
def tabular_random_forest_pipeline(
    platform_job_id: str = "manual-tabular-rf-001",
    recipe_id: str = "tabular-random-forest",
    recipe_version: str = "1.0",
    random_seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    mlflow_parent_run_id: str = "",
    mlflow_experiment_name: str = "tabular_random_forest_recipe_demo",
    registered_model_name: str = "tabular_random_forest_classifier",
    katib_experiment_id: str = "",
):
    validation_task = configure_local_task(
        validate_dataset(random_seed=random_seed),
        needs_mlflow=False,
    )
    validation_task.set_display_name("validate-dataset")
    validation_task.set_cpu_request("100m")
    validation_task.set_cpu_limit("1")
    validation_task.set_memory_request("256Mi")
    validation_task.set_memory_limit("1Gi")

    training_task = configure_local_task(
        final_train_and_evaluate(
            platform_job_id=platform_job_id,
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            mlflow_parent_run_id=mlflow_parent_run_id,
            mlflow_experiment_name=mlflow_experiment_name,
            katib_experiment_id=katib_experiment_id,
        ),
        needs_mlflow=True,
    )
    training_task.after(validation_task)
    training_task.set_display_name("final-train-and-evaluate")
    training_task.set_cpu_request("250m")
    training_task.set_cpu_limit("2")
    training_task.set_memory_request("512Mi")
    training_task.set_memory_limit("2Gi")

    registration_task = configure_local_task(
        register_model(
            platform_job_id=platform_job_id,
            mlflow_parent_run_id=mlflow_parent_run_id,
            registered_model_name=registered_model_name,
            result_json=training_task.outputs["result_json"],
        ),
        needs_mlflow=True,
    )
    registration_task.after(training_task)
    registration_task.set_display_name("register-model")
    registration_task.set_cpu_request("100m")
    registration_task.set_cpu_limit("1")
    registration_task.set_memory_request("256Mi")
    registration_task.set_memory_limit("1Gi")


if __name__ == "__main__":
    output_path = (
        Path(__file__).resolve().parent
        / "compiled"
        / "tabular_random_forest_pipeline.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=tabular_random_forest_pipeline,
        package_path=str(output_path),
    )
    print(f"Compiled pipeline: {output_path}")
