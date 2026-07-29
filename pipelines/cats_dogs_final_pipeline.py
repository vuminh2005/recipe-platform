from pathlib import Path

from kfp import compiler, dsl, kubernetes
from kfp.dsl import OutputPath


TRAINER_IMAGE = "docker.io/library/cats-dogs-trainer:0.5"
PLATFORM_SECRET = "cats-dogs-platform-secrets"

SECRET_ENV = {
    "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME": "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME": "MLFLOW_REGISTERED_MODEL_NAME",
    "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
    "DATASET_URI": "DATASET_URI",
}


@dsl.container_component
def validate_dataset(
    validation_json: OutputPath(str),
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "trainer.validate"],
        args=[
            "--output-path",
            validation_json,
        ],
    )


@dsl.container_component
def final_train_and_evaluate(
    platform_job_id: str,
    recipe_id: str,
    recipe_version: str,
    mlflow_parent_run_id: str,
    katib_experiment_name: str,
    learning_rate: float,
    dropout_rate: float,
    dense_units: int,
    batch_size: int,
    epochs: int,
    image_size: int,
    trainable_backbone: bool,
    result_json: OutputPath(str),
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,

        # $0 là tên giả "runner".
        # $1, $2, $3 là các tham số platform.
        command=["sh", "-c"],

        args=[
            """
export PLATFORM_JOB_ID="$1"
export MLFLOW_PARENT_RUN_ID="$2"
export KATIB_EXPERIMENT_NAME="$3"

shift 3

exec python -m trainer.train "$@"
""",
            "runner",
            platform_job_id,
            mlflow_parent_run_id,
            katib_experiment_name,

            "--mode=final",
            "--recipe-id",
            recipe_id,
            "--recipe-version",
            recipe_version,

            "--learning-rate",
            learning_rate,

            "--dropout-rate",
            dropout_rate,

            "--dense-units",
            dense_units,

            "--batch-size",
            batch_size,

            "--epochs",
            epochs,

            "--image-size",
            image_size,

            "--trainable-backbone",
            trainable_backbone,

            "--result-path",
            result_json,

            "--skip-registration",
        ],
    )


@dsl.container_component
def register_model(
    platform_job_id: str,
    mlflow_parent_run_id: str,
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
  --output-path "$4"
""",
            "runner",
            platform_job_id,
            mlflow_parent_run_id,
            result_json,
            registered_result_json,
        ],
    )


def configure_local_task(
    task: dsl.PipelineTask,
) -> dsl.PipelineTask:
    kubernetes.set_image_pull_policy(task, "Never")

    kubernetes.use_secret_as_env(
        task=task,
        secret_name=PLATFORM_SECRET,
        secret_key_to_env=SECRET_ENV,
    )

    return task


@dsl.pipeline(
    name="cats-dogs-final-pipeline",
    description=(
        "Validate the dataset, train and evaluate using Katib best "
        "parameters, then register the model in MLflow."
    ),
)
def cats_dogs_final_pipeline(
    platform_job_id: str = "manual-final-001",
    recipe_id: str = "cats-dogs",
    recipe_version: str = "1.0",
    mlflow_parent_run_id: str = "",
    katib_experiment_name: str = "cats-dogs-mobilenet-tuning",
    learning_rate: float = 0.0003,
    dropout_rate: float = 0.25,
    dense_units: int = 128,
    batch_size: int = 8,
    final_epochs: int = 5,
    image_size: int = 224,
    trainable_backbone: bool = False,
):
    # ---------------------------------------------------------
    # Task 1: Validate dataset
    # ---------------------------------------------------------
    validation_task = configure_local_task(
        validate_dataset()
    )

    validation_task.set_display_name("validate-dataset")
    validation_task.set_cpu_request("500m")
    validation_task.set_cpu_limit("2")
    validation_task.set_memory_request("1Gi")
    validation_task.set_memory_limit("2Gi")
    validation_task.set_caching_options(False)

    # ---------------------------------------------------------
    # Task 2: Final training + threshold optimization + evaluate
    # ---------------------------------------------------------
    train_task = configure_local_task(
        final_train_and_evaluate(
            platform_job_id=platform_job_id,
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            mlflow_parent_run_id=mlflow_parent_run_id,
            katib_experiment_name=katib_experiment_name,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            batch_size=batch_size,
            epochs=final_epochs,
            image_size=image_size,
            trainable_backbone=trainable_backbone,
        )
    )

    train_task.after(validation_task)
    train_task.set_display_name("final-train-and-evaluate")
    train_task.set_cpu_request("2")
    train_task.set_cpu_limit("6")
    train_task.set_memory_request("3Gi")
    train_task.set_memory_limit("6Gi")
    train_task.set_caching_options(False)

    # ---------------------------------------------------------
    # Task 3: Register the model logged by the previous task
    # ---------------------------------------------------------
    register_task = configure_local_task(
        register_model(
            platform_job_id=platform_job_id,
            mlflow_parent_run_id=mlflow_parent_run_id,
            result_json=train_task.outputs["result_json"],
        )
    )

    register_task.after(train_task)
    register_task.set_display_name("register-model")
    register_task.set_cpu_request("250m")
    register_task.set_cpu_limit("1")
    register_task.set_memory_request("512Mi")
    register_task.set_memory_limit("1Gi")
    register_task.set_caching_options(False)


if __name__ == "__main__":
    output_path = (
        Path(__file__).resolve().parent
        / "compiled"
        / "cats_dogs_final_pipeline.yaml"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    compiler.Compiler().compile(
        pipeline_func=cats_dogs_final_pipeline,
        package_path=str(output_path),
    )

    print(f"Compiled pipeline: {output_path}")
