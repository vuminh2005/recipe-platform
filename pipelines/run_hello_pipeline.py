import os

from kfp import Client


KFP_ENDPOINT = os.getenv(
    "KFP_ENDPOINT",
    "http://127.0.0.1:8080",
)

PIPELINE_PACKAGE = "pipelines/compiled/hello_pipeline.yaml"


def main() -> None:
    client = Client(host=KFP_ENDPOINT)

    run = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_PACKAGE,
        arguments={
            "recipient": "Recipe Platform",
        },
        run_name="hello-recipe-platform",
        experiment_name="recipe-platform-development",
    )

    print(f"KFP run ID: {run.run_id}")

    completed_run = client.wait_for_run_completion(
        run_id=run.run_id,
        timeout=600,
        sleep_duration=5,
    )

    print(f"Completed run: {completed_run}")


if __name__ == "__main__":
    main()
