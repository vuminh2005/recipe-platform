from datetime import datetime

import kubeflow.katib as katib


def objective(parameters):
    import time

    a = int(parameters["a"])
    b = float(parameters["b"])

    time.sleep(2)

    result = 4 * a - b**2

    print(f"result={result}", flush=True)


def main():
    experiment_name = (
        "objective-demo-"
        + datetime.now().strftime("%m%d-%H%M%S")
    )

    parameters = {
        "a": katib.search.int(min=10, max=20),
        "b": katib.search.double(min=0.1, max=0.2),
    }

    client = katib.KatibClient(namespace="kubeflow")

    client.tune(
        name=experiment_name,
        objective=objective,
        parameters=parameters,
        objective_metric_name="result",
        objective_type="maximize",
        algorithm_name="random",
        max_trial_count=3,
        parallel_trial_count=1,
        max_failed_trial_count=1,
        resources_per_trial={
            "cpu": "500m",
            "memory": "256Mi",
        },
    )

    print(f"Created experiment: {experiment_name}")


if __name__ == "__main__":
    main()
