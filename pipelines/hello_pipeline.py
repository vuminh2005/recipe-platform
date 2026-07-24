from kfp import compiler, dsl


@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message


@dsl.pipeline(name="hello-recipe-platform")
def hello_pipeline(recipient: str = "World"):
    say_hello(name=recipient)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path="pipelines/compiled/hello_pipeline.yaml",
    )

    print("Compiled: pipelines/compiled/hello_pipeline.yaml")
