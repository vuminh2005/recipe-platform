# Recipe Platform Contributor Guide

## Scope and direction

This repository is a graduation-thesis prototype that connects a browser
control plane to a local K3s execution environment through Katib, Kubeflow
Pipelines (KFP), and MLflow.

The platform currently has one working ML recipe, Cats & Dogs image
classification, plus the internal `hello` smoke recipe. The agent selects
these built-in recipes through an explicit registry; `hello` verifies platform
dispatch and KFP connectivity and must not be presented as an ML workload.
Promotion, serving, and inference are out of scope until a later task
explicitly adds them. MLflow model registration that is already part of the
Cats & Dogs training flow is not the same as promotion and should continue to
work.

These instructions apply to the whole repository.

## Architecture and ownership

- `frontend/` is the React/Vite dashboard. It creates jobs through the backend,
  lists and polls jobs, renders lifecycle details and metrics, and links to
  external UIs. It is a client of the control-plane API; it must not talk
  directly to Kubernetes, Katib, KFP, or MLflow APIs.
- `backend/` is the FastAPI control plane and job store. It validates and
  persists job requests, exposes job read APIs, and provides authenticated
  claim/update endpoints for the agent. It is the source of truth for platform
  job state, but it does not execute training or require direct access to the
  local cluster. `backend/app/recipe_catalog.py` owns typed public/internal
  recipe metadata, while `backend/app/recipe_normalization.py` owns
  recipe-scoped request validation, normalized configuration, versions, and
  immutable job snapshots.
- `agent/` is the trusted bridge to local K3s. It polls and claims backend jobs,
  dispatches them to a recipe handler, invokes Katib and KFP, coordinates
  MLflow metadata, and reports status and result references back to the
  backend. Keep generic polling, dispatch, HTTP, and integration clients
  separate from recipe executors. `agent/recipe_registry.py` owns the explicit
  built-in dispatch table. `agent/cats_dogs_executor.py` owns the Cats & Dogs
  orchestration, `agent/cats_dogs_katib.py` owns its Katib manifest and result
  interpretation, and `agent/hello_executor.py` owns the KFP-only smoke flow.
  These handlers are not templates for putting recipe assumptions into
  `agent/main.py`.
- `pipelines/` contains KFP pipeline source code. A recipe pipeline describes
  the in-cluster steps and their inputs/outputs. The current Cats & Dogs final
  pipeline validates data, trains/evaluates the selected configuration, and
  records/registers the resulting model in MLflow. `pipelines/compiled/`
  contains generated packages, not source.
- `workloads/<recipe>/` owns runnable ML code, dependency pins, image build
  definitions, and Kubernetes manifests for that recipe. The current
  `workloads/cats-dogs/` trainer also owns its data preparation, model,
  metrics, validation, and MLflow logging/registration behavior.
- Katib is the hyperparameter-tuning engine. A recipe handler or recipe-owned
  manifest defines the search space, objective metrics, trial image, and trial
  arguments. Katib launches trial jobs and returns the optimal parameters; it
  should not define the platform's shared job contract.
- KFP is the workflow engine for recipe pipelines. The agent submits compiled
  packages and follows run state, while pipeline components execute
  recipe-owned validation/training steps in K3s.
- MLflow is the experiment and model metadata store. The current flow records a
  parent platform run, Katib trial runs, final metrics and artifacts, and the
  registered-model reference. The backend stores the IDs and summary fields
  needed by the dashboard; it should not duplicate MLflow's full tracking
  model.

The current high-level flow is:

1. The frontend submits a recipe job to the backend.
2. The backend persists it as `PENDING`; the agent claims it.
3. The agent resolves `recipe.recipe_id`, or the legacy `recipe.workload`
   fallback, through the explicit registry.
4. The Cats & Dogs handler creates/reuses an MLflow parent run. When AutoML is
   enabled, it also builds a recipe-owned Katib Experiment and extracts the
   best parameters; when disabled, it uses the recipe defaults.
5. The selected handler submits its own KFP package and arguments. The `hello`
   handler only performs this KFP smoke flow and reports the actual run status.
6. For Cats & Dogs, MLflow stores final metrics, artifacts, and registration
   metadata.
7. The handler patches the backend job to its terminal state, and the frontend
   displays the persisted summary and external references.

Agent handlers publish newly available result data through incremental
`result_patch` objects. The backend merges those patches into the existing
physical columns and reconstructs the workload-agnostic `JobResponse.result`.
Omitted patch fields preserve stored data, and normal execution patches do not
clear result values. Existing top-level result fields remain compatibility
projections for the current dashboard, not a second independently written
result.

## Multi-recipe boundaries

Platform-core code must be workload-agnostic. Shared frontend infrastructure,
backend persistence/API code, the agent polling loop and dispatcher, and
generic Katib/KFP/MLflow clients must not assume image classification, Cats &
Dogs, MobileNetV2, or any other recipe.

Treat a job as a common envelope with a recipe/workload discriminator,
recipe-owned configuration, lifecycle state, external-run references, errors,
and timestamps. Do not add recipe-specific database columns, request fields,
form controls, status logic, metric names, pipeline paths, image names, or
hyperparameters to shared components.

Keep workload-specific behavior in these places:

- a recipe schema/definition and its frontend form or renderer;
- a recipe handler/executor selected through an explicit registry or
  dispatcher;
- `pipelines/<recipe...>.py` and recipe-owned pipeline components;
- `workloads/<recipe>/`, including trainer code, dependencies, Dockerfile, and
  manifests.

Generic integration wrappers may live in platform core, but the recipe handler
must supply its Katib search space, objective, pipeline package and arguments,
expected metrics, and result interpretation. An unknown recipe must fail
clearly; never silently execute it as Cats & Dogs.

The backend Recipe Catalog and agent execution registry have different
responsibilities and must remain independently deployable. Backend runtime
code must not import agent handlers. Keep cross-component identifier drift
checks in tests and use `agent/recipe_ids.py` when a lightweight agent-side
identifier import is needed.

When adding a recipe, extend registries/schemas and add recipe-owned code rather
than adding another recipe-name conditional throughout shared files. The
frontend should render recipe definitions or recipe-specific forms rather than
growing one shared form with fields for every workload.

## Compatibility and change discipline

- Preserve backward compatibility for existing persisted jobs and API payloads
  where practical. Current compatibility details include the legacy `hello`
  workload, `RUNNING` status, `training_config`/`automl_config` fallbacks in the
  Cats & Dogs executor, and the legacy `training.epochs` field. Do not remove
  or reinterpret them without a migration and an explicit compatibility plan.
- Preserve the existing Cats & Dogs lifecycle, retry/idempotency behavior,
  field names, external IDs, and dashboard rendering while extracting generic
  abstractions.
- New jobs store `recipe_id`, `recipe_version`, normalized `configuration`, and
  a `recipe_snapshot` in the existing recipe JSON column. Historical rows are
  read through identifier and configuration fallbacks; never rewrite them
  automatically.
- For AutoML-disabled Cats & Dogs jobs, the canonical objective value and
  `best_params` remain null. Effective fixed learning-rate and dropout values
  belong to normalized configuration/snapshots, not tuning results.
- Do not hard-code recipe-specific fields in shared components. Prefer opaque
  or validated recipe configuration plus handler-owned interpretation.
- Do not implement promotion, deployment, serving endpoints, or inference
  behavior as part of the multi-recipe refactor.
- Make small, reviewable changes. Avoid mixing schema migrations, orchestration
  changes, UI redesigns, and workload changes in one task unless they are
  inseparable.
- Add or update focused tests when behavior changes, and run relevant tests or
  static checks after each implementation task. Report checks that could not be
  run because they require K3s, credentials, datasets, or remote services.

## Files that are off limits

Do not inspect, search, or modify dependency, build, log, archive, or cache
content, including:

- `node_modules/`, `dist/`, `.venv*/`, and dependency-managed contents;
- `logs/`, `*.log`, Docker image tar files, and other large archives;
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, coverage
  output, and similar generated caches;
- generated KFP packages in `pipelines/compiled/`.

Never hand-edit generated outputs or vendored dependencies. Modify pipeline
Python sources instead. If a task explicitly requires regenerated artifacts,
generate them with the compiler and keep the generated change separate; for
ordinary validation, compile to `/tmp`.

Do not commit secrets or local state. Treat `.env*`, local databases, datasets,
model files, and credentials as local-only even when an existing untracked file
is present.

## Commands

Run commands from the repository root unless a command changes directory.
Environment-specific services require their existing `.env` values,
credentials, K3s context, and port-forwards.

### Start services

```bash
# Backend API (defaults to SQLite when DATABASE_URL is unset)
python -m uvicorn backend.app.main:app --reload

# Local polling agent; loads agent/.env
python -m agent.main

# Frontend development server
cd frontend
npm run dev
```

The agent requires `BACKEND_URL`, `AGENT_TOKEN`, and `MLFLOW_TRACKING_URI`.
Relevant optional settings include `AGENT_ID`, `KFP_ENDPOINT`,
`KATIB_NAMESPACE`, `MLFLOW_EXPERIMENT_NAME`, `POLL_INTERVAL_SECONDS`, and
`CATS_DOGS_PIPELINE_PATH`. The Hello handler also accepts
`HELLO_PIPELINE_PATH` and `HELLO_KFP_EXPERIMENT_NAME`. The frontend requires
`VITE_API_BASE_URL`; its other `VITE_*` values only configure external UI
links and must never contain secrets.

### Frontend validation and build

The frontend is JavaScript/JSX, not TypeScript. No `tsc` or frontend test script
is currently configured.

```bash
cd frontend
npm run lint

# Validate a production build without writing frontend/dist.
npm run build -- --outDir /tmp/recipe-platform-frontend-dist --emptyOutDir

# Existing production/preview commands; npm run build writes generated dist/.
npm run build
npm run preview
```

Do not run install commands merely to validate a change. If dependencies are
missing, report that instead of inspecting or modifying `node_modules/` unless
the task explicitly authorizes dependency installation.

### Python validation

Use the isolated validation environments documented by
`requirements-dev.txt` and `workloads/cats-dogs/requirements-test.txt`:

```bash
python -m venv /tmp/recipe-platform-validation-venv
/tmp/recipe-platform-validation-venv/bin/pip install -r requirements-dev.txt
/tmp/recipe-platform-validation-venv/bin/python -m unittest discover -s tests/backend_tests -v
/tmp/recipe-platform-validation-venv/bin/python -m unittest discover -s tests/agent_tests -v

python -m venv /tmp/recipe-platform-mlflow-test-venv
/tmp/recipe-platform-mlflow-test-venv/bin/pip install -r workloads/cats-dogs/requirements-test.txt
/tmp/recipe-platform-mlflow-test-venv/bin/python -m unittest discover -s tests/workload_tests -v
```

No repository-wide Python formatter, linter, or type checker is configured.
Use a cache-free syntax check for Python sources:

```bash
python -c "import ast,pathlib; roots=('backend','agent','pipelines','workloads/cats-dogs/trainer'); files=[p for r in roots for p in pathlib.Path(r).rglob('*.py')]; [ast.parse(p.read_text(encoding='utf-8'),filename=str(p)) for p in files]"
```

If a task adds tests, run the narrow relevant tests first and then the
applicable suite. Do not claim cluster-integrated behavior is tested by a
syntax check.

### Compile and run pipelines

The existing compiler entry points are:

```bash
python pipelines/hello_pipeline.py
python pipelines/cats_dogs_final_pipeline.py
```

They write generated YAML under `pipelines/compiled/`. For validation that
respects the generated-directory rule, compile to `/tmp` instead:

```bash
python -c "from kfp import compiler; from pipelines.hello_pipeline import hello_pipeline; compiler.Compiler().compile(pipeline_func=hello_pipeline,package_path='/tmp/hello_pipeline.yaml')"
python -c "from kfp import compiler; from pipelines.cats_dogs_final_pipeline import cats_dogs_final_pipeline; compiler.Compiler().compile(pipeline_func=cats_dogs_final_pipeline,package_path='/tmp/cats_dogs_final_pipeline.yaml')"
```

With KFP reachable at `KFP_ENDPOINT` and an existing compiled Hello package,
the repository's sample submission command is:

```bash
python pipelines/run_hello_pipeline.py
```

Pipeline compilation proves that the DSL is valid; it does not prove that
images, secrets, datasets, Katib, KFP, or MLflow are available in K3s.
