function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function snapshotFirst(snapshot, key, ...fallbacks) {
  return Object.hasOwn(snapshot, key)
    ? snapshot[key]
    : firstDefined(...fallbacks);
}

export function getRecipeId(job) {
  const recipe = job?.recipe || {};
  const snapshot = isRecord(recipe.recipe_snapshot)
    ? recipe.recipe_snapshot
    : {};

  return firstDefined(
    snapshot.recipe_id,
    recipe.recipe_id,
    recipe.workload,
    null,
  );
}

export function getRecipeConfiguration(job) {
  const recipe = job?.recipe || {};
  const snapshot = isRecord(recipe.recipe_snapshot)
    ? recipe.recipe_snapshot
    : null;

  if (isRecord(snapshot?.configuration)) {
    return snapshot.configuration;
  }
  if (isRecord(recipe.configuration)) {
    return recipe.configuration;
  }

  const training = recipe.training || recipe.training_config;
  const automl = recipe.automl || recipe.automl_config;

  if (isRecord(training) || isRecord(automl)) {
    return {
      training: isRecord(training) ? training : {},
      automl: isRecord(automl) ? automl : {},
      effective_final_parameters: null,
    };
  }

  return {};
}

export function getRecipeMetadata(job, catalogById = {}) {
  const recipe = job?.recipe || {};
  const snapshot = isRecord(recipe.recipe_snapshot)
    ? recipe.recipe_snapshot
    : {};
  const recipeId = getRecipeId(job);
  const catalog = (recipeId && catalogById?.[recipeId]) || {};
  const configuration = getRecipeConfiguration(job);
  const normalizedTraining = configuration.training || {};
  const legacyTraining = recipe.training || recipe.training_config || {};

  return {
    recipe_id: recipeId,
    recipe_version: snapshotFirst(
      snapshot,
      "recipe_version",
      recipe.recipe_version,
      catalog.version,
      null,
    ),
    display_name: snapshotFirst(
      snapshot,
      "display_name",
      catalog.display_name,
      recipeId,
      "Unknown recipe",
    ),
    description: snapshotFirst(
      snapshot,
      "description",
      catalog.description,
      null,
    ),
    task_type: snapshotFirst(snapshot, "task_type", catalog.task_type, null),
    framework: snapshotFirst(snapshot, "framework", catalog.framework, null),
    model: snapshotFirst(
      snapshot,
      "model",
      normalizedTraining.model,
      catalog.model,
      legacyTraining.model,
      null,
    ),
    supports_automl: snapshotFirst(
      snapshot,
      "supports_automl",
      catalog.supports_automl,
      isRecord(configuration.automl),
      false,
    ),
    objective: snapshotFirst(snapshot, "objective", catalog.objective, null),
  };
}

export function isAutoMLEnabled(job) {
  const configuration = getRecipeConfiguration(job);
  return configuration.automl?.enabled === true;
}

function hasLegacyResult(job) {
  return [
    "best_metric",
    "best_params",
    "katib_experiment_name",
    "kfp_run_id",
    "mlflow_parent_run_id",
    "mlflow_final_run_id",
    "model_uri",
    "registered_model_name",
    "registered_model_version",
    "final_metrics",
  ].some((field) => job?.[field] !== null && job?.[field] !== undefined);
}

function synthesizeLegacyResult(job, catalogById) {
  if (!hasLegacyResult(job)) {
    return null;
  }

  const metadata = getRecipeMetadata(job, catalogById);
  const automlEnabled = isAutoMLEnabled(job);
  const isHello = metadata.recipe_id === "hello";
  const objective = metadata.objective
    ? {
        ...metadata.objective,
        value: automlEnabled ? (job.best_metric ?? null) : null,
      }
    : null;
  const modelValues = {
    uri: job.model_uri ?? null,
    registered_name: job.registered_model_name ?? null,
    version: job.registered_model_version ?? null,
  };
  const hasModel = Object.values(modelValues).some((value) => value !== null);

  return {
    objective: isHello ? null : objective,
    best_params:
      isHello || !automlEnabled ? null : (job.best_params ?? null),
    final_metrics: isHello ? null : (job.final_metrics ?? null),
    external_ids: {
      katib_experiment_id: isHello
        ? null
        : (job.katib_experiment_name ?? null),
      kfp_run_id: job.kfp_run_id ?? null,
      mlflow_parent_run_id: isHello
        ? null
        : (job.mlflow_parent_run_id ?? null),
      mlflow_run_id: isHello ? null : (job.mlflow_final_run_id ?? null),
    },
    model: !isHello && hasModel ? modelValues : null,
  };
}

export function getJobResult(job, catalogById = {}) {
  if (job?.result !== null && job?.result !== undefined) {
    return job.result;
  }

  return synthesizeLegacyResult(job, catalogById);
}

export function getObjective(job, catalogById = {}) {
  if (job?.result !== null && job?.result !== undefined) {
    return job.result.objective ?? null;
  }

  const result = getJobResult(job, catalogById);
  if (result?.objective) {
    return result.objective;
  }

  const definition = getRecipeMetadata(job, catalogById).objective;
  return definition ? { ...definition, value: null } : null;
}

export function getEffectiveFinalParameters(job) {
  const configuration = getRecipeConfiguration(job);
  return isRecord(configuration.effective_final_parameters)
    ? configuration.effective_final_parameters
    : null;
}

export function getJobPresentation(job, catalogById = {}) {
  const configuration = getRecipeConfiguration(job);
  const metadata = getRecipeMetadata(job, catalogById);
  const result = getJobResult(job, catalogById);

  return {
    name: job?.recipe?.name || "Unnamed recipe",
    metadata,
    configuration,
    automlEnabled: configuration.automl?.enabled === true,
    result,
    objective: getObjective(job, catalogById),
    bestParams: result?.best_params ?? null,
    finalMetrics: result?.final_metrics ?? null,
    externalIds: result?.external_ids || {
      katib_experiment_id: null,
      kfp_run_id: null,
      mlflow_parent_run_id: null,
      mlflow_run_id: null,
    },
    model: result?.model ?? null,
    effectiveFinalParameters: getEffectiveFinalParameters(job),
  };
}
