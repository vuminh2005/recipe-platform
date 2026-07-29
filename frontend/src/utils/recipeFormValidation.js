function issue(path, message) {
  return { path, message };
}

function findField(recipe, collection, name) {
  return (
    recipe?.[collection]?.find((field) => field.name === name) || null
  );
}

function validateScalar(value, field, path) {
  const issues = [];
  const isMissing = value === "" || value === null || value === undefined;

  if (isMissing) {
    if (field.required) {
      issues.push(issue(path, `${field.label} is required.`));
    }
    return issues;
  }

  if (field.type === "boolean") {
    if (
      typeof value !== "boolean" &&
      !["true", "false", "1", "0"].includes(String(value))
    ) {
      issues.push(issue(path, `${field.label} must be true or false.`));
    }
    return issues;
  }

  if (field.type === "string") {
    const allowed = (field.options || []).map((option) => String(option.value));
    if (allowed.length > 0 && !allowed.includes(String(value))) {
      issues.push(issue(path, `${field.label} is not supported.`));
    }
    return issues;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    issues.push(issue(path, `${field.label} must be a number.`));
    return issues;
  }

  if (field.type === "integer" && !Number.isInteger(numeric)) {
    issues.push(issue(path, `${field.label} must be an integer.`));
  }

  if (
    field.minimum !== null &&
    field.minimum !== undefined &&
    (field.exclusive_minimum
      ? numeric <= field.minimum
      : numeric < field.minimum)
  ) {
    const operator = field.exclusive_minimum ? "greater than" : "at least";
    issues.push(
      issue(path, `${field.label} must be ${operator} ${field.minimum}.`),
    );
  }

  if (
    field.maximum !== null &&
    field.maximum !== undefined &&
    (field.exclusive_maximum
      ? numeric >= field.maximum
      : numeric > field.maximum)
  ) {
    const operator = field.exclusive_maximum ? "less than" : "at most";
    issues.push(
      issue(path, `${field.label} must be ${operator} ${field.maximum}.`),
    );
  }

  return issues;
}

function rangeEndpointType(recipeId) {
  return recipeId === "tabular-random-forest" ? "integer" : "number";
}

export function validateRecipeForm(form, recipe) {
  const issues = [];
  const name = String(form.name || "").trim();

  if (name.length < 3) {
    issues.push(issue("name", "Job name must contain at least 3 characters."));
  } else if (name.length > 100) {
    issues.push(issue("name", "Job name must contain at most 100 characters."));
  }

  const configuration = form.configuration || {};
  const training = configuration.training || {};
  const automl = configuration.automl || {};

  for (const field of recipe.configurable_training_fields || []) {
    issues.push(
      ...validateScalar(
        training[field.name],
        field,
        `configuration.training.${field.name}`,
      ),
    );
  }

  for (const field of recipe.configurable_automl_fields || []) {
    issues.push(
      ...validateScalar(
        automl[field.name],
        field,
        `configuration.automl.${field.name}`,
      ),
    );
  }

  if (
    Number.isFinite(Number(automl.parallel_trials)) &&
    Number.isFinite(Number(automl.max_trials)) &&
    Number(automl.parallel_trials) > Number(automl.max_trials)
  ) {
    issues.push(
      issue(
        "configuration.automl.parallel_trials",
        "Parallel trials must not exceed maximum trials.",
      ),
    );
  }

  if (
    !recipe.supported_algorithms.includes(String(automl.algorithm || ""))
  ) {
    issues.push(
      issue(
        "configuration.automl.algorithm",
        "The selected AutoML algorithm is not supported.",
      ),
    );
  }

  const endpointType = rangeEndpointType(recipe.recipe_id);
  for (const field of recipe.configurable_search_space || []) {
    const range = automl.search_space?.[field.name] || {};
    const endpointField = { ...field, type: endpointType };
    const path = `configuration.automl.search_space.${field.name}`;

    issues.push(
      ...validateScalar(range.min, endpointField, `${path}.min`),
      ...validateScalar(range.max, endpointField, `${path}.max`),
    );

    const minimum = Number(range.min);
    const maximum = Number(range.max);
    if (
      Number.isFinite(minimum) &&
      Number.isFinite(maximum) &&
      minimum >= maximum
    ) {
      issues.push(
        issue(`${path}.min`, `${field.label} minimum must be less than maximum.`),
      );
    }
  }

  const enabledField = findField(
    recipe,
    "configurable_automl_fields",
    "enabled",
  );
  if (recipe.supports_automl && !enabledField) {
    issues.push(
      issue("configuration.automl.enabled", "AutoML metadata is incomplete."),
    );
  }

  return issues;
}
