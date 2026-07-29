const SUPPORTED_RENDERER_IDS = new Set([
  "cats-dogs",
  "tabular-random-forest",
]);

const RENDERER_REQUIREMENTS = {
  "cats-dogs": {
    training: [
      "image_size",
      "trial_epochs",
      "final_epochs",
      "batch_size",
      "dense_units",
      "trainable_backbone",
    ],
    automl: ["enabled", "max_trials", "parallel_trials", "algorithm"],
    searchSpace: ["learning_rate", "dropout_rate"],
    effectiveParameters: ["learning_rate", "dropout_rate"],
  },
  "tabular-random-forest": {
    training: ["random_seed"],
    automl: ["enabled", "max_trials", "parallel_trials", "algorithm"],
    searchSpace: ["n_estimators", "max_depth", "min_samples_split"],
    effectiveParameters: [
      "n_estimators",
      "max_depth",
      "min_samples_split",
      "max_features",
      "random_seed",
    ],
  },
};

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function hasString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function validateFieldDefinitions(fields) {
  return (
    Array.isArray(fields) &&
    fields.every(
      (field) =>
        isRecord(field) &&
        hasString(field.name) &&
        hasString(field.label) &&
        hasString(field.type) &&
        typeof field.required === "boolean" &&
        Object.hasOwn(field, "default"),
    )
  );
}

function missingFieldNames(fields, requiredNames) {
  const names = new Set(fields.map((field) => field.name));
  return requiredNames.filter((name) => !names.has(name));
}

function missingObjectKeys(value, requiredNames) {
  if (!isRecord(value)) {
    return [...requiredNames];
  }
  return requiredNames.filter((name) => !Object.hasOwn(value, name));
}

function validateRendererRequirements(recipe) {
  const requirements = RENDERER_REQUIREMENTS[recipe.recipe_id];
  if (!requirements) {
    return "This dashboard version has no renderer for the recipe.";
  }

  if (!hasString(recipe.framework) || !hasString(recipe.model)) {
    return "Framework and model metadata are required by this renderer.";
  }

  const sections = [
    ["training", recipe.configurable_training_fields, requirements.training],
    ["AutoML", recipe.configurable_automl_fields, requirements.automl],
    [
      "search-space",
      recipe.configurable_search_space,
      requirements.searchSpace,
    ],
  ];
  for (const [label, fields, requiredNames] of sections) {
    const missing = missingFieldNames(fields, requiredNames);
    if (missing.length > 0) {
      return `Missing required ${label} metadata: ${missing.join(", ")}.`;
    }
  }

  const defaults = recipe.default_configuration;
  const missingTraining = missingObjectKeys(
    defaults.training,
    requirements.training,
  );
  const missingAutoml = missingObjectKeys(defaults.automl, requirements.automl);
  const missingSearch = missingObjectKeys(
    defaults.automl?.search_space,
    requirements.searchSpace,
  );
  const missingEffective = missingObjectKeys(
    defaults.effective_final_parameters,
    requirements.effectiveParameters,
  );
  if (
    missingTraining.length ||
    missingAutoml.length ||
    missingSearch.length ||
    missingEffective.length
  ) {
    return "Default configuration is incomplete for this renderer.";
  }

  const algorithm = defaults.automl.algorithm;
  const algorithmField = recipe.configurable_automl_fields.find(
    (field) => field.name === "algorithm",
  );
  const optionValues = new Set(
    (algorithmField?.options || []).map((option) => option.value),
  );
  if (
    !recipe.supported_algorithms.includes(algorithm) ||
    !optionValues.has(algorithm)
  ) {
    return "Algorithm defaults and options are inconsistent.";
  }

  return "";
}

export function validateRecipeDefinition(recipe) {
  if (!isRecord(recipe)) {
    return "Recipe definition must be an object.";
  }

  const requiredStrings = [
    "recipe_id",
    "version",
    "display_name",
    "description",
    "visibility",
    "task_type",
  ];
  const missing = requiredStrings.filter((field) => !hasString(recipe[field]));

  if (missing.length > 0) {
    return `Missing required metadata: ${missing.join(", ")}.`;
  }

  if (recipe.visibility !== "public") {
    return "Recipe is not public.";
  }

  if (!SUPPORTED_RENDERER_IDS.has(recipe.recipe_id)) {
    return "This dashboard version has no renderer for the recipe.";
  }

  if (
    !isRecord(recipe.default_configuration) ||
    !isRecord(recipe.default_configuration.training) ||
    !isRecord(recipe.default_configuration.automl)
  ) {
    return "Default training and AutoML configuration is required.";
  }

  if (
    typeof recipe.supports_automl !== "boolean" ||
    !Array.isArray(recipe.supported_algorithms)
  ) {
    return "AutoML capability metadata is malformed.";
  }

  if (
    !isRecord(recipe.objective) ||
    !hasString(recipe.objective.name) ||
    !["maximize", "minimize"].includes(recipe.objective.direction)
  ) {
    return "Objective metadata is malformed.";
  }

  if (
    !validateFieldDefinitions(recipe.configurable_training_fields) ||
    !validateFieldDefinitions(recipe.configurable_automl_fields) ||
    !validateFieldDefinitions(recipe.configurable_search_space)
  ) {
    return "Configurable field metadata is malformed.";
  }

  return validateRendererRequirements(recipe);
}

export function prepareRecipeCatalog(payload) {
  if (!Array.isArray(payload)) {
    return {
      recipes: [],
      issues: ["Recipe Catalog response must be an array."],
    };
  }

  const recipes = [];
  const issues = [];
  const publicRecipes = payload.filter(
    (recipe) => recipe?.visibility === "public",
  );
  const identityCounts = new Map();
  for (const recipe of publicRecipes) {
    if (hasString(recipe?.recipe_id)) {
      identityCounts.set(
        recipe.recipe_id,
        (identityCounts.get(recipe.recipe_id) || 0) + 1,
      );
    }
  }
  const duplicateIds = new Set(
    [...identityCounts.entries()]
      .filter(([, count]) => count > 1)
      .map(([recipeId]) => recipeId),
  );
  const reportedDuplicates = new Set();

  for (const recipe of publicRecipes) {
    if (duplicateIds.has(recipe?.recipe_id)) {
      if (!reportedDuplicates.has(recipe.recipe_id)) {
        issues.push(`${recipe.recipe_id}: Duplicate recipe definition.`);
        reportedDuplicates.add(recipe.recipe_id);
      }
      continue;
    }

    const reason = validateRecipeDefinition(recipe);
    const label = recipe?.recipe_id || "unknown recipe";

    if (reason) {
      issues.push(`${label}: ${reason}`);
      continue;
    }

    recipes.push(recipe);
  }

  return { recipes, issues };
}

export function getDefaultRecipe(recipes) {
  return (
    recipes.find((recipe) => recipe.recipe_id === "cats-dogs") ||
    recipes[0] ||
    null
  );
}

export function createConfigurationFromDefaults(recipe) {
  return {
    training: cloneJson(recipe.default_configuration.training),
    automl: cloneJson(recipe.default_configuration.automl),
  };
}

export function getFieldDefinition(recipe, section, fieldName) {
  const fields = recipe?.[section];
  return Array.isArray(fields)
    ? fields.find((field) => field.name === fieldName) || null
    : null;
}

export function getEffectiveParametersForDisplay(recipe, configuration) {
  const defaults = recipe?.default_configuration?.effective_final_parameters;

  if (!isRecord(defaults)) {
    return null;
  }

  const effective = cloneJson(defaults);
  const training = configuration?.training;

  if (isRecord(training)) {
    for (const key of Object.keys(effective)) {
      if (Object.hasOwn(training, key)) {
        effective[key] = training[key];
      }
    }
  }

  return effective;
}

export function indexRecipes(recipes) {
  return Object.fromEntries(
    (recipes || []).map((recipe) => [recipe.recipe_id, recipe]),
  );
}
