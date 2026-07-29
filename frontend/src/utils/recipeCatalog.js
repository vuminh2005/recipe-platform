const SUPPORTED_RENDERER_IDS = new Set([
  "cats-dogs",
  "tabular-random-forest",
]);

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

  return "";
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
  const seenIds = new Set();

  for (const recipe of payload) {
    if (recipe?.visibility !== "public") {
      continue;
    }

    const reason = validateRecipeDefinition(recipe);
    const label = recipe?.recipe_id || "unknown recipe";

    if (reason) {
      issues.push(`${label}: ${reason}`);
      continue;
    }

    if (seenIds.has(recipe.recipe_id)) {
      issues.push(`${recipe.recipe_id}: Duplicate recipe definition.`);
      continue;
    }

    seenIds.add(recipe.recipe_id);
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
