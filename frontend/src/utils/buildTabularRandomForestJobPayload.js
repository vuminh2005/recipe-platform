import {
  toBoolean,
  toInteger,
  valueOrDefault,
} from "./formValues.js";
import { validateRecipeDefinition } from "./recipeCatalog.js";

function assertTabularRecipe(recipe) {
  const validationError = validateRecipeDefinition(recipe);
  if (
    recipe?.recipe_id !== "tabular-random-forest" ||
    !recipe.version ||
    validationError
  ) {
    throw new Error(
      "A valid Tabular Random Forest catalog definition is required.",
    );
  }
}

export function buildTabularRandomForestJobPayload(form, recipe) {
  assertTabularRecipe(recipe);

  const configuration = form.configuration || {};
  const training = configuration.training || {};
  const automl = configuration.automl || {};
  const searchSpace = automl.search_space || {};
  const defaults = recipe.default_configuration;

  return {
    name: String(form.name || "").trim(),
    recipe_id: recipe.recipe_id,
    recipe_version: recipe.version,
    configuration: {
      training: {
        random_seed: toInteger(
          valueOrDefault(
            training.random_seed,
            defaults.training.random_seed,
          ),
          "Random seed",
        ),
      },
      automl: {
        enabled: toBoolean(
          valueOrDefault(automl.enabled, defaults.automl.enabled),
        ),
        max_trials: toInteger(
          valueOrDefault(automl.max_trials, defaults.automl.max_trials),
          "Maximum trials",
        ),
        parallel_trials: toInteger(
          valueOrDefault(
            automl.parallel_trials,
            defaults.automl.parallel_trials,
          ),
          "Parallel trials",
        ),
        algorithm: String(
          valueOrDefault(automl.algorithm, defaults.automl.algorithm),
        ),
        search_space: {
          n_estimators: {
            min: toInteger(
              valueOrDefault(
                searchSpace.n_estimators?.min,
                defaults.automl.search_space.n_estimators.min,
              ),
              "Tree count minimum",
            ),
            max: toInteger(
              valueOrDefault(
                searchSpace.n_estimators?.max,
                defaults.automl.search_space.n_estimators.max,
              ),
              "Tree count maximum",
            ),
          },
          max_depth: {
            min: toInteger(
              valueOrDefault(
                searchSpace.max_depth?.min,
                defaults.automl.search_space.max_depth.min,
              ),
              "Maximum tree depth minimum",
            ),
            max: toInteger(
              valueOrDefault(
                searchSpace.max_depth?.max,
                defaults.automl.search_space.max_depth.max,
              ),
              "Maximum tree depth maximum",
            ),
          },
          min_samples_split: {
            min: toInteger(
              valueOrDefault(
                searchSpace.min_samples_split?.min,
                defaults.automl.search_space.min_samples_split.min,
              ),
              "Minimum samples per split minimum",
            ),
            max: toInteger(
              valueOrDefault(
                searchSpace.min_samples_split?.max,
                defaults.automl.search_space.min_samples_split.max,
              ),
              "Minimum samples per split maximum",
            ),
          },
        },
      },
    },
  };
}
