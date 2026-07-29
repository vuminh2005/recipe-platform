import {
  toBoolean,
  toInteger,
  toNumber,
  valueOrDefault,
} from "./formValues.js";

function assertCatsDogsRecipe(recipe) {
  if (recipe?.recipe_id !== "cats-dogs" || !recipe.version) {
    throw new Error("A valid Cats & Dogs catalog definition is required.");
  }
}

export function buildCatsDogsJobPayload(form, recipe) {
  assertCatsDogsRecipe(recipe);

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
        image_size: toInteger(
          valueOrDefault(
            training.image_size,
            defaults.training.image_size,
          ),
          "Image size",
        ),
        trial_epochs: toInteger(
          valueOrDefault(
            training.trial_epochs,
            defaults.training.trial_epochs,
          ),
          "Trial epochs",
        ),
        final_epochs: toInteger(
          valueOrDefault(
            training.final_epochs,
            defaults.training.final_epochs,
          ),
          "Final epochs",
        ),
        batch_size: toInteger(
          valueOrDefault(
            training.batch_size,
            defaults.training.batch_size,
          ),
          "Batch size",
        ),
        dense_units: toInteger(
          valueOrDefault(
            training.dense_units,
            defaults.training.dense_units,
          ),
          "Dense units",
        ),
        trainable_backbone: toBoolean(
          valueOrDefault(
            training.trainable_backbone,
            defaults.training.trainable_backbone,
          ),
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
          learning_rate: {
            min: toNumber(
              valueOrDefault(
                searchSpace.learning_rate?.min,
                defaults.automl.search_space.learning_rate.min,
              ),
              "Learning rate minimum",
            ),
            max: toNumber(
              valueOrDefault(
                searchSpace.learning_rate?.max,
                defaults.automl.search_space.learning_rate.max,
              ),
              "Learning rate maximum",
            ),
          },
          dropout_rate: {
            min: toNumber(
              valueOrDefault(
                searchSpace.dropout_rate?.min,
                defaults.automl.search_space.dropout_rate.min,
              ),
              "Dropout rate minimum",
            ),
            max: toNumber(
              valueOrDefault(
                searchSpace.dropout_rate?.max,
                defaults.automl.search_space.dropout_rate.max,
              ),
              "Dropout rate maximum",
            ),
          },
        },
      },
    },
  };
}
