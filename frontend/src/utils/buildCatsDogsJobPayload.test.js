import assert from "node:assert/strict";
import test from "node:test";

import { buildCatsDogsJobPayload } from "./buildCatsDogsJobPayload.js";
import { catsDogsRecipe } from "./testRecipes.js";

test("builds the canonical Cats & Dogs payload and preserves search ranges", () => {
  const payload = buildCatsDogsJobPayload(
    {
      name: "  cats-job  ",
      configuration: {
        training: {
          image_size: "256",
          trial_epochs: "3",
          final_epochs: "8",
          batch_size: "16",
          dense_units: "192",
          trainable_backbone: "true",
          random_seed: 999,
        },
        automl: {
          enabled: "true",
          max_trials: "6",
          parallel_trials: "2",
          algorithm: "random",
          search_space: {
            learning_rate: { min: "0.0001", max: "0.0004" },
            dropout_rate: { min: "0.2", max: "0.4" },
            n_estimators: { min: 50, max: 300 },
          },
        },
      },
      recipe_snapshot: { forbidden: true },
      result: { forbidden: true },
    },
    catsDogsRecipe,
  );

  assert.deepEqual(payload, {
    name: "cats-job",
    recipe_id: "cats-dogs",
    recipe_version: "1.0",
    configuration: {
      training: {
        image_size: 256,
        trial_epochs: 3,
        final_epochs: 8,
        batch_size: 16,
        dense_units: 192,
        trainable_backbone: true,
      },
      automl: {
        enabled: true,
        max_trials: 6,
        parallel_trials: 2,
        algorithm: "random",
        search_space: {
          learning_rate: { min: 0.0001, max: 0.0004 },
          dropout_rate: { min: 0.2, max: 0.4 },
        },
      },
    },
  });
});

test("disabled Cats & Dogs payload contains no result or effective fields", () => {
  const configuration = structuredClone(
    catsDogsRecipe.default_configuration,
  );
  configuration.automl.enabled = false;

  const payload = buildCatsDogsJobPayload(
    { name: "direct-cats", configuration },
    catsDogsRecipe,
  );

  assert.equal(payload.configuration.automl.enabled, false);
  assert.equal("effective_final_parameters" in payload.configuration, false);
  assert.equal("best_params" in payload, false);
  assert.equal("objective" in payload, false);
});
