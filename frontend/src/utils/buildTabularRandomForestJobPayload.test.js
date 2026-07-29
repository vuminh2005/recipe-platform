import assert from "node:assert/strict";
import test from "node:test";

import { buildTabularRandomForestJobPayload } from "./buildTabularRandomForestJobPayload.js";
import { tabularRecipe } from "./testRecipes.js";

test("builds a canonical typed Tabular payload without Cats fields", () => {
  const configuration = structuredClone(tabularRecipe.default_configuration);
  configuration.training.random_seed = "7";
  configuration.automl.enabled = "false";
  configuration.automl.max_trials = "5";
  configuration.automl.search_space.n_estimators = {
    min: "80",
    max: "240",
  };
  configuration.training.image_size = 224;

  const payload = buildTabularRandomForestJobPayload(
    { name: " tabular-job ", configuration },
    tabularRecipe,
  );

  assert.deepEqual(payload, {
    name: "tabular-job",
    recipe_id: "tabular-random-forest",
    recipe_version: "1.0",
    configuration: {
      training: { random_seed: 7 },
      automl: {
        enabled: false,
        max_trials: 5,
        parallel_trials: 1,
        algorithm: "random",
        search_space: {
          n_estimators: { min: 80, max: 240 },
          max_depth: { min: 2, max: 20 },
          min_samples_split: { min: 2, max: 10 },
        },
      },
    },
  });
});

test("rejects non-integer Tabular search endpoints", () => {
  const configuration = structuredClone(tabularRecipe.default_configuration);
  configuration.automl.search_space.max_depth.min = "2.5";

  assert.throws(
    () =>
      buildTabularRandomForestJobPayload(
        { name: "tabular-job", configuration },
        tabularRecipe,
      ),
    /must be an integer/,
  );
});

test("rejects an incomplete Tabular catalog definition", () => {
  const incomplete = structuredClone(tabularRecipe);
  incomplete.configurable_automl_fields =
    incomplete.configurable_automl_fields.filter(
      (field) => field.name !== "algorithm",
    );

  assert.throws(
    () =>
      buildTabularRandomForestJobPayload(
        {
          name: "invalid-catalog",
          configuration: structuredClone(incomplete.default_configuration),
        },
        incomplete,
      ),
    /valid Tabular Random Forest catalog definition/,
  );
});
