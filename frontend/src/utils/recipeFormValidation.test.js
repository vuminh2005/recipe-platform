import assert from "node:assert/strict";
import test from "node:test";

import { createConfigurationFromDefaults } from "./recipeCatalog.js";
import { validateRecipeForm } from "./recipeFormValidation.js";
import { catsDogsRecipe, tabularRecipe } from "./testRecipes.js";

function paths(issues) {
  return issues.map((item) => item.path);
}

test("rejects parallel trials greater than maximum trials", () => {
  const configuration = createConfigurationFromDefaults(catsDogsRecipe);
  configuration.automl.max_trials = "2";
  configuration.automl.parallel_trials = "3";

  assert.ok(
    paths(
      validateRecipeForm(
        { name: "cats-job", configuration },
        catsDogsRecipe,
      ),
    ).includes("configuration.automl.parallel_trials"),
  );
});

test("rejects unsupported algorithm and invalid floating range", () => {
  const configuration = createConfigurationFromDefaults(catsDogsRecipe);
  configuration.automl.algorithm = "tpe";
  configuration.automl.search_space.learning_rate = {
    min: "0.5",
    max: "0.1",
  };
  const issuePaths = paths(
    validateRecipeForm(
      { name: "cats-job", configuration },
      catsDogsRecipe,
    ),
  );

  assert.ok(issuePaths.includes("configuration.automl.algorithm"));
  assert.ok(
    issuePaths.includes(
      "configuration.automl.search_space.learning_rate.min",
    ),
  );
});

test("requires integer Tabular values and respects catalog bounds", () => {
  const configuration = createConfigurationFromDefaults(tabularRecipe);
  configuration.training.random_seed = "-1";
  configuration.automl.search_space.n_estimators.min = "50.5";
  const issuePaths = paths(
    validateRecipeForm(
      { name: "tabular-job", configuration },
      tabularRecipe,
    ),
  );

  assert.ok(issuePaths.includes("configuration.training.random_seed"));
  assert.ok(
    issuePaths.includes(
      "configuration.automl.search_space.n_estimators.min",
    ),
  );
});

test("valid disabled AutoML configuration remains accepted", () => {
  const configuration = createConfigurationFromDefaults(tabularRecipe);
  configuration.automl.enabled = false;

  assert.deepEqual(
    validateRecipeForm(
      { name: "tabular-job", configuration },
      tabularRecipe,
    ),
    [],
  );
});
