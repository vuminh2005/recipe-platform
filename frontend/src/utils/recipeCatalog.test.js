import assert from "node:assert/strict";
import test from "node:test";

import {
  createConfigurationFromDefaults,
  getDefaultRecipe,
  getEffectiveParametersForDisplay,
  prepareRecipeCatalog,
} from "./recipeCatalog.js";
import { catsDogsRecipe, tabularRecipe } from "./testRecipes.js";

test("accepts both supported public recipes and excludes Hello", () => {
  const hello = {
    ...catsDogsRecipe,
    recipe_id: "hello",
    visibility: "internal",
  };
  const prepared = prepareRecipeCatalog([
    catsDogsRecipe,
    hello,
    tabularRecipe,
  ]);

  assert.deepEqual(
    prepared.recipes.map((recipe) => recipe.recipe_id),
    ["cats-dogs", "tabular-random-forest"],
  );
  assert.deepEqual(prepared.issues, []);
});

test("malformed and unsupported catalog entries are not selectable", () => {
  const unknown = { ...catsDogsRecipe, recipe_id: "future-recipe" };
  const malformed = { ...tabularRecipe, objective: null };
  const prepared = prepareRecipeCatalog([unknown, malformed]);

  assert.deepEqual(prepared.recipes, []);
  assert.equal(prepared.issues.length, 2);
});

test("rejects a malformed non-array catalog response", () => {
  const prepared = prepareRecipeCatalog({ recipes: [catsDogsRecipe] });

  assert.deepEqual(prepared.recipes, []);
  assert.match(prepared.issues[0], /must be an array/);
});

test("defaults to Cats & Dogs only when returned by the catalog", () => {
  assert.equal(
    getDefaultRecipe([tabularRecipe, catsDogsRecipe]).recipe_id,
    "cats-dogs",
  );
  assert.equal(
    getDefaultRecipe([tabularRecipe]).recipe_id,
    "tabular-random-forest",
  );
});

test("switching recipes creates a fresh configuration without stale fields", () => {
  const cats = createConfigurationFromDefaults(catsDogsRecipe);
  cats.training.image_size = 512;
  const tabular = createConfigurationFromDefaults(tabularRecipe);

  assert.deepEqual(tabular.training, { random_seed: 42 });
  assert.equal("image_size" in tabular.training, false);
  assert.equal("learning_rate" in tabular.automl.search_space, false);
  assert.equal("effective_final_parameters" in tabular, false);
});

test("effective Tabular display defaults use the selected random seed", () => {
  const configuration = createConfigurationFromDefaults(tabularRecipe);
  configuration.training.random_seed = "17";

  assert.deepEqual(
    getEffectiveParametersForDisplay(tabularRecipe, configuration),
    {
      n_estimators: 200,
      max_depth: 8,
      min_samples_split: 2,
      max_features: "sqrt",
      random_seed: "17",
    },
  );
});

test("rejects Cats definitions missing required renderer metadata", () => {
  const cases = [
    ["image_size", "configurable_training_fields"],
    ["algorithm", "configurable_automl_fields"],
    ["learning_rate", "configurable_search_space"],
  ];

  for (const [fieldName, section] of cases) {
    const recipe = structuredClone(catsDogsRecipe);
    recipe[section] = recipe[section].filter(
      (field) => field.name !== fieldName,
    );
    const prepared = prepareRecipeCatalog([recipe]);
    assert.deepEqual(prepared.recipes, [], fieldName);
    assert.equal(prepared.issues.length, 1, fieldName);
  }

  const missingObjective = structuredClone(catsDogsRecipe);
  missingObjective.objective = null;
  assert.deepEqual(prepareRecipeCatalog([missingObjective]).recipes, []);
});

test("rejects every entry for duplicate supported recipe identities", () => {
  for (const recipe of [catsDogsRecipe, tabularRecipe]) {
    const prepared = prepareRecipeCatalog([
      structuredClone(recipe),
      structuredClone(recipe),
    ]);
    assert.deepEqual(prepared.recipes, []);
    assert.equal(prepared.issues.length, 1);
    assert.match(prepared.issues[0], /Duplicate recipe definition/);
  }
});

test("keeps a valid recipe usable alongside an invalid recipe", () => {
  const invalidTabular = structuredClone(tabularRecipe);
  invalidTabular.configurable_search_space =
    invalidTabular.configurable_search_space.filter(
      (field) => field.name !== "max_depth",
    );

  const prepared = prepareRecipeCatalog([catsDogsRecipe, invalidTabular]);
  assert.deepEqual(
    prepared.recipes.map((recipe) => recipe.recipe_id),
    ["cats-dogs"],
  );
  assert.equal(prepared.issues.length, 1);
});
