import assert from "node:assert/strict";
import test from "node:test";

import {
  getJobPresentation,
  getJobResult,
  getRecipeConfiguration,
  getRecipeMetadata,
} from "./jobPresentation.js";
import { catsDogsRecipe, tabularRecipe } from "./testRecipes.js";

test("historical snapshot metadata and configuration outrank current catalog", () => {
  const snapshotConfiguration = {
    training: { image_size: 160 },
    automl: { enabled: false },
    effective_final_parameters: {
      learning_rate: 0.0002,
      dropout_rate: 0.2,
    },
  };
  const job = {
    recipe: {
      recipe_id: "cats-dogs",
      recipe_version: "9.0",
      configuration: {
        training: { image_size: 999 },
        automl: { enabled: true },
      },
      recipe_snapshot: {
        recipe_id: "cats-dogs",
        recipe_version: "0.9",
        display_name: "Historical Cats",
        model: "historical_model",
        objective: { name: "old_auc", direction: "maximize" },
        configuration: snapshotConfiguration,
      },
      training: { image_size: 224, model: "legacy_model" },
    },
  };

  const metadata = getRecipeMetadata(job, {
    "cats-dogs": catsDogsRecipe,
  });
  assert.equal(metadata.display_name, "Historical Cats");
  assert.equal(metadata.recipe_version, "0.9");
  assert.equal(metadata.model, "historical_model");
  assert.equal(getRecipeConfiguration(job), snapshotConfiguration);
});

test("normalized configuration is used before legacy fields", () => {
  const job = {
    recipe: {
      recipe_id: "cats-dogs",
      configuration: {
        training: { image_size: 320, model: "normalized-model" },
        automl: { enabled: false },
      },
      training: { image_size: 224 },
      automl: { enabled: true },
    },
  };

  assert.equal(getRecipeConfiguration(job).training.image_size, 320);
  assert.equal(
    getRecipeMetadata(job, {
      "cats-dogs": { ...catsDogsRecipe, model: "current-catalog-model" },
    }).model,
    "normalized-model",
  );
});

test("non-null canonical result is never filled from legacy fields", () => {
  const job = {
    recipe: {
      recipe_id: "cats-dogs",
      configuration: { automl: { enabled: true } },
    },
    result: {
      objective: { name: "val_auc", direction: "maximize", value: null },
      best_params: null,
      final_metrics: null,
      external_ids: { kfp_run_id: "canonical-kfp" },
      model: null,
    },
    best_metric: 0.99,
    best_params: { legacy: true },
    registered_model_name: "legacy-model",
  };

  const result = getJobResult(job, { "cats-dogs": catsDogsRecipe });
  assert.equal(result.objective.value, null);
  assert.equal(result.best_params, null);
  assert.equal(result.external_ids.kfp_run_id, "canonical-kfp");
  assert.equal(result.model, null);
});

test("canonical null objective is not replaced from catalog metadata", () => {
  const job = {
    recipe: {
      recipe_id: "cats-dogs",
      configuration: { automl: { enabled: true } },
    },
    result: {
      objective: null,
      best_params: null,
      final_metrics: null,
      external_ids: {},
      model: null,
    },
    best_metric: 0.99,
  };

  assert.equal(
    getJobPresentation(job, { "cats-dogs": catsDogsRecipe }).objective,
    null,
  );
});

test("legacy fields synthesize a result only when canonical result is absent", () => {
  const job = {
    recipe: {
      workload: "cats-dogs",
      automl: { enabled: true },
    },
    best_metric: 0.91,
    best_params: { arbitrary: ["json", 1, true] },
    katib_experiment_name: "legacy-katib",
  };

  const result = getJobResult(job, { "cats-dogs": catsDogsRecipe });
  assert.equal(result.objective.value, 0.91);
  assert.deepEqual(result.best_params.arbitrary, ["json", 1, true]);
  assert.equal(result.external_ids.katib_experiment_id, "legacy-katib");
});

test("AutoML-disabled result is not tuned and uses effective configuration", () => {
  const configuration = {
    training: { random_seed: 7 },
    automl: { enabled: false },
    effective_final_parameters: {
      n_estimators: 200,
      max_depth: 8,
      min_samples_split: 2,
      max_features: "sqrt",
      random_seed: 7,
    },
  };
  const job = {
    recipe: {
      recipe_id: "tabular-random-forest",
      configuration,
      recipe_snapshot: {
        recipe_id: "tabular-random-forest",
        objective: { name: "val_f1", direction: "maximize" },
        configuration,
      },
    },
    result: {
      objective: { name: "val_f1", direction: "maximize", value: null },
      best_params: null,
      final_metrics: { test_f1: 0.94, context: { split: "test" } },
      external_ids: { kfp_run_id: "tabular-kfp" },
      model: null,
    },
  };

  const presentation = getJobPresentation(job, {
    "tabular-random-forest": tabularRecipe,
  });
  assert.equal(presentation.objective.value, null);
  assert.equal(presentation.bestParams, null);
  assert.equal(presentation.effectiveFinalParameters.random_seed, 7);
  assert.deepEqual(presentation.finalMetrics.context, { split: "test" });
});

test("Hello result remains KFP-only", () => {
  const job = {
    recipe: {
      recipe_id: "hello",
      recipe_snapshot: {
        recipe_id: "hello",
        objective: null,
        configuration: {},
      },
    },
    result: {
      objective: null,
      best_params: null,
      final_metrics: null,
      external_ids: { kfp_run_id: "hello-kfp" },
      model: null,
    },
  };

  const presentation = getJobPresentation(job);
  assert.equal(presentation.objective, null);
  assert.equal(presentation.externalIds.kfp_run_id, "hello-kfp");
  assert.equal(presentation.model, null);
});
