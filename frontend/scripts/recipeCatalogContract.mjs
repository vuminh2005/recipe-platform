import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { prepareRecipeCatalog } from "../src/utils/recipeCatalog.js";

const frontendRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const repositoryRoot = path.dirname(frontendRoot);
const python = process.env.RECIPE_PLATFORM_PYTHON || "python3";
const source = [
  "import json",
  "from backend.app.recipe_catalog import list_public_recipes",
  "print(json.dumps([item.model_dump(mode='json') for item in list_public_recipes()]))",
].join("; ");

const output = execFileSync(python, ["-c", source], {
  cwd: repositoryRoot,
  encoding: "utf8",
});
const prepared = prepareRecipeCatalog(JSON.parse(output));

assert.deepEqual(prepared.issues, []);
assert.deepEqual(
  prepared.recipes.map((recipe) => recipe.recipe_id),
  ["cats-dogs", "tabular-random-forest"],
);
assert.equal(
  prepared.recipes.some((recipe) => recipe.recipe_id === "hello"),
  false,
);

const cats = prepared.recipes.find((recipe) => recipe.recipe_id === "cats-dogs");
const tabular = prepared.recipes.find(
  (recipe) => recipe.recipe_id === "tabular-random-forest",
);

assert.equal(cats.version, "1.0");
assert.equal(cats.objective.name, "val_auc");
assert.equal(cats.default_configuration.training.image_size, 224);
assert.equal(
  cats.default_configuration.effective_final_parameters.learning_rate,
  0.0003,
);
assert.equal(tabular.version, "1.0");
assert.equal(tabular.objective.name, "val_f1");
assert.equal(tabular.default_configuration.training.random_seed, 42);
assert.equal(
  tabular.default_configuration.effective_final_parameters.n_estimators,
  200,
);

console.log(
  "Recipe Catalog contract passed for cats-dogs and tabular-random-forest.",
);
