import assert from "node:assert/strict";
import test from "node:test";

import { parseApiError } from "./client.js";

test("preserves nested FastAPI validation paths", () => {
  const parsed = parseApiError(
    {
      detail: [
        {
          loc: [
            "body",
            "configuration",
            "automl",
            "search_space",
            "n_estimators",
            "min",
          ],
          msg: "Input should be greater than or equal to 1",
        },
      ],
    },
    422,
  );

  assert.deepEqual(parsed.issues, [
    {
      path: "configuration.automl.search_space.n_estimators.min",
      message: "Input should be greater than or equal to 1",
    },
  ]);
});

test("keeps backend normalization string details readable", () => {
  const parsed = parseApiError(
    { detail: "parallel_trials must be less than or equal to max_trials" },
    422,
  );

  assert.equal(
    parsed.message,
    "parallel_trials must be less than or equal to max_trials",
  );
});

test("adapts multiple structured recipe validation issues", () => {
  const parsed = parseApiError(
    {
      detail: [
        {
          loc: ["body", "configuration", "training", "random_seed"],
          msg: "Input should be greater than or equal to 0",
        },
        {
          loc: [
            "body",
            "configuration",
            "automl",
            "parallel_trials",
          ],
          msg: "parallel_trials must be less than or equal to max_trials",
        },
      ],
    },
    422,
  );

  assert.deepEqual(
    parsed.issues.map((issue) => issue.path),
    [
      "configuration.training.random_seed",
      "configuration.automl.parallel_trials",
    ],
  );
});
