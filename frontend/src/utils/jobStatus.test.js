import assert from "node:assert/strict";
import test from "node:test";

import {
  getTimelineState,
  getTimelineSteps,
  inferFailureStage,
} from "./jobStatus.js";

function job({
  recipeId = "cats-dogs",
  automl = true,
  status = "FAILED",
  agentId = "agent-1",
  result = null,
} = {}) {
  return {
    status,
    agent_id: agentId,
    recipe: {
      recipe_id: recipeId,
      configuration: { automl: { enabled: automl } },
    },
    result,
  };
}

function result(externalIds = {}, values = {}) {
  return {
    objective: null,
    best_params: values.best_params ?? null,
    final_metrics: values.final_metrics ?? null,
    external_ids: externalIds,
    model: values.model ?? null,
  };
}

test("timeline omits tuning when AutoML is disabled", () => {
  assert.deepEqual(getTimelineSteps(job({ automl: false, status: "PENDING" })), [
    "PENDING",
    "CLAIMED",
    "TRAINING",
    "REGISTERING",
    "SUCCEEDED",
  ]);
});

test("Hello uses only its KFP smoke lifecycle", () => {
  assert.deepEqual(
    getTimelineSteps(job({ recipeId: "hello", status: "RUNNING" })),
    ["PENDING", "CLAIMED", "RUNNING", "SUCCEEDED"],
  );
});

test("failure before Katib is conservatively assigned to claimed", () => {
  const failed = job();
  const steps = getTimelineSteps(failed);

  assert.equal(inferFailureStage(failed, steps), "CLAIMED");
  assert.equal(getTimelineState("PENDING", failed, steps), "complete");
  assert.equal(getTimelineState("CLAIMED", failed, steps), "failed");
  assert.equal(getTimelineState("TUNING", failed, steps), "upcoming");
});

test("failure without claim evidence does not complete pending", () => {
  const failed = job({ agentId: null });
  const steps = getTimelineSteps(failed);

  assert.equal(inferFailureStage(failed, steps), "PENDING");
  assert.equal(getTimelineState("PENDING", failed, steps), "failed");
  assert.equal(getTimelineState("CLAIMED", failed, steps), "upcoming");
});

test("failure during or after Katib does not complete training", () => {
  const failed = job({
    result: result({ katib_experiment_id: "katib-1" }),
  });
  const steps = getTimelineSteps(failed);

  assert.equal(inferFailureStage(failed, steps), "TUNING");
  assert.equal(getTimelineState("TUNING", failed, steps), "failed");
  assert.equal(getTimelineState("TRAINING", failed, steps), "upcoming");
});

test("failure after KFP submission reaches training only", () => {
  const failed = job({
    result: result({
      katib_experiment_id: "katib-1",
      kfp_run_id: "kfp-1",
    }),
  });
  const steps = getTimelineSteps(failed);

  assert.equal(inferFailureStage(failed, steps), "TRAINING");
  assert.equal(getTimelineState("TUNING", failed, steps), "complete");
  assert.equal(getTimelineState("TRAINING", failed, steps), "failed");
  assert.equal(getTimelineState("REGISTERING", failed, steps), "upcoming");
});

test("partial final MLflow/model data reaches registration only", () => {
  const failed = job({
    result: result(
      { kfp_run_id: "kfp-1", mlflow_run_id: "run-1" },
      { model: { registered_name: "partial-model" } },
    ),
  });
  const steps = getTimelineSteps(failed);

  assert.equal(inferFailureStage(failed, steps), "REGISTERING");
  assert.equal(getTimelineState("TRAINING", failed, steps), "complete");
  assert.equal(getTimelineState("REGISTERING", failed, steps), "failed");
  assert.equal(getTimelineState("SUCCEEDED", failed, steps), "upcoming");
});
