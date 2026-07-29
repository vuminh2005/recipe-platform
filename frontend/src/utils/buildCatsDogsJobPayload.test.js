import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCatsDogsJobPayload,
  toPositiveInteger,
} from "./buildCatsDogsJobPayload.js";

test("builds the current Cats & Dogs API payload", () => {
  const payload = buildCatsDogsJobPayload({
    name: "  cats-dogs-recipe  ",
    imageSize: "224",
    trialEpochs: "2",
    finalEpochs: "5",
    batchSize: "8",
    denseUnits: "128",
    trainableBackbone: true,
    automlEnabled: false,
    maxTrials: "3",
    parallelTrials: "1",
    algorithm: "random",
  });

  assert.deepEqual(payload, {
    name: "cats-dogs-recipe",
    workload: "cats-dogs",
    training: {
      model: "mobilenet_v2",
      image_size: 224,
      trial_epochs: 2,
      final_epochs: 5,
      batch_size: 8,
      dense_units: 128,
      trainable_backbone: true,
    },
    automl: {
      enabled: false,
      max_trials: 3,
      parallel_trials: 1,
      algorithm: "random",
    },
  });
});

test("keeps the existing invalid-number sentinel", () => {
  assert.equal(toPositiveInteger("not-a-number"), 0);
});
