import assert from "node:assert/strict";
import test from "node:test";

import {
  formatJsonValue,
  formatMetric,
  formatObjectiveLabel,
} from "./format.js";

test("null metrics never render as zero", () => {
  assert.equal(formatMetric(null), "—");
  assert.equal(formatMetric(undefined), "—");
  assert.equal(formatMetric(""), "—");
});

test("objective labels include direction", () => {
  assert.equal(
    formatObjectiveLabel({ name: "val_f1", direction: "maximize" }),
    "val_f1 ↑",
  );
  assert.equal(
    formatObjectiveLabel({ name: "loss", direction: "minimize" }),
    "loss ↓",
  );
});

test("arbitrary JSON-compatible metric values remain readable", () => {
  assert.equal(formatJsonValue({ split: "test" }), '{"split":"test"}');
  assert.equal(formatJsonValue(["a", 1]), '["a",1]');
});
