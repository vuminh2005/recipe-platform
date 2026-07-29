import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import {
  buildCreateJobRequest,
  buildPublicJobReadRequest,
  getJobSubmissionErrorIssues,
  JOB_SUBMISSION_HEADER,
} from "./jobs.js";

const PAYLOAD = {
  name: "frontend-token-contract",
  recipe_id: "cats-dogs",
  recipe_version: "1.0",
  configuration: {
    training: { image_size: 224 },
    automl: { enabled: false },
  },
};

test("adds the runtime token only to the create-job header", () => {
  const request = buildCreateJobRequest(PAYLOAD, "  runtime-only-token  ");

  assert.equal(request.path, "/api/jobs");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(request.options.headers, {
    [JOB_SUBMISSION_HEADER]: "runtime-only-token",
  });
  assert.deepEqual(JSON.parse(request.options.body), PAYLOAD);
  assert.doesNotMatch(request.options.body, /runtime-only-token/);
  assert.deepEqual(Object.keys(JSON.parse(request.options.body)).sort(), [
    "configuration",
    "name",
    "recipe_id",
    "recipe_version",
  ]);
});

test("public job reads never receive the submission token", () => {
  for (const path of ["/health", "/api/jobs", "/api/jobs/job-123"]) {
    const request = buildPublicJobReadRequest(path);
    assert.deepEqual(request, {
      path,
      options: { method: "GET" },
    });
    assert.equal(request.options.headers, undefined);
  }
});

test("missing submission token blocks request construction", () => {
  for (const token of [undefined, null, "", "   "]) {
    assert.throws(
      () => buildCreateJobRequest(PAYLOAD, token),
      /Enter the job submission token/,
    );
  }
});

test("maps 401 and 403 to clear token-specific messages", () => {
  const unauthorized = getJobSubmissionErrorIssues({ status: 401 });
  const forbidden = getJobSubmissionErrorIssues({ status: 403 });

  assert.equal(unauthorized[0].path, "submission_token");
  assert.match(unauthorized[0].message, /requires a job submission token/);
  assert.equal(forbidden[0].path, "submission_token");
  assert.match(forbidden[0].message, /rejected the job submission token/);
});

test("active token handling introduces no browser persistence", () => {
  const sources = [
    new URL("./jobs.js", import.meta.url),
    new URL("../components/CreateJobForm.jsx", import.meta.url),
    new URL("../components/CommonJobFields.jsx", import.meta.url),
  ].map((source) => fs.readFileSync(source, "utf8"));

  for (const source of sources) {
    assert.doesNotMatch(source, /localStorage/);
    assert.doesNotMatch(source, /sessionStorage/);
    assert.doesNotMatch(source, /document\.cookie/);
    assert.doesNotMatch(source, /VITE_.*TOKEN/i);
  }
});
