import { apiRequest } from "./client.js";

export const JOB_SUBMISSION_HEADER = "X-Job-Submission-Token";

export function buildPublicJobReadRequest(path) {
  return {
    path,
    options: { method: "GET" },
  };
}

export function buildCreateJobRequest(recipe, submissionToken) {
  const token = String(submissionToken ?? "").trim();
  if (!token) {
    throw new Error("Enter the job submission token before creating a job.");
  }

  return {
    path: "/api/jobs",
    options: {
      method: "POST",
      headers: {
        [JOB_SUBMISSION_HEADER]: token,
      },
      body: JSON.stringify(recipe),
    },
  };
}

export function getJobSubmissionErrorIssues(error) {
  if (error?.status === 401) {
    return [
      {
        path: "submission_token",
        message:
          "The Backend requires a job submission token. Enter it and try again.",
      },
    ];
  }
  if (error?.status === 403) {
    return [
      {
        path: "submission_token",
        message:
          "The Backend rejected the job submission token. Check it and try again.",
      },
    ];
  }
  return error?.issues?.length
    ? error.issues
    : [{ path: "", message: error?.message || "Job submission failed." }];
}

export function getBackendHealth() {
  const request = buildPublicJobReadRequest("/health");
  return apiRequest(request.path, request.options);
}

export function listJobs() {
  const request = buildPublicJobReadRequest("/api/jobs");
  return apiRequest(request.path, request.options);
}

export function getJob(jobId) {
  const request = buildPublicJobReadRequest(
    `/api/jobs/${encodeURIComponent(jobId)}`,
  );
  return apiRequest(request.path, request.options);
}

export function createJob(recipe, submissionToken) {
  const request = buildCreateJobRequest(recipe, submissionToken);
  return apiRequest(request.path, request.options);
}

// These endpoints belong to the promotion phase and are intentionally
// not used by the first dashboard release.
export function promoteJob(jobId, adminToken) {
  return apiRequest(`/api/jobs/${encodeURIComponent(jobId)}/promote`, {
    method: "POST",
    headers: {
      "X-Admin-Token": adminToken,
    },
  });
}

export function getProductionStatus() {
  return apiRequest("/api/production/status");
}
