import { apiRequest } from "./client";

export function getBackendHealth() {
  return apiRequest("/health");
}

export function listJobs() {
  return apiRequest("/api/jobs");
}

export function getJob(jobId) {
  return apiRequest(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export function createJob(recipe) {
  return apiRequest("/api/jobs", {
    method: "POST",
    body: JSON.stringify(recipe),
  });
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
