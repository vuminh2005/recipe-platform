const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");

function assertApiBaseUrl() {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL is not configured. Check the frontend environment file.",
    );
  }
}

function formatApiError(payload, status) {
  if (Array.isArray(payload?.detail)) {
    return payload.detail
      .map((item) => {
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "request";
        return `${location}: ${item.msg}`;
      })
      .join("; ");
  }

  return (
    payload?.detail ||
    payload?.message ||
    `Request failed with HTTP ${status}`
  );
}

async function apiRequest(path, options = {}) {
  assertApiBaseUrl();

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  if (response.status === 204) {
    return null;
  }

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(formatApiError(payload, response.status));
  }

  return payload;
}

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
