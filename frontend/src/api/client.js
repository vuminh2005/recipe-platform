const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL?.replace(/\/$/, "");

function assertApiBaseUrl() {
  if (!API_BASE_URL) {
    throw new Error(
      "VITE_API_BASE_URL is not configured. Check the frontend environment file.",
    );
  }
}

function normalizeLocation(location) {
  if (!Array.isArray(location)) {
    return "";
  }

  return location
    .filter((part, index) => !(index === 0 && part === "body"))
    .join(".");
}

export function parseApiError(payload, status) {
  if (Array.isArray(payload?.detail)) {
    const issues = payload.detail.map((item) => ({
      path: normalizeLocation(item.loc),
      message: item.msg || "Invalid value",
    }));

    return {
      message: issues
        .map(({ path, message }) => (path ? `${path}: ${message}` : message))
        .join("; "),
      issues,
    };
  }

  const message =
    (typeof payload?.detail === "string" && payload.detail) ||
    (typeof payload?.message === "string" && payload.message) ||
    `Request failed with HTTP ${status}`;

  return {
    message,
    issues: [{ path: "", message }],
  };
}

export class ApiError extends Error {
  constructor(message, { status, issues, payload } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.issues = issues || [];
    this.payload = payload;
  }
}

export async function apiRequest(path, options = {}) {
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
    const parsed = parseApiError(payload, response.status);
    throw new ApiError(parsed.message, {
      status: response.status,
      issues: parsed.issues,
      payload,
    });
  }

  return payload;
}
