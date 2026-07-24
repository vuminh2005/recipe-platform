const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export async function createJob(recipe) {
  const response = await fetch(`${API_BASE_URL}/api/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(recipe),
  });

  if (!response.ok) {
    throw new Error(`Create job failed: ${response.status}`);
  }

  return response.json();
}

export async function listJobs() {
  const response = await fetch(`${API_BASE_URL}/api/jobs`);

  if (!response.ok) {
    throw new Error(`List jobs failed: ${response.status}`);
  }

  return response.json();
}
