from __future__ import annotations

from typing import Any

import requests


class BackendClient:
    def __init__(self, *, base_url: str, agent_id: str, agent_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-Agent-Token": agent_token,
                "Content-Type": "application/json",
            }
        )

    def _raise(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Backend HTTP {response.status_code}: {response.text[:1000]}"
            ) from exc

    def claim_next_job(self) -> dict[str, Any] | None:
        response = self.session.post(
            f"{self.base_url}/api/agent/jobs/claim",
            json={"agent_id": self.agent_id},
            timeout=(10, 60),
        )
        if response.status_code == 204:
            return None
        self._raise(response)
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/jobs/{job_id}",
            timeout=(10, 60),
        )
        self._raise(response)
        return response.json()

    def patch_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        payload = {key: value for key, value in updates.items() if value is not None}
        response = self.session.patch(
            f"{self.base_url}/api/agent/jobs/{job_id}",
            json=payload,
            timeout=(10, 60),
        )
        self._raise(response)
        return response.json()
