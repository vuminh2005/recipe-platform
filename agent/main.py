from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow both `python -m agent.main` and the existing `python agent/main.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from agent.backend_client import BackendClient
from agent.recipe_registry import execute_job
from agent.settings import Settings


def main() -> None:
    load_dotenv("agent/.env")
    settings = Settings.from_env()
    backend = BackendClient(
        base_url=settings.backend_url,
        agent_id=settings.agent_id,
        agent_token=settings.agent_token,
    )

    logging.info("Agent %s started", settings.agent_id)
    while True:
        try:
            job = backend.claim_next_job()
            if job is None:
                time.sleep(settings.poll_interval_seconds)
                continue

            logging.info("Claimed platform job %s", job["id"])
            execute_job(job, settings=settings, backend=backend)
        except KeyboardInterrupt:
            raise
        except Exception:
            logging.exception("Agent iteration failed")
            time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
