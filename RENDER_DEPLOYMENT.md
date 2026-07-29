# Render Deployment

Deploy the Backend before the Frontend and before moving the local Agent from
its local Backend. Use separate, randomly generated values for
`AGENT_TOKEN` and `JOB_SUBMISSION_TOKEN`; never put either value in Git or a
`VITE_*` variable.

## Backend web service

Configure a Render Python web service with:

- Root Directory: `backend`
- Python version: `3.12.3`
- Build Command: `pip install -r requirements.txt`
- Start Command:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
- Health Check Path: `/health`

Set these environment variables in Render:

- `DATABASE_URL`: a persistent PostgreSQL connection URL. Render-style
  `postgresql://` URLs are normalized to SQLAlchemy's psycopg 3 driver.
- `AGENT_TOKEN`: a secret shared only with the local Agent.
- `JOB_SUBMISSION_TOKEN`: a separate secret entered by a user at job-submission
  time.
- `CORS_ORIGINS`: the exact deployed Frontend origin, for example
  `https://<frontend-service>.onrender.com`. Multiple origins are
  comma-separated.
- `ALLOW_INSECURE_DEVELOPMENT_TOKEN=false`.

Do not use SQLite on Render. The Backend initializes the current schema on an
empty PostgreSQL database at startup. Keep one Backend worker because the
prototype does not yet include a migration system or a concurrency redesign.

Verify `/health`, `/api/recipes`, and `/openapi.json` before starting an Agent
against the deployed URL. The OpenAPI document must expose the Recipe Catalog,
`AgentUpdate.result_patch`, and `JobResponse.result`.

## Frontend static site

Configure a Render Static Site with:

- Root Directory: `frontend`
- Node version: `22.23.1`
- Build Command: `npm ci && npm run build`
- Publish Directory: `dist`
- Rewrite: `/*` to `/index.html`

Set `VITE_API_BASE_URL` to the deployed Backend URL. Set
`VITE_MLFLOW_UI_URL` only when it is a safe, publicly reachable UI.
Set `VITE_SHOW_LOCAL_TOOLS=false` and leave `VITE_KFP_UI_URL` and
`VITE_KATIB_UI_URL` unset in production because laptop-local UIs are not
reachable from a deployed browser session.

All `VITE_*` values are compiled into public browser assets and must never
contain credentials. The job-submission token is entered at runtime, retained
only in React memory, and sent only in `X-Job-Submission-Token` for
`POST /api/jobs`.

## Safe rollout order

1. Deploy the Backend with PostgreSQL and both distinct tokens.
2. Verify health, Catalog, OpenAPI, and database persistence.
3. Deploy the Frontend with the Backend URL and exact CORS origin.
4. Stop the Agent that targets the local Backend.
5. Start exactly one Agent with the deployed Backend URL and matching
   `AGENT_TOKEN`.
6. Verify authorized empty claim polls, then submit one `hello` smoke job with
   the runtime submission token.

This token gate is intentionally small. It is not user authentication, RBAC,
rate limiting, or a production security boundary; use private demo access or
an external access layer in addition to the token for any Internet-facing
deployment.
