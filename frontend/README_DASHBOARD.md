# Recipe Platform Dashboard

## 1. Install the only added dependency

From the existing `frontend` directory:

```bash
npm install react-router-dom
```

## 2. Copy files

Copy the provided `src/` directory and environment files into the existing
Vite frontend. Remove or stop importing the old `App.css` and `index.css`.

## 3. Configure the backend and optional tool links

Set `VITE_API_BASE_URL` to the Recipe Platform backend. Optional
`VITE_MLFLOW_UI_URL`, `VITE_KFP_UI_URL`, and `VITE_KATIB_UI_URL` values enable
links for integration IDs that are actually present on a job.

Never place tokens, R2 credentials, database URLs, or other secrets in a
`VITE_*` variable.

The create-job form loads public recipe metadata from `GET /api/recipes`.
Job history remains readable when the catalog is unavailable, but new
submissions are disabled until a valid supported catalog response is loaded.

## 4. Local development

```bash
npm run dev
```

The development build uses `.env.development`.

## 5. Production build

```bash
npm run build
npm run preview
```

The production build uses `.env.production`.

## 6. Render

Use:

- Root Directory: `frontend`
- Build Command: `npm ci && npm run build`
- Publish Directory: `dist`

Add a rewrite rule for React Router:

- Source: `/*`
- Destination: `/index.html`
- Action: Rewrite

Without this rewrite, directly opening `/jobs/<job-id>` can return 404 from the
static host.
