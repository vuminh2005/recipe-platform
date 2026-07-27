# Recipe Platform Dashboard

## 1. Install the only added dependency

From the existing `frontend` directory:

```bash
npm install react-router-dom
```

## 2. Copy files

Copy the provided `src/` directory and environment files into the existing
Vite frontend. Remove or stop importing the old `App.css` and `index.css`.

## 3. Configure inference URL

Set `VITE_INFERENCE_URL` in `.env.development` and `.env.production` when the
inference service URL is known. Leave it blank for now.

Never place tokens, R2 credentials, database URLs, or other secrets in a
`VITE_*` variable.

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
