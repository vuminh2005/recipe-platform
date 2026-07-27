const MLFLOW_UI_URL = import.meta.env.VITE_MLFLOW_UI_URL;
const INFERENCE_URL = import.meta.env.VITE_INFERENCE_URL;

export default function ProductionPage() {
  return (
    <div className="page-stack">
      <section className="dashboard-hero dashboard-hero--compact">
        <div>
          <p className="eyebrow">Model lifecycle</p>
          <h1>Production</h1>
          <p>
            This page is prepared for the next phase: candidate review, champion
            promotion, inference reload, monitoring, and rollback.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Current scope</p>
            <h2>Promotion is not connected yet</h2>
          </div>
          <span className="model-stage-badge">Planned</span>
        </div>

        <div className="production-checklist">
          <div>
            <span>1</span>
            <p>Standardize one registered model and candidate/champion aliases.</p>
          </div>
          <div>
            <span>2</span>
            <p>Add backend promotion and production-status endpoints.</p>
          </div>
          <div>
            <span>3</span>
            <p>Reload the inference service atomically after promotion.</p>
          </div>
          <div>
            <span>4</span>
            <p>Display monitoring, recent predictions, and rollback controls.</p>
          </div>
        </div>

        <div className="tool-links">
          {MLFLOW_UI_URL ? (
            <a
              className="button button--ghost"
              href={MLFLOW_UI_URL}
              target="_blank"
              rel="noreferrer"
            >
              Open MLflow Registry
            </a>
          ) : null}

          {INFERENCE_URL ? (
            <a
              className="button button--ghost"
              href={INFERENCE_URL}
              target="_blank"
              rel="noreferrer"
            >
              Open Inference
            </a>
          ) : null}
        </div>
      </section>
    </div>
  );
}
