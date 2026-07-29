const MLFLOW_UI_URL = import.meta.env.VITE_MLFLOW_UI_URL;

export default function ProductionPage() {
  return (
    <div className="page-stack">
      <section className="dashboard-hero dashboard-hero--compact">
        <div>
          <p className="eyebrow">Model lifecycle</p>
          <h1>Production</h1>
          <p>
            Registered candidates can be inspected in MLflow. Production
            promotion and serving controls are not implemented.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Current scope</p>
            <h2>Production actions are unavailable</h2>
          </div>
          <span className="model-stage-badge">Read only</span>
        </div>

        <p className="helper-text">
          This dashboard currently creates training jobs and displays registered
          candidate metadata. It does not expose promotion, rollback, inference,
          or endpoint controls.
        </p>

        {MLFLOW_UI_URL ? (
          <div className="tool-links">
            <a
              className="button button--ghost"
              href={MLFLOW_UI_URL}
              target="_blank"
              rel="noreferrer"
            >
              Open MLflow Registry
            </a>
          </div>
        ) : null}
      </section>
    </div>
  );
}
