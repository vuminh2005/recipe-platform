import { Link } from "react-router-dom";
import { formatMetric } from "../utils/format";

export default function ProductionCard({ jobs }) {
  const latestCandidate = jobs.find(
    (job) =>
      job.status === "SUCCEEDED" &&
      job.registered_model_name &&
      job.registered_model_version,
  );

  return (
    <section className="panel production-card">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Model lifecycle</p>
          <h2>Production</h2>
        </div>
        <span className="model-stage-badge">Phase 19</span>
      </div>

      {latestCandidate ? (
        <>
          <p>
            Latest candidate: <strong>{latestCandidate.registered_model_name}</strong>{" "}
            version <strong>{latestCandidate.registered_model_version}</strong>.
          </p>
          <p className="helper-text">
            Best validation AUC: {formatMetric(latestCandidate.best_metric)}
          </p>
          <Link className="button button--ghost" to={`/jobs/${latestCandidate.id}`}>
            Review candidate
          </Link>
        </>
      ) : (
        <p className="empty-state">
          No successfully registered candidate model is available yet.
        </p>
      )}
    </section>
  );
}
