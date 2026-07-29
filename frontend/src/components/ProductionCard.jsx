import { Link } from "react-router-dom";
import {
  formatMetric,
  formatObjectiveLabel,
} from "../utils/format";
import { getJobPresentation } from "../utils/jobPresentation";

export default function ProductionCard({ jobs, catalogById = {} }) {
  const latestCandidateJob = jobs.find((job) => {
    const model = getJobPresentation(job, catalogById).model;
    return (
      job.status === "SUCCEEDED" &&
      model?.registered_name &&
      model?.version
    );
  });
  const candidate = latestCandidateJob
    ? getJobPresentation(latestCandidateJob, catalogById)
    : null;

  return (
    <section className="panel production-card">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Model lifecycle</p>
          <h2>Registered candidates</h2>
        </div>
        <span className="model-stage-badge">Promotion unavailable</span>
      </div>

      {candidate ? (
        <>
          <p>
            Latest candidate: <strong>{candidate.model.registered_name}</strong>{" "}
            version <strong>{candidate.model.version}</strong>.
          </p>
          <p className="helper-text">
            {candidate.metadata.display_name}
            {candidate.objective
              ? ` · ${formatObjectiveLabel(candidate.objective)}: ${
                  candidate.automlEnabled
                    ? formatMetric(candidate.objective.value, "N/A")
                    : "Not tuned"
                }`
              : ""}
          </p>
          <Link
            className="button button--ghost"
            to={`/jobs/${latestCandidateJob.id}`}
          >
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
