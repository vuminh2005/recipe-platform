import { Link } from "react-router-dom";
import {
  formatDate,
  formatMetric,
  formatObjectiveLabel,
} from "../utils/format";
import { getJobPresentation } from "../utils/jobPresentation";
import JobStatusBadge from "./JobStatusBadge";

function objectiveValue(presentation) {
  if (!presentation.objective) {
    return "N/A";
  }
  if (!presentation.automlEnabled) {
    return "Not tuned";
  }
  return formatMetric(presentation.objective.value, "N/A");
}

export default function JobTable({ jobs, loading, catalogById = {} }) {
  if (loading && jobs.length === 0) {
    return <p className="empty-state">Loading platform jobs...</p>;
  }

  if (jobs.length === 0) {
    return <p className="empty-state">No jobs have been created yet.</p>;
  }

  return (
    <div className="table-wrapper">
      <table className="jobs-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Recipe</th>
            <th>Version</th>
            <th>Status</th>
            <th>Objective</th>
            <th>Model</th>
            <th>Created</th>
            <th aria-label="Actions" />
          </tr>
        </thead>

        <tbody>
          {jobs.map((job) => {
            const presentation = getJobPresentation(job, catalogById);
            const registeredModel = presentation.model?.registered_name;
            const modelLabel = registeredModel
              ? `${registeredModel} · v${presentation.model.version || "?"}`
              : "—";

            return (
              <tr key={job.id}>
                <td>
                  <strong>{presentation.name}</strong>
                  <small className="table-subtitle">{job.id.slice(0, 8)}</small>
                </td>
                <td>
                  <strong>{presentation.metadata.display_name}</strong>
                  <small className="table-subtitle">
                    {presentation.metadata.recipe_id || "Unknown"}
                  </small>
                </td>
                <td>{presentation.metadata.recipe_version || "—"}</td>
                <td>
                  <JobStatusBadge status={job.status} />
                </td>
                <td>
                  <strong>
                    {presentation.objective
                      ? formatObjectiveLabel(presentation.objective)
                      : "N/A"}
                  </strong>
                  <small className="table-subtitle">
                    {objectiveValue(presentation)}
                  </small>
                </td>
                <td>{modelLabel}</td>
                <td>{formatDate(job.created_at)}</td>
                <td>
                  <Link
                    className="button button--small button--ghost"
                    to={`/jobs/${job.id}`}
                  >
                    View details
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
