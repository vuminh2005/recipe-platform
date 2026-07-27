import { Link } from "react-router-dom";
import JobStatusBadge from "./JobStatusBadge";
import { formatDate, formatMetric } from "../utils/format";

export default function JobTable({ jobs, loading }) {
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
            <th>Workload</th>
            <th>Status</th>
            <th>Best AUC</th>
            <th>Model</th>
            <th>Created</th>
            <th aria-label="Actions" />
          </tr>
        </thead>

        <tbody>
          {jobs.map((job) => {
            const modelLabel = job.registered_model_name
              ? `${job.registered_model_name} · v${
                  job.registered_model_version || "?"
                }`
              : "—";

            return (
              <tr key={job.id}>
                <td>
                  <strong>{job.recipe?.name || "Unnamed recipe"}</strong>
                  <small className="table-subtitle">{job.id.slice(0, 8)}</small>
                </td>
                <td>{job.recipe?.workload || "—"}</td>
                <td>
                  <JobStatusBadge status={job.status} />
                </td>
                <td>{formatMetric(job.best_metric)}</td>
                <td>{modelLabel}</td>
                <td>{formatDate(job.created_at)}</td>
                <td>
                  <Link className="button button--small button--ghost" to={`/jobs/${job.id}`}>
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
