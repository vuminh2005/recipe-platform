import { getJobStatusMeta } from "../utils/jobStatus";

export default function JobStatusBadge({ status }) {
  const meta = getJobStatusMeta(status);

  return (
    <span className={`status-badge status-badge--${meta.tone}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      {meta.label}
    </span>
  );
}
