import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getJob } from "../api/jobs";
import JobDetails from "../components/JobDetails";
import JobTimeline from "../components/JobTimeline";
import { isTerminalStatus } from "../utils/jobStatus";

const DETAIL_POLL_INTERVAL_MS = 5_000;

export default function JobDetailsPage() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refreshJob = useCallback(async () => {
    try {
      const payload = await getJob(jobId);
      setJob(payload);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    const timer = window.setTimeout(refreshJob, 0);
    return () => window.clearTimeout(timer);
  }, [refreshJob]);

  useEffect(() => {
    if (!job || isTerminalStatus(job.status)) {
      return undefined;
    }

    const timer = window.setInterval(refreshJob, DETAIL_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [job, refreshJob]);

  if (loading) {
    return <p className="empty-state">Loading job details...</p>;
  }

  if (error && !job) {
    return (
      <div className="page-stack">
        <Link className="back-link" to="/jobs">
          ← Back to jobs
        </Link>
        <div className="alert alert--danger">{error}</div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <Link className="back-link" to="/jobs">
        ← Back to jobs
      </Link>

      {error ? <div className="alert alert--warning">{error}</div> : null}

      <JobDetails job={job} />
      <JobTimeline job={job} />
    </div>
  );
}
