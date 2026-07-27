import JobStatusBadge from "./JobStatusBadge";
import {
  getJobStatusMeta,
  getTimelineState,
  getTimelineSteps,
} from "../utils/jobStatus";

export default function JobTimeline({ job }) {
  const steps = getTimelineSteps(job);

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Orchestration lifecycle</p>
          <h2>Job Timeline</h2>
        </div>
        <JobStatusBadge status={job.status} />
      </div>

      <ol className="timeline">
        {steps.map((step) => {
          const state = getTimelineState(step, job.status, steps);
          const meta = getJobStatusMeta(step);

          return (
            <li className={`timeline__step timeline__step--${state}`} key={step}>
              <span className="timeline__marker" aria-hidden="true" />
              <div>
                <strong>{meta.label}</strong>
                <small>{meta.description}</small>
              </div>
            </li>
          );
        })}

        {job.status === "FAILED" ? (
          <li className="timeline__step timeline__step--failed">
            <span className="timeline__marker" aria-hidden="true" />
            <div>
              <strong>Failed</strong>
              <small>{job.error_message || "No error message was recorded."}</small>
            </div>
          </li>
        ) : null}
      </ol>
    </section>
  );
}
