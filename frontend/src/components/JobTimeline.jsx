import JobStatusBadge from "./JobStatusBadge";
import {
  getJobStatusMeta,
  inferFailureStage,
  getTimelineState,
  getTimelineSteps,
} from "../utils/jobStatus";

export default function JobTimeline({ job }) {
  const steps = getTimelineSteps(job);
  const failureStage =
    job.status === "FAILED" ? inferFailureStage(job, steps) : null;

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
          const state = getTimelineState(step, job, steps);
          const meta = getJobStatusMeta(step);

          return (
            <li className={`timeline__step timeline__step--${state}`} key={step}>
              <span className="timeline__marker" aria-hidden="true" />
              <div>
                <strong>{meta.label}</strong>
                <small>
                  {state === "failed"
                    ? `Failure reached this stage (conservative inference). ${
                        job.error_message || "No error message was recorded."
                      }`
                    : meta.description}
                </small>
              </div>
            </li>
          );
        })}
      </ol>
      {failureStage ? (
        <p className="helper-text">
          Failure stage is inferred from persisted integration IDs and partial
          results because the API does not store complete status history.
        </p>
      ) : null}
    </section>
  );
}
