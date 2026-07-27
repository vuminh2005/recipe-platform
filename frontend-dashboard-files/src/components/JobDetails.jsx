import ExternalToolLinks from "./ExternalToolLinks";
import JobStatusBadge from "./JobStatusBadge";
import MetricsGrid from "./MetricsGrid";
import {
  formatBoolean,
  formatDate,
  formatKey,
  formatMetric,
  truncateMiddle,
} from "../utils/format";

function DefinitionList({ entries }) {
  return (
    <dl className="definition-list">
      {entries.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

function JsonDetails({ title, value }) {
  const entries = Object.entries(value || {});

  return (
    <section className="detail-card">
      <h3>{title}</h3>
      {entries.length ? (
        <DefinitionList
          entries={entries.map(([key, item]) => [
            formatKey(key),
            typeof item === "number" ? formatMetric(item) : String(item),
          ])}
        />
      ) : (
        <p className="empty-state">Not available yet.</p>
      )}
    </section>
  );
}

export default function JobDetails({ job }) {
  const training = job.recipe?.training || {};
  const automl = job.recipe?.automl || {};

  return (
    <div className="details-layout">
      <section className="panel details-hero">
        <div>
          <p className="eyebrow">Platform Job</p>
          <h1>{job.recipe?.name || "Unnamed recipe"}</h1>
          <p className="job-id">{job.id}</p>
        </div>

        <div className="details-hero__meta">
          <JobStatusBadge status={job.status} />
          <small>Updated {formatDate(job.updated_at)}</small>
        </div>
      </section>

      {job.error_message ? (
        <div className="alert alert--danger">
          <strong>Job error:</strong> {job.error_message}
        </div>
      ) : null}

      <section className="details-grid">
        <section className="detail-card">
          <h3>Recipe configuration</h3>
          <DefinitionList
            entries={[
              ["Workload", job.recipe?.workload],
              ["Model", training.model],
              ["Image size", training.image_size],
              ["Trial epochs", training.trial_epochs ?? training.epochs],
              ["Final epochs", training.final_epochs ?? training.epochs],
              ["Batch size", training.batch_size],
              ["Dense units", training.dense_units],
              [
                "Trainable backbone",
                formatBoolean(Boolean(training.trainable_backbone)),
              ],
              ["AutoML enabled", formatBoolean(Boolean(automl.enabled))],
              ["Max trials", automl.max_trials],
              ["Parallel trials", automl.parallel_trials],
              ["Algorithm", automl.algorithm],
            ]}
          />
        </section>

        <section className="detail-card">
          <h3>Katib tuning</h3>
          <DefinitionList
            entries={[
              ["Experiment", job.katib_experiment_name],
              ["Best objective", formatMetric(job.best_metric)],
            ]}
          />
          <MetricsGrid
            metrics={job.best_params}
            emptyMessage="Best parameters will appear after Katib succeeds."
          />
        </section>

        <section className="detail-card">
          <h3>KFP execution</h3>
          <DefinitionList
            entries={[
              ["KFP run ID", truncateMiddle(job.kfp_run_id, 16)],
              ["Agent ID", job.agent_id],
            ]}
          />
        </section>

        <section className="detail-card">
          <h3>MLflow tracking</h3>
          <DefinitionList
            entries={[
              [
                "Parent run ID",
                truncateMiddle(job.mlflow_parent_run_id, 16),
              ],
              [
                "Final run ID",
                truncateMiddle(job.mlflow_final_run_id, 16),
              ],
              ["Model URI", job.model_uri],
            ]}
          />
        </section>

        <section className="detail-card">
          <div className="detail-card__title-row">
            <h3>Model Registry</h3>
            {job.registered_model_version ? (
              <span className="model-stage-badge">Candidate</span>
            ) : null}
          </div>

          <DefinitionList
            entries={[
              ["Registered model", job.registered_model_name],
              ["Version", job.registered_model_version],
            ]}
          />
        </section>

        <section className="detail-card detail-card--wide">
          <h3>Final metrics</h3>
          <MetricsGrid
            metrics={job.final_metrics}
            emptyMessage="Final evaluation metrics are not available yet."
          />
        </section>
      </section>

      <ExternalToolLinks job={job} />
    </div>
  );
}
