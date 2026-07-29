import {
  formatBoolean,
  formatDate,
  formatJsonValue,
  formatKey,
  formatMetric,
  formatObjectiveLabel,
} from "../utils/format";
import { getJobPresentation } from "../utils/jobPresentation";
import ExternalToolLinks from "./ExternalToolLinks";
import JobStatusBadge from "./JobStatusBadge";
import MetricsGrid from "./MetricsGrid";

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

function SearchSpaceGrid({ searchSpace }) {
  const ranges = Object.fromEntries(
    Object.entries(searchSpace || {}).map(([name, range]) => [
      name,
      range && typeof range === "object"
        ? `${formatJsonValue(range.min)} – ${formatJsonValue(range.max)}`
        : range,
    ]),
  );

  return (
    <MetricsGrid
      metrics={ranges}
      emptyMessage="No search-space ranges were recorded."
    />
  );
}

function objectiveValue(presentation) {
  if (!presentation.objective) {
    return "N/A";
  }
  if (!presentation.automlEnabled) {
    return "Not tuned";
  }
  return formatMetric(presentation.objective.value, "N/A");
}

export default function JobDetails({ job, catalogById = {} }) {
  const presentation = getJobPresentation(job, catalogById);
  const {
    metadata,
    configuration,
    externalIds,
    model,
    effectiveFinalParameters,
  } = presentation;
  const training = configuration.training || {};
  const automl = configuration.automl || {};
  const isHello = metadata.recipe_id === "hello";
  const hasMlflowData = Boolean(
    externalIds.mlflow_parent_run_id ||
      externalIds.mlflow_run_id ||
      model?.uri,
  );

  return (
    <div className="details-layout">
      <section className="panel details-hero">
        <div>
          <p className="eyebrow">Platform Job</p>
          <h1>{presentation.name}</h1>
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
          <h3>Recipe</h3>
          <DefinitionList
            entries={[
              ["Display name", metadata.display_name],
              ["Recipe ID", metadata.recipe_id],
              ["Version", metadata.recipe_version],
              ["Task type", metadata.task_type && formatKey(metadata.task_type)],
              ["Framework", metadata.framework && formatKey(metadata.framework)],
              ["Model", metadata.model],
            ]}
          />
        </section>

        {!isHello ? (
          <section className="detail-card">
            <h3>Training configuration</h3>
            <MetricsGrid
              metrics={training}
              emptyMessage="No normalized training configuration is available."
            />
          </section>
        ) : null}

        {!isHello ? (
          <section className="detail-card detail-card--wide">
            <h3>AutoML configuration</h3>
            <DefinitionList
              entries={[
                ["Enabled", formatBoolean(presentation.automlEnabled)],
                ["Maximum trials", automl.max_trials],
                ["Parallel trials", automl.parallel_trials],
                ["Algorithm", automl.algorithm],
              ]}
            />
            {presentation.automlEnabled ? (
              <SearchSpaceGrid searchSpace={automl.search_space} />
            ) : (
              <>
                <p className="helper-text">
                  Katib was skipped. Final parameters came from the normalized
                  recipe configuration.
                </p>
                <MetricsGrid
                  metrics={effectiveFinalParameters}
                  emptyMessage="Effective final parameters are unavailable for this historical job."
                />
              </>
            )}
          </section>
        ) : null}

        {presentation.objective ? (
          <section className="detail-card">
            <h3>Objective</h3>
            <DefinitionList
              entries={[
                ["Metric", formatObjectiveLabel(presentation.objective)],
                ["Direction", presentation.objective.direction],
                ["Value", objectiveValue(presentation)],
              ]}
            />
          </section>
        ) : null}

        {metadata.supports_automl ? (
          <section className="detail-card">
            <h3>Katib tuning</h3>
            {presentation.automlEnabled ? (
              <>
                <DefinitionList
                  entries={[
                    [
                      "Experiment",
                      externalIds.katib_experiment_id || "Not available yet",
                    ],
                  ]}
                />
                <MetricsGrid
                  metrics={presentation.bestParams}
                  emptyMessage="Best parameters are not available yet."
                />
              </>
            ) : (
              <p className="empty-state">
                Not tuned. AutoML was disabled for this job.
              </p>
            )}
          </section>
        ) : null}

        <section className="detail-card">
          <h3>KFP execution</h3>
          <DefinitionList
            entries={[
              ...(externalIds.kfp_run_id
                ? [["KFP run ID", externalIds.kfp_run_id]]
                : []),
              ["Agent ID", job.agent_id],
            ]}
          />
        </section>

        {!isHello && hasMlflowData ? (
          <section className="detail-card">
            <h3>MLflow tracking</h3>
            <DefinitionList
              entries={[
                ...(externalIds.mlflow_parent_run_id
                  ? [["Parent run ID", externalIds.mlflow_parent_run_id]]
                  : []),
                ...(externalIds.mlflow_run_id
                  ? [["Final run ID", externalIds.mlflow_run_id]]
                  : []),
                ...(model?.uri ? [["Model URI", model.uri]] : []),
              ]}
            />
          </section>
        ) : null}

        {!isHello && model ? (
          <section className="detail-card">
            <div className="detail-card__title-row">
              <h3>Model Registry</h3>
              {model.version ? (
                <span className="model-stage-badge">Candidate</span>
              ) : null}
            </div>
            <DefinitionList
              entries={[
                ["Registered model", model.registered_name],
                ["Version", model.version],
              ]}
            />
          </section>
        ) : null}

        {!isHello ? (
          <section className="detail-card detail-card--wide">
            <h3>Final metrics</h3>
            <MetricsGrid
              metrics={presentation.finalMetrics}
              emptyMessage="Final evaluation metrics are not available yet."
            />
          </section>
        ) : null}
      </section>

      <ExternalToolLinks job={job} catalogById={catalogById} />
    </div>
  );
}
