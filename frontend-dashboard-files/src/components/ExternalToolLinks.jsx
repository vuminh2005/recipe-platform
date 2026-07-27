const SHOW_LOCAL_TOOLS =
  String(import.meta.env.VITE_SHOW_LOCAL_TOOLS).toLowerCase() === "true";

const MLFLOW_UI_URL = import.meta.env.VITE_MLFLOW_UI_URL?.replace(/\/$/, "");
const KFP_UI_URL = import.meta.env.VITE_KFP_UI_URL?.replace(/\/$/, "");
const KATIB_UI_URL = import.meta.env.VITE_KATIB_UI_URL?.replace(/\/$/, "");
const INFERENCE_URL = import.meta.env.VITE_INFERENCE_URL?.replace(/\/$/, "");

function ExternalLink({ href, children, disabled = false }) {
  if (!href || disabled) {
    return (
      <span className="button button--ghost button--disabled" aria-disabled="true">
        {children}
      </span>
    );
  }

  return (
    <a className="button button--ghost" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

export default function ExternalToolLinks({ job }) {
  const kfpRunUrl =
    SHOW_LOCAL_TOOLS && KFP_UI_URL && job.kfp_run_id
      ? `${KFP_UI_URL}/#/runs/details/${job.kfp_run_id}`
      : "";

  const katibUrl =
    SHOW_LOCAL_TOOLS && KATIB_UI_URL && job.katib_experiment_name
      ? KATIB_UI_URL
      : "";

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Specialized interfaces</p>
          <h2>External Tools</h2>
        </div>
      </div>

      <div className="tool-links">
        <ExternalLink href={kfpRunUrl} disabled={!job.kfp_run_id}>
          Open KFP Run
        </ExternalLink>

        <ExternalLink href={katibUrl} disabled={!job.katib_experiment_name}>
          Open Katib
        </ExternalLink>

        <ExternalLink href={MLFLOW_UI_URL}>Open MLflow</ExternalLink>

        <ExternalLink href={INFERENCE_URL}>Open Inference</ExternalLink>
      </div>

      {SHOW_LOCAL_TOOLS ? (
        <p className="helper-text">
          KFP and Katib links point to localhost. They work only on a machine with
          the required port-forwards running.
        </p>
      ) : null}
    </section>
  );
}
