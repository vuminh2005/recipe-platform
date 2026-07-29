import { getJobPresentation } from "../utils/jobPresentation";

const SHOW_LOCAL_TOOLS =
  String(import.meta.env.VITE_SHOW_LOCAL_TOOLS).toLowerCase() === "true";

const MLFLOW_UI_URL = import.meta.env.VITE_MLFLOW_UI_URL?.replace(/\/$/, "");
const KFP_UI_URL = import.meta.env.VITE_KFP_UI_URL?.replace(/\/$/, "");
const KATIB_UI_URL = import.meta.env.VITE_KATIB_UI_URL?.replace(/\/$/, "");

function ExternalLink({ href, children }) {
  return (
    <a className="button button--ghost" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

export default function ExternalToolLinks({ job, catalogById = {} }) {
  const { externalIds } = getJobPresentation(job, catalogById);
  const kfpRunUrl =
    SHOW_LOCAL_TOOLS && KFP_UI_URL && externalIds.kfp_run_id
      ? `${KFP_UI_URL}/#/runs/details/${externalIds.kfp_run_id}`
      : "";
  const katibUrl =
    SHOW_LOCAL_TOOLS && KATIB_UI_URL && externalIds.katib_experiment_id
      ? KATIB_UI_URL
      : "";
  const mlflowUrl =
    MLFLOW_UI_URL &&
    (externalIds.mlflow_parent_run_id || externalIds.mlflow_run_id)
      ? MLFLOW_UI_URL
      : "";
  const links = [
    kfpRunUrl && { href: kfpRunUrl, label: "Open KFP Run" },
    katibUrl && { href: katibUrl, label: "Open Katib" },
    mlflowUrl && { href: mlflowUrl, label: "Open MLflow" },
  ].filter(Boolean);

  if (links.length === 0) {
    return null;
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Specialized interfaces</p>
          <h2>External Tools</h2>
        </div>
      </div>

      <div className="tool-links">
        {links.map((link) => (
          <ExternalLink href={link.href} key={link.label}>
            {link.label}
          </ExternalLink>
        ))}
      </div>

      {SHOW_LOCAL_TOOLS ? (
        <p className="helper-text">
          KFP and Katib links point to configured local interfaces and require
          the corresponding port-forwards.
        </p>
      ) : null}
    </section>
  );
}
