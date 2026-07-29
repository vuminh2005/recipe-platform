import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getBackendHealth, listJobs } from "../api/jobs";
import { listRecipes } from "../api/recipes";
import CreateJobForm from "../components/CreateJobForm";
import JobTable from "../components/JobTable";
import ProductionCard from "../components/ProductionCard";
import { formatDate } from "../utils/format";
import { isTerminalStatus } from "../utils/jobStatus";
import {
  indexRecipes,
  prepareRecipeCatalog,
} from "../utils/recipeCatalog";

const LIST_POLL_INTERVAL_MS = 10_000;
const ACTIVE_AGENT_WINDOW_MS = 2 * 60 * 1000;

function inferAgentState(jobs) {
  const activeJob = jobs.find((job) => !isTerminalStatus(job.status));

  if (!activeJob) {
    return {
      label: "Unknown / idle",
      detail: "No active job is available to infer Agent activity.",
      tone: "neutral",
    };
  }

  const updatedAt = new Date(activeJob.updated_at).getTime();
  const isRecent =
    Number.isFinite(updatedAt) &&
    Date.now() - updatedAt <= ACTIVE_AGENT_WINDOW_MS;

  return {
    label: isRecent ? "Active (inferred)" : "Unknown",
    detail: `Latest active job update: ${formatDate(activeJob.updated_at)}`,
    tone: isRecent ? "success" : "warning",
  };
}

function SystemCard({ title, value, detail, tone = "neutral" }) {
  return (
    <article className={`system-card system-card--${tone}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

export default function JobsPage() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [backendHealth, setBackendHealth] = useState("checking");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [recipes, setRecipes] = useState([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [catalogIssues, setCatalogIssues] = useState([]);

  const refreshJobs = useCallback(async ({ silent = false } = {}) => {
    if (!silent) {
      setLoading(true);
    }

    try {
      const payload = await listJobs();
      setJobs(Array.isArray(payload) ? payload : []);
      setError("");
      setBackendHealth("online");
    } catch (requestError) {
      setError(requestError.message);
      setBackendHealth("offline");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const initialRefreshTimer = window.setTimeout(refreshJobs, 0);

    const pollingTimer = window.setInterval(
      () => refreshJobs({ silent: true }),
      LIST_POLL_INTERVAL_MS,
    );

    return () => {
      window.clearTimeout(initialRefreshTimer);
      window.clearInterval(pollingTimer);
    };
  }, [refreshJobs]);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const health = await getBackendHealth();

        if (!cancelled) {
          setBackendHealth(health?.status === "ok" ? "online" : "degraded");
        }
      } catch {
        if (!cancelled) {
          setBackendHealth("offline");
        }
      }
    }

    checkHealth();
    const timer = window.setInterval(checkHealth, 30_000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadCatalog() {
      try {
        const payload = await listRecipes();
        const prepared = prepareRecipeCatalog(payload);

        if (!cancelled) {
          setRecipes(prepared.recipes);
          setCatalogIssues(prepared.issues);
          setCatalogError("");
        }
      } catch (requestError) {
        if (!cancelled) {
          setRecipes([]);
          setCatalogIssues([]);
          setCatalogError(
            `Recipe Catalog is unavailable: ${requestError.message}`,
          );
        }
      } finally {
        if (!cancelled) {
          setCatalogLoading(false);
        }
      }
    }

    loadCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  const agentState = useMemo(() => inferAgentState(jobs), [jobs]);
  const catalogById = useMemo(() => indexRecipes(recipes), [recipes]);

  const backendCard =
    backendHealth === "online"
      ? {
          value: "Online",
          detail: "Recipe Platform API is responding.",
          tone: "success",
        }
      : backendHealth === "offline"
        ? {
            value: "Offline",
            detail: "The browser could not reach the backend.",
            tone: "danger",
          }
        : {
            value: "Checking",
            detail: "Waiting for the backend health response.",
            tone: "neutral",
          };

  function handleCreated(job) {
    refreshJobs({ silent: true });
    navigate(`/jobs/${job.id}`);
  }

  return (
    <div className="page-stack">
      <section className="dashboard-hero">
        <div>
          <p className="eyebrow">Unified MLOps control plane</p>
          <h1>Recipe Platform Dashboard</h1>
          <p>
            Create recipes and follow Katib tuning, KFP execution, and MLflow
            tracking from one place.
          </p>
        </div>
      </section>

      <section className="system-grid" aria-label="System status">
        <SystemCard
          title="Cloud Backend"
          value={backendCard.value}
          detail={backendCard.detail}
          tone={backendCard.tone}
        />
        <SystemCard
          title="Local Agent"
          value={agentState.label}
          detail={agentState.detail}
          tone={agentState.tone}
        />
        <SystemCard
          title="Execution Cluster"
          value="Local k3s"
          detail="KFP and Katib require local services and port-forwards."
          tone="info"
        />
      </section>

      <div className="dashboard-columns">
        <CreateJobForm
          recipes={recipes}
          catalogLoading={catalogLoading}
          catalogError={catalogError}
          catalogIssues={catalogIssues}
          onCreated={handleCreated}
        />
        <ProductionCard jobs={jobs} catalogById={catalogById} />
      </div>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="eyebrow">Backend metadata</p>
            <h2>Platform Jobs</h2>
          </div>

          <button
            className="button button--ghost"
            type="button"
            onClick={() => refreshJobs()}
            disabled={loading}
          >
            Refresh
          </button>
        </div>

        {error ? <div className="alert alert--danger">{error}</div> : null}

        <JobTable
          jobs={jobs}
          loading={loading}
          catalogById={catalogById}
        />
      </section>
    </div>
  );
}
