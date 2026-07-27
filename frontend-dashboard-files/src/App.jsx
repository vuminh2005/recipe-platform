import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import JobDetailsPage from "./pages/JobDetailsPage";
import JobsPage from "./pages/JobsPage";
import ProductionPage from "./pages/ProductionPage";

function NavigationLink({ to, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `nav-link${isActive ? " nav-link--active" : ""}`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink className="brand" to="/jobs">
          <span className="brand__mark">RP</span>
          <span>
            <strong>Recipe Platform</strong>
            <small>MLOps Control Plane</small>
          </span>
        </NavLink>

        <nav aria-label="Primary navigation">
          <NavigationLink to="/jobs">Jobs</NavigationLink>
          <NavigationLink to="/production">Production</NavigationLink>
        </nav>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate replace to="/jobs" />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/jobs/:jobId" element={<JobDetailsPage />} />
          <Route path="/production" element={<ProductionPage />} />
          <Route path="*" element={<Navigate replace to="/jobs" />} />
        </Routes>
      </main>

      <footer className="app-footer">
        Recipe Platform · Katib + Kubeflow Pipelines + MLflow
      </footer>
    </div>
  );
}
