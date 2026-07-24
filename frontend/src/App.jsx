import { useEffect, useState } from "react";
import { createJob, listJobs } from "./api";

function App() {
  const [name, setName] = useState("cats-dogs-demo");
  const [workload, setWorkload] = useState("hello");
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState("");

  async function refreshJobs() {
    try {
      setJobs(await listJobs());
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    try {
      await createJob({
        name,
        workload,
        automl: {
          enabled: workload === "cats-dogs",
          max_trials: 3,
          parallel_trials: 1,
          algorithm: "random",
        },
        training: {
          model: "tiny_cnn",
          epochs: 1,
          batch_size: 8,
        },
      });

      await refreshJobs();
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshJobs();

    const timer = setInterval(refreshJobs, 5000);
    return () => clearInterval(timer);
  }, []);

  return (
    <main style={{ maxWidth: 900, margin: "40px auto" }}>
      <h1>Recipe Platform</h1>

      <form onSubmit={handleSubmit}>
        <label>
          Recipe name
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </label>

        <label>
          Workload
          <select
            value={workload}
            onChange={(event) => setWorkload(event.target.value)}
          >
            <option value="hello">Hello Pipeline</option>
            <option value="cats-dogs">Cats & Dogs</option>
          </select>
        </label>

        <button type="submit">Create Job</button>
      </form>

      {error && <p>{error}</p>}

      <h2>Jobs</h2>

      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Recipe</th>
            <th>Status</th>
            <th>KFP Run</th>
            <th>Katib Experiment</th>
          </tr>
        </thead>

        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td>{job.id.slice(0, 8)}</td>
              <td>{job.recipe.name}</td>
              <td>{job.status}</td>
              <td>{job.kfp_run_id ?? "-"}</td>
              <td>{job.katib_experiment_name ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

export default App;
