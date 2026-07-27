import { useState } from "react";
import { createJob } from "../api/jobs";

const INITIAL_FORM = {
  name: "cats-dogs-recipe",
  imageSize: 224,
  trialEpochs: 2,
  finalEpochs: 5,
  batchSize: 8,
  denseUnits: 128,
  trainableBackbone: false,
  automlEnabled: true,
  maxTrials: 3,
  parallelTrials: 1,
  algorithm: "random",
};

function toPositiveInteger(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function CreateJobForm({ onCreated }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  function updateField(name, value) {
    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (submitting) {
      return;
    }

    setSubmitting(true);
    setError("");

    const payload = {
      name: form.name.trim(),
      workload: "cats-dogs",
      training: {
        model: "mobilenet_v2",
        image_size: toPositiveInteger(form.imageSize),
        trial_epochs: toPositiveInteger(form.trialEpochs),
        final_epochs: toPositiveInteger(form.finalEpochs),
        batch_size: toPositiveInteger(form.batchSize),
        dense_units: toPositiveInteger(form.denseUnits),
        trainable_backbone: form.trainableBackbone,
      },
      automl: {
        enabled: form.automlEnabled,
        max_trials: toPositiveInteger(form.maxTrials),
        parallel_trials: toPositiveInteger(form.parallelTrials),
        algorithm: form.algorithm,
      },
    };

    try {
      const createdJob = await createJob(payload);
      onCreated?.(createdJob);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="panel create-job-form" onSubmit={handleSubmit}>
      <div className="panel__header">
        <div>
          <p className="eyebrow">New orchestration run</p>
          <h2>Create Cats &amp; Dogs Recipe</h2>
        </div>
        <span className="model-pill">MobileNetV2</span>
      </div>

      <div className="form-section">
        <h3>Recipe</h3>

        <div className="form-grid form-grid--two">
          <label className="field">
            <span>Recipe name</span>
            <input
              required
              minLength={3}
              maxLength={100}
              value={form.name}
              onChange={(event) => updateField("name", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Workload</span>
            <input value="cats-dogs" disabled />
          </label>
        </div>
      </div>

      <div className="form-section">
        <h3>Training</h3>

        <div className="form-grid">
          <label className="field">
            <span>Image size</span>
            <input
              type="number"
              min={32}
              max={512}
              value={form.imageSize}
              onChange={(event) => updateField("imageSize", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Trial epochs</span>
            <input
              type="number"
              min={1}
              max={5}
              value={form.trialEpochs}
              onChange={(event) => updateField("trialEpochs", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Final epochs</span>
            <input
              type="number"
              min={1}
              max={20}
              value={form.finalEpochs}
              onChange={(event) => updateField("finalEpochs", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              max={32}
              value={form.batchSize}
              onChange={(event) => updateField("batchSize", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Dense units</span>
            <input
              type="number"
              min={32}
              max={512}
              value={form.denseUnits}
              onChange={(event) => updateField("denseUnits", event.target.value)}
            />
          </label>

          <label className="switch-field">
            <input
              type="checkbox"
              checked={form.trainableBackbone}
              onChange={(event) =>
                updateField("trainableBackbone", event.target.checked)
              }
            />
            <span>
              <strong>Trainable backbone</strong>
              <small>Keep disabled for the first CPU-based demo.</small>
            </span>
          </label>
        </div>
      </div>

      <div className="form-section">
        <h3>AutoML</h3>

        <div className="form-grid">
          <label className="switch-field">
            <input
              type="checkbox"
              checked={form.automlEnabled}
              onChange={(event) =>
                updateField("automlEnabled", event.target.checked)
              }
            />
            <span>
              <strong>Enable Katib tuning</strong>
              <small>Search learning rate and dropout rate.</small>
            </span>
          </label>

          <label className="field">
            <span>Maximum trials</span>
            <input
              type="number"
              min={1}
              max={20}
              value={form.maxTrials}
              disabled={!form.automlEnabled}
              onChange={(event) => updateField("maxTrials", event.target.value)}
            />
          </label>

          <label className="field">
            <span>Parallel trials</span>
            <input
              type="number"
              min={1}
              max={4}
              value={form.parallelTrials}
              disabled={!form.automlEnabled}
              onChange={(event) =>
                updateField("parallelTrials", event.target.value)
              }
            />
          </label>

          <label className="field">
            <span>Algorithm</span>
            <select
              value={form.algorithm}
              disabled={!form.automlEnabled}
              onChange={(event) => updateField("algorithm", event.target.value)}
            >
              <option value="random">Random search</option>
            </select>
          </label>
        </div>
      </div>

      {error ? <div className="alert alert--danger">{error}</div> : null}

      <div className="form-actions">
        <button className="button button--primary" type="submit" disabled={submitting}>
          {submitting ? "Creating job..." : "Run Recipe"}
        </button>
      </div>
    </form>
  );
}
