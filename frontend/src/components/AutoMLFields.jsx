import { formatJsonValue, formatKey } from "../utils/format";
import CatalogField from "./CatalogField";

function RangeField({ field, range, integer, onChange }) {
  return (
    <fieldset className="range-field">
      <legend>{field.label}</legend>
      {field.description ? <p>{field.description}</p> : null}
      <div className="range-field__inputs">
        {["min", "max"].map((endpoint) => (
          <label className="field" key={endpoint}>
            <span>{formatKey(endpoint)}</span>
            <input
              type="number"
              value={range?.[endpoint] ?? ""}
              min={field.minimum ?? undefined}
              max={field.maximum ?? undefined}
              step={integer ? 1 : (field.step ?? "any")}
              onChange={(event) => onChange(endpoint, event.target.value)}
            />
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function EffectiveParameters({ parameters }) {
  return (
    <div className="effective-parameters">
      <strong>Effective final parameters</strong>
      <p>
        Katib will be skipped. KFP will train directly with these recipe-owned
        values.
      </p>
      <dl>
        {Object.entries(parameters || {}).map(([key, value]) => (
          <div key={key}>
            <dt>{formatKey(key)}</dt>
            <dd>{formatJsonValue(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function AutoMLFields({
  recipe,
  automl,
  effectiveParameters,
  onFieldChange,
  onRangeChange,
}) {
  const fields = Object.fromEntries(
    recipe.configurable_automl_fields.map((field) => [field.name, field]),
  );
  const enabled = Boolean(automl.enabled);
  const integerRanges = recipe.recipe_id === "tabular-random-forest";

  return (
    <div className="form-section">
      <div className="section-heading">
        <div>
          <h3>AutoML</h3>
          <p className="helper-text">
            Objective: <strong>{recipe.objective.name}</strong> ·{" "}
            {recipe.objective.direction}
          </p>
        </div>
        <span className={`automl-mode automl-mode--${enabled ? "on" : "off"}`}>
          {enabled ? "Katib enabled" : "Direct KFP"}
        </span>
      </div>

      <div className="form-grid">
        <CatalogField
          field={fields.enabled}
          value={automl.enabled}
          onChange={(value) => onFieldChange("enabled", value)}
        />
        {["max_trials", "parallel_trials", "algorithm"].map((name) => (
          <CatalogField
            key={name}
            field={fields[name]}
            value={automl[name]}
            disabled={!enabled}
            onChange={(value) => onFieldChange(name, value)}
          />
        ))}
      </div>

      {enabled ? (
        <>
          <p className="helper-text">
            Katib will tune the following ranges before the final KFP run.
          </p>
          <div className="search-space-grid">
            {recipe.configurable_search_space.map((field) => (
              <RangeField
                key={field.name}
                field={field}
                integer={integerRanges}
                range={automl.search_space?.[field.name]}
                onChange={(endpoint, value) =>
                  onRangeChange(field.name, endpoint, value)
                }
              />
            ))}
          </div>
        </>
      ) : (
        <EffectiveParameters parameters={effectiveParameters} />
      )}
    </div>
  );
}
