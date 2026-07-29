import CatalogField from "./CatalogField";

export default function TabularRandomForestRecipeFields({
  recipe,
  training,
  onChange,
}) {
  const randomSeedField = recipe.configurable_training_fields.find(
    (field) => field.name === "random_seed",
  );

  return (
    <div className="form-section">
      <div className="section-heading">
        <div>
          <h3>Tabular training</h3>
          <p className="helper-text">
            Uses scikit-learn&apos;s built-in breast cancer dataset. No download
            or dataset upload is required.
          </p>
        </div>
        <span className="model-pill">{recipe.model}</span>
      </div>

      <div className="read-only-note">
        <strong>Dataset</strong>
        <span>Breast cancer Wisconsin diagnostic dataset</span>
        <small>Binary tabular classification · CPU only</small>
      </div>

      <div className="form-grid">
        <CatalogField
          field={randomSeedField}
          value={training.random_seed}
          onChange={(value) => onChange("random_seed", value)}
        />
      </div>
    </div>
  );
}
