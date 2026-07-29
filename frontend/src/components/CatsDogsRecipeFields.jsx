import CatalogField from "./CatalogField";

const TRAINING_FIELDS = [
  "image_size",
  "trial_epochs",
  "final_epochs",
  "batch_size",
  "dense_units",
  "trainable_backbone",
];

export default function CatsDogsRecipeFields({
  recipe,
  training,
  onChange,
}) {
  const fields = Object.fromEntries(
    recipe.configurable_training_fields.map((field) => [field.name, field]),
  );

  return (
    <div className="form-section">
      <div className="section-heading">
        <div>
          <h3>Image training</h3>
          <p className="helper-text">
            The selected recipe uses the built-in Cats &amp; Dogs dataset flow.
          </p>
        </div>
        <span className="model-pill">{recipe.model}</span>
      </div>

      <div className="form-grid">
        {TRAINING_FIELDS.map((name) => (
          <CatalogField
            key={name}
            field={fields[name]}
            value={training[name]}
            onChange={(value) => onChange(name, value)}
          />
        ))}
      </div>
    </div>
  );
}
