export default function CatalogField({
  field,
  value,
  onChange,
  disabled = false,
}) {
  if (!field) {
    return null;
  }

  if (field.type === "boolean") {
    return (
      <label className="switch-field">
        <input
          type="checkbox"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          <strong>{field.label}</strong>
          {field.description ? <small>{field.description}</small> : null}
        </span>
      </label>
    );
  }

  const options = Array.isArray(field.options) ? field.options : [];

  return (
    <label className="field">
      <span>{field.label}</span>
      {options.length > 0 ? (
        <select
          value={value ?? ""}
          disabled={disabled}
          required={field.required}
          onChange={(event) => onChange(event.target.value)}
        >
          {options.map((option) => (
            <option key={String(option.value)} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={field.type === "string" ? "text" : "number"}
          value={value ?? ""}
          disabled={disabled}
          required={field.required}
          min={field.minimum ?? undefined}
          max={field.maximum ?? undefined}
          step={field.step ?? undefined}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
      {field.description ? (
        <small className="field__description">{field.description}</small>
      ) : null}
    </label>
  );
}
