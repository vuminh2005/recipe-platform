export default function CommonJobFields({ name, onChange }) {
  return (
    <div className="form-section">
      <h3>Job</h3>
      <div className="form-grid form-grid--two">
        <label className="field">
          <span>Job name</span>
          <input
            required
            minLength={3}
            maxLength={100}
            value={name}
            onChange={(event) => onChange(event.target.value)}
          />
        </label>
      </div>
    </div>
  );
}
