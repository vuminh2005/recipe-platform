export default function CommonJobFields({
  name,
  onChange,
  submissionToken,
  onSubmissionTokenChange,
}) {
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
        <label className="field">
          <span>Job submission token</span>
          <input
            required
            type="password"
            autoComplete="off"
            value={submissionToken}
            onChange={(event) =>
              onSubmissionTokenChange(event.target.value)
            }
          />
          <small className="field__description">
            Kept only in this page&apos;s memory and sent only when creating a
            job.
          </small>
        </label>
      </div>
    </div>
  );
}
