export default function ValidationErrors({ issues, tone = "danger" }) {
  if (!issues?.length) {
    return null;
  }

  return (
    <div className={`alert alert--${tone}`} role="alert">
      <strong>Please review the following:</strong>
      <ul className="validation-list">
        {issues.map((issue, index) => (
          <li key={`${issue.path || "request"}-${index}`}>
            {issue.path ? <code>{issue.path}</code> : null}
            {issue.path ? ": " : ""}
            {issue.message}
          </li>
        ))}
      </ul>
    </div>
  );
}
