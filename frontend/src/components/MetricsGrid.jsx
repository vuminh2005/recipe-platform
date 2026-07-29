import { formatJsonValue, formatKey } from "../utils/format";

export default function MetricsGrid({
  metrics,
  emptyMessage = "No values have been recorded yet.",
}) {
  const entries =
    metrics && typeof metrics === "object" ? Object.entries(metrics) : [];

  if (entries.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>;
  }

  return (
    <dl className="metrics-grid">
      {entries.map(([key, value]) => (
        <div className="metric-card" key={key}>
          <dt>{formatKey(key)}</dt>
          <dd>{formatJsonValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
