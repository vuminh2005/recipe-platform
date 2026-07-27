import { formatKey, formatMetric } from "../utils/format";

export default function MetricsGrid({
  metrics,
  emptyMessage = "No metrics have been recorded yet.",
}) {
  const entries = Object.entries(metrics || {});

  if (entries.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>;
  }

  return (
    <dl className="metrics-grid">
      {entries.map(([key, value]) => (
        <div className="metric-card" key={key}>
          <dt>{formatKey(key)}</dt>
          <dd>{typeof value === "number" ? formatMetric(value) : String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
