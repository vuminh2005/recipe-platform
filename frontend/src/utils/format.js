const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 6,
});

const dateFormatter = new Intl.DateTimeFormat("vi-VN", {
  dateStyle: "medium",
  timeStyle: "medium",
});

export function formatNumber(value, fallback = "—") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return fallback;
  }

  return numberFormatter.format(numericValue);
}

export function formatMetric(value, fallback = "—") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return fallback;
  }

  return numericValue.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

export function formatPercent(value, fallback = "—") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return fallback;
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

export function formatDate(value, fallback = "—") {
  if (!value) {
    return fallback;
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return fallback;
  }

  return dateFormatter.format(date);
}

export function formatBoolean(value) {
  return value ? "Enabled" : "Disabled";
}

export function formatKey(key) {
  return String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function formatObjectiveLabel(objective) {
  if (!objective?.name) {
    return "Objective";
  }

  const direction =
    objective.direction === "maximize"
      ? "↑"
      : objective.direction === "minimize"
        ? "↓"
        : "";

  return `${objective.name}${direction ? ` ${direction}` : ""}`;
}

export function formatJsonValue(value, fallback = "—") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  if (typeof value === "number") {
    return formatMetric(value, fallback);
  }

  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

export function truncateMiddle(value, visibleCharacters = 12) {
  if (!value) {
    return "—";
  }

  const text = String(value);

  if (text.length <= visibleCharacters * 2 + 3) {
    return text;
  }

  return `${text.slice(0, visibleCharacters)}...${text.slice(
    -visibleCharacters,
  )}`;
}
