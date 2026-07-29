function numericValue(value, label) {
  if (value === "" || value === null || value === undefined) {
    throw new Error(`${label} is required.`);
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${label} must be a number.`);
  }
  return numeric;
}

export function toNumber(value, label) {
  return numericValue(value, label);
}

export function toInteger(value, label) {
  const numeric = numericValue(value, label);
  if (!Number.isInteger(numeric)) {
    throw new Error(`${label} must be an integer.`);
  }
  return numeric;
}

export function toBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }
  if (value === "true" || value === 1 || value === "1") {
    return true;
  }
  if (value === "false" || value === 0 || value === "0") {
    return false;
  }
  return Boolean(value);
}

export function valueOrDefault(value, fallback) {
  return value === undefined || value === null || value === ""
    ? fallback
    : value;
}
