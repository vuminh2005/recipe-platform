export const TERMINAL_JOB_STATUSES = new Set(["SUCCEEDED", "FAILED"]);

export const JOB_STATUS_META = {
  PENDING: {
    label: "Pending",
    tone: "neutral",
    description: "Waiting for an Agent to claim the job.",
  },
  CLAIMED: {
    label: "Claimed",
    tone: "info",
    description: "A local Agent has claimed the job.",
  },
  RUNNING: {
    label: "Running",
    tone: "info",
    description: "The legacy workload is running.",
  },
  TUNING: {
    label: "Tuning",
    tone: "purple",
    description: "Katib is evaluating hyperparameter trials.",
  },
  TRAINING: {
    label: "Training",
    tone: "info",
    description: "KFP is running final training and evaluation.",
  },
  REGISTERING: {
    label: "Registering",
    tone: "warning",
    description: "The trained model is being registered in MLflow.",
  },
  SUCCEEDED: {
    label: "Succeeded",
    tone: "success",
    description: "The end-to-end job completed successfully.",
  },
  FAILED: {
    label: "Failed",
    tone: "danger",
    description: "The job stopped because one of its stages failed.",
  },
};

export function getJobStatusMeta(status) {
  return (
    JOB_STATUS_META[status] || {
      label: status || "Unknown",
      tone: "neutral",
      description: "Unknown platform state.",
    }
  );
}

export function isTerminalStatus(status) {
  return TERMINAL_JOB_STATUSES.has(status);
}

export function getTimelineSteps(job) {
  if (job?.recipe?.workload === "hello") {
    return ["PENDING", "CLAIMED", "RUNNING", "SUCCEEDED"];
  }

  return [
    "PENDING",
    "CLAIMED",
    "TUNING",
    "TRAINING",
    "REGISTERING",
    "SUCCEEDED",
  ];
}

export function getTimelineState(step, currentStatus, steps) {
  if (currentStatus === "FAILED") {
    if (step === "PENDING" || step === "CLAIMED") {
      return "complete";
    }

    return "upcoming";
  }

  const currentIndex = steps.indexOf(currentStatus);
  const stepIndex = steps.indexOf(step);

  if (currentIndex === -1) {
    return "upcoming";
  }

  if (stepIndex < currentIndex) {
    return "complete";
  }

  if (stepIndex === currentIndex) {
    return currentStatus === "SUCCEEDED" ? "complete" : "current";
  }

  return "upcoming";
}
