import {
  getJobPresentation,
  getRecipeId,
} from "./jobPresentation.js";

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
    description: "KFP is running the internal smoke workflow.",
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
  if (getRecipeId(job) === "hello") {
    return ["PENDING", "CLAIMED", "RUNNING", "SUCCEEDED"];
  }

  const steps = ["PENDING", "CLAIMED"];
  if (getJobPresentation(job).automlEnabled) {
    steps.push("TUNING");
  }
  steps.push("TRAINING", "REGISTERING", "SUCCEEDED");
  return steps;
}

export function inferFailureStage(job, steps = getTimelineSteps(job)) {
  const presentation = getJobPresentation(job);
  const { externalIds, model, finalMetrics, bestParams } = presentation;

  if (
    externalIds.mlflow_run_id ||
    model ||
    (finalMetrics && Object.keys(finalMetrics).length > 0)
  ) {
    return steps.includes("REGISTERING") ? "REGISTERING" : "TRAINING";
  }

  if (externalIds.kfp_run_id) {
    return steps.includes("TRAINING") ? "TRAINING" : "CLAIMED";
  }

  if (
    steps.includes("TUNING") &&
    (externalIds.katib_experiment_id || bestParams)
  ) {
    return "TUNING";
  }

  if (job?.agent_id) {
    return "CLAIMED";
  }

  return "PENDING";
}

export function getTimelineState(step, job, steps) {
  if (job?.status === "FAILED") {
    const failureStage = inferFailureStage(job, steps);
    const failureIndex = steps.indexOf(failureStage);
    const stepIndex = steps.indexOf(step);

    if (stepIndex < failureIndex) {
      return "complete";
    }
    if (step === failureStage) {
      return "failed";
    }
    return "upcoming";
  }

  const currentIndex = steps.indexOf(job?.status);
  const stepIndex = steps.indexOf(step);

  if (job?.status === "SUCCEEDED") {
    return "complete";
  }
  if (currentIndex === -1) {
    return "upcoming";
  }
  if (stepIndex < currentIndex) {
    return "complete";
  }
  if (stepIndex === currentIndex) {
    return "current";
  }
  return "upcoming";
}
