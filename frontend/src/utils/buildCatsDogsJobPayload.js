export function toPositiveInteger(value) {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : 0
}

export function buildCatsDogsJobPayload(form) {
  return {
    name: form.name.trim(),
    workload: "cats-dogs",
    training: {
      model: "mobilenet_v2",
      image_size: toPositiveInteger(form.imageSize),
      trial_epochs: toPositiveInteger(form.trialEpochs),
      final_epochs: toPositiveInteger(form.finalEpochs),
      batch_size: toPositiveInteger(form.batchSize),
      dense_units: toPositiveInteger(form.denseUnits),
      trainable_backbone: form.trainableBackbone,
    },
    automl: {
      enabled: form.automlEnabled,
      max_trials: toPositiveInteger(form.maxTrials),
      parallel_trials: toPositiveInteger(form.parallelTrials),
      algorithm: form.algorithm,
    },
  }
}
