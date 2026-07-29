function field(name, type, defaultValue, values = {}) {
  return {
    name,
    label: values.label || name,
    description: values.description || null,
    type,
    required: true,
    default: defaultValue,
    minimum: values.minimum ?? null,
    maximum: values.maximum ?? null,
    step: values.step ?? null,
    exclusive_minimum: values.exclusive_minimum || false,
    exclusive_maximum: values.exclusive_maximum || false,
    options: values.options || [],
  };
}

const commonAutoMLFields = [
  field("enabled", "boolean", true),
  field("max_trials", "integer", 3, { minimum: 1, maximum: 20 }),
  field("parallel_trials", "integer", 1, { minimum: 1, maximum: 4 }),
  field("algorithm", "string", "random", {
    options: [{ value: "random", label: "Random search" }],
  }),
];

export const catsDogsRecipe = {
  recipe_id: "cats-dogs",
  version: "1.0",
  display_name: "Cats & Dogs Image Classification",
  description: "Cats and Dogs test recipe.",
  visibility: "public",
  task_type: "binary_image_classification",
  framework: "tensorflow_keras",
  model: "mobilenet_v2",
  supports_automl: true,
  supported_algorithms: ["random"],
  objective: { name: "val_auc", direction: "maximize" },
  default_configuration: {
    training: {
      image_size: 224,
      trial_epochs: 2,
      final_epochs: 5,
      batch_size: 8,
      dense_units: 128,
      trainable_backbone: false,
    },
    automl: {
      enabled: true,
      max_trials: 3,
      parallel_trials: 1,
      algorithm: "random",
      search_space: {
        learning_rate: { min: 0.00005, max: 0.0005 },
        dropout_rate: { min: 0.15, max: 0.45 },
      },
    },
    effective_final_parameters: {
      learning_rate: 0.0003,
      dropout_rate: 0.25,
    },
  },
  configurable_training_fields: [
    field("image_size", "integer", 224, { minimum: 32, maximum: 512 }),
    field("trial_epochs", "integer", 2, { minimum: 1, maximum: 5 }),
    field("final_epochs", "integer", 5, { minimum: 1, maximum: 20 }),
    field("batch_size", "integer", 8, { minimum: 1, maximum: 32 }),
    field("dense_units", "integer", 128, { minimum: 32, maximum: 512 }),
    field("trainable_backbone", "boolean", false),
  ],
  configurable_automl_fields: commonAutoMLFields,
  configurable_search_space: [
    field("learning_rate", "range", { min: 0.00005, max: 0.0005 }, {
      minimum: 0,
      exclusive_minimum: true,
    }),
    field("dropout_rate", "range", { min: 0.15, max: 0.45 }, {
      minimum: 0,
      maximum: 1,
      exclusive_maximum: true,
    }),
  ],
};

export const tabularRecipe = {
  recipe_id: "tabular-random-forest",
  version: "1.0",
  display_name: "Tabular Random Forest Classification",
  description: "Tabular test recipe.",
  visibility: "public",
  task_type: "binary_tabular_classification",
  framework: "scikit_learn",
  model: "RandomForestClassifier",
  supports_automl: true,
  supported_algorithms: ["random"],
  objective: { name: "val_f1", direction: "maximize" },
  default_configuration: {
    training: { random_seed: 42 },
    automl: {
      enabled: true,
      max_trials: 3,
      parallel_trials: 1,
      algorithm: "random",
      search_space: {
        n_estimators: { min: 50, max: 300 },
        max_depth: { min: 2, max: 20 },
        min_samples_split: { min: 2, max: 10 },
      },
    },
    effective_final_parameters: {
      n_estimators: 200,
      max_depth: 8,
      min_samples_split: 2,
      max_features: "sqrt",
      random_seed: 42,
    },
  },
  configurable_training_fields: [
    field("random_seed", "integer", 42, {
      minimum: 0,
      maximum: 4294967295,
    }),
  ],
  configurable_automl_fields: commonAutoMLFields,
  configurable_search_space: [
    field("n_estimators", "range", { min: 50, max: 300 }, { minimum: 1 }),
    field("max_depth", "range", { min: 2, max: 20 }, { minimum: 1 }),
    field(
      "min_samples_split",
      "range",
      { min: 2, max: 10 },
      { minimum: 2 },
    ),
  ],
};
