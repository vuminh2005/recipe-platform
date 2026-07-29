"""Typed metadata for the platform's explicit built-in recipes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, JsonValue


CATS_DOGS_RECIPE_ID = "cats-dogs"
HELLO_RECIPE_ID = "hello"
TABULAR_RANDOM_FOREST_RECIPE_ID = "tabular-random-forest"
RECIPE_VERSION = "1.0"

CATS_DOGS_DEFAULT_LEARNING_RATE = 0.0003
CATS_DOGS_DEFAULT_DROPOUT_RATE = 0.25
CATS_DOGS_LEARNING_RATE_MIN = 0.00005
CATS_DOGS_LEARNING_RATE_MAX = 0.0005
CATS_DOGS_DROPOUT_RATE_MIN = 0.15
CATS_DOGS_DROPOUT_RATE_MAX = 0.45

TABULAR_DEFAULT_N_ESTIMATORS = 200
TABULAR_DEFAULT_MAX_DEPTH = 8
TABULAR_DEFAULT_MIN_SAMPLES_SPLIT = 2
TABULAR_DEFAULT_MAX_FEATURES = "sqrt"
TABULAR_DEFAULT_RANDOM_SEED = 42
TABULAR_N_ESTIMATORS_MIN = 50
TABULAR_N_ESTIMATORS_MAX = 300
TABULAR_MAX_DEPTH_MIN = 2
TABULAR_MAX_DEPTH_MAX = 20
TABULAR_MIN_SAMPLES_SPLIT_MIN = 2
TABULAR_MIN_SAMPLES_SPLIT_MAX = 10

RecipeVisibility = Literal["public", "internal"]
FieldType = Literal["integer", "number", "boolean", "string", "range"]
ObjectiveDirection = Literal["maximize", "minimize"]


class ObjectiveDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    direction: ObjectiveDirection


class FieldOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: JsonValue
    label: str


class ConfigurableFieldDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    description: str | None = None
    type: FieldType
    required: bool
    default: JsonValue
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    options: tuple[FieldOption, ...] = ()


class RecipeDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    recipe_id: str
    version: str
    display_name: str
    description: str
    visibility: RecipeVisibility
    task_type: str
    framework: str | None
    model: str | None
    supports_automl: bool
    supported_algorithms: tuple[str, ...]
    objective: ObjectiveDefinition | None
    default_configuration: dict[str, JsonValue]
    configurable_training_fields: tuple[ConfigurableFieldDefinition, ...]
    configurable_automl_fields: tuple[ConfigurableFieldDefinition, ...]
    configurable_search_space: tuple[ConfigurableFieldDefinition, ...]


CATS_DOGS_DEFINITION = RecipeDefinition(
    recipe_id=CATS_DOGS_RECIPE_ID,
    version=RECIPE_VERSION,
    display_name="Cats & Dogs Image Classification",
    description=(
        "Binary image classification with a MobileNetV2 transfer-learning model."
    ),
    visibility="public",
    task_type="binary_image_classification",
    framework="tensorflow_keras",
    model="mobilenet_v2",
    supports_automl=True,
    supported_algorithms=("random",),
    objective=ObjectiveDefinition(name="val_auc", direction="maximize"),
    default_configuration={
        "training": {
            "image_size": 224,
            "trial_epochs": 2,
            "final_epochs": 5,
            "batch_size": 8,
            "dense_units": 128,
            "trainable_backbone": False,
        },
        "automl": {
            "enabled": True,
            "max_trials": 3,
            "parallel_trials": 1,
            "algorithm": "random",
            "search_space": {
                "learning_rate": {
                    "min": CATS_DOGS_LEARNING_RATE_MIN,
                    "max": CATS_DOGS_LEARNING_RATE_MAX,
                },
                "dropout_rate": {
                    "min": CATS_DOGS_DROPOUT_RATE_MIN,
                    "max": CATS_DOGS_DROPOUT_RATE_MAX,
                },
            },
        },
        "effective_final_parameters": {
            "learning_rate": CATS_DOGS_DEFAULT_LEARNING_RATE,
            "dropout_rate": CATS_DOGS_DEFAULT_DROPOUT_RATE,
        },
    },
    configurable_training_fields=(
        ConfigurableFieldDefinition(
            name="image_size",
            label="Image size",
            description="Square input image dimension in pixels.",
            type="integer",
            required=True,
            default=224,
            minimum=32,
            maximum=512,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="trial_epochs",
            label="Trial epochs",
            description="Training epochs used by each Katib trial.",
            type="integer",
            required=True,
            default=2,
            minimum=1,
            maximum=5,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="final_epochs",
            label="Final epochs",
            description="Training epochs used by the final KFP run.",
            type="integer",
            required=True,
            default=5,
            minimum=1,
            maximum=20,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="batch_size",
            label="Batch size",
            type="integer",
            required=True,
            default=8,
            minimum=1,
            maximum=32,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="dense_units",
            label="Dense units",
            description="Units in the recipe's dense classification layer.",
            type="integer",
            required=True,
            default=128,
            minimum=32,
            maximum=512,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="trainable_backbone",
            label="Trainable backbone",
            description="Whether MobileNetV2 backbone weights are fine-tuned.",
            type="boolean",
            required=True,
            default=False,
        ),
    ),
    configurable_automl_fields=(
        ConfigurableFieldDefinition(
            name="enabled",
            label="Enable AutoML",
            description="Run Katib before final training.",
            type="boolean",
            required=True,
            default=True,
        ),
        ConfigurableFieldDefinition(
            name="max_trials",
            label="Maximum trials",
            type="integer",
            required=True,
            default=3,
            minimum=1,
            maximum=20,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="parallel_trials",
            label="Parallel trials",
            type="integer",
            required=True,
            default=1,
            minimum=1,
            maximum=4,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="algorithm",
            label="Search algorithm",
            description="Katib algorithm used for hyperparameter search.",
            type="string",
            required=True,
            default="random",
            options=(
                FieldOption(value="random", label="Random search"),
            ),
        ),
    ),
    configurable_search_space=(
        ConfigurableFieldDefinition(
            name="learning_rate",
            label="Learning rate range",
            description="Learning-rate range searched by Katib.",
            type="range",
            required=True,
            default={
                "min": CATS_DOGS_LEARNING_RATE_MIN,
                "max": CATS_DOGS_LEARNING_RATE_MAX,
            },
            minimum=0,
            step=0.00001,
            exclusive_minimum=True,
        ),
        ConfigurableFieldDefinition(
            name="dropout_rate",
            label="Dropout rate range",
            description="Dropout-rate range searched by Katib.",
            type="range",
            required=True,
            default={
                "min": CATS_DOGS_DROPOUT_RATE_MIN,
                "max": CATS_DOGS_DROPOUT_RATE_MAX,
            },
            minimum=0,
            maximum=1,
            step=0.01,
            exclusive_maximum=True,
        ),
    ),
)

HELLO_DEFINITION = RecipeDefinition(
    recipe_id=HELLO_RECIPE_ID,
    version=RECIPE_VERSION,
    display_name="Hello KFP Smoke Test",
    description="Internal KFP connectivity smoke test; it is not an ML workload.",
    visibility="internal",
    task_type="smoke_test",
    framework=None,
    model=None,
    supports_automl=False,
    supported_algorithms=(),
    objective=None,
    default_configuration={},
    configurable_training_fields=(),
    configurable_automl_fields=(),
    configurable_search_space=(),
)

TABULAR_RANDOM_FOREST_DEFINITION = RecipeDefinition(
    recipe_id=TABULAR_RANDOM_FOREST_RECIPE_ID,
    version=RECIPE_VERSION,
    display_name="Tabular Random Forest Classification",
    description=(
        "CPU-only binary tabular classification with RandomForestClassifier "
        "and the built-in scikit-learn breast-cancer dataset."
    ),
    visibility="public",
    task_type="binary_tabular_classification",
    framework="scikit_learn",
    model="RandomForestClassifier",
    supports_automl=True,
    supported_algorithms=("random",),
    objective=ObjectiveDefinition(name="val_f1", direction="maximize"),
    default_configuration={
        "training": {
            "random_seed": TABULAR_DEFAULT_RANDOM_SEED,
        },
        "automl": {
            "enabled": True,
            "max_trials": 3,
            "parallel_trials": 1,
            "algorithm": "random",
            "search_space": {
                "n_estimators": {
                    "min": TABULAR_N_ESTIMATORS_MIN,
                    "max": TABULAR_N_ESTIMATORS_MAX,
                },
                "max_depth": {
                    "min": TABULAR_MAX_DEPTH_MIN,
                    "max": TABULAR_MAX_DEPTH_MAX,
                },
                "min_samples_split": {
                    "min": TABULAR_MIN_SAMPLES_SPLIT_MIN,
                    "max": TABULAR_MIN_SAMPLES_SPLIT_MAX,
                },
            },
        },
        "effective_final_parameters": {
            "n_estimators": TABULAR_DEFAULT_N_ESTIMATORS,
            "max_depth": TABULAR_DEFAULT_MAX_DEPTH,
            "min_samples_split": TABULAR_DEFAULT_MIN_SAMPLES_SPLIT,
            "max_features": TABULAR_DEFAULT_MAX_FEATURES,
            "random_seed": TABULAR_DEFAULT_RANDOM_SEED,
        },
    },
    configurable_training_fields=(
        ConfigurableFieldDefinition(
            name="random_seed",
            label="Random seed",
            description=(
                "Seed used for deterministic data splitting and model training."
            ),
            type="integer",
            required=True,
            default=TABULAR_DEFAULT_RANDOM_SEED,
            minimum=0,
            maximum=4_294_967_295,
            step=1,
        ),
    ),
    configurable_automl_fields=(
        ConfigurableFieldDefinition(
            name="enabled",
            label="Enable AutoML",
            description="Run Katib before final training.",
            type="boolean",
            required=True,
            default=True,
        ),
        ConfigurableFieldDefinition(
            name="max_trials",
            label="Maximum trials",
            type="integer",
            required=True,
            default=3,
            minimum=1,
            maximum=20,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="parallel_trials",
            label="Parallel trials",
            type="integer",
            required=True,
            default=1,
            minimum=1,
            maximum=4,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="algorithm",
            label="Search algorithm",
            description="Katib algorithm used for hyperparameter search.",
            type="string",
            required=True,
            default="random",
            options=(FieldOption(value="random", label="Random search"),),
        ),
    ),
    configurable_search_space=(
        ConfigurableFieldDefinition(
            name="n_estimators",
            label="Tree count range",
            description="Number of trees searched by Katib.",
            type="range",
            required=True,
            default={
                "min": TABULAR_N_ESTIMATORS_MIN,
                "max": TABULAR_N_ESTIMATORS_MAX,
            },
            minimum=1,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="max_depth",
            label="Maximum tree depth range",
            description="Maximum tree depth searched by Katib.",
            type="range",
            required=True,
            default={
                "min": TABULAR_MAX_DEPTH_MIN,
                "max": TABULAR_MAX_DEPTH_MAX,
            },
            minimum=1,
            step=1,
        ),
        ConfigurableFieldDefinition(
            name="min_samples_split",
            label="Minimum samples per split range",
            description=(
                "Minimum sample count required to split a node, searched by Katib."
            ),
            type="range",
            required=True,
            default={
                "min": TABULAR_MIN_SAMPLES_SPLIT_MIN,
                "max": TABULAR_MIN_SAMPLES_SPLIT_MAX,
            },
            minimum=2,
            step=1,
        ),
    ),
)

CATALOG: dict[str, RecipeDefinition] = {
    CATS_DOGS_RECIPE_ID: CATS_DOGS_DEFINITION,
    HELLO_RECIPE_ID: HELLO_DEFINITION,
    TABULAR_RANDOM_FOREST_RECIPE_ID: TABULAR_RANDOM_FOREST_DEFINITION,
}
CATALOG_RECIPE_IDS = frozenset(CATALOG)


def get_recipe_definition(recipe_id: str) -> RecipeDefinition | None:
    return CATALOG.get(recipe_id)


def list_public_recipes() -> list[RecipeDefinition]:
    return [
        definition
        for definition in CATALOG.values()
        if definition.visibility == "public"
    ]


router = APIRouter(prefix="/api/recipes", tags=["recipes"])


@router.get("", response_model=list[RecipeDefinition])
async def list_recipes() -> list[RecipeDefinition]:
    return list_public_recipes()


@router.get("/{recipe_id}", response_model=RecipeDefinition)
async def get_public_recipe(recipe_id: str) -> RecipeDefinition:
    definition = get_recipe_definition(recipe_id)
    if definition is None or definition.visibility != "public":
        raise HTTPException(status_code=404, detail="Recipe not found")
    return definition
