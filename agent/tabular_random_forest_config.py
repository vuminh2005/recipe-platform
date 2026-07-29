"""Lightweight recipe-owned constants and persisted configuration validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RECIPE_VERSION = "1.0"
SUPPORTED_ALGORITHMS = frozenset({"random"})

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 8
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_RANDOM_SEED = 42

DEFAULT_N_ESTIMATORS_MIN = 50
DEFAULT_N_ESTIMATORS_MAX = 300
DEFAULT_MAX_DEPTH_MIN = 2
DEFAULT_MAX_DEPTH_MAX = 20
DEFAULT_MIN_SAMPLES_SPLIT_MIN = 2
DEFAULT_MIN_SAMPLES_SPLIT_MAX = 10


@dataclass(frozen=True)
class TabularExecutionConfig:
    automl_enabled: bool
    max_trial_count: int
    parallel_trial_count: int
    algorithm_name: str
    n_estimators_min: int
    n_estimators_max: int
    max_depth_min: int
    max_depth_max: int
    min_samples_split_min: int
    min_samples_split_max: int
    random_seed: int
    final_n_estimators: int
    final_max_depth: int
    final_min_samples_split: int
    final_max_features: str

    @property
    def effective_final_parameters(self) -> dict[str, int | str]:
        return {
            "n_estimators": self.final_n_estimators,
            "max_depth": self.final_max_depth,
            "min_samples_split": self.final_min_samples_split,
            "max_features": self.final_max_features,
            "random_seed": self.random_seed,
        }


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed != value and not (
        isinstance(value, str) and value.strip() == str(parsed)
    ):
        raise ValueError(f"{field} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return parsed


def _range(
    search_space: dict[str, Any],
    *,
    name: str,
    default_minimum: int,
    default_maximum: int,
    lower_bound: int,
) -> tuple[int, int]:
    raw = search_space.get(name) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"automl.search_space.{name} must be an object")
    minimum = _integer(
        raw.get("min", default_minimum),
        field=f"automl.search_space.{name}.min",
        minimum=lower_bound,
    )
    maximum = _integer(
        raw.get("max", default_maximum),
        field=f"automl.search_space.{name}.max",
        minimum=lower_bound,
    )
    if minimum >= maximum:
        raise ValueError(
            f"automl.search_space.{name} must satisfy min < max"
        )
    return minimum, maximum


def validate_execution_config(
    configuration: dict[str, Any],
) -> TabularExecutionConfig:
    if not isinstance(configuration, dict):
        raise ValueError("recipe.configuration must be an object")

    training = configuration.get("training") or {}
    automl = configuration.get("automl") or {}
    if not isinstance(training, dict):
        raise ValueError("configuration.training must be an object")
    if not isinstance(automl, dict):
        raise ValueError("configuration.automl must be an object")

    enabled_value = automl.get("enabled", True)
    if not isinstance(enabled_value, bool):
        raise ValueError("automl.enabled must be a boolean")

    random_seed = _integer(
        training.get("random_seed", DEFAULT_RANDOM_SEED),
        field="training.random_seed",
        minimum=0,
    )
    if random_seed > 4_294_967_295:
        raise ValueError("training.random_seed must be at most 4294967295")

    max_trials = _integer(
        automl.get("max_trials", 3),
        field="automl.max_trials",
        minimum=1,
    )
    if max_trials > 20:
        raise ValueError("automl.max_trials must be at most 20")
    parallel_trials = _integer(
        automl.get("parallel_trials", 1),
        field="automl.parallel_trials",
        minimum=1,
    )
    if parallel_trials > 4:
        raise ValueError("automl.parallel_trials must be at most 4")
    if parallel_trials > max_trials:
        raise ValueError(
            "automl.parallel_trials must be less than or equal to "
            "automl.max_trials"
        )

    algorithm = str(automl.get("algorithm", "random"))
    if algorithm not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(
            f"Unsupported automl.algorithm {algorithm!r}; supported: {supported}"
        )

    search_space = automl.get("search_space") or {}
    if not isinstance(search_space, dict):
        raise ValueError("automl.search_space must be an object")
    n_estimators_min, n_estimators_max = _range(
        search_space,
        name="n_estimators",
        default_minimum=DEFAULT_N_ESTIMATORS_MIN,
        default_maximum=DEFAULT_N_ESTIMATORS_MAX,
        lower_bound=1,
    )
    max_depth_min, max_depth_max = _range(
        search_space,
        name="max_depth",
        default_minimum=DEFAULT_MAX_DEPTH_MIN,
        default_maximum=DEFAULT_MAX_DEPTH_MAX,
        lower_bound=1,
    )
    min_samples_split_min, min_samples_split_max = _range(
        search_space,
        name="min_samples_split",
        default_minimum=DEFAULT_MIN_SAMPLES_SPLIT_MIN,
        default_maximum=DEFAULT_MIN_SAMPLES_SPLIT_MAX,
        lower_bound=2,
    )

    effective = configuration.get("effective_final_parameters")
    if enabled_value:
        effective = {}
    elif effective is None:
        effective = {}
    if not isinstance(effective, dict):
        raise ValueError("effective_final_parameters must be an object or null")

    final_n_estimators = _integer(
        effective.get("n_estimators", DEFAULT_N_ESTIMATORS),
        field="effective_final_parameters.n_estimators",
        minimum=1,
    )
    final_max_depth = _integer(
        effective.get("max_depth", DEFAULT_MAX_DEPTH),
        field="effective_final_parameters.max_depth",
        minimum=1,
    )
    final_min_samples_split = _integer(
        effective.get("min_samples_split", DEFAULT_MIN_SAMPLES_SPLIT),
        field="effective_final_parameters.min_samples_split",
        minimum=2,
    )
    final_max_features = str(
        effective.get("max_features", DEFAULT_MAX_FEATURES)
    )
    if final_max_features != DEFAULT_MAX_FEATURES:
        raise ValueError(
            "effective_final_parameters.max_features must be 'sqrt'"
        )
    effective_seed = _integer(
        effective.get("random_seed", random_seed),
        field="effective_final_parameters.random_seed",
        minimum=0,
    )
    if effective_seed != random_seed:
        raise ValueError(
            "effective_final_parameters.random_seed must match "
            "training.random_seed"
        )

    return TabularExecutionConfig(
        automl_enabled=enabled_value,
        max_trial_count=max_trials,
        parallel_trial_count=parallel_trials,
        algorithm_name=algorithm,
        n_estimators_min=n_estimators_min,
        n_estimators_max=n_estimators_max,
        max_depth_min=max_depth_min,
        max_depth_max=max_depth_max,
        min_samples_split_min=min_samples_split_min,
        min_samples_split_max=min_samples_split_max,
        random_seed=random_seed,
        final_n_estimators=final_n_estimators,
        final_max_depth=final_max_depth,
        final_min_samples_split=final_min_samples_split,
        final_max_features=final_max_features,
    )
