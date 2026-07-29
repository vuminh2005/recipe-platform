"""Random Forest construction and parameter validation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_model(
    *,
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    random_seed: int = 42,
) -> RandomForestClassifier:
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if min_samples_split < 2:
        raise ValueError("min_samples_split must be at least 2")
    if max_features != "sqrt":
        raise ValueError("max_features must be 'sqrt'")
    if not 0 <= random_seed <= 4_294_967_295:
        raise ValueError("random_seed is outside the supported range")

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=1,
    )
