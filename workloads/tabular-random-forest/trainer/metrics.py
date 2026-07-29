"""Binary classification metric calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model: Any,
    features: Any,
    target: Any,
    *,
    prefix: str,
) -> tuple[dict[str, float], list[list[int]]]:
    predictions = np.asarray(model.predict(features), dtype=int)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Binary classifier predict_proba output must have two columns"
        )
    positive_probabilities = probabilities[:, 1]

    metrics = {
        f"{prefix}_accuracy": float(
            accuracy_score(target, predictions)
        ),
        f"{prefix}_precision": float(
            precision_score(target, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(target, predictions, zero_division=0)
        ),
        f"{prefix}_f1": float(
            f1_score(target, predictions, zero_division=0)
        ),
        f"{prefix}_roc_auc": float(
            roc_auc_score(target, positive_probabilities)
        ),
    }
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return metrics, matrix.astype(int).tolist()
