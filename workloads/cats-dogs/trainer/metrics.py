from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.10, 0.90, 81)

    rows = []

    for threshold in thresholds:
        predictions = (
            y_proba >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(
                    accuracy_score(y_true, predictions)
                ),
                "f1": float(
                    f1_score(y_true, predictions)
                ),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            ["f1", "accuracy"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return float(results.iloc[0]["threshold"]), results
