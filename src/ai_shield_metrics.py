"""Shared metrics for AI Shield binary detectors and fusion models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float | int]:
    """Compute the binary metrics used across the AI Shield notebooks."""
    y_true_array = np.asarray(y_true).astype(int)
    y_prob_array = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true_array, y_prob_array)),
        "average_precision": float(average_precision_score(y_true_array, y_prob_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true_array, y_prob_array)),
        "threshold": float(threshold),
        "n": int(len(y_true_array)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true, y_prob, thresholds=None, metric: str = "f1") -> tuple[float, list[dict[str, float | int]]]:
    """Evaluate metrics across thresholds and return the best threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [binary_metrics(y_true, y_prob, float(threshold)) for threshold in thresholds]
    best = max(rows, key=lambda row: float(row[metric]))
    return float(best["threshold"]), rows
