"""Plotting helpers for AI Shield reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_reliability_plot(y_true, y_prob, title: str, output_path: str | Path, n_bins: int = 10) -> Path:
    """Save a reliability curve for a probability-producing detector."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fake fraction")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_pr_comparison(y_true, model_probs: dict[str, np.ndarray], output_path: str | Path) -> Path:
    """Save side-by-side ROC and precision-recall comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, probs in model_probs.items():
        RocCurveDisplay.from_predictions(y_true, probs, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, probs, name=name, ax=axes[1])
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix(y_true, y_pred, title: str, output_path: str | Path) -> Path:
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["real", "fake"], ax=ax)
    ax.set_title(title)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
