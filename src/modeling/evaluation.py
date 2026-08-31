"""Shared classification and ranking metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def top_fraction_metrics(
    y_true,
    probability,
    fraction: float = 0.10,
) -> dict[str, float]:
    """Calculate CTR, lift, and click capture among the highest scores."""
    y = np.asarray(y_true)
    scores = np.asarray(probability)
    cutoff = max(1, int(len(y) * fraction))
    top_indices = np.argsort(scores)[-cutoff:]
    base_ctr = float(y.mean())
    top_ctr = float(y[top_indices].mean())
    captured = float(y[top_indices].sum() / y.sum()) if y.sum() else np.nan
    return {
        "top_fraction_ctr": top_ctr,
        "top_fraction_lift": top_ctr / base_ctr if base_ctr else np.nan,
        "top_fraction_click_capture": captured,
    }


def probability_metrics(y_true, probability) -> dict[str, float]:
    """Return the core probability and ranking metrics for click prediction."""
    ranking = top_fraction_metrics(y_true, probability)
    return {
        "pr_auc": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "log_loss": log_loss(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "top_10pct_ctr": ranking["top_fraction_ctr"],
        "top_10pct_lift": ranking["top_fraction_lift"],
        "top_10pct_click_capture": ranking["top_fraction_click_capture"],
    }

