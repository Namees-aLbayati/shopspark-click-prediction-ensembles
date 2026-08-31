"""Fast unit tests for reusable modeling helpers."""

import numpy as np
import pandas as pd

from src.modeling.evaluation import probability_metrics
from src.modeling.ranking import add_score_deciles, decile_summary


def test_probability_metrics_rank_good_scores_above_prevalence():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.01, 0.05, 0.10, 0.80, 0.90])
    metrics = probability_metrics(y, scores)
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_score_deciles_preserve_rows_and_rank_high_scores_highest():
    frame = pd.DataFrame({"clicked": [0, 0, 1, 1], "predicted_click_probability": [0.1, 0.2, 0.8, 0.9]})
    scored = add_score_deciles(frame, bins=2)
    summary = decile_summary(scored)
    assert len(scored) == len(frame)
    assert summary.loc[2, "actual_ctr"] > summary.loc[1, "actual_ctr"]

