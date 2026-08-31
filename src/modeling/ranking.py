"""Convert click probabilities into ranking outputs."""

import pandas as pd


def add_score_deciles(
    frame: pd.DataFrame,
    probability_column: str = "predicted_click_probability",
    bins: int = 10,
) -> pd.DataFrame:
    """Add stable equal-sized score groups, where the highest bin ranks best."""
    result = frame.copy()
    result["score_decile"] = pd.qcut(
        result[probability_column].rank(method="first"),
        q=bins,
        labels=range(1, bins + 1),
    ).astype("int8")
    return result


def decile_summary(
    frame: pd.DataFrame,
    target: str = "clicked",
    probability_column: str = "predicted_click_probability",
) -> pd.DataFrame:
    """Summarize CTR, lift, and click capture by prediction-score decile."""
    summary = (
        frame.groupby("score_decile", observed=True)
        .agg(
            impressions=(target, "size"),
            clicks=(target, "sum"),
            actual_ctr=(target, "mean"),
            average_score=(probability_column, "mean"),
        )
        .sort_index(ascending=False)
    )
    summary["ctr_lift"] = summary["actual_ctr"] / frame[target].mean()
    summary["click_capture_rate"] = summary["clicks"] / frame[target].sum()
    summary["cumulative_click_capture"] = summary["click_capture_rate"].cumsum()
    return summary

