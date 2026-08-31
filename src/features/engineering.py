"""Leakage-aware feature engineering functions."""

import numpy as np
import pandas as pd


def add_row_features(
    frame: pd.DataFrame,
    training_start: pd.Timestamp,
    category_price_median: pd.Series,
    category_rating_mean: pd.Series,
) -> pd.DataFrame:
    """Create time, value, demographic, and context-interaction features."""
    result = frame.copy()
    result["event_month"] = result["event_ts"].dt.month.astype("int8")
    result["event_weekday_number"] = result["event_ts"].dt.dayofweek.astype("int8")
    result["is_weekend"] = result["event_weekday_number"].isin([5, 6]).astype("int8")
    result["hour_sin"] = np.sin(2 * np.pi * result["hour_of_day"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour_of_day"] / 24)
    result["days_since_training_start"] = (
        result["event_ts"] - training_start
    ).dt.total_seconds() / 86_400
    result["has_discount"] = result["discount_pct"].gt(0).astype("int8")
    result["discount_amount_usd"] = result["price_usd"] * result["discount_pct"] / 100
    result["discounted_price_usd"] = result["price_usd"] - result["discount_amount_usd"]
    result["log_price_usd"] = np.log1p(result["price_usd"].clip(lower=0))
    result["log_review_count"] = np.log1p(result["review_count"].clip(lower=0))
    result["category_median_price"] = result["category"].map(category_price_median)
    result["price_to_category_median"] = (
        result["price_usd"] / result["category_median_price"].replace(0, np.nan)
    )
    result["rating_minus_category_mean"] = (
        result["avg_rating"] - result["category"].map(category_rating_mean)
    )
    result["platform_device"] = result["platform"].astype(str) + "__" + result["device_os"].astype(str)
    result["page_slot"] = result["page_type"].astype(str) + "__slot_" + result["slot_position"].astype(str)
    result["category_page"] = result["category"].astype(str) + "__" + result["page_type"].astype(str)
    return result


def add_training_history(
    frame: pd.DataFrame,
    keys: list[str],
    prefix: str,
    target: str = "clicked",
    smoothing_strength: float = 50,
    global_ctr: float | None = None,
) -> pd.DataFrame:
    """Add prior-only counts and smoothed CTR to chronologically sorted training data."""
    result = frame.copy()
    prior = global_ctr if global_ctr is not None else float(result[target].mean())
    grouped = result.groupby(keys, dropna=False, sort=False)[target]
    prior_count = grouped.cumcount()
    prior_clicks = grouped.cumsum() - result[target]
    result[f"{prefix}_past_impressions"] = prior_count.astype("int32")
    result[f"{prefix}_past_clicks"] = prior_clicks.astype("int32")
    result[f"{prefix}_past_ctr"] = (
        prior_clicks + smoothing_strength * prior
    ) / (prior_count + smoothing_strength)
    return result

