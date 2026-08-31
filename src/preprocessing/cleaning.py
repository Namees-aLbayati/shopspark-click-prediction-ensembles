"""Reusable data-quality and outlier helpers."""

import numpy as np
import pandas as pd


NUMERIC_RANGES = {
    "age": (18, 100),
    "price_usd": (0, None),
    "discount_pct": (0, 100),
    "avg_rating": (0, 5),
    "review_count": (0, None),
    "slot_position": (1, None),
    "hour_of_day": (0, 23),
}
MISSING_MARKERS = {"", "na", "n/a", "null", "none", "unknown", "?"}


def clean_modeling_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardize missing markers and replace impossible numeric values with NaN."""
    result = frame.copy().replace([np.inf, -np.inf], np.nan)
    missing_before = int(result.isna().sum().sum())

    for column in result.select_dtypes(include=["object", "string", "category"]):
        values = result[column].astype("string").str.strip()
        result[column] = values.mask(values.str.lower().isin(MISSING_MARKERS), pd.NA)

    invalid_numeric = 0
    for column, (lower, upper) in NUMERIC_RANGES.items():
        if column not in result:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].lt(lower)
        if upper is not None:
            invalid |= result[column].gt(upper)
        invalid_numeric += int(invalid.sum())
        result.loc[invalid, column] = np.nan

    report = {
        "rows": len(result),
        "missing_before": missing_before,
        "missing_after": int(result.isna().sum().sum()),
        "invalid_numeric_changed_to_missing": invalid_numeric,
    }
    return result, report


def iqr_outlier_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Describe 1.5-IQR outliers without deleting valid observations."""
    rows = []
    for column in columns:
        values = frame[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append({
            "feature": column,
            "lower_fence": lower,
            "upper_fence": upper,
            "outlier_rows": count,
            "outlier_pct": count / len(values) if len(values) else np.nan,
        })
    return pd.DataFrame(rows).set_index("feature")

