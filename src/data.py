"""Data loading, joining, and split validation."""

from pathlib import Path

import pandas as pd

from src.config import TARGET, project_paths


def load_raw_data(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the three source datasets with parsed impression timestamps."""
    raw = project_paths(root)["raw"]
    return {
        "impressions": pd.read_csv(raw / "impressions.csv", parse_dates=["event_ts"]),
        "products": pd.read_csv(raw / "product_catalog.csv"),
        "users": pd.read_csv(raw / "user_profile.csv"),
    }


def join_source_data(
    impressions: pd.DataFrame,
    products: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Join product and user attributes to impressions with key validation."""
    return (
        impressions.merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(users, on="user_id", how="left", validate="many_to_one")
    )


def load_splits(
    directory: Path,
    suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """Load chronological train, validation, and test CSV files."""
    ending = f"_{suffix}" if suffix else ""
    return {
        name: pd.read_csv(directory / f"{name}{ending}.csv", parse_dates=["event_ts"])
        for name in ("train", "validation", "test")
    }


def validate_binary_target(frame: pd.DataFrame, target: str = TARGET) -> None:
    """Raise when the target is absent, missing, or not binary."""
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    if frame[target].isna().any() or not set(frame[target].unique()).issubset({0, 1}):
        raise ValueError(f"{target} must contain only non-null 0 and 1 values.")


def validate_chronological_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Validate target, impression uniqueness, overlap, and chronological order."""
    for name, frame in splits.items():
        validate_binary_target(frame)
        if frame["impression_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate impression IDs.")
        if frame["event_ts"].isna().any():
            raise ValueError(f"{name} contains invalid timestamps.")

    combined_ids = pd.concat(
        [frame[["impression_id"]].assign(split=name) for name, frame in splits.items()]
    )
    if combined_ids["impression_id"].duplicated().any():
        raise ValueError("An impression appears in more than one split.")

    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    if not (
        train["event_ts"].max()
        <= validation["event_ts"].min()
        <= validation["event_ts"].max()
        <= test["event_ts"].min()
    ):
        raise ValueError("The splits are not chronological.")

