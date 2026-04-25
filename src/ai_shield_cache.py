"""CSV/cache utilities for AI Shield branch-output tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_required(path: str | Path) -> pd.DataFrame:
    """Read a required CSV artifact with a clear error if it is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required cache does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def save_csv_if_allowed(df: pd.DataFrame, path: str | Path, force: bool = False) -> Path:
    """Save a CSV artifact without silently overwriting existing work."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Reusing existing CSV artifact: {output_path}")
        return output_path
    df.to_csv(output_path, index=False)
    print(f"Saved CSV artifact: {output_path}")
    return output_path


def assert_columns(df: pd.DataFrame, required_columns: Iterable[str], name: str = "dataframe") -> None:
    """Validate that a table contains the columns a later stage needs."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def merge_branch_outputs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str] | None = None,
    suffixes: tuple[str, str] = ("_forensic", "_semantic"),
) -> pd.DataFrame:
    """Merge branch-output tables and fail loudly if labels do not align."""
    merge_keys = keys or ["image_id", "label"]
    merged = left.merge(right, on=merge_keys, how="inner", suffixes=suffixes)
    if merged.empty:
        raise ValueError("Branch-output merge produced zero rows; check image_id/label alignment.")
    return merged


def make_train_val_test_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_size: float = 0.60,
    val_size: float = 0.20,
    random_state: int = 156,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation/test split for a fusion table."""
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be less than 1.")
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[label_column],
        random_state=random_state,
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[label_column],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
