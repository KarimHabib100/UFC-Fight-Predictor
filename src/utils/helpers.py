"""
UFC Fight Predictor - Helper Utilities Module

This module contains reusable helper functions used across the project.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any
import os
from datetime import datetime

# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def time_based_split(
    df: pd.DataFrame,
    date_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to prevent data leakage.

    In fight prediction, we can't use future data to predict past fights.
    This function ensures training data is always older than validation,
    and validation is older than test data.

    Args:
        df: DataFrame to split.
        date_column: Name of the date column.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # TODO: Implement in Phase 4
    pass


def get_train_val_test_splits(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    date_column: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Get X and y splits for train, validation, and test sets.

    Args:
        df: Full DataFrame.
        feature_columns: List of feature column names.
        target_column: Name of target column.
        date_column: Name of date column for splitting.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    # TODO: Implement in Phase 4
    pass


# =============================================================================
# DATE UTILITIES
# =============================================================================

def parse_date(date_string: str) -> pd.Timestamp:
    """
    Parse date string to pandas Timestamp.

    Handles various date formats commonly found in UFC data.

    Args:
        date_string: Date string to parse.

    Returns:
        Pandas Timestamp object.
    """
    # TODO: Implement in Phase 2
    pass


def get_days_between(date1: pd.Timestamp, date2: pd.Timestamp) -> int:
    """
    Calculate days between two dates.

    Args:
        date1: First date.
        date2: Second date.

    Returns:
        Number of days between dates (positive if date2 > date1).
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> str:
    """
    Get the project root directory path.

    Returns:
        Absolute path to project root.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_data_path(filename: str, processed: bool = True) -> str:
    """
    Get full path to a data file.

    Args:
        filename: Name of the data file.
        processed: If True, looks in processed/ folder; else raw/.

    Returns:
        Full path to the data file.
    """
    root = get_project_root()
    folder = 'processed' if processed else 'raw'
    return os.path.join(root, 'data', folder, filename)


def get_model_path(filename: str) -> str:
    """
    Get full path to a model file.

    Args:
        filename: Name of the model file.

    Returns:
        Full path to the model file.
    """
    root = get_project_root()
    return os.path.join(root, 'models', filename)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_fighter_names(names: List[str], valid_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate fighter names against known fighters.

    Args:
        names: List of names to validate.
        valid_names: List of valid fighter names.

    Returns:
        Tuple of (valid names found, invalid names).
    """
    # TODO: Implement in Phase 2
    pass


def check_data_leakage(df: pd.DataFrame, date_column: str) -> bool:
    """
    Check if there's potential data leakage in the dataset.

    Args:
        df: DataFrame to check.
        date_column: Name of date column.

    Returns:
        True if potential leakage detected.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# STATISTICS UTILITIES
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value if denominator is zero.

    Returns:
        Division result or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def compute_ema(values: pd.Series, span: int = 5) -> pd.Series:
    """
    Compute exponential moving average.

    Args:
        values: Series of values.
        span: EMA span parameter.

    Returns:
        EMA series.
    """
    return values.ewm(span=span, adjust=False).mean()


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_message(message: str, level: str = "INFO") -> None:
    """
    Print a formatted log message.

    Args:
        message: Message to log.
        level: Log level (INFO, WARNING, ERROR).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Helper utilities module ready.")
    print(f"Project root: {get_project_root()}")
