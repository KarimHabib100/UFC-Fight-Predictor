"""
UFC Fight Predictor - Helper Utilities Module

This module contains reusable helper functions used across the project.
Includes critical time-based splitting to prevent data leakage.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
import os
import json
from datetime import datetime
import joblib

# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def time_based_split(
    df: pd.DataFrame,
    date_column: str = 'Date',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to prevent data leakage.

    CRITICAL: In fight prediction, we can't use future data to predict
    past fights. This function ensures:
    - Training data contains the OLDEST fights
    - Validation data contains MIDDLE-period fights
    - Test data contains the MOST RECENT fights

    This mimics real-world usage where we train on historical data
    and predict upcoming fights.

    Args:
        df: DataFrame to split (must be sorted by date).
        date_column: Name of the date column.
        train_ratio: Proportion for training set (default 70%).
        val_ratio: Proportion for validation set (default 15%).

    Returns:
        Tuple of (train_df, val_df, test_df).

    Example:
        >>> train, val, test = time_based_split(df, 'Date', 0.7, 0.15)
        >>> # train: 2010-2020, val: 2020-2022, test: 2022-2024
    """
    df = df.copy()

    # Ensure data is sorted by date
    if date_column in df.columns:
        df = df.sort_values(date_column).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Print split information
    print("\n" + "="*60)
    print("CHRONOLOGICAL DATA SPLIT")
    print("="*60)
    print(f"Total samples: {n:,}")
    print(f"\nTraining set:   {len(train_df):,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"Validation set: {len(val_df):,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"Test set:       {len(test_df):,} samples ({len(test_df)/n*100:.1f}%)")

    if date_column in df.columns:
        print(f"\nDate ranges:")
        print(f"  Train: {train_df[date_column].min()} to {train_df[date_column].max()}")
        print(f"  Val:   {val_df[date_column].min()} to {val_df[date_column].max()}")
        print(f"  Test:  {test_df[date_column].min()} to {test_df[date_column].max()}")

    print("="*60)

    return train_df, val_df, test_df


def get_train_val_test_splits(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'Target',
    date_column: str = 'Date',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Get X and y splits for train, validation, and test sets.

    This is the main function for preparing data for model training.
    It ensures chronological splitting to prevent data leakage.

    Args:
        df: Full DataFrame with features and target.
        feature_columns: List of feature column names to use.
        target_column: Name of target column.
        date_column: Name of date column for splitting.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    # Split chronologically
    train_df, val_df, test_df = time_based_split(
        df, date_column, train_ratio, val_ratio
    )

    # Get only existing feature columns
    existing_features = [col for col in feature_columns if col in df.columns]

    if len(existing_features) < len(feature_columns):
        missing = set(feature_columns) - set(existing_features)
        print(f"Warning: {len(missing)} features not found in data")

    # Extract X and y
    X_train = train_df[existing_features].copy()
    y_train = train_df[target_column].copy()

    X_val = val_df[existing_features].copy()
    y_val = val_df[target_column].copy()

    X_test = test_df[existing_features].copy()
    y_test = test_df[target_column].copy()

    # Print class distribution
    print("\nTarget distribution:")
    print(f"  Train - Red wins: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"  Val   - Red wins: {(y_val == 1).sum():,} ({(y_val == 1).mean()*100:.1f}%)")
    print(f"  Test  - Red wins: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_column: str = 'Date'
) -> bool:
    """
    Verify that there's no temporal data leakage between splits.

    Checks that:
    1. All training dates are before validation dates
    2. All validation dates are before test dates

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        date_column: Name of date column.

    Returns:
        True if no leakage detected, raises ValueError otherwise.
    """
    if date_column not in train_df.columns:
        print("Warning: Cannot verify leakage - no date column")
        return True

    train_max = train_df[date_column].max()
    val_min = val_df[date_column].min()
    val_max = val_df[date_column].max()
    test_min = test_df[date_column].min()

    leakage_detected = False

    if train_max > val_min:
        print(f"ERROR: Training data ({train_max}) overlaps with validation ({val_min})")
        leakage_detected = True

    if val_max > test_min:
        print(f"ERROR: Validation data ({val_max}) overlaps with test ({test_min})")
        leakage_detected = True

    if leakage_detected:
        raise ValueError("Temporal data leakage detected!")

    print("✓ No temporal data leakage detected")
    return True


def save_split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    date_column: str = 'Date'
) -> str:
    """
    Save split information for reproducibility.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        output_dir: Directory to save split info.
        date_column: Name of date column.

    Returns:
        Path to saved split info file.
    """
    os.makedirs(output_dir, exist_ok=True)

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(val_df) + len(test_df),
        'train_ratio': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
        'val_ratio': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
        'test_ratio': len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
        'created_at': datetime.now().isoformat(),
    }

    if date_column in train_df.columns:
        split_info['date_ranges'] = {
            'train': {
                'min': str(train_df[date_column].min()),
                'max': str(train_df[date_column].max())
            },
            'val': {
                'min': str(val_df[date_column].min()),
                'max': str(val_df[date_column].max())
            },
            'test': {
                'min': str(test_df[date_column].min()),
                'max': str(test_df[date_column].max())
            }
        }

    filepath = os.path.join(output_dir, 'split_info.json')
    with open(filepath, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split info to: {filepath}")
    return filepath


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
    if pd.isna(date_string):
        return pd.NaT

    # Try common formats
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            continue

    # Fallback to pandas auto-detection
    return pd.to_datetime(date_string, errors='coerce')


def get_days_between(date1: pd.Timestamp, date2: pd.Timestamp) -> int:
    """
    Calculate days between two dates.

    Args:
        date1: First date.
        date2: Second date.

    Returns:
        Number of days between dates (positive if date2 > date1).
    """
    if pd.isna(date1) or pd.isna(date2):
        return 0
    return (date2 - date1).days


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
    valid_set = set(valid_names)
    found_valid = [n for n in names if n in valid_set]
    not_found = [n for n in names if n not in valid_set]

    return found_valid, not_found


def check_data_leakage(
    df: pd.DataFrame,
    date_column: str,
    feature_columns: List[str]
) -> Dict[str, Any]:
    """
    Check for potential data leakage issues in the dataset.

    Checks for:
    1. Future information in features
    2. Target leakage
    3. Duplicate rows

    Args:
        df: DataFrame to check.
        date_column: Name of date column.
        feature_columns: List of feature columns.

    Returns:
        Dictionary with leakage check results.
    """
    results = {
        'duplicates': df.duplicated().sum(),
        'missing_dates': df[date_column].isna().sum() if date_column in df.columns else 0,
        'warnings': []
    }

    # Check for suspicious column names that might indicate leakage
    leakage_keywords = ['result', 'winner', 'outcome', 'final', 'finish']
    for col in feature_columns:
        if any(keyword in col.lower() for keyword in leakage_keywords):
            results['warnings'].append(f"Potential leakage column: {col}")

    return results


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
    if denominator == 0 or pd.isna(denominator):
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
# DATA PREPARATION PIPELINE
# =============================================================================

def prepare_data_for_training(
    features_path: str = None,
    feature_columns_path: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Complete data preparation pipeline for model training.

    Loads features, splits data chronologically, and returns
    ready-to-use training, validation, and test sets.

    Args:
        features_path: Path to features CSV. If None, uses default.
        feature_columns_path: Path to feature columns file. If None, uses default.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        save_splits: Whether to save split information.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_columns).
    """
    # Set default paths
    if features_path is None:
        features_path = get_data_path('features.csv')
    if feature_columns_path is None:
        feature_columns_path = get_data_path('feature_columns.txt')

    print("Loading feature data...")
    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df):,} samples")

    # Load feature columns
    with open(feature_columns_path, 'r') as f:
        feature_columns = f.read().strip().split('\n')
    print(f"Loaded {len(feature_columns)} feature columns")

    # Get splits
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(
        df, feature_columns, 'Target', 'Date', train_ratio, val_ratio
    )

    # Verify no leakage
    train_df, val_df, test_df = time_based_split(df, 'Date', train_ratio, val_ratio)
    verify_no_leakage(train_df, val_df, test_df, 'Date')

    # Save split info
    if save_splits:
        output_dir = os.path.dirname(features_path)
        save_split_info(train_df, val_df, test_df, output_dir, 'Date')

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Helper utilities module ready.")
    print(f"Project root: {get_project_root()}")

    # Test the data preparation pipeline
    try:
        print("\nTesting data preparation pipeline...")
        X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data_for_training()

        print(f"\nFinal shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")

    except Exception as e:
        print(f"Error: {e}")
