"""
UFC Fight Predictor - Data Cleaning Module

This module handles all data cleaning operations including:
- Removing post-fight stats (to prevent data leakage)
- Standardizing fighter names
- Handling missing values
- Normalizing units (weight, height, reach)

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================
# Columns that contain post-fight information (must be removed to prevent leakage)
POST_FIGHT_COLUMNS = []  # TODO: Define in Phase 2

# Standard weight class mappings
WEIGHT_CLASSES = {
    'Strawweight': 115,
    'Flyweight': 125,
    'Bantamweight': 135,
    'Featherweight': 145,
    'Lightweight': 155,
    'Welterweight': 170,
    'Middleweight': 185,
    'Light Heavyweight': 205,
    'Heavyweight': 265,
    "Women's Strawweight": 115,
    "Women's Flyweight": 125,
    "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
}

# =============================================================================
# DATA LEAKAGE PREVENTION
# =============================================================================

def remove_post_fight_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns containing post-fight statistics to prevent data leakage.

    In a real prediction scenario, we wouldn't have access to how the fight
    ended or post-fight statistics, so we must remove these.

    Args:
        df: DataFrame with potential post-fight columns.

    Returns:
        DataFrame with post-fight columns removed.
    """
    # TODO: Implement in Phase 2
    pass


def remove_future_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any information that would only be known after the fight date.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame without future information.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# NAME STANDARDIZATION
# =============================================================================

def standardize_fighter_names(df: pd.DataFrame, name_columns: List[str]) -> pd.DataFrame:
    """
    Standardize fighter names for consistency.

    Handles issues like:
    - Different spellings
    - Nickname variations
    - Case inconsistencies

    Args:
        df: DataFrame with fighter name columns.
        name_columns: List of column names containing fighter names.

    Returns:
        DataFrame with standardized names.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_numerical(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in numerical columns.

    Args:
        df: DataFrame with potential missing values.
        strategy: Imputation strategy ('median', 'mean', 'zero').

    Returns:
        DataFrame with imputed numerical values.
    """
    # TODO: Implement in Phase 2
    pass


def handle_missing_categorical(df: pd.DataFrame, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    Handle missing values in categorical columns.

    Args:
        df: DataFrame with potential missing values.
        fill_value: Value to use for missing categorical data.

    Returns:
        DataFrame with filled categorical values.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# UNIT NORMALIZATION
# =============================================================================

def normalize_height(height_series: pd.Series) -> pd.Series:
    """
    Normalize height to inches.

    Handles formats like "5'11" or "180 cm".

    Args:
        height_series: Series containing height values.

    Returns:
        Series with heights in inches.
    """
    # TODO: Implement in Phase 2
    pass


def normalize_weight(weight_series: pd.Series) -> pd.Series:
    """
    Normalize weight to pounds.

    Args:
        weight_series: Series containing weight values.

    Returns:
        Series with weights in pounds.
    """
    # TODO: Implement in Phase 2
    pass


def normalize_reach(reach_series: pd.Series) -> pd.Series:
    """
    Normalize reach to inches.

    Args:
        reach_series: Series containing reach values.

    Returns:
        Series with reach in inches.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_fight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline that applies all cleaning steps.

    Args:
        df: Raw fight DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    # TODO: Implement in Phase 2
    pass


def save_cleaned_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save cleaned data to the processed directory.

    Args:
        df: Cleaned DataFrame to save.
        filepath: Output filepath.
    """
    # TODO: Implement in Phase 2
    pass


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Data cleaning module ready.")
