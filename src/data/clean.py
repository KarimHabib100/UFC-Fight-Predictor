"""
UFC Fight Predictor - Data Cleaning Module

This module handles all data cleaning operations including:
- Removing post-fight stats (to prevent data leakage)
- Standardizing fighter names
- Handling missing values
- Normalizing units (weight, height, reach)

IMPORTANT: Data leakage prevention is critical in fight prediction.
We must ensure no post-fight information is used for prediction.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import os
import re
from typing import Optional, List, Tuple, Dict

# =============================================================================
# CONSTANTS
# =============================================================================
# Columns that contain POST-FIGHT information (must be removed to prevent leakage)
# These are statistics that are only known AFTER the fight happens
POST_FIGHT_COLUMNS = [
    # Fight outcome details - only known after fight
    'Finish',
    'FinishDetails',
    'FinishRound',
    'FinishRoundTime',
    'TotalFightTimeSecs',
    # Winner is our TARGET, not a feature (but we keep it for labeling)
    # 'Winner',  # Keep this - it's our prediction target
]

# Columns that could cause data leakage (odds might be okay as pre-fight info)
# Method/finish odds could hint at outcome - be careful
POTENTIAL_LEAKAGE_COLUMNS = [
    'RedDecOdds', 'BlueDecOdds',  # Decision odds
    'RSubOdds', 'BSubOdds',        # Submission odds
    'RKOOdds', 'BKOOdds',          # KO odds
    'BetterRank',                   # This might be derived from outcome
]

# Standard weight class mappings (in pounds)
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
    'Catch Weight': None,  # Variable weight
    'Open Weight': None,
}

# Stance encoding
STANCE_ENCODING = {
    'Orthodox': 0,
    'Southpaw': 1,
    'Switch': 2,
    'Open Stance': 3,
    'Sideways': 3,
}

# Columns we want to keep for modeling
COLUMNS_TO_KEEP = [
    # Identifiers
    'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country',
    # Fight context
    'WeightClass', 'Gender', 'NumberOfRounds', 'TitleBout',
    # Target variable
    'Winner',
    # Pre-fight odds (these are known before the fight)
    'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
    # Red fighter pre-fight stats
    'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws',
    'RedAvgSigStrLanded', 'RedAvgSigStrPct', 'RedAvgSubAtt',
    'RedAvgTDLanded', 'RedAvgTDPct', 'RedLongestWinStreak',
    'RedLosses', 'RedTotalRoundsFought', 'RedTotalTitleBouts',
    'RedWinsByDecisionMajority', 'RedWinsByDecisionSplit',
    'RedWinsByDecisionUnanimous', 'RedWinsByKO', 'RedWinsBySubmission',
    'RedWinsByTKODoctorStoppage', 'RedWins', 'RedStance',
    'RedHeightCms', 'RedReachCms', 'RedWeightLbs',
    # Blue fighter pre-fight stats
    'BlueCurrentLoseStreak', 'BlueCurrentWinStreak', 'BlueDraws',
    'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 'BlueAvgSubAtt',
    'BlueAvgTDLanded', 'BlueAvgTDPct', 'BlueLongestWinStreak',
    'BlueLosses', 'BlueTotalRoundsFought', 'BlueTotalTitleBouts',
    'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit',
    'BlueWinsByDecisionUnanimous', 'BlueWinsByKO', 'BlueWinsBySubmission',
    'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueStance',
    'BlueHeightCms', 'BlueReachCms', 'BlueWeightLbs',
    # Rankings (pre-fight rankings)
    'RMatchWCRank', 'BMatchWCRank', 'RPFPRank', 'BPFPRank',
]

# =============================================================================
# DATA LEAKAGE PREVENTION
# =============================================================================

def remove_post_fight_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns containing post-fight statistics to prevent data leakage.

    In a real prediction scenario, we wouldn't have access to how the fight
    ended or post-fight statistics, so we must remove these.

    Post-fight stats include:
    - Finish type and details
    - Round and time of finish
    - Total fight time

    Args:
        df: DataFrame with potential post-fight columns.

    Returns:
        DataFrame with post-fight columns removed.
    """
    df = df.copy()

    # Remove post-fight columns
    cols_to_drop = [col for col in POST_FIGHT_COLUMNS if col in df.columns]

    if cols_to_drop:
        print(f"Removing {len(cols_to_drop)} post-fight columns to prevent leakage:")
        for col in cols_to_drop:
            print(f"  - {col}")
        df = df.drop(columns=cols_to_drop)

    return df


def remove_potential_leakage_columns(
    df: pd.DataFrame,
    remove_method_odds: bool = True
) -> pd.DataFrame:
    """
    Remove columns that could potentially leak fight outcome information.

    Method-specific odds (KO odds, submission odds, decision odds) might
    encode predictions about fight outcome. These can be optionally removed.

    Args:
        df: DataFrame to clean.
        remove_method_odds: If True, removes method-specific odds columns.

    Returns:
        DataFrame without potential leakage columns.
    """
    df = df.copy()

    if remove_method_odds:
        cols_to_drop = [col for col in POTENTIAL_LEAKAGE_COLUMNS if col in df.columns]
        if cols_to_drop:
            print(f"Removing {len(cols_to_drop)} potential leakage columns:")
            for col in cols_to_drop:
                print(f"  - {col}")
            df = df.drop(columns=cols_to_drop)

    return df


def select_valid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns that are safe to use for prediction.

    Args:
        df: DataFrame with all columns.

    Returns:
        DataFrame with only valid prediction columns.
    """
    df = df.copy()

    # Get columns that exist in both the dataframe and our allowed list
    valid_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]

    print(f"Keeping {len(valid_cols)} valid columns for modeling")

    return df[valid_cols]


# =============================================================================
# NAME STANDARDIZATION
# =============================================================================

def standardize_fighter_names(
    df: pd.DataFrame,
    name_columns: List[str] = ['RedFighter', 'BlueFighter']
) -> pd.DataFrame:
    """
    Standardize fighter names for consistency.

    Handles issues like:
    - Extra whitespace
    - Case inconsistencies
    - Special characters

    Args:
        df: DataFrame with fighter name columns.
        name_columns: List of column names containing fighter names.

    Returns:
        DataFrame with standardized names.
    """
    df = df.copy()

    for col in name_columns:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()

            # Normalize multiple spaces to single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            # Title case for consistency
            df[col] = df[col].str.title()

    print(f"Standardized names in columns: {name_columns}")

    return df


def create_fighter_name_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping for common fighter name variations.

    This helps handle cases where the same fighter appears with
    slightly different name spellings.

    Args:
        df: DataFrame with fighter names.

    Returns:
        Dictionary mapping variations to canonical names.
    """
    # Get all unique fighter names
    all_names = set()
    for col in ['RedFighter', 'BlueFighter']:
        if col in df.columns:
            all_names.update(df[col].dropna().unique())

    # Common name corrections (can be extended)
    name_mapping = {
        'Tj Dillashaw': 'T.J. Dillashaw',
        'Bj Penn': 'B.J. Penn',
        'Cb Dollaway': 'C.B. Dollaway',
        'Jj Aldrich': 'J.J. Aldrich',
    }

    return name_mapping


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_numerical(
    df: pd.DataFrame,
    strategy: str = 'median',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in numerical columns.

    For fighter stats, missing values often mean:
    - First fight (no history)
    - Data not recorded

    Using median is preferred over mean as it's more robust to outliers.

    Args:
        df: DataFrame with potential missing values.
        strategy: Imputation strategy ('median', 'mean', 'zero').
        columns: Specific columns to impute. If None, imputes all numeric.

    Returns:
        DataFrame with imputed numerical values.
    """
    df = df.copy()

    # Get numeric columns to impute
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in df.columns]

    imputed_count = 0

    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            df[col] = df[col].fillna(fill_value)
            imputed_count += 1

    print(f"Imputed missing values in {imputed_count} numeric columns using {strategy}")

    return df


def handle_missing_categorical(
    df: pd.DataFrame,
    fill_value: str = 'Unknown',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in categorical columns.

    Args:
        df: DataFrame with potential missing values.
        fill_value: Value to use for missing categorical data.
        columns: Specific columns to fill. If None, fills all object dtype.

    Returns:
        DataFrame with filled categorical values.
    """
    df = df.copy()

    # Get categorical columns to fill
    if columns is None:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Exclude fighter name columns - these shouldn't be filled
        cat_cols = [c for c in cat_cols if c not in ['RedFighter', 'BlueFighter']]
    else:
        cat_cols = [c for c in columns if c in df.columns]

    filled_count = 0

    for col in cat_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(fill_value)
            filled_count += 1

    print(f"Filled missing values in {filled_count} categorical columns with '{fill_value}'")

    return df


def handle_missing_stance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing stance values specifically.

    Missing stance is common and often means "Orthodox" (most common stance).

    Args:
        df: DataFrame with stance columns.

    Returns:
        DataFrame with filled stance values.
    """
    df = df.copy()

    for col in ['RedStance', 'BlueStance']:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                # Fill with most common stance (typically Orthodox)
                most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Orthodox'
                df[col] = df[col].fillna(most_common)
                print(f"Filled {missing} missing values in {col} with '{most_common}'")

    return df


# =============================================================================
# UNIT NORMALIZATION
# =============================================================================

def normalize_height(height_series: pd.Series) -> pd.Series:
    """
    Normalize height to centimeters.

    The dataset already has heights in cm (HeightCms columns),
    but this function handles edge cases.

    Args:
        height_series: Series containing height values.

    Returns:
        Series with heights in centimeters.
    """
    # Heights are already in cm in our dataset
    # Just ensure they're numeric and handle any outliers
    heights = pd.to_numeric(height_series, errors='coerce')

    # Sanity check: heights should be between 140-220 cm for fighters
    heights = heights.clip(lower=140, upper=220)

    return heights


def normalize_reach(reach_series: pd.Series) -> pd.Series:
    """
    Normalize reach to centimeters.

    Args:
        reach_series: Series containing reach values.

    Returns:
        Series with reach in centimeters.
    """
    # Reach is already in cm in our dataset
    reach = pd.to_numeric(reach_series, errors='coerce')

    # Sanity check: reach should be between 140-220 cm
    reach = reach.clip(lower=140, upper=220)

    return reach


def normalize_weight(weight_series: pd.Series) -> pd.Series:
    """
    Normalize weight to pounds.

    Args:
        weight_series: Series containing weight values.

    Returns:
        Series with weights in pounds.
    """
    # Weight is already in lbs in our dataset
    weight = pd.to_numeric(weight_series, errors='coerce')

    # Sanity check: weight should be between 100-350 lbs
    weight = weight.clip(lower=100, upper=350)

    return weight


# =============================================================================
# DATA TYPE CONVERSIONS
# =============================================================================

def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column to datetime type.

    Args:
        df: DataFrame with date column.

    Returns:
        DataFrame with properly typed date column.
    """
    df = df.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print(f"Converted Date column to datetime")

    return df


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean-like columns to proper boolean type.

    Args:
        df: DataFrame with boolean columns.

    Returns:
        DataFrame with properly typed boolean columns.
    """
    df = df.copy()

    bool_columns = ['TitleBout']

    for col in bool_columns:
        if col in df.columns:
            # Convert True/False strings or 1/0 to boolean
            df[col] = df[col].map({
                True: 1, False: 0,
                'True': 1, 'False': 0,
                1: 1, 0: 0,
                1.0: 1, 0.0: 0
            }).fillna(0).astype(int)

    return df


def encode_stance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode stance categorical columns to numeric.

    Args:
        df: DataFrame with stance columns.

    Returns:
        DataFrame with encoded stance columns.
    """
    df = df.copy()

    for col in ['RedStance', 'BlueStance']:
        if col in df.columns:
            new_col = col.replace('Stance', 'StanceEncoded')
            df[new_col] = df[col].map(STANCE_ENCODING).fillna(0).astype(int)

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable from Winner column.

    Target: 1 if Red fighter wins, 0 if Blue fighter wins.

    Args:
        df: DataFrame with Winner column.

    Returns:
        DataFrame with Target column added.
    """
    df = df.copy()

    if 'Winner' in df.columns:
        # Create binary target: 1 = Red wins, 0 = Blue wins
        df['Target'] = (df['Winner'] == 'Red').astype(int)

        # Handle draws and no contests
        draw_mask = df['Winner'].isin(['Draw', 'NC', 'No Contest', 'draw'])
        df.loc[draw_mask, 'Target'] = np.nan

        print(f"Created Target variable: Red wins = 1, Blue wins = 0")
        print(f"  Red wins: {(df['Target'] == 1).sum()}")
        print(f"  Blue wins: {(df['Target'] == 0).sum()}")
        print(f"  Draws/NC: {draw_mask.sum()} (will be removed)")

    return df


# =============================================================================
# OUTLIER HANDLING
# =============================================================================

def clip_statistical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip extreme outliers in statistical columns.

    Args:
        df: DataFrame with statistical columns.

    Returns:
        DataFrame with clipped values.
    """
    df = df.copy()

    # Columns that should be percentages (0-100 or 0-1)
    pct_columns = [col for col in df.columns if 'Pct' in col]
    for col in pct_columns:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=100)

    # Columns that should be non-negative
    non_neg_columns = [col for col in df.columns if any(x in col for x in
                       ['Wins', 'Losses', 'Streak', 'Rounds', 'Avg'])]
    for col in non_neg_columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].clip(lower=0)

    return df


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_fight_data(
    df: pd.DataFrame,
    remove_draws: bool = True,
    remove_method_odds: bool = True
) -> pd.DataFrame:
    """
    Main cleaning pipeline that applies all cleaning steps.

    This is the primary function to call for data cleaning. It:
    1. Removes post-fight stats (leakage prevention)
    2. Selects valid columns
    3. Standardizes names
    4. Converts data types
    5. Handles missing values
    6. Creates target variable
    7. Removes invalid fights (draws, missing essential data)

    Args:
        df: Raw fight DataFrame.
        remove_draws: If True, removes draws and no contests.
        remove_method_odds: If True, removes method-specific odds columns.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    print("\n" + "="*60)
    print("STARTING DATA CLEANING PIPELINE")
    print("="*60)

    original_rows = len(df)
    df = df.copy()

    # Step 1: Remove post-fight stats (CRITICAL for preventing leakage)
    print("\n[1/8] Removing post-fight statistics...")
    df = remove_post_fight_stats(df)
    df = remove_potential_leakage_columns(df, remove_method_odds)

    # Step 2: Select valid columns
    print("\n[2/8] Selecting valid prediction columns...")
    df = select_valid_columns(df)

    # Step 3: Standardize fighter names
    print("\n[3/8] Standardizing fighter names...")
    df = standardize_fighter_names(df)

    # Step 4: Convert data types
    print("\n[4/8] Converting data types...")
    df = convert_date_column(df)
    df = convert_boolean_columns(df)

    # Step 5: Handle missing stance values
    print("\n[5/8] Handling missing stance values...")
    df = handle_missing_stance(df)
    df = encode_stance_columns(df)

    # Step 6: Handle missing numerical values
    print("\n[6/8] Imputing missing numerical values...")
    df = handle_missing_numerical(df, strategy='median')

    # Step 7: Create target variable
    print("\n[7/8] Creating target variable...")
    df = create_target_variable(df)

    # Step 8: Remove invalid fights
    print("\n[8/8] Removing invalid fights...")

    # Remove rows with missing fighter names
    df = df.dropna(subset=['RedFighter', 'BlueFighter'])

    # Remove draws and no contests if specified
    if remove_draws:
        df = df.dropna(subset=['Target'])
        print(f"  Removed draws/no contests")

    # Remove rows with missing dates
    df = df.dropna(subset=['Date'])

    # Clip outliers
    df = clip_statistical_outliers(df)

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    final_rows = len(df)
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print(f"Original rows: {original_rows:,}")
    print(f"Final rows: {final_rows:,}")
    print(f"Removed: {original_rows - final_rows:,} rows")
    print(f"Columns: {len(df.columns)}")

    return df


def save_cleaned_data(df: pd.DataFrame, filepath: Optional[str] = None) -> str:
    """
    Save cleaned data to the processed directory.

    Args:
        df: Cleaned DataFrame to save.
        filepath: Output filepath. If None, uses default.

    Returns:
        Path where data was saved.
    """
    if filepath is None:
        # Get processed directory path
        processed_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed'
        )
        os.makedirs(processed_dir, exist_ok=True)
        filepath = os.path.join(processed_dir, 'processed_fights.csv')

    filepath = os.path.abspath(filepath)

    df.to_csv(filepath, index=False)
    print(f"\nSaved cleaned data to: {filepath}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    return filepath


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Data cleaning module ready.")

    # Import load function
    from load import load_raw_fights, print_data_summary

    try:
        # Load raw data
        print("\nLoading raw data...")
        df_raw = load_raw_fights()
        print_data_summary(df_raw)

        # Clean data
        print("\nCleaning data...")
        df_clean = clean_fight_data(df_raw)

        # Save cleaned data
        save_cleaned_data(df_clean)

        # Show sample of cleaned data
        print("\nSample of cleaned data:")
        print(df_clean[['RedFighter', 'BlueFighter', 'Date', 'WeightClass', 'Target']].head(10))

    except Exception as e:
        print(f"Error: {e}")
