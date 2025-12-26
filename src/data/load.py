"""
UFC Fight Predictor - Data Loading Module

This module handles loading raw UFC fight data from various sources.
It provides functions to read CSV files, validate data structure,
and prepare data for the cleaning pipeline.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import os
from typing import Optional, Dict, Any, List

# =============================================================================
# CONSTANTS
# =============================================================================
# Directory paths relative to this file
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# Default filenames
DEFAULT_FIGHTS_FILE = 'ufc-master.csv'

# Required columns for the fights dataset
# These are essential columns that must be present for the model to work
REQUIRED_COLUMNS = [
    'RedFighter',
    'BlueFighter',
    'Winner',
    'Date',
    'WeightClass',
]

# Fighter statistics columns (these contain pre-fight averages)
FIGHTER_STAT_COLUMNS = {
    'red': [
        'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws',
        'RedAvgSigStrLanded', 'RedAvgSigStrPct', 'RedAvgSubAtt',
        'RedAvgTDLanded', 'RedAvgTDPct', 'RedLongestWinStreak',
        'RedLosses', 'RedTotalRoundsFought', 'RedTotalTitleBouts',
        'RedWinsByDecisionMajority', 'RedWinsByDecisionSplit',
        'RedWinsByDecisionUnanimous', 'RedWinsByKO', 'RedWinsBySubmission',
        'RedWinsByTKODoctorStoppage', 'RedWins', 'RedStance',
        'RedHeightCms', 'RedReachCms', 'RedWeightLbs',
    ],
    'blue': [
        'BlueCurrentLoseStreak', 'BlueCurrentWinStreak', 'BlueDraws',
        'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 'BlueAvgSubAtt',
        'BlueAvgTDLanded', 'BlueAvgTDPct', 'BlueLongestWinStreak',
        'BlueLosses', 'BlueTotalRoundsFought', 'BlueTotalTitleBouts',
        'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit',
        'BlueWinsByDecisionUnanimous', 'BlueWinsByKO', 'BlueWinsBySubmission',
        'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueStance',
        'BlueHeightCms', 'BlueReachCms', 'BlueWeightLbs',
    ]
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_raw_fights(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw UFC fight data from CSV file.

    This function loads the main UFC fights dataset containing:
    - Fight metadata (date, location, weight class)
    - Fighter names and corners (Red/Blue)
    - Pre-fight statistics for both fighters
    - Fight outcomes and odds

    Args:
        filepath: Path to the CSV file. If None, uses default location.

    Returns:
        DataFrame containing raw fight data.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If required columns are missing.
    """
    # Use default path if none provided
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, DEFAULT_FIGHTS_FILE)

    # Normalize path
    filepath = os.path.abspath(filepath)

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fight data file not found: {filepath}\n"
            f"Please ensure the UFC dataset is placed in: {RAW_DATA_DIR}"
        )

    # Load the CSV file
    print(f"Loading fight data from: {filepath}")
    df = pd.read_csv(filepath)

    # Basic validation
    print(f"Loaded {len(df)} fights with {len(df.columns)} columns")

    # Validate required columns are present
    validate_data_schema(df, REQUIRED_COLUMNS)

    return df


def load_processed_fights(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed/cleaned UFC fight data.

    Args:
        filepath: Path to the processed CSV file. If None, uses default.

    Returns:
        DataFrame containing processed fight data.
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DATA_DIR, 'processed_fights.csv')

    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data file not found: {filepath}\n"
            "Please run the data cleaning pipeline first."
        )

    print(f"Loading processed data from: {filepath}")
    df = pd.read_csv(filepath)

    # Parse date column if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    print(f"Loaded {len(df)} processed fights")
    return df


def load_fighter_stats(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load fighter statistics database.

    This function loads the pre-computed fighter statistics used
    for making predictions on new matchups.

    Args:
        filepath: Path to the CSV file. If None, uses default location.

    Returns:
        DataFrame containing fighter statistics.
    """
    if filepath is None:
        filepath = os.path.join(PROCESSED_DATA_DIR, 'fighter_stats.csv')

    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fighter stats file not found: {filepath}\n"
            "Please run the feature engineering pipeline first."
        )

    print(f"Loading fighter statistics from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded statistics for {len(df)} fighters")

    return df


# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_data_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Validate that the DataFrame has the expected columns.

    Args:
        df: DataFrame to validate.
        expected_columns: List of column names that should be present.

    Returns:
        True if valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)[:20]}..."
        )

    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data quality checks on the loaded dataset.

    Checks for:
    - Missing values per column
    - Duplicate rows
    - Date range
    - Invalid values

    Args:
        df: DataFrame to check.

    Returns:
        Dictionary with quality metrics.
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': {},
        'date_range': {},
        'unique_fighters': 0,
    }

    # Calculate missing values percentage for each column
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > 0:
            quality_report['missing_values'][col] = round(missing_pct, 2)

    # Date range if Date column exists
    if 'Date' in df.columns:
        df_dates = pd.to_datetime(df['Date'], errors='coerce')
        quality_report['date_range'] = {
            'min': str(df_dates.min()),
            'max': str(df_dates.max()),
        }

    # Count unique fighters
    if 'RedFighter' in df.columns and 'BlueFighter' in df.columns:
        all_fighters = set(df['RedFighter'].unique()) | set(df['BlueFighter'].unique())
        quality_report['unique_fighters'] = len(all_fighters)

    return quality_report


# =============================================================================
# DATA INFO FUNCTIONS
# =============================================================================

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary information about the loaded data.

    Provides a comprehensive overview of the dataset including
    shape, dtypes, memory usage, and basic statistics.

    Args:
        df: DataFrame to summarize.

    Returns:
        Dictionary with data statistics.
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'null_counts': df.isnull().sum().to_dict(),
    }

    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        info['numeric_stats'] = df[numeric_cols].describe().to_dict()

    return info


def get_column_categories() -> Dict[str, List[str]]:
    """
    Get categorized column names for reference.

    Returns:
        Dictionary mapping category names to column lists.
    """
    return {
        'fighter_names': ['RedFighter', 'BlueFighter'],
        'fight_metadata': ['Date', 'Location', 'Country', 'WeightClass',
                          'Gender', 'NumberOfRounds', 'TitleBout'],
        'outcome': ['Winner', 'Finish', 'FinishDetails', 'FinishRound',
                   'FinishRoundTime', 'TotalFightTimeSecs'],
        'odds': ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
                'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds',
                'RKOOdds', 'BKOOdds'],
        'red_fighter_stats': FIGHTER_STAT_COLUMNS['red'],
        'blue_fighter_stats': FIGHTER_STAT_COLUMNS['blue'],
    }


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of the dataset.

    Args:
        df: DataFrame to summarize.
    """
    print("\n" + "="*60)
    print("UFC FIGHT DATA SUMMARY")
    print("="*60)

    print(f"\nTotal fights: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Date range
    if 'Date' in df.columns:
        df_dates = pd.to_datetime(df['Date'], errors='coerce')
        print(f"Date range: {df_dates.min()} to {df_dates.max()}")

    # Unique fighters
    if 'RedFighter' in df.columns and 'BlueFighter' in df.columns:
        all_fighters = set(df['RedFighter'].unique()) | set(df['BlueFighter'].unique())
        print(f"Unique fighters: {len(all_fighters):,}")

    # Weight classes
    if 'WeightClass' in df.columns:
        print(f"Weight classes: {df['WeightClass'].nunique()}")
        print(f"  - {', '.join(df['WeightClass'].value_counts().head(5).index.tolist())}")

    # Missing values summary
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"\nColumns with missing values: {len(missing_cols)}")

    print("\n" + "="*60)


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Data loading module ready.")
    print(f"Raw data directory: {os.path.abspath(RAW_DATA_DIR)}")
    print(f"Processed data directory: {os.path.abspath(PROCESSED_DATA_DIR)}")

    # Test loading if file exists
    try:
        df = load_raw_fights()
        print_data_summary(df)

        # Run quality check
        quality = check_data_quality(df)
        print(f"\nData Quality Report:")
        print(f"  Duplicate rows: {quality['duplicate_rows']}")
        print(f"  Unique fighters: {quality['unique_fighters']}")
        print(f"  Columns with missing data: {len(quality['missing_values'])}")

    except FileNotFoundError as e:
        print(f"\n{e}")
