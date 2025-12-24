"""
UFC Fight Predictor - Feature Engineering Module

This module creates features for the ML model including:
- Fighter-level statistics
- Contextual/historical features
- Differential features (Fighter A - Fighter B)

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONSTANTS
# =============================================================================
# Features that should be computed per fighter
FIGHTER_LEVEL_FEATURES = [
    'strikes_landed_per_min',
    'strikes_absorbed_per_min',
    'strike_accuracy',
    'strike_defense',
    'takedown_avg',
    'takedown_defense',
    'submission_avg',
    'age',
    'height_inches',
    'reach_inches',
]

# =============================================================================
# FIGHTER-LEVEL STATISTICS
# =============================================================================

def compute_strikes_per_minute(fight_history: pd.DataFrame) -> float:
    """
    Calculate average strikes landed per minute.

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Average strikes landed per minute.
    """
    # TODO: Implement in Phase 3
    pass


def compute_strike_accuracy(fight_history: pd.DataFrame) -> float:
    """
    Calculate strike accuracy (landed / attempted).

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Strike accuracy as percentage.
    """
    # TODO: Implement in Phase 3
    pass


def compute_strike_defense(fight_history: pd.DataFrame) -> float:
    """
    Calculate strike defense (1 - absorbed/opponent_attempted).

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Strike defense as percentage.
    """
    # TODO: Implement in Phase 3
    pass


def compute_takedown_avg(fight_history: pd.DataFrame) -> float:
    """
    Calculate average takedowns per 15 minutes.

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Average takedowns.
    """
    # TODO: Implement in Phase 3
    pass


def compute_takedown_defense(fight_history: pd.DataFrame) -> float:
    """
    Calculate takedown defense percentage.

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Takedown defense percentage.
    """
    # TODO: Implement in Phase 3
    pass


def compute_submission_avg(fight_history: pd.DataFrame) -> float:
    """
    Calculate average submission attempts per 15 minutes.

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Average submission attempts.
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# CONTEXTUAL/HISTORICAL FEATURES
# =============================================================================

def compute_last_n_fights_performance(fight_history: pd.DataFrame, n: int = 5) -> dict:
    """
    Compute performance metrics from last N fights.

    Args:
        fight_history: DataFrame of fighter's historical fights.
        n: Number of recent fights to consider.

    Returns:
        Dictionary of performance metrics.
    """
    # TODO: Implement in Phase 3
    pass


def compute_win_streak(fight_history: pd.DataFrame) -> int:
    """
    Calculate current win/loss streak.

    Positive = win streak, Negative = loss streak.

    Args:
        fight_history: DataFrame of fighter's historical fights.

    Returns:
        Current streak value.
    """
    # TODO: Implement in Phase 3
    pass


def compute_days_since_last_fight(fight_history: pd.DataFrame, current_date: pd.Timestamp) -> int:
    """
    Calculate days since fighter's last bout.

    Args:
        fight_history: DataFrame of fighter's historical fights.
        current_date: Date of the upcoming fight.

    Returns:
        Number of days since last fight.
    """
    # TODO: Implement in Phase 3
    pass


def compute_age_at_fight(birth_date: pd.Timestamp, fight_date: pd.Timestamp) -> float:
    """
    Calculate fighter's age at the time of fight.

    Args:
        birth_date: Fighter's date of birth.
        fight_date: Date of the fight.

    Returns:
        Age in years.
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# DIFFERENTIAL FEATURES
# =============================================================================

def compute_differential_features(fighter_a_stats: dict, fighter_b_stats: dict) -> dict:
    """
    Compute differential features (Fighter A - Fighter B).

    These features capture the relative advantage/disadvantage
    between the two fighters.

    Args:
        fighter_a_stats: Dictionary of Fighter A statistics.
        fighter_b_stats: Dictionary of Fighter B statistics.

    Returns:
        Dictionary of differential features.
    """
    # TODO: Implement in Phase 3
    pass


def compute_ratio_features(fighter_a_stats: dict, fighter_b_stats: dict) -> dict:
    """
    Compute ratio features where meaningful (A / B).

    Args:
        fighter_a_stats: Dictionary of Fighter A statistics.
        fighter_b_stats: Dictionary of Fighter B statistics.

    Returns:
        Dictionary of ratio features.
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# CATEGORICAL FEATURES
# =============================================================================

def encode_stance(stance: str) -> int:
    """
    Encode fighter stance as numeric value.

    Args:
        stance: Fighter's stance (Orthodox, Southpaw, Switch).

    Returns:
        Encoded stance value.
    """
    # TODO: Implement in Phase 3
    pass


def encode_weight_class(weight_class: str) -> int:
    """
    Encode weight class as numeric value.

    Args:
        weight_class: Fight weight class.

    Returns:
        Encoded weight class.
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# FEATURE SCALING
# =============================================================================

def scale_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        df: DataFrame with features to scale.
        scaler: Pre-fitted scaler (for inference). If None, fits new scaler.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# MAIN FEATURE PIPELINE
# =============================================================================

def build_feature_set(fights_df: pd.DataFrame, fighter_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete feature set for all fights.

    This is the main function that orchestrates feature engineering.

    Args:
        fights_df: DataFrame of cleaned fights.
        fighter_stats_df: DataFrame of fighter statistics.

    Returns:
        DataFrame with all engineered features.
    """
    # TODO: Implement in Phase 3
    pass


def save_features(df: pd.DataFrame, filepath: str) -> None:
    """
    Save feature-engineered data.

    Args:
        df: DataFrame to save.
        filepath: Output filepath.
    """
    # TODO: Implement in Phase 3
    pass


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Feature engineering module ready.")
