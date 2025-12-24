"""
UFC Fight Predictor - Feature Engineering Module

This module creates features for the ML model including:
- Fighter-level statistics (from pre-computed data)
- Contextual/historical features (experience, streaks)
- Differential features (Fighter A - Fighter B)
- Ratio features for meaningful comparisons

The dataset already contains pre-fight averages for each fighter.
This module focuses on creating comparative features between fighters.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import os
import joblib
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =============================================================================
# CONSTANTS
# =============================================================================
# Base statistics columns for each fighter (Red/Blue prefixes)
STAT_COLUMNS = [
    'CurrentLoseStreak', 'CurrentWinStreak', 'Draws',
    'AvgSigStrLanded', 'AvgSigStrPct', 'AvgSubAtt',
    'AvgTDLanded', 'AvgTDPct', 'LongestWinStreak',
    'Losses', 'TotalRoundsFought', 'TotalTitleBouts',
    'WinsByDecisionMajority', 'WinsByDecisionSplit',
    'WinsByDecisionUnanimous', 'WinsByKO', 'WinsBySubmission',
    'WinsByTKODoctorStoppage', 'Wins',
    'HeightCms', 'ReachCms', 'WeightLbs',
]

# Physical attribute columns (for differential features)
PHYSICAL_COLUMNS = ['HeightCms', 'ReachCms', 'WeightLbs']

# Performance columns (for differential features)
PERFORMANCE_COLUMNS = [
    'AvgSigStrLanded', 'AvgSigStrPct', 'AvgSubAtt',
    'AvgTDLanded', 'AvgTDPct',
]

# Record columns
RECORD_COLUMNS = ['Wins', 'Losses', 'Draws', 'CurrentWinStreak', 'CurrentLoseStreak']

# Weight class encoding (ordered by weight limit)
WEIGHT_CLASS_ENCODING = {
    "Women's Strawweight": 0,
    'Strawweight': 1,
    "Women's Flyweight": 2,
    'Flyweight': 3,
    "Women's Bantamweight": 4,
    'Bantamweight': 5,
    "Women's Featherweight": 6,
    'Featherweight': 7,
    'Lightweight': 8,
    'Welterweight': 9,
    'Middleweight': 10,
    'Light Heavyweight': 11,
    'Heavyweight': 12,
    'Catch Weight': 13,
    'Open Weight': 13,
}

# Stance encoding
STANCE_ENCODING = {
    'Orthodox': 0,
    'Southpaw': 1,
    'Switch': 2,
    'Open Stance': 3,
    'Sideways': 3,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value to return if b is zero.

    Returns:
        a / b or default if b is zero.
    """
    if b == 0 or pd.isna(b):
        return default
    return a / b


def safe_ratio(a: float, b: float, default: float = 1.0) -> float:
    """
    Compute safe ratio with bounds to prevent extreme values.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value to return if b is zero.

    Returns:
        Bounded ratio value.
    """
    if b == 0 or pd.isna(b):
        return default
    ratio = a / b
    # Clip to reasonable bounds to prevent extreme values
    return np.clip(ratio, 0.1, 10.0)


# =============================================================================
# FIGHTER-LEVEL STATISTICS
# =============================================================================

def compute_total_fights(row: pd.Series, prefix: str) -> int:
    """
    Calculate total number of fights for a fighter.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Total fights (wins + losses + draws).
    """
    wins = row.get(f'{prefix}Wins', 0) or 0
    losses = row.get(f'{prefix}Losses', 0) or 0
    draws = row.get(f'{prefix}Draws', 0) or 0
    return int(wins + losses + draws)


def compute_win_rate(row: pd.Series, prefix: str) -> float:
    """
    Calculate win rate for a fighter.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Win rate as decimal (0-1).
    """
    wins = row.get(f'{prefix}Wins', 0) or 0
    total = compute_total_fights(row, prefix)
    return safe_divide(wins, total, default=0.5)


def compute_finish_rate(row: pd.Series, prefix: str) -> float:
    """
    Calculate finish rate (KO + Sub wins / total wins).

    A higher finish rate indicates a more dangerous finisher.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Finish rate as decimal (0-1).
    """
    ko_wins = row.get(f'{prefix}WinsByKO', 0) or 0
    tko_wins = row.get(f'{prefix}WinsByTKODoctorStoppage', 0) or 0
    sub_wins = row.get(f'{prefix}WinsBySubmission', 0) or 0
    total_wins = row.get(f'{prefix}Wins', 0) or 0

    finishes = ko_wins + tko_wins + sub_wins
    return safe_divide(finishes, total_wins, default=0.0)


def compute_ko_rate(row: pd.Series, prefix: str) -> float:
    """
    Calculate KO/TKO win rate.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        KO rate as decimal.
    """
    ko_wins = row.get(f'{prefix}WinsByKO', 0) or 0
    tko_wins = row.get(f'{prefix}WinsByTKODoctorStoppage', 0) or 0
    total_wins = row.get(f'{prefix}Wins', 0) or 0

    return safe_divide(ko_wins + tko_wins, total_wins, default=0.0)


def compute_submission_rate(row: pd.Series, prefix: str) -> float:
    """
    Calculate submission win rate.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Submission rate as decimal.
    """
    sub_wins = row.get(f'{prefix}WinsBySubmission', 0) or 0
    total_wins = row.get(f'{prefix}Wins', 0) or 0

    return safe_divide(sub_wins, total_wins, default=0.0)


def compute_decision_rate(row: pd.Series, prefix: str) -> float:
    """
    Calculate decision win rate.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Decision rate as decimal.
    """
    dec_maj = row.get(f'{prefix}WinsByDecisionMajority', 0) or 0
    dec_split = row.get(f'{prefix}WinsByDecisionSplit', 0) or 0
    dec_unan = row.get(f'{prefix}WinsByDecisionUnanimous', 0) or 0
    total_wins = row.get(f'{prefix}Wins', 0) or 0

    decisions = dec_maj + dec_split + dec_unan
    return safe_divide(decisions, total_wins, default=0.0)


def compute_experience_score(row: pd.Series, prefix: str) -> float:
    """
    Calculate experience score based on fights and rounds.

    Combines total fights and rounds fought for experience metric.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Experience score (normalized).
    """
    total_fights = compute_total_fights(row, prefix)
    total_rounds = row.get(f'{prefix}TotalRoundsFought', 0) or 0
    title_bouts = row.get(f'{prefix}TotalTitleBouts', 0) or 0

    # Weight: fights + rounds/3 + title_bouts*2
    score = total_fights + (total_rounds / 3) + (title_bouts * 2)
    return score


# =============================================================================
# CONTEXTUAL FEATURES
# =============================================================================

def compute_streak_momentum(row: pd.Series, prefix: str) -> float:
    """
    Calculate momentum based on current streak.

    Positive for win streak, negative for lose streak.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Momentum score.
    """
    win_streak = row.get(f'{prefix}CurrentWinStreak', 0) or 0
    lose_streak = row.get(f'{prefix}CurrentLoseStreak', 0) or 0

    # Positive momentum for wins, negative for losses
    return win_streak - lose_streak


def compute_title_experience(row: pd.Series, prefix: str) -> int:
    """
    Get title bout experience.

    Args:
        row: DataFrame row.
        prefix: 'Red' or 'Blue'.

    Returns:
        Number of title bouts.
    """
    return int(row.get(f'{prefix}TotalTitleBouts', 0) or 0)


# =============================================================================
# DIFFERENTIAL FEATURES
# =============================================================================

def compute_all_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all differential features (Red - Blue).

    Differential features capture the relative advantage/disadvantage
    between fighters. Positive values favor Red, negative favor Blue.

    Args:
        df: DataFrame with Red and Blue fighter stats.

    Returns:
        DataFrame with added differential columns.
    """
    df = df.copy()

    print("Computing differential features...")

    # Physical differentials
    df['HeightDiff'] = df['RedHeightCms'] - df['BlueHeightCms']
    df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']
    df['WeightDiff'] = df['RedWeightLbs'] - df['BlueWeightLbs']

    # Experience differentials
    df['RedTotalFights'] = df.apply(lambda x: compute_total_fights(x, 'Red'), axis=1)
    df['BlueTotalFights'] = df.apply(lambda x: compute_total_fights(x, 'Blue'), axis=1)
    df['ExperienceDiff'] = df['RedTotalFights'] - df['BlueTotalFights']

    # Win/Loss differentials
    df['WinsDiff'] = df['RedWins'] - df['BlueWins']
    df['LossesDiff'] = df['RedLosses'] - df['BlueLosses']

    # Performance differentials
    df['SigStrDiff'] = df['RedAvgSigStrLanded'] - df['BlueAvgSigStrLanded']
    df['SigStrPctDiff'] = df['RedAvgSigStrPct'] - df['BlueAvgSigStrPct']
    df['TDLandedDiff'] = df['RedAvgTDLanded'] - df['BlueAvgTDLanded']
    df['TDPctDiff'] = df['RedAvgTDPct'] - df['BlueAvgTDPct']
    df['SubAttDiff'] = df['RedAvgSubAtt'] - df['BlueAvgSubAtt']

    # Streak differentials
    df['WinStreakDiff'] = df['RedCurrentWinStreak'] - df['BlueCurrentWinStreak']
    df['LoseStreakDiff'] = df['RedCurrentLoseStreak'] - df['BlueCurrentLoseStreak']
    df['LongestWinStreakDiff'] = df['RedLongestWinStreak'] - df['BlueLongestWinStreak']

    # Title experience differential
    df['TitleBoutsDiff'] = df['RedTotalTitleBouts'] - df['BlueTotalTitleBouts']

    print(f"  Added {15} differential features")

    return df


def compute_all_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ratio features where meaningful (Red / Blue).

    Ratios are useful for comparing relative strength.
    Values > 1 favor Red, < 1 favor Blue.

    Args:
        df: DataFrame with Red and Blue fighter stats.

    Returns:
        DataFrame with added ratio columns.
    """
    df = df.copy()

    print("Computing ratio features...")

    # Win rate ratio
    df['RedWinRate'] = df.apply(lambda x: compute_win_rate(x, 'Red'), axis=1)
    df['BlueWinRate'] = df.apply(lambda x: compute_win_rate(x, 'Blue'), axis=1)
    df['WinRateRatio'] = df.apply(
        lambda x: safe_ratio(x['RedWinRate'], x['BlueWinRate']), axis=1
    )

    # Finish rate ratio
    df['RedFinishRate'] = df.apply(lambda x: compute_finish_rate(x, 'Red'), axis=1)
    df['BlueFinishRate'] = df.apply(lambda x: compute_finish_rate(x, 'Blue'), axis=1)
    df['FinishRateRatio'] = df.apply(
        lambda x: safe_ratio(x['RedFinishRate'] + 0.01, x['BlueFinishRate'] + 0.01), axis=1
    )

    # KO rate ratio
    df['RedKORate'] = df.apply(lambda x: compute_ko_rate(x, 'Red'), axis=1)
    df['BlueKORate'] = df.apply(lambda x: compute_ko_rate(x, 'Blue'), axis=1)
    df['KORateRatio'] = df.apply(
        lambda x: safe_ratio(x['RedKORate'] + 0.01, x['BlueKORate'] + 0.01), axis=1
    )

    # Submission rate ratio
    df['RedSubRate'] = df.apply(lambda x: compute_submission_rate(x, 'Red'), axis=1)
    df['BlueSubRate'] = df.apply(lambda x: compute_submission_rate(x, 'Blue'), axis=1)
    df['SubRateRatio'] = df.apply(
        lambda x: safe_ratio(x['RedSubRate'] + 0.01, x['BlueSubRate'] + 0.01), axis=1
    )

    # Experience ratio
    df['RedExperience'] = df.apply(lambda x: compute_experience_score(x, 'Red'), axis=1)
    df['BlueExperience'] = df.apply(lambda x: compute_experience_score(x, 'Blue'), axis=1)
    df['ExperienceRatio'] = df.apply(
        lambda x: safe_ratio(x['RedExperience'] + 1, x['BlueExperience'] + 1), axis=1
    )

    # Striking accuracy ratio
    df['StrikeAccuracyRatio'] = df.apply(
        lambda x: safe_ratio(x['RedAvgSigStrPct'] + 1, x['BlueAvgSigStrPct'] + 1), axis=1
    )

    # Takedown accuracy ratio
    df['TDAccuracyRatio'] = df.apply(
        lambda x: safe_ratio(x['RedAvgTDPct'] + 1, x['BlueAvgTDPct'] + 1), axis=1
    )

    print(f"  Added ratio features and intermediate stats")

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum-related features.

    Args:
        df: DataFrame with fighter stats.

    Returns:
        DataFrame with momentum features.
    """
    df = df.copy()

    print("Computing momentum features...")

    # Momentum scores
    df['RedMomentum'] = df.apply(lambda x: compute_streak_momentum(x, 'Red'), axis=1)
    df['BlueMomentum'] = df.apply(lambda x: compute_streak_momentum(x, 'Blue'), axis=1)
    df['MomentumDiff'] = df['RedMomentum'] - df['BlueMomentum']

    print(f"  Added 3 momentum features")

    return df


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_weight_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode weight class as numeric feature.

    Args:
        df: DataFrame with WeightClass column.

    Returns:
        DataFrame with encoded weight class.
    """
    df = df.copy()

    if 'WeightClass' in df.columns:
        df['WeightClassEncoded'] = df['WeightClass'].map(WEIGHT_CLASS_ENCODING)
        # Fill unknown weight classes with middle value
        df['WeightClassEncoded'] = df['WeightClassEncoded'].fillna(8)
        print(f"Encoded WeightClass")

    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode gender as numeric feature.

    Args:
        df: DataFrame with Gender column.

    Returns:
        DataFrame with encoded gender.
    """
    df = df.copy()

    if 'Gender' in df.columns:
        df['GenderEncoded'] = (df['Gender'] == 'MALE').astype(int)
        print(f"Encoded Gender")

    return df


def encode_stance_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stance matchup features.

    Stance matchups can affect fight dynamics:
    - Orthodox vs Southpaw is a common "open stance" matchup
    - Same stance fighters have different dynamics

    Args:
        df: DataFrame with stance columns.

    Returns:
        DataFrame with stance matchup features.
    """
    df = df.copy()

    # Check if we have the encoded stance columns
    if 'RedStanceEncoded' in df.columns and 'BlueStanceEncoded' in df.columns:
        # Same stance indicator
        df['SameStance'] = (df['RedStanceEncoded'] == df['BlueStanceEncoded']).astype(int)

        # Open stance matchup (Orthodox vs Southpaw)
        df['OpenStanceMatchup'] = (
            ((df['RedStanceEncoded'] == 0) & (df['BlueStanceEncoded'] == 1)) |
            ((df['RedStanceEncoded'] == 1) & (df['BlueStanceEncoded'] == 0))
        ).astype(int)

        # Stance difference (useful for ML)
        df['StanceDiff'] = df['RedStanceEncoded'] - df['BlueStanceEncoded']

        print(f"Created stance matchup features")

    return df


# =============================================================================
# ODDS-BASED FEATURES
# =============================================================================

def compute_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features derived from betting odds.

    Betting odds encode market expectations and can be predictive.

    Args:
        df: DataFrame with odds columns.

    Returns:
        DataFrame with odds-derived features.
    """
    df = df.copy()

    print("Computing odds-based features...")

    if 'RedOdds' in df.columns and 'BlueOdds' in df.columns:
        # Convert American odds to implied probability
        def american_to_prob(odds):
            if pd.isna(odds):
                return 0.5
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)

        df['RedImpliedProb'] = df['RedOdds'].apply(american_to_prob)
        df['BlueImpliedProb'] = df['BlueOdds'].apply(american_to_prob)

        # Odds difference (positive favors Red)
        df['OddsDiff'] = df['RedOdds'] - df['BlueOdds']

        # Implied probability difference
        df['ImpliedProbDiff'] = df['RedImpliedProb'] - df['BlueImpliedProb']

        # Favorite indicator (1 = Red favorite, 0 = Blue favorite)
        df['RedIsFavorite'] = (df['RedImpliedProb'] > df['BlueImpliedProb']).astype(int)

        print(f"  Added 5 odds-based features")

    return df


# =============================================================================
# RANKING FEATURES
# =============================================================================

def compute_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from UFC rankings.

    Args:
        df: DataFrame with ranking columns.

    Returns:
        DataFrame with ranking features.
    """
    df = df.copy()

    print("Computing ranking features...")

    rank_cols_red = ['RMatchWCRank', 'RPFPRank']
    rank_cols_blue = ['BMatchWCRank', 'BPFPRank']

    # Fill missing rankings with 99 (unranked)
    for col in rank_cols_red + rank_cols_blue:
        if col in df.columns:
            df[col] = df[col].fillna(99)

    # Weight class rank difference
    if 'RMatchWCRank' in df.columns and 'BMatchWCRank' in df.columns:
        df['WCRankDiff'] = df['RMatchWCRank'] - df['BMatchWCRank']
        # Ranked indicator
        df['RedIsRanked'] = (df['RMatchWCRank'] < 16).astype(int)
        df['BlueIsRanked'] = (df['BMatchWCRank'] < 16).astype(int)
        df['BothRanked'] = (df['RedIsRanked'] & df['BlueIsRanked']).astype(int)

    # Pound-for-pound rank difference
    if 'RPFPRank' in df.columns and 'BPFPRank' in df.columns:
        df['PFPRankDiff'] = df['RPFPRank'] - df['BPFPRank']

    print(f"  Added ranking features")

    return df


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for modeling.

    Excludes identifiers, dates, and target variable.

    Args:
        df: DataFrame with all columns.

    Returns:
        List of feature column names.
    """
    # Columns to exclude from features
    exclude_cols = [
        'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country',
        'WeightClass', 'Gender', 'Winner', 'Target',
        'RedStance', 'BlueStance',  # Keep encoded versions
    ]

    # Get numeric columns that aren't excluded
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Only keep numeric columns
    feature_cols = [col for col in feature_cols
                   if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    return feature_cols


# =============================================================================
# FEATURE SCALING
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Scale numerical features using StandardScaler.

    Standardization is important for many ML algorithms.

    Args:
        df: DataFrame with features to scale.
        feature_columns: List of columns to scale.
        scaler: Pre-fitted scaler (for inference). If None, fits new scaler.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler, feature columns).
    """
    df = df.copy()

    # Get only existing columns
    existing_cols = [col for col in feature_columns if col in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        print(f"Fitted and applied StandardScaler to {len(existing_cols)} features")
    else:
        df[existing_cols] = scaler.transform(df[existing_cols])
        print(f"Applied pre-fitted StandardScaler to {len(existing_cols)} features")

    return df, scaler, existing_cols


# =============================================================================
# MAIN FEATURE PIPELINE
# =============================================================================

def build_feature_set(
    df: pd.DataFrame,
    scale: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Build complete feature set for all fights.

    This is the main function that orchestrates feature engineering.
    It creates all derived features from the cleaned fight data.

    Args:
        df: DataFrame of cleaned fights.
        scale: Whether to apply feature scaling.
        scaler: Pre-fitted scaler (for inference).

    Returns:
        Tuple of (feature DataFrame, scaler, feature column names).
    """
    print("\n" + "="*60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60)

    original_cols = len(df.columns)
    df = df.copy()

    # Step 1: Compute differential features
    print("\n[1/7] Computing differential features...")
    df = compute_all_differential_features(df)

    # Step 2: Compute ratio features
    print("\n[2/7] Computing ratio features...")
    df = compute_all_ratio_features(df)

    # Step 3: Compute momentum features
    print("\n[3/7] Computing momentum features...")
    df = compute_momentum_features(df)

    # Step 4: Encode categorical features
    print("\n[4/7] Encoding categorical features...")
    df = encode_weight_class(df)
    df = encode_gender(df)
    df = encode_stance_matchup(df)

    # Step 5: Compute odds features
    print("\n[5/7] Computing odds features...")
    df = compute_odds_features(df)

    # Step 6: Compute ranking features
    print("\n[6/7] Computing ranking features...")
    df = compute_ranking_features(df)

    # Step 7: Get feature columns and optionally scale
    print("\n[7/7] Finalizing features...")
    feature_columns = get_feature_columns(df)

    if scale:
        df, scaler, feature_columns = scale_features(df, feature_columns, scaler)
    else:
        scaler = None

    final_cols = len(df.columns)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Original columns: {original_cols}")
    print(f"Final columns: {final_cols}")
    print(f"New features created: {final_cols - original_cols}")
    print(f"Total feature columns: {len(feature_columns)}")

    return df, scaler, feature_columns


def save_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: List[str],
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Save feature-engineered data and artifacts.

    Args:
        df: Feature DataFrame to save.
        scaler: Fitted StandardScaler.
        feature_columns: List of feature column names.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary of saved file paths.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed'
        )

    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Save feature data
    features_path = os.path.join(output_dir, 'features.csv')
    df.to_csv(features_path, index=False)
    saved_files['features'] = features_path
    print(f"Saved features to: {features_path}")

    # Save scaler
    if scaler is not None:
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        saved_files['scaler'] = scaler_path
        print(f"Saved scaler to: {scaler_path}")

    # Save feature column names
    columns_path = os.path.join(output_dir, 'feature_columns.txt')
    with open(columns_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    saved_files['columns'] = columns_path
    print(f"Saved feature columns to: {columns_path}")

    return saved_files


def load_feature_artifacts(
    input_dir: Optional[str] = None
) -> Tuple[StandardScaler, List[str]]:
    """
    Load saved feature engineering artifacts.

    Args:
        input_dir: Directory containing artifacts.

    Returns:
        Tuple of (scaler, feature_columns).
    """
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed'
        )

    # Load scaler
    scaler_path = os.path.join(input_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)

    # Load feature columns
    columns_path = os.path.join(input_dir, 'feature_columns.txt')
    with open(columns_path, 'r') as f:
        feature_columns = f.read().strip().split('\n')

    return scaler, feature_columns


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Feature engineering module ready.")

    # Import load functions
    from load import load_processed_fights

    try:
        # Load processed data
        print("\nLoading processed fight data...")
        df = load_processed_fights()

        # Build features
        print("\nBuilding features...")
        df_features, scaler, feature_cols = build_feature_set(df)

        # Save features
        saved = save_features(df_features, scaler, feature_cols)

        # Show sample
        print("\nSample of features:")
        print(df_features[['RedFighter', 'BlueFighter', 'HeightDiff', 'ReachDiff',
                          'WinRateRatio', 'MomentumDiff', 'Target']].head(10))

        print(f"\nFeature columns ({len(feature_cols)}):")
        print(feature_cols[:20])
        print("...")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
