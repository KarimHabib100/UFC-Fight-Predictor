"""
UFC Fight Predictor - Prediction Interface Module

This module provides the main interface for making fight predictions.
It handles:
- Loading trained models
- Looking up fighter statistics
- Building feature vectors for new matchups
- Making predictions with confidence scores
- Generating SHAP explanations

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib
import os
import warnings
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Weight class mappings
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
    "Women's Featherweight": 145
}

# Stance mappings
STANCES = ['Orthodox', 'Southpaw', 'Switch', 'Open Stance', 'Sideways']


# =============================================================================
# MODEL LOADING
# =============================================================================

class FightPredictor:
    """
    Main class for making UFC fight predictions.

    This class encapsulates model loading, fighter lookup,
    feature engineering, and prediction generation.
    """

    def __init__(
        self,
        model_name: str = 'lightgbm',
        models_dir: str = None,
        data_dir: str = None
    ):
        """
        Initialize the fight predictor.

        Args:
            model_name: Name of the model to load ('lightgbm', 'xgboost', 'logistic_regression').
            models_dir: Directory containing trained models.
            data_dir: Directory containing processed data.
        """
        self.models_dir = models_dir or MODELS_DIR
        self.data_dir = data_dir or DATA_DIR
        self.model_name = model_name

        # Load model and metadata
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.scaler = None
        self.fighter_stats = None
        self.explainer = None

        self._load_model()
        self._load_scaler()
        self._load_fighter_stats()

    def _load_model(self) -> None:
        """Load the trained model and its metadata."""
        model_path = os.path.join(self.models_dir, f'{self.model_name}.joblib')
        metadata_path = os.path.join(self.models_dir, f'{self.model_name}_metadata.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        print(f"Loaded model: {self.model_name}")

        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            if 'feature_columns' in self.metadata:
                self.feature_columns = self.metadata['feature_columns']

        # Load feature columns from file if not in metadata
        if self.feature_columns is None:
            feature_cols_path = os.path.join(self.data_dir, 'feature_columns.txt')
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]

    def _load_scaler(self) -> None:
        """Load the feature scaler."""
        scaler_path = os.path.join(self.data_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded feature scaler")

    def _load_fighter_stats(self) -> None:
        """Load fighter statistics from the processed data."""
        # Load the features dataset to extract fighter stats
        features_path = os.path.join(self.data_dir, 'features.csv')

        if os.path.exists(features_path):
            df = pd.read_csv(features_path)

            # Build fighter stats lookup from most recent fights
            self.fighter_stats = self._build_fighter_lookup(df)
            print(f"Loaded stats for {len(self.fighter_stats)} fighters")

    def _build_fighter_lookup(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Build a lookup dictionary of fighter statistics.

        Uses each fighter's most recent fight data.
        """
        fighter_stats = {}

        # Sort by date to get most recent stats
        if 'Date' in df.columns:
            df = df.sort_values('Date', ascending=False)

        # Extract Red fighter stats
        red_cols = [c for c in df.columns if c.startswith('Red') and c != 'RedFighter']
        for _, row in df.iterrows():
            fighter = row.get('RedFighter', '')
            if fighter and fighter not in fighter_stats:
                stats = {}
                for col in red_cols:
                    base_col = col.replace('Red', '')
                    stats[base_col] = row[col]
                fighter_stats[fighter] = stats

        # Extract Blue fighter stats
        blue_cols = [c for c in df.columns if c.startswith('Blue') and c != 'BlueFighter']
        for _, row in df.iterrows():
            fighter = row.get('BlueFighter', '')
            if fighter and fighter not in fighter_stats:
                stats = {}
                for col in blue_cols:
                    base_col = col.replace('Blue', '')
                    stats[base_col] = row[col]
                fighter_stats[fighter] = stats

        return fighter_stats

    def get_fighter_names(self) -> List[str]:
        """Get list of all known fighter names."""
        if self.fighter_stats:
            return sorted(self.fighter_stats.keys())
        return []

    def search_fighter(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for fighters by name.

        Args:
            query: Search query (partial name).
            limit: Maximum number of results.

        Returns:
            List of matching fighter names.
        """
        query = query.lower()
        matches = [
            name for name in self.get_fighter_names()
            if query in name.lower()
        ]
        return matches[:limit]

    def get_fighter_stats(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific fighter.

        Args:
            fighter_name: Fighter's full name.

        Returns:
            Dictionary of fighter statistics, or None if not found.
        """
        if self.fighter_stats and fighter_name in self.fighter_stats:
            return self.fighter_stats[fighter_name].copy()
        return None

    def _build_feature_vector(
        self,
        red_fighter: str,
        blue_fighter: str,
        weight_class: str = 'Lightweight',
        title_bout: bool = False,
        num_rounds: int = 3,
        red_odds: float = -150,
        blue_odds: float = 130
    ) -> pd.DataFrame:
        """
        Build a feature vector for a fight prediction.

        Args:
            red_fighter: Name of red corner fighter.
            blue_fighter: Name of blue corner fighter.
            weight_class: Weight class of the fight.
            title_bout: Whether this is a title fight.
            num_rounds: Number of rounds (3 or 5).
            red_odds: American odds for red fighter.
            blue_odds: American odds for blue fighter.

        Returns:
            DataFrame with feature vector ready for prediction.
        """
        # Get fighter stats
        red_stats = self.get_fighter_stats(red_fighter) or {}
        blue_stats = self.get_fighter_stats(blue_fighter) or {}

        # Initialize feature dictionary
        features = {}

        # Basic fight info
        features['NumberOfRounds'] = num_rounds
        features['TitleBout'] = 1 if title_bout else 0

        # Odds features
        features['RedOdds'] = red_odds
        features['BlueOdds'] = blue_odds

        # Convert odds to implied probability
        if red_odds < 0:
            red_implied = abs(red_odds) / (abs(red_odds) + 100)
        else:
            red_implied = 100 / (red_odds + 100)

        if blue_odds < 0:
            blue_implied = abs(blue_odds) / (abs(blue_odds) + 100)
        else:
            blue_implied = 100 / (blue_odds + 100)

        features['RedImpliedProb'] = red_implied
        features['BlueImpliedProb'] = blue_implied
        features['ImpliedProbDiff'] = red_implied - blue_implied
        features['OddsDiff'] = red_odds - blue_odds

        # Expected value
        if red_odds > 0:
            red_decimal = 1 + red_odds / 100
        else:
            red_decimal = 1 + 100 / abs(red_odds)

        if blue_odds > 0:
            blue_decimal = 1 + blue_odds / 100
        else:
            blue_decimal = 1 + 100 / abs(blue_odds)

        features['RedExpectedValue'] = red_implied * red_decimal - 1
        features['BlueExpectedValue'] = blue_implied * blue_decimal - 1
        features['RedIsFavorite'] = 1 if red_odds < blue_odds else 0

        # Fighter stats - Red
        stat_cols = [
            'Odds', 'ExpectedValue', 'CurrentLoseStreak', 'CurrentWinStreak',
            'Draws', 'AvgSigStrLanded', 'AvgSigStrPct', 'AvgSubAtt',
            'AvgTDLanded', 'AvgTDPct', 'LongestWinStreak', 'Losses',
            'TotalRoundsFought', 'TotalTitleBouts', 'WinsByDecisionMajority',
            'WinsByDecisionSplit', 'WinsByDecisionUnanimous', 'WinsByKO',
            'WinsBySubmission', 'WinsByTKODoctorStoppage', 'Wins',
            'HeightCms', 'ReachCms', 'WeightLbs'
        ]

        for col in stat_cols:
            red_val = red_stats.get(col, 0)
            blue_val = blue_stats.get(col, 0)

            features[f'Red{col}'] = red_val if pd.notna(red_val) else 0
            features[f'Blue{col}'] = blue_val if pd.notna(blue_val) else 0

        # Differential features
        diff_cols = ['AvgSigStrLanded', 'AvgSigStrPct', 'AvgTDLanded', 'AvgTDPct',
                     'Wins', 'Losses', 'CurrentWinStreak', 'HeightCms', 'ReachCms']

        for col in diff_cols:
            red_val = features.get(f'Red{col}', 0)
            blue_val = features.get(f'Blue{col}', 0)

            diff_name = col.replace('Avg', '').replace('Cms', '').replace('Pct', 'Pct')
            features[f'{diff_name}Diff'] = red_val - blue_val

        # Ratio features
        red_wins = features.get('RedWins', 0)
        red_losses = features.get('RedLosses', 0)
        blue_wins = features.get('BlueWins', 0)
        blue_losses = features.get('BlueLosses', 0)

        red_total = red_wins + red_losses
        blue_total = blue_wins + blue_losses

        red_win_rate = red_wins / red_total if red_total > 0 else 0.5
        blue_win_rate = blue_wins / blue_total if blue_total > 0 else 0.5

        features['WinRateRatio'] = red_win_rate / blue_win_rate if blue_win_rate > 0 else 1.0

        # Finish rates
        red_ko = features.get('RedWinsByKO', 0)
        red_sub = features.get('RedWinsBySubmission', 0)
        blue_ko = features.get('BlueWinsByKO', 0)
        blue_sub = features.get('BlueWinsBySubmission', 0)

        features['FinishRateRatio'] = ((red_ko + red_sub) / red_wins if red_wins > 0 else 0) / \
                                      ((blue_ko + blue_sub) / blue_wins if blue_wins > 0 else 0.5)

        # Momentum features
        features['RedMomentum'] = features.get('RedCurrentWinStreak', 0) - features.get('RedCurrentLoseStreak', 0)
        features['BlueMomentum'] = features.get('BlueCurrentWinStreak', 0) - features.get('BlueCurrentLoseStreak', 0)
        features['MomentumDiff'] = features['RedMomentum'] - features['BlueMomentum']

        # Weight class encoding (simplified)
        weight = WEIGHT_CLASSES.get(weight_class, 155)
        features['WeightClass_encoded'] = weight / 265  # Normalize

        # Gender (assume male unless women's division)
        features['Gender_encoded'] = 0 if "Women's" in weight_class else 1

        # Stance matchup (default Orthodox vs Orthodox)
        red_stance = red_stats.get('Stance', 'Orthodox')
        blue_stance = blue_stats.get('Stance', 'Orthodox')
        features['StanceMatchup_encoded'] = self._encode_stance_matchup(red_stance, blue_stance)

        # Ranking features (default to unranked)
        features['RMatchWCRank'] = 0
        features['BMatchWCRank'] = 0
        features['RPFPRank'] = 0
        features['BPFPRank'] = 0
        features['WCRankDiff'] = 0
        features['RedIsRanked'] = 0
        features['BlueIsRanked'] = 0
        features['BothRanked'] = 0
        features['PFPRankDiff'] = 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([features])

        # Ensure all required columns exist
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Reorder columns to match training
            df = df[self.feature_columns]

        return df

    def _encode_stance_matchup(self, red_stance: str, blue_stance: str) -> float:
        """Encode the stance matchup as a numeric value."""
        # Simple encoding: same stance = 0, different = 1
        if red_stance == blue_stance:
            return 0.0
        elif red_stance == 'Southpaw' or blue_stance == 'Southpaw':
            return 0.5  # Southpaw matchups are interesting
        else:
            return 0.25

    def predict(
        self,
        red_fighter: str,
        blue_fighter: str,
        weight_class: str = 'Lightweight',
        title_bout: bool = False,
        num_rounds: int = 3,
        red_odds: float = -150,
        blue_odds: float = 130,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make a fight prediction.

        Args:
            red_fighter: Name of red corner fighter.
            blue_fighter: Name of blue corner fighter.
            weight_class: Weight class of the fight.
            title_bout: Whether this is a title fight.
            num_rounds: Number of rounds (3 or 5).
            red_odds: American odds for red fighter.
            blue_odds: American odds for blue fighter.
            explain: Whether to include SHAP explanation.

        Returns:
            Dictionary with prediction results.
        """
        # Build feature vector
        X = self._build_feature_vector(
            red_fighter, blue_fighter,
            weight_class, title_bout, num_rounds,
            red_odds, blue_odds
        )

        # Make prediction
        pred_proba = self.model.predict_proba(X)[0]
        pred_class = self.model.predict(X)[0]

        result = {
            'red_fighter': red_fighter,
            'blue_fighter': blue_fighter,
            'weight_class': weight_class,
            'title_bout': title_bout,
            'num_rounds': num_rounds,
            'prediction': 'Red Wins' if pred_class == 1 else 'Blue Wins',
            'predicted_winner': red_fighter if pred_class == 1 else blue_fighter,
            'red_win_probability': float(pred_proba[1]),
            'blue_win_probability': float(pred_proba[0]),
            'confidence': float(max(pred_proba)),
            'red_odds': red_odds,
            'blue_odds': blue_odds,
            'model_used': self.model_name,
            'timestamp': datetime.now().isoformat()
        }

        # Add confidence level
        confidence = result['confidence']
        if confidence >= 0.7:
            result['confidence_level'] = 'High'
        elif confidence >= 0.6:
            result['confidence_level'] = 'Medium'
        else:
            result['confidence_level'] = 'Low'

        # Add SHAP explanation if requested
        if explain:
            result['explanation'] = self._get_explanation(X)

        return result

    def _get_explanation(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction."""
        try:
            import shap

            # Create explainer if not exists
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)

            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (Red wins)

            shap_values = shap_values[0]  # Single sample

            # Get top contributing features
            feature_importance = list(zip(self.feature_columns, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            top_for_red = [(f, v) for f, v in feature_importance if v > 0][:5]
            top_for_blue = [(f, v) for f, v in feature_importance if v < 0][:5]

            return {
                'top_factors_red': [{'feature': f, 'impact': float(v)} for f, v in top_for_red],
                'top_factors_blue': [{'feature': f, 'impact': float(abs(v))} for f, v in top_for_blue],
                'all_shap_values': dict(zip(self.feature_columns, shap_values.tolist()))
            }

        except Exception as e:
            return {'error': str(e)}

    def predict_card(
        self,
        fights: List[Dict[str, Any]],
        explain: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict outcomes for an entire fight card.

        Args:
            fights: List of fight dictionaries with fighter names and details.
            explain: Whether to include SHAP explanations.

        Returns:
            List of prediction results.
        """
        results = []

        for fight in fights:
            try:
                result = self.predict(
                    red_fighter=fight.get('red_fighter', 'Unknown'),
                    blue_fighter=fight.get('blue_fighter', 'Unknown'),
                    weight_class=fight.get('weight_class', 'Lightweight'),
                    title_bout=fight.get('title_bout', False),
                    num_rounds=fight.get('num_rounds', 3),
                    red_odds=fight.get('red_odds', -150),
                    blue_odds=fight.get('blue_odds', 130),
                    explain=explain
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'red_fighter': fight.get('red_fighter', 'Unknown'),
                    'blue_fighter': fight.get('blue_fighter', 'Unknown')
                })

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_predictor(model_name: str = 'lightgbm') -> FightPredictor:
    """
    Load a fight predictor with the specified model.

    Args:
        model_name: Name of the model ('lightgbm', 'xgboost', 'logistic_regression').

    Returns:
        Initialized FightPredictor instance.
    """
    return FightPredictor(model_name=model_name)


def quick_predict(
    red_fighter: str,
    blue_fighter: str,
    red_odds: float = -150,
    blue_odds: float = 130
) -> Dict[str, Any]:
    """
    Make a quick prediction with default settings.

    Args:
        red_fighter: Name of red corner fighter.
        blue_fighter: Name of blue corner fighter.
        red_odds: American odds for red fighter.
        blue_odds: American odds for blue fighter.

    Returns:
        Prediction result dictionary.
    """
    predictor = FightPredictor()
    return predictor.predict(
        red_fighter=red_fighter,
        blue_fighter=blue_fighter,
        red_odds=red_odds,
        blue_odds=blue_odds
    )


def print_prediction(result: Dict[str, Any]) -> None:
    """
    Print a formatted prediction result.

    Args:
        result: Prediction result from predict().
    """
    print("\n" + "="*60)
    print("UFC FIGHT PREDICTION")
    print("="*60)

    print(f"\n{result['red_fighter']} vs {result['blue_fighter']}")
    print(f"Weight Class: {result['weight_class']}")
    if result['title_bout']:
        print("*** TITLE FIGHT ***")

    print(f"\n--- Prediction ---")
    print(f"Winner: {result['predicted_winner']}")
    print(f"Confidence: {result['confidence']:.1%} ({result['confidence_level']})")

    print(f"\n--- Probabilities ---")
    print(f"{result['red_fighter']}: {result['red_win_probability']:.1%}")
    print(f"{result['blue_fighter']}: {result['blue_win_probability']:.1%}")

    print(f"\n--- Betting Odds ---")
    print(f"{result['red_fighter']}: {result['red_odds']:+d}")
    print(f"{result['blue_fighter']}: {result['blue_odds']:+d}")

    if 'explanation' in result and 'error' not in result['explanation']:
        print(f"\n--- Key Factors ---")
        print(f"Favoring {result['red_fighter']}:")
        for f in result['explanation']['top_factors_red'][:3]:
            print(f"  + {f['feature']}: {f['impact']:+.3f}")

        print(f"Favoring {result['blue_fighter']}:")
        for f in result['explanation']['top_factors_blue'][:3]:
            print(f"  - {f['feature']}: {f['impact']:.3f}")

    print("\n" + "="*60)


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("UFC Fight Predictor - Prediction Module")
    print("="*50)

    # Load predictor
    predictor = load_predictor('lightgbm')

    # Show available fighters
    print(f"\nLoaded {len(predictor.get_fighter_names())} fighters")

    # Search example
    print("\nSearching for 'McGregor':")
    matches = predictor.search_fighter('McGregor')
    for m in matches:
        print(f"  - {m}")
