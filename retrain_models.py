"""
UFC Fight Predictor - Retrain Models Script

Run this script if you get numpy version errors when loading models.
This will retrain the models with your current package versions.

Usage: python retrain_models.py
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def retrain_models():
    """Retrain all models with current package versions."""
    print("="*60)
    print("RETRAINING MODELS")
    print("="*60)
    print("\nThis will retrain the models with your current numpy/sklearn versions.")
    print("This may take a few minutes...\n")

    try:
        from src.models.train import train_all_models
        from src.utils.helpers import prepare_data_for_training

        # Load data
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_columns = prepare_data_for_training()

        # Train all models
        print("\nTraining models...")
        results = train_all_models(
            X_train, y_train, X_val, y_val,
            feature_columns, save_models=True
        )

        print("\n" + "="*60)
        print("MODELS RETRAINED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run: streamlit run app.py")

        return True

    except Exception as e:
        print(f"\nError during retraining: {e}")
        print("\nTry running: pip install -r requirements.txt --force-reinstall")
        return False

if __name__ == "__main__":
    retrain_models()
