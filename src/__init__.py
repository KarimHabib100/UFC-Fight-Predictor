"""
UFC Fight Predictor - Source Package

This package contains all modules for the UFC fight prediction system.

Modules:
    data: Data loading, cleaning, and feature engineering
    models: Model training, evaluation, tuning, and explainability
    prediction: Prediction interface for making fight predictions
    utils: Helper functions and utilities

Example usage:
    from src.prediction import FightPredictor
    predictor = FightPredictor('lightgbm')
    result = predictor.predict('Jon Jones', 'Stipe Miocic')
"""

__version__ = '1.0.0'
__author__ = 'UFC Fight Predictor Team'
