"""
UFC Fight Predictor - Prediction Module

This module provides the main interface for making fight predictions.
"""

from .predict import (
    FightPredictor,
    load_predictor,
    quick_predict,
    print_prediction
)

__all__ = [
    'FightPredictor',
    'load_predictor',
    'quick_predict',
    'print_prediction'
]
