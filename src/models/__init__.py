"""
UFC Fight Predictor - Models Module

Contains functions for training, evaluating, tuning, and explaining ML models.

Modules:
    train: Train baseline and advanced models
    evaluate: Evaluate model performance
    tune: Hyperparameter optimization with Optuna
    explain: SHAP-based model explanations
"""

from .train import (
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
    train_all_models,
    save_model,
    load_model
)
from .evaluate import (
    compute_all_metrics,
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_calibration_curve
)
from .explain import (
    create_explainer,
    compute_shap_values,
    explain_model,
    explain_single_fight,
    plot_feature_importance,
    plot_summary
)

__all__ = [
    # Training
    'train_logistic_regression',
    'train_xgboost',
    'train_lightgbm',
    'train_all_models',
    'save_model',
    'load_model',
    # Evaluation
    'compute_all_metrics',
    'evaluate_model',
    'compare_models',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_calibration_curve',
    # Explainability
    'create_explainer',
    'compute_shap_values',
    'explain_model',
    'explain_single_fight',
    'plot_feature_importance',
    'plot_summary'
]
