"""
UFC Fight Predictor - Model Training Module

This module handles training of ML models including:
- Baseline models (Logistic Regression)
- Advanced models (XGBoost, LightGBM)
- Model persistence with joblib

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

# =============================================================================
# CONSTANTS
# =============================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
RANDOM_STATE = 42

# Default hyperparameters
DEFAULT_LR_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'logloss',
}

DEFAULT_LGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
}

# =============================================================================
# BASELINE MODELS
# =============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a Logistic Regression baseline model.

    Logistic Regression is a good baseline because:
    1. It's interpretable - coefficients show feature importance
    2. It outputs calibrated probabilities
    3. It's fast to train and predict
    4. It establishes a performance floor for comparison

    Args:
        X_train: Training features.
        y_train: Training labels (1 = Red wins, 0 = Blue wins).
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        params: Model hyperparameters.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)

    # Use default params if none provided
    if params is None:
        params = DEFAULT_LR_PARAMS.copy()

    print(f"Parameters: {params}")

    # Initialize and train
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Compute metrics
    metrics = {}

    # Training metrics
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)
    metrics['train_log_loss'] = log_loss(y_train, y_train_prob)

    print(f"\nTraining metrics:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['train_roc_auc']:.4f}")
    print(f"  Log Loss: {metrics['train_log_loss']:.4f}")

    # Validation metrics
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_prob)
        metrics['val_log_loss'] = log_loss(y_val, y_val_prob)

        print(f"\nValidation metrics:")
        print(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['val_roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['val_log_loss']:.4f}")

    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features by importance:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        params: Model hyperparameters.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)

    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }

    print(f"Parameters: {params}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Compute metrics
    metrics = {}
    y_train_prob = model.predict_proba(X_train)[:, 1]
    metrics['train_accuracy'] = model.score(X_train, y_train)
    metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)

    print(f"\nTraining metrics:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['train_roc_auc']:.4f}")

    if X_val is not None and y_val is not None:
        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics['val_accuracy'] = model.score(X_val, y_val)
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_prob)

        print(f"\nValidation metrics:")
        print(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['val_roc_auc']:.4f}")

    return model, metrics


# =============================================================================
# ADVANCED MODELS
# =============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50
) -> Tuple['xgb.XGBClassifier', Dict[str, float]]:
    """
    Train an XGBoost model.

    XGBoost advantages:
    1. Handles missing values natively
    2. Built-in regularization (L1/L2)
    3. Excellent performance on tabular data
    4. Early stopping to prevent overfitting

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features for early stopping.
        y_val: Validation labels.
        params: XGBoost hyperparameters.
        early_stopping_rounds: Rounds for early stopping.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")

    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)

    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    print(f"Key parameters:")
    print(f"  n_estimators: {params.get('n_estimators', 500)}")
    print(f"  max_depth: {params.get('max_depth', 6)}")
    print(f"  learning_rate: {params.get('learning_rate', 0.05)}")

    # Add early stopping to params if validation data will be provided
    if X_val is not None and y_val is not None:
        params = params.copy()
        params['early_stopping_rounds'] = early_stopping_rounds

    model = xgb.XGBClassifier(**params)

    # Training with early stopping if validation data provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        print(f"  Best iteration: {model.best_iteration}")
    else:
        model.fit(X_train, y_train)

    # Compute metrics
    metrics = {}
    y_train_prob = model.predict_proba(X_train)[:, 1]
    metrics['train_accuracy'] = model.score(X_train, y_train)
    metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)
    metrics['train_log_loss'] = log_loss(y_train, y_train_prob)

    print(f"\nTraining metrics:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['train_roc_auc']:.4f}")
    print(f"  Log Loss: {metrics['train_log_loss']:.4f}")

    if X_val is not None and y_val is not None:
        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics['val_accuracy'] = model.score(X_val, y_val)
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_prob)
        metrics['val_log_loss'] = log_loss(y_val, y_val_prob)

        print(f"\nValidation metrics:")
        print(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['val_roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['val_log_loss']:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50
) -> Tuple['lgb.LGBMClassifier', Dict[str, float]]:
    """
    Train a LightGBM model.

    LightGBM advantages:
    1. Faster training than XGBoost
    2. Lower memory usage
    3. Better with categorical features
    4. Handles large datasets efficiently

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features for early stopping.
        y_val: Validation labels.
        params: LightGBM hyperparameters.
        early_stopping_rounds: Rounds for early stopping.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)

    if params is None:
        params = DEFAULT_LGB_PARAMS.copy()

    print(f"Key parameters:")
    print(f"  n_estimators: {params.get('n_estimators', 500)}")
    print(f"  max_depth: {params.get('max_depth', 6)}")
    print(f"  learning_rate: {params.get('learning_rate', 0.05)}")

    model = lgb.LGBMClassifier(**params)

    # Training with early stopping if validation data provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        print(f"  Best iteration: {model.best_iteration_}")
    else:
        model.fit(X_train, y_train)

    # Compute metrics
    metrics = {}
    y_train_prob = model.predict_proba(X_train)[:, 1]
    metrics['train_accuracy'] = model.score(X_train, y_train)
    metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_prob)
    metrics['train_log_loss'] = log_loss(y_train, y_train_prob)

    print(f"\nTraining metrics:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['train_roc_auc']:.4f}")
    print(f"  Log Loss: {metrics['train_log_loss']:.4f}")

    if X_val is not None and y_val is not None:
        y_val_prob = model.predict_proba(X_val)[:, 1]
        metrics['val_accuracy'] = model.score(X_val, y_val)
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_prob)
        metrics['val_log_loss'] = log_loss(y_val, y_val_prob)

        print(f"\nValidation metrics:")
        print(f"  Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  ROC-AUC:  {metrics['val_roc_auc']:.4f}")
        print(f"  Log Loss: {metrics['val_log_loss']:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, metrics


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, float] = None,
    feature_columns: List[str] = None,
    output_dir: str = None
) -> str:
    """
    Save trained model and metadata to disk.

    Args:
        model: Trained model object.
        model_name: Name for the saved model file.
        metrics: Training/validation metrics.
        feature_columns: List of feature column names.
        output_dir: Directory to save model.

    Returns:
        Full path to saved model.
    """
    if output_dir is None:
        output_dir = MODELS_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'created_at': datetime.now().isoformat(),
        'metrics': metrics or {},
        'feature_columns': feature_columns or [],
    }

    metadata_path = os.path.join(output_dir, f'{model_name}_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to: {metadata_path}")

    return model_path


def load_model(model_name: str, input_dir: str = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained model and metadata from disk.

    Args:
        model_name: Name of the model to load.
        input_dir: Directory containing model files.

    Returns:
        Tuple of (model, metadata).
    """
    if input_dir is None:
        input_dir = MODELS_DIR

    model_path = os.path.join(input_dir, f'{model_name}.joblib')
    metadata_path = os.path.join(input_dir, f'{model_name}_metadata.joblib')

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    metadata = {}
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        print(f"Loaded metadata from: {metadata_path}")

    return model, metadata


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_columns: List[str] = None,
    save_models: bool = True
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all models (baseline and advanced).

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        feature_columns: Feature column names.
        save_models: Whether to save trained models.

    Returns:
        Dictionary mapping model names to (model, metrics) tuples.
    """
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)

    results = {}

    # 1. Logistic Regression (baseline)
    print("\n[1/3] Training Logistic Regression baseline...")
    lr_model, lr_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    results['logistic_regression'] = (lr_model, lr_metrics)

    if save_models:
        save_model(lr_model, 'logistic_regression', lr_metrics,
                  feature_columns or list(X_train.columns))

    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n[2/3] Training XGBoost...")
        xgb_model, xgb_metrics = train_xgboost(
            X_train, y_train, X_val, y_val
        )
        results['xgboost'] = (xgb_model, xgb_metrics)

        if save_models:
            save_model(xgb_model, 'xgboost', xgb_metrics,
                      feature_columns or list(X_train.columns))
    else:
        print("\n[2/3] Skipping XGBoost (not installed)")

    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n[3/3] Training LightGBM...")
        lgb_model, lgb_metrics = train_lightgbm(
            X_train, y_train, X_val, y_val
        )
        results['lightgbm'] = (lgb_model, lgb_metrics)

        if save_models:
            save_model(lgb_model, 'lightgbm', lgb_metrics,
                      feature_columns or list(X_train.columns))
    else:
        print("\n[3/3] Skipping LightGBM (not installed)")

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Val Accuracy':<15} {'Val ROC-AUC':<15} {'Val Log Loss':<15}")
    print("-" * 70)

    for name, (model, metrics) in results.items():
        val_acc = metrics.get('val_accuracy', 0)
        val_auc = metrics.get('val_roc_auc', 0)
        val_ll = metrics.get('val_log_loss', 0)
        print(f"{name:<25} {val_acc:<15.4f} {val_auc:<15.4f} {val_ll:<15.4f}")

    # Find best model
    best_model = max(results.items(),
                    key=lambda x: x[1][1].get('val_roc_auc', 0))
    print(f"\nBest model: {best_model[0]} (ROC-AUC: {best_model[1][1].get('val_roc_auc', 0):.4f})")

    return results


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Model training module ready.")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")

    # Import data preparation
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from helpers import prepare_data_for_training

    try:
        # Load data
        print("\nLoading data...")
        X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data_for_training()

        # Train all models
        results = train_all_models(
            X_train, y_train, X_val, y_val,
            feature_columns=features,
            save_models=True
        )

        print("\nTraining complete!")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
