"""
UFC Fight Predictor - Hyperparameter Tuning Module

This module handles hyperparameter optimization using Optuna.
Supports tuning for:
- Logistic Regression
- XGBoost
- LightGBM

Optuna uses Bayesian optimization (TPE sampler) to efficiently
search the hyperparameter space.

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, Callable, Optional, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

# Conditional imports for gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================
RANDOM_STATE = 42
N_TRIALS_DEFAULT = 50  # Reduced for faster tuning
CV_FOLDS = 3  # Reduced for faster tuning

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

def create_lr_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Callable:
    """
    Create Optuna objective function for Logistic Regression.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Objective function for Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

        # Use saga solver for l1, lbfgs for l2
        solver = 'saga' if penalty == 'l1' else 'lbfgs'

        # Create and train model
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_prob)

        return roc_auc

    return objective


def create_xgb_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Callable:
    """
    Create Optuna objective function for XGBoost.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Objective function for Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'eval_metric': 'logloss',
        }

        # Create and train model with early stopping
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate on validation set
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_prob)

        return roc_auc

    return objective


def create_lgb_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Callable:
    """
    Create Optuna objective function for LightGBM.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Objective function for Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }

        # Create and train model with early stopping
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Evaluate on validation set
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_prob)

        return roc_auc

    return objective


# =============================================================================
# TUNING FUNCTIONS
# =============================================================================

def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = N_TRIALS_DEFAULT
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune Logistic Regression hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (best parameters, study object).
    """
    print("\n" + "="*60)
    print("TUNING LOGISTIC REGRESSION")
    print("="*60)
    print(f"Trials: {n_trials}")

    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize ROC-AUC
        sampler=TPESampler(seed=RANDOM_STATE)
    )

    # Create objective function
    objective = create_lr_objective(X_train, y_train, X_val, y_val)

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  ROC-AUC: {study.best_value:.4f}")
    print(f"  Parameters: {study.best_params}")

    return study.best_params, study


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = N_TRIALS_DEFAULT
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune XGBoost hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (best parameters, study object).
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed")

    print("\n" + "="*60)
    print("TUNING XGBOOST")
    print("="*60)
    print(f"Trials: {n_trials}")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5)
    )

    # Create objective function
    objective = create_xgb_objective(X_train, y_train, X_val, y_val)

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Add fixed parameters
    best_params = study.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'logloss'

    print(f"\nBest trial:")
    print(f"  ROC-AUC: {study.best_value:.4f}")
    print(f"  Key parameters:")
    print(f"    n_estimators: {best_params.get('n_estimators')}")
    print(f"    max_depth: {best_params.get('max_depth')}")
    print(f"    learning_rate: {best_params.get('learning_rate'):.4f}")

    return best_params, study


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = N_TRIALS_DEFAULT
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Tune LightGBM hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (best parameters, study object).
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed")

    print("\n" + "="*60)
    print("TUNING LIGHTGBM")
    print("="*60)
    print(f"Trials: {n_trials}")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5)
    )

    # Create objective function
    objective = create_lgb_objective(X_train, y_train, X_val, y_val)

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Add fixed parameters
    best_params = study.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1

    print(f"\nBest trial:")
    print(f"  ROC-AUC: {study.best_value:.4f}")
    print(f"  Key parameters:")
    print(f"    n_estimators: {best_params.get('n_estimators')}")
    print(f"    max_depth: {best_params.get('max_depth')}")
    print(f"    learning_rate: {best_params.get('learning_rate'):.4f}")

    return best_params, study


# =============================================================================
# STUDY ANALYSIS
# =============================================================================

def get_best_params(study: optuna.Study) -> Dict[str, Any]:
    """
    Extract best parameters from completed study.

    Args:
        study: Completed Optuna study.

    Returns:
        Dictionary of best parameters.
    """
    return study.best_params.copy()


def print_study_summary(study: optuna.Study, model_name: str) -> None:
    """
    Print summary of optimization study.

    Args:
        study: Completed Optuna study.
        model_name: Name of the model.
    """
    print(f"\n{model_name} Optimization Summary:")
    print(f"  Best ROC-AUC: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Total trials: {len(study.trials)}")

    # Parameter importance (if available)
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n  Parameter importance:")
        for param, importance in sorted(importances.items(),
                                        key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {param}: {importance:.4f}")
    except Exception:
        pass


def save_study(
    study: optuna.Study,
    model_name: str,
    output_dir: str = None
) -> str:
    """
    Save Optuna study for later analysis.

    Args:
        study: Completed Optuna study.
        model_name: Name of the model.
        output_dir: Directory to save study.

    Returns:
        Path to saved study.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'models'
        )

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f'{model_name}_study.joblib')
    joblib.dump(study, filepath)
    print(f"Saved study to: {filepath}")

    return filepath


# =============================================================================
# MAIN TUNING PIPELINE
# =============================================================================

def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = N_TRIALS_DEFAULT,
    save_studies: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Tune all models and return best parameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of trials per model.
        save_studies: Whether to save Optuna studies.

    Returns:
        Dictionary mapping model names to their best parameters.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING FOR ALL MODELS")
    print("="*60)
    print(f"Trials per model: {n_trials}")

    best_params = {}

    # 1. Logistic Regression
    print("\n[1/3] Tuning Logistic Regression...")
    lr_params, lr_study = tune_logistic_regression(
        X_train, y_train, X_val, y_val, n_trials
    )
    best_params['logistic_regression'] = lr_params
    if save_studies:
        save_study(lr_study, 'logistic_regression')

    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n[2/3] Tuning XGBoost...")
        xgb_params, xgb_study = tune_xgboost(
            X_train, y_train, X_val, y_val, n_trials
        )
        best_params['xgboost'] = xgb_params
        if save_studies:
            save_study(xgb_study, 'xgboost')
    else:
        print("\n[2/3] Skipping XGBoost (not installed)")

    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n[3/3] Tuning LightGBM...")
        lgb_params, lgb_study = tune_lightgbm(
            X_train, y_train, X_val, y_val, n_trials
        )
        best_params['lightgbm'] = lgb_params
        if save_studies:
            save_study(lgb_study, 'lightgbm')
    else:
        print("\n[3/3] Skipping LightGBM (not installed)")

    # Summary
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)

    return best_params


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Hyperparameter tuning module ready.")
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

        # Run tuning (with fewer trials for testing)
        best_params = tune_all_models(
            X_train, y_train, X_val, y_val,
            n_trials=20,  # Reduced for testing
            save_studies=True
        )

        print("\nBest parameters found:")
        for model, params in best_params.items():
            print(f"\n{model}:")
            for k, v in params.items():
                print(f"  {k}: {v}")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
