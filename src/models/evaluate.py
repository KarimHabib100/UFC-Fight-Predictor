"""
UFC Fight Predictor - Model Evaluation Module

This module handles comprehensive model evaluation including:
- Standard metrics (Accuracy, ROC-AUC, Log Loss, Brier Score)
- Calibration analysis (how well probabilities match actual outcomes)
- Confusion matrix and classification reports
- ROC and Precision-Recall curves
- Time-based evaluation (performance on recent fights)
- Betting simulation (theoretical returns)

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve

# =============================================================================
# STANDARD METRICS
# =============================================================================

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Accuracy = (True Positives + True Negatives) / Total

    For UFC: Percentage of fights where we correctly predicted the winner.

    Args:
        y_true: True labels (1 = Red wins, 0 = Blue wins).
        y_pred: Predicted labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    return accuracy_score(y_true, y_pred)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute ROC-AUC score.

    ROC-AUC measures the model's ability to rank positive examples
    higher than negative examples. A score of 0.5 = random guessing,
    1.0 = perfect separation.

    For UFC: How well the model ranks Red corner wins higher than
    Blue corner wins in terms of predicted probability.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for class 1.

    Returns:
        ROC-AUC score between 0 and 1.
    """
    return roc_auc_score(y_true, y_prob)


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute log loss (cross-entropy).

    Log loss measures how well-calibrated the probabilities are.
    Lower is better. Heavily penalizes confident wrong predictions.

    For UFC: Important for betting - we want probabilities that
    accurately reflect the true likelihood of outcomes.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Log loss value (lower is better).
    """
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_prob)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score.

    Brier score = mean((predicted_prob - actual_outcome)^2)
    Measures the mean squared error of probability predictions.
    Range: 0 (perfect) to 1 (worst).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Brier score (lower is better).
    """
    return brier_score_loss(y_true, y_prob)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute all standard metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary of all metrics.
    """
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'roc_auc': compute_roc_auc(y_true, y_prob),
        'log_loss': compute_log_loss(y_true, y_prob),
        'brier_score': compute_brier_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'average_precision': average_precision_score(y_true, y_prob)
    }
    return metrics


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    A well-calibrated model should have predicted probabilities
    that match actual win rates (e.g., 70% predicted = 70% actual wins).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        Tuple of (fraction of positives, mean predicted values, bin counts).
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    # Compute bin counts for confidence intervals
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(y_prob, bins=bin_edges)[0]

    return fraction_of_positives, mean_predicted_value, bin_counts


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|accuracy_in_bin - confidence_in_bin| * bin_weight)

    Lower ECE means better calibration.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        ECE value.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    total_samples = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = np.sum(mask)

        if bin_size > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            ece += (bin_size / total_samples) * np.abs(bin_accuracy - bin_confidence)

    return ece


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot calibration curve with histogram.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        model_name: Name for the plot title.
        n_bins: Number of bins for calibration.
        save_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Calibration curve
    fraction_pos, mean_pred, bin_counts = compute_calibration_curve(
        y_true, y_prob, n_bins
    )

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Model calibration
    ax1.plot(mean_pred, fraction_pos, 's-', label=model_name,
             color='#2196F3', linewidth=2, markersize=8)

    # Compute and display ECE
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins)
    ax1.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Actual)', fontsize=12)
    ax1.set_title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), color='#2196F3',
             edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot to: {save_path}")

    return fig


# =============================================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# =============================================================================

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Matrix layout:
                    Predicted
                    Blue    Red
    Actual  Blue    TN      FP
            Red     FN      TP

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix array.
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix heatmap with counts and percentages.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name for plot title.
        save_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Proportion')

    # Labels
    classes = ['Blue Wins', 'Red Wins']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='Actual',
           xlabel='Predicted')

    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")

    return fig


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dict: bool = False
) -> Any:
    """
    Generate classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dict: Return as dictionary if True.

    Returns:
        Classification report (string or dict).
    """
    target_names = ['Blue Wins', 'Red Wins']
    return classification_report(y_true, y_pred,
                                 target_names=target_names,
                                 output_dict=output_dict)


# =============================================================================
# ROC & PRECISION-RECALL CURVES
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        model_name: Name for the plot title.
        save_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)

    # ROC curve
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'{model_name} (AUC = {roc_auc:.4f})')

    # Random classifier
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to: {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve with Average Precision.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        model_name: Name for the plot title.
        save_path: Path to save the plot (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Baseline (proportion of positive class)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=figsize)

    # PR curve
    ax.plot(recall, precision, color='#4CAF50', lw=2,
            label=f'{model_name} (AP = {avg_precision:.4f})')

    # Baseline
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2,
               label=f'Baseline (Red win rate = {baseline:.1%})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curve to: {save_path}")

    return fig


def plot_all_roc_curves(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same plot.

    Args:
        models_data: Dict of {model_name: (y_true, y_prob)}.
        save_path: Path to save the plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    # Random classifier
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison ROC curves to: {save_path}")

    return fig


# =============================================================================
# TIME-BASED EVALUATION
# =============================================================================

def evaluate_recent_fights(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates: pd.Series,
    n_recent: int = 100
) -> Dict[str, float]:
    """
    Evaluate model on most recent N fights.

    This provides a more realistic assessment of how the model
    would perform on upcoming fights.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        dates: Fight dates for sorting.
        n_recent: Number of most recent fights to evaluate.

    Returns:
        Dictionary of metrics on recent fights.
    """
    # Sort by date (most recent last)
    sorted_indices = dates.argsort()
    recent_indices = sorted_indices[-n_recent:]

    X_recent = X_test.iloc[recent_indices]
    y_recent = y_test.iloc[recent_indices]

    y_pred = model.predict(X_recent)
    y_prob = model.predict_proba(X_recent)[:, 1]

    metrics = compute_all_metrics(y_recent.values, y_pred, y_prob)
    metrics['n_fights'] = n_recent
    metrics['date_range'] = f"{dates.iloc[recent_indices[0]]} to {dates.iloc[recent_indices[-1]]}"

    return metrics


def evaluate_by_time_periods(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates: pd.Series,
    period: str = 'quarter'
) -> pd.DataFrame:
    """
    Evaluate model performance across different time periods.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        dates: Fight dates.
        period: 'month', 'quarter', or 'year'.

    Returns:
        DataFrame with metrics per time period.
    """
    df = pd.DataFrame({
        'date': dates,
        'y_true': y_test.values,
        'y_pred': model.predict(X_test),
        'y_prob': model.predict_proba(X_test)[:, 1]
    })

    # Create period column
    if period == 'month':
        df['period'] = df['date'].dt.to_period('M')
    elif period == 'quarter':
        df['period'] = df['date'].dt.to_period('Q')
    else:
        df['period'] = df['date'].dt.to_period('Y')

    # Compute metrics per period
    results = []
    for p, group in df.groupby('period'):
        if len(group) >= 10:  # Minimum sample size
            metrics = {
                'period': str(p),
                'n_fights': len(group),
                'accuracy': compute_accuracy(group['y_true'].values, group['y_pred'].values),
                'roc_auc': compute_roc_auc(group['y_true'].values, group['y_prob'].values)
            }
            results.append(metrics)

    return pd.DataFrame(results)


# =============================================================================
# BETTING SIMULATION
# =============================================================================

def simulate_betting_returns(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds_red: np.ndarray,
    odds_blue: np.ndarray,
    threshold: float = 0.55,
    stake: float = 100
) -> Dict[str, float]:
    """
    Simulate betting returns based on model predictions.

    Strategy: Bet on a fighter when predicted probability > threshold
    and there's positive expected value.

    Args:
        y_true: True labels (1 = Red wins).
        y_prob: Predicted probability of Red winning.
        odds_red: American odds for Red.
        odds_blue: American odds for Blue.
        threshold: Minimum probability to place a bet.
        stake: Base stake per bet.

    Returns:
        Dictionary with betting simulation results.
    """
    total_bets = 0
    total_staked = 0
    total_returns = 0
    winning_bets = 0

    for i in range(len(y_true)):
        pred_prob_red = y_prob[i]
        pred_prob_blue = 1 - pred_prob_red

        # Convert American odds to decimal
        if odds_red[i] > 0:
            decimal_red = 1 + odds_red[i] / 100
        else:
            decimal_red = 1 + 100 / abs(odds_red[i])

        if odds_blue[i] > 0:
            decimal_blue = 1 + odds_blue[i] / 100
        else:
            decimal_blue = 1 + 100 / abs(odds_blue[i])

        # Check for value bets
        ev_red = pred_prob_red * decimal_red - 1
        ev_blue = pred_prob_blue * decimal_blue - 1

        bet_placed = False
        bet_won = False
        returns = 0

        # Bet on Red if confident and positive EV
        if pred_prob_red >= threshold and ev_red > 0:
            total_bets += 1
            total_staked += stake
            bet_placed = True

            if y_true[i] == 1:  # Red won
                returns = stake * decimal_red
                winning_bets += 1
                bet_won = True

        # Bet on Blue if confident and positive EV
        elif pred_prob_blue >= threshold and ev_blue > 0:
            total_bets += 1
            total_staked += stake
            bet_placed = True

            if y_true[i] == 0:  # Blue won
                returns = stake * decimal_blue
                winning_bets += 1
                bet_won = True

        total_returns += returns

    roi = ((total_returns - total_staked) / total_staked * 100) if total_staked > 0 else 0

    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': winning_bets / total_bets if total_bets > 0 else 0,
        'total_staked': total_staked,
        'total_returns': total_returns,
        'profit': total_returns - total_staked,
        'roi': roi,
        'threshold': threshold
    }


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save_plots: bool = False,
    plots_dir: str = "plots/",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name for reports/plots.
        save_plots: Whether to save plots.
        plots_dir: Directory to save plots.
        verbose: Print results.

    Returns:
        Dictionary with all evaluation results.
    """
    # Create plots directory if needed
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute all metrics
    metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

    # Compute ECE
    metrics['expected_calibration_error'] = compute_expected_calibration_error(
        y_test.values, y_prob
    )

    # Get classification report
    clf_report = get_classification_report(y_test.values, y_pred, output_dict=True)

    # Get confusion matrix
    cm = compute_confusion_matrix(y_test.values, y_pred)

    if verbose:
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - {model_name}")
        print("="*60)

        print(f"\nTest Set Size: {len(y_test)} fights")
        print(f"Red Win Rate: {y_test.mean():.1%}")

        print(f"\n--- Standard Metrics ---")
        print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
        print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"Log Loss:           {metrics['log_loss']:.4f}")
        print(f"Brier Score:        {metrics['brier_score']:.4f}")

        print(f"\n--- Additional Metrics ---")
        print(f"Precision (Red):    {metrics['precision']:.4f}")
        print(f"Recall (Red):       {metrics['recall']:.4f}")
        print(f"F1 Score:           {metrics['f1_score']:.4f}")
        print(f"Average Precision:  {metrics['average_precision']:.4f}")

        print(f"\n--- Calibration ---")
        print(f"Expected Cal. Error: {metrics['expected_calibration_error']:.4f}")

        print(f"\n--- Confusion Matrix ---")
        print(f"                  Predicted Blue  Predicted Red")
        print(f"Actual Blue       {cm[0, 0]:>12}  {cm[0, 1]:>13}")
        print(f"Actual Red        {cm[1, 0]:>12}  {cm[1, 1]:>13}")

        print(f"\n--- Classification Report ---")
        print(get_classification_report(y_test.values, y_pred))

    # Generate plots
    if save_plots:
        plot_confusion_matrix(
            y_test.values, y_pred, model_name,
            save_path=os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_confusion.png")
        )
        plot_roc_curve(
            y_test.values, y_prob, model_name,
            save_path=os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_roc.png")
        )
        plot_precision_recall_curve(
            y_test.values, y_prob, model_name,
            save_path=os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_pr.png")
        )
        plot_calibration_curve(
            y_test.values, y_prob, model_name,
            save_path=os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_calibration.png")
        )
        plt.close('all')  # Close all figures to free memory

    return {
        'metrics': metrics,
        'classification_report': clf_report,
        'confusion_matrix': cm.tolist(),
        'predictions': {
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist()
        }
    }


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_plots: bool = False,
    plots_dir: str = "plots/"
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary of {model_name: model}.
        X_test: Test features.
        y_test: Test labels.
        save_plots: Whether to save comparison plots.
        plots_dir: Directory to save plots.

    Returns:
        DataFrame with comparison results.
    """
    results = []
    roc_data = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test.values, y_pred, y_prob)
        metrics['model'] = model_name
        metrics['ece'] = compute_expected_calibration_error(y_test.values, y_prob)

        results.append(metrics)
        roc_data[model_name] = (y_test.values, y_prob)

    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.set_index('model')

    # Sort by ROC-AUC
    df = df.sort_values('roc_auc', ascending=False)

    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON - TEST SET")
    print("="*80)
    print(f"\nTest Set Size: {len(y_test)} fights")
    print(f"Red Win Rate: {y_test.mean():.1%}")

    print("\n" + "-"*80)
    print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Log Loss':>10} {'Brier':>10} {'ECE':>10}")
    print("-"*80)
    for model_name in df.index:
        row = df.loc[model_name]
        print(f"{model_name:<25} {row['accuracy']:>10.4f} {row['roc_auc']:>10.4f} "
              f"{row['log_loss']:>10.4f} {row['brier_score']:>10.4f} {row['ece']:>10.4f}")
    print("-"*80)

    # Identify best model
    best_model = df['roc_auc'].idxmax()
    print(f"\nBest Model (by ROC-AUC): {best_model}")

    # Save comparison plot
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        plot_all_roc_curves(
            roc_data,
            save_path=os.path.join(plots_dir, "model_comparison_roc.png")
        )
        plt.close('all')

    return df


def generate_evaluation_report(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "evaluation_report.json"
) -> None:
    """
    Generate a comprehensive evaluation report.

    Args:
        results: Dictionary of evaluation results per model.
        output_path: Path to save the report.
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'models': {}
    }

    for model_name, model_results in results.items():
        report['models'][model_name] = {
            'metrics': model_results['metrics'],
            'classification_report': model_results['classification_report'],
            'confusion_matrix': model_results['confusion_matrix']
        }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nSaved evaluation report to: {output_path}")


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Model evaluation module ready.")
    print("\nAvailable functions:")
    print("  - compute_all_metrics(y_true, y_pred, y_prob)")
    print("  - evaluate_model(model, X_test, y_test, model_name)")
    print("  - compare_models(models, X_test, y_test)")
    print("  - plot_calibration_curve(y_true, y_prob)")
    print("  - plot_confusion_matrix(y_true, y_pred)")
    print("  - plot_roc_curve(y_true, y_prob)")
