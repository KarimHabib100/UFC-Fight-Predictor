"""
UFC Fight Predictor - SHAP Explainability Module

This module provides model interpretability using SHAP (SHapley Additive exPlanations).
SHAP values explain how each feature contributes to individual predictions.

Key concepts:
- SHAP values show the contribution of each feature to pushing the prediction
  away from the base value (average prediction)
- Positive SHAP = pushes toward Red win
- Negative SHAP = pushes toward Blue win

Author: UFC Fight Predictor Team
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import os
import warnings

# SHAP for model explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Run: pip install shap")


# =============================================================================
# SHAP EXPLAINER CREATION
# =============================================================================

def create_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_type: str = 'tree'
) -> 'shap.Explainer':
    """
    Create a SHAP explainer for the given model.

    Different model types require different SHAP explainers:
    - Tree-based (XGBoost, LightGBM, RF): TreeExplainer (fast, exact)
    - Linear models: LinearExplainer
    - Any model: KernelExplainer (slow but universal)

    Args:
        model: Trained model.
        X_background: Background dataset for SHAP (typically training data sample).
        model_type: 'tree', 'linear', or 'kernel'.

    Returns:
        SHAP Explainer object.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    if model_type == 'tree':
        # Fast exact computation for tree-based models
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        # For logistic regression and linear models
        explainer = shap.LinearExplainer(model, X_background)
    else:
        # Universal but slow - use for any model
        # Sample background data for efficiency
        if len(X_background) > 100:
            X_sample = shap.sample(X_background, 100)
        else:
            X_sample = X_background
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)

    return explainer


def compute_shap_values(
    explainer: 'shap.Explainer',
    X: pd.DataFrame,
    check_additivity: bool = False
) -> np.ndarray:
    """
    Compute SHAP values for given samples.

    Args:
        explainer: SHAP explainer object.
        X: Samples to explain.
        check_additivity: Whether to verify SHAP values sum correctly.

    Returns:
        SHAP values array (n_samples, n_features) for binary classification.
    """
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # For binary classification, shap_values may be a list [class_0, class_1]
    # We want class 1 (Red wins) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


# =============================================================================
# GLOBAL EXPLANATIONS
# =============================================================================

def get_feature_importance_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get global feature importance based on mean absolute SHAP values.

    Unlike model-native feature importance, SHAP importance shows
    the actual impact on predictions, not just how often features are used.

    Args:
        shap_values: SHAP values array.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with feature importance rankings.
    """
    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'mean_shap': shap_values.mean(axis=0),  # Direction of effect
        'std_shap': shap_values.std(axis=0)
    })

    df = df.sort_values('importance', ascending=False)
    df['rank'] = range(1, len(df) + 1)

    return df.head(top_n)


def plot_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot global feature importance bar chart.

    Args:
        shap_values: SHAP values array.
        feature_names: List of feature names.
        top_n: Number of top features to show.
        save_path: Path to save the plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    importance_df = get_feature_importance_shap(shap_values, feature_names, top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by direction of effect
    colors = ['#E53935' if x > 0 else '#1E88E5' for x in importance_df['mean_shap']]

    bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                   color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()  # Top feature at top

    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E53935', label='Favors Red Win'),
        Patch(facecolor='#1E88E5', label='Favors Blue Win')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to: {save_path}")

    return fig


def plot_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 20,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Plot SHAP summary plot (beeswarm plot).

    This shows:
    - Feature importance (y-axis order)
    - Feature effect direction (x-axis position)
    - Feature value (color)

    Args:
        shap_values: SHAP values array.
        X: Feature values for coloring.
        top_n: Number of top features to show.
        save_path: Path to save the plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X, max_display=top_n, show=False)

    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {save_path}")

    fig = plt.gcf()
    return fig


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot SHAP dependence plot for a single feature.

    Shows how a feature's value affects the prediction,
    optionally colored by an interaction feature.

    Args:
        shap_values: SHAP values array.
        X: Feature values.
        feature: Feature to plot.
        interaction_feature: Feature to color by (optional).
        save_path: Path to save the plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    plt.figure(figsize=figsize)

    if interaction_feature:
        shap.dependence_plot(feature, shap_values, X,
                            interaction_index=interaction_feature, show=False)
    else:
        shap.dependence_plot(feature, shap_values, X, show=False)

    plt.title(f'SHAP Dependence: {feature}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dependence plot to: {save_path}")

    fig = plt.gcf()
    return fig


# =============================================================================
# INDIVIDUAL EXPLANATIONS
# =============================================================================

def explain_prediction(
    model: Any,
    explainer: 'shap.Explainer',
    X_single: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Explain a single prediction.

    Args:
        model: Trained model.
        explainer: SHAP explainer.
        X_single: Single sample to explain (1 row DataFrame).
        feature_names: List of feature names.
        top_n: Number of top contributing features to show.

    Returns:
        Dictionary with prediction explanation.
    """
    # Get prediction
    pred_prob = model.predict_proba(X_single)[0]
    pred_class = model.predict(X_single)[0]

    # Get SHAP values
    shap_values = compute_shap_values(explainer, X_single)
    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    # Get base value (expected value)
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]  # Class 1 base value
    else:
        base_value = 0.5

    # Create feature contribution DataFrame
    contributions = pd.DataFrame({
        'feature': feature_names,
        'value': X_single.values[0],
        'shap_value': shap_values,
        'abs_shap': np.abs(shap_values)
    })

    contributions = contributions.sort_values('abs_shap', ascending=False)
    top_features = contributions.head(top_n)

    # Categorize contributions
    pushing_red = top_features[top_features['shap_value'] > 0]
    pushing_blue = top_features[top_features['shap_value'] < 0]

    return {
        'prediction': 'Red Wins' if pred_class == 1 else 'Blue Wins',
        'probability_red': float(pred_prob[1]),
        'probability_blue': float(pred_prob[0]),
        'base_value': float(base_value),
        'top_features': top_features.to_dict('records'),
        'pushing_red': pushing_red[['feature', 'value', 'shap_value']].to_dict('records'),
        'pushing_blue': pushing_blue[['feature', 'value', 'shap_value']].to_dict('records'),
        'all_shap_values': dict(zip(feature_names, shap_values.tolist()))
    }


def print_prediction_explanation(
    explanation: Dict[str, Any],
    fighter_red: str = "Red Fighter",
    fighter_blue: str = "Blue Fighter"
) -> None:
    """
    Print a human-readable prediction explanation.

    Args:
        explanation: Output from explain_prediction().
        fighter_red: Name of red corner fighter.
        fighter_blue: Name of blue corner fighter.
    """
    print("\n" + "="*60)
    print("FIGHT PREDICTION EXPLANATION")
    print("="*60)

    print(f"\n{fighter_red} (Red) vs {fighter_blue} (Blue)")
    print("-"*40)

    print(f"\nPrediction: {explanation['prediction']}")
    print(f"Probability: {fighter_red} wins: {explanation['probability_red']:.1%}")
    print(f"            {fighter_blue} wins: {explanation['probability_blue']:.1%}")

    print(f"\n--- Factors Favoring {fighter_red} (Red) ---")
    for f in explanation['pushing_red'][:5]:
        print(f"  + {f['feature']}: {f['value']:.2f} (SHAP: +{f['shap_value']:.4f})")

    print(f"\n--- Factors Favoring {fighter_blue} (Blue) ---")
    for f in explanation['pushing_blue'][:5]:
        print(f"  - {f['feature']}: {f['value']:.2f} (SHAP: {f['shap_value']:.4f})")

    print("\n" + "="*60)


def plot_prediction_explanation(
    explanation: Dict[str, Any],
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot waterfall-style explanation for a single prediction.

    Args:
        explanation: Output from explain_prediction().
        save_path: Path to save the plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    top_features = explanation['top_features']

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by SHAP value for waterfall effect
    features = [f['feature'] for f in top_features]
    shap_vals = [f['shap_value'] for f in top_features]

    # Color by direction
    colors = ['#E53935' if v > 0 else '#1E88E5' for v in shap_vals]

    y_pos = range(len(features))
    ax.barh(y_pos, shap_vals, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)

    # Add prediction info
    pred = explanation['prediction']
    prob = explanation['probability_red'] if 'Red' in pred else explanation['probability_blue']
    ax.set_title(f"Prediction: {pred} ({prob:.1%} confidence)", fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E53935', label='Pushes toward Red Win'),
        Patch(facecolor='#1E88E5', label='Pushes toward Blue Win')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction explanation to: {save_path}")

    return fig


def plot_force(
    explainer: 'shap.Explainer',
    shap_values: np.ndarray,
    X_single: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Generate SHAP force plot for a single prediction.

    Force plots show how each feature pushes the prediction
    from the base value to the final prediction.

    Args:
        explainer: SHAP explainer.
        shap_values: SHAP values for the sample.
        X_single: Single sample features.
        save_path: Path to save HTML plot.
    """
    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    force_plot = shap.force_plot(
        base_value,
        shap_values,
        X_single.iloc[0],
        matplotlib=False
    )

    if save_path:
        shap.save_html(save_path, force_plot)
        print(f"Saved force plot to: {save_path}")

    return force_plot


# =============================================================================
# EXPLANATION PIPELINE
# =============================================================================

def explain_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    model_type: str = 'tree',
    save_plots: bool = True,
    plots_dir: str = 'plots/'
) -> Dict[str, Any]:
    """
    Complete model explanation pipeline.

    Args:
        model: Trained model.
        X_train: Training data for background.
        X_test: Test data to explain.
        feature_names: List of feature names.
        model_type: 'tree', 'linear', or 'kernel'.
        save_plots: Whether to save plots.
        plots_dir: Directory for plots.

    Returns:
        Dictionary with all explanation results.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)

    # Create output directory
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    # Sample training data for efficiency
    n_background = min(500, len(X_train))
    X_background = X_train.sample(n=n_background, random_state=42)

    print(f"\nCreating SHAP explainer ({model_type})...")
    explainer = create_explainer(model, X_background, model_type)

    # Compute SHAP values for test set (sample if large)
    n_explain = min(500, len(X_test))
    X_explain = X_test.sample(n=n_explain, random_state=42) if len(X_test) > n_explain else X_test

    print(f"Computing SHAP values for {n_explain} samples...")
    shap_values = compute_shap_values(explainer, X_explain)

    # Global feature importance
    print("\nComputing global feature importance...")
    importance_df = get_feature_importance_shap(shap_values, feature_names, top_n=20)

    print("\nTop 10 Most Important Features:")
    print("-"*50)
    for _, row in importance_df.head(10).iterrows():
        direction = "Red" if row['mean_shap'] > 0 else "Blue"
        print(f"  {row['rank']:2d}. {row['feature']:<30} (favors {direction})")

    # Generate plots
    if save_plots:
        print("\nGenerating plots...")

        # Feature importance bar chart
        plot_feature_importance(
            shap_values, feature_names, top_n=20,
            save_path=os.path.join(plots_dir, 'shap_importance.png')
        )
        plt.close()

        # Summary plot
        plot_summary(
            shap_values, X_explain, top_n=20,
            save_path=os.path.join(plots_dir, 'shap_summary.png')
        )
        plt.close()

        # Dependence plots for top 3 features
        top_3_features = importance_df.head(3)['feature'].tolist()
        for feature in top_3_features:
            safe_name = feature.replace('/', '_').replace(' ', '_')
            plot_dependence(
                shap_values, X_explain, feature,
                save_path=os.path.join(plots_dir, f'shap_dependence_{safe_name}.png')
            )
            plt.close()

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_explained': X_explain,
        'feature_importance': importance_df,
        'feature_names': feature_names
    }


def explain_single_fight(
    model: Any,
    explainer: 'shap.Explainer',
    X_fight: pd.DataFrame,
    feature_names: List[str],
    red_fighter: str = "Red Fighter",
    blue_fighter: str = "Blue Fighter",
    save_plot: bool = False,
    plot_path: str = None
) -> Dict[str, Any]:
    """
    Explain a single fight prediction.

    Args:
        model: Trained model.
        explainer: SHAP explainer.
        X_fight: Fight features (single row).
        feature_names: Feature names.
        red_fighter: Name of red corner fighter.
        blue_fighter: Name of blue corner fighter.
        save_plot: Whether to save explanation plot.
        plot_path: Path for the plot.

    Returns:
        Fight explanation dictionary.
    """
    explanation = explain_prediction(model, explainer, X_fight, feature_names)
    print_prediction_explanation(explanation, red_fighter, blue_fighter)

    if save_plot and plot_path:
        plot_prediction_explanation(explanation, save_path=plot_path)
        plt.close()

    return explanation


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    print("SHAP Explainability module ready.")
    print("\nAvailable functions:")
    print("  - explain_model(model, X_train, X_test, feature_names)")
    print("  - explain_single_fight(model, explainer, X_fight, feature_names)")
    print("  - plot_summary(shap_values, X)")
    print("  - plot_feature_importance(shap_values, feature_names)")
    print("  - print_prediction_explanation(explanation)")
