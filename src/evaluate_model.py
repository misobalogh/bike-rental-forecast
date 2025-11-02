"""
Model evaluation and visualization pipeline for bike rental forecasting.
Generates comprehensive metrics and plots for model performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path

from utils import evaluate_model


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_model_comparison(results, save_path=None):
    """
    Create comparison plot of all models' R2 scores.

    Args:
        results: Dictionary of evaluation results
        save_path: Path to save the plot
    """
    model_names = list(results.keys())
    train_r2 = [results[name]['train_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='coral')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_residuals(results, model_name, save_path=None):
    """
    Create residual plots for a specific model.

    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model to plot
        save_path: Path to save the plot
    """
    res = results[model_name]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Residuals vs predicted
    axes[0].scatter(res['test_pred'], res['test_residuals'], alpha=0.4, s=15, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0].set_title(f'{model_name} - Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1].hist(res['test_residuals'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title(f'{model_name} - Residuals Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_predictions_vs_actual(results, model_name, y_test, save_path=None):
    """
    Create scatter plot of predictions vs actual values.

    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model to plot
        y_test: Actual test values
        save_path: Path to save the plot
    """
    res = results[model_name]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, res['test_pred'], alpha=0.4, s=15, color='darkgreen')

    min_val = min(y_test.min(), res['test_pred'].min())
    max_val = max(y_test.max(), res['test_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add R² score as text
    r2 = res['test_r2']
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_feature_importance(feature_names, feature_importances, top_n=10, save_path=None):
    """
    Plot feature importances from Random Forest model.

    Args:
        feature_names: List of feature names
        feature_importances: Array of feature importances
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", top_n)

    plt.bar(range(top_n), feature_importances[indices], color=colors, alpha=0.8)
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_error_distribution(results, model_name, y_test, save_path=None):
    """
    Plot percentage error distribution.

    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model to plot
        y_test: Actual test values
        save_path: Path to save the plot
    """
    res = results[model_name]
    test_errors = np.abs(res['test_residuals'])
    test_pct_errors = (test_errors / y_test) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(test_pct_errors, bins=50, edgecolor='black', alpha=0.7, color='indianred')
    plt.axvline(x=np.median(test_pct_errors), color='blue', linestyle='--',
                linewidth=2, label=f'Median: {np.median(test_pct_errors):.1f}%')
    plt.axvline(x=np.mean(test_pct_errors), color='green', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(test_pct_errors):.1f}%')

    plt.xlabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Percentage Error Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f'Error < 10%: {(test_pct_errors < 10).sum() / len(test_pct_errors) * 100:.1f}% samples\n'
    stats_text += f'Error < 20%: {(test_pct_errors < 20).sum() / len(test_pct_errors) * 100:.1f}% samples'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def create_metrics_table(results):
    """
    Create a formatted metrics table for all models.

    Args:
        results: Dictionary of evaluation results

    Returns:
        pd.DataFrame: Metrics table
    """
    metrics_data = []

    for model_name, res in results.items():
        metrics_data.append({
            'Model': model_name,
            'Train R²': f"{res['train_r2']:.4f}",
            'Test R²': f"{res['test_r2']:.4f}",
            'Train RMSE': f"{res['train_rmse']:.2f}",
            'Test RMSE': f"{res['test_rmse']:.2f}",
            'Train MAE': f"{res['train_mae']:.2f}",
            'Test MAE': f"{res['test_mae']:.2f}"
        })

    df = pd.DataFrame(metrics_data)
    return df


def evaluate_pipeline(data_path='../data/processed_data.pkl',
                      models_dir='../models',
                      results_dir='../results'):
    """
    Complete evaluation pipeline.

    Args:
        data_path: Path to processed data
        models_dir: Directory containing trained models
        results_dir: Directory to save evaluation results
    """
    print("=" * 60)
    print("BIKE RENTAL MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    figures_path = results_path / 'figures'
    figures_path.mkdir(exist_ok=True)

    # Load processed data
    print("\nLoading processed data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # Load training results
    print("Loading training results...")
    with open(Path(models_dir) / 'training_results.json', 'r') as f:
        training_results = json.load(f)

    best_model_name = training_results['best_model']

    # Load models
    print("Loading trained models...")
    models = {}
    for model_name in ['Decision Tree', 'Random Forest', 'SVM', 'Gradient Boosting']:
        model_filename = model_name.lower().replace(' ', '_') + '.pkl'
        with open(Path(models_dir) / model_filename, 'rb') as f:
            models[model_name] = pickle.load(f)

    # Evaluate all models
    print("\nEvaluating models...")
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Create metrics table
    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    metrics_df = create_metrics_table(results)
    print(metrics_df.to_string(index=False))

    # Save metrics table
    metrics_df.to_csv(results_path / 'metrics_table.csv', index=False)
    print(f"\nMetrics saved to: {results_path / 'metrics_table.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Model comparison
    print("  [1/5] Model comparison plot...")
    plot_model_comparison(results, save_path=figures_path / '01_model_comparison.png')

    # 2. Residuals for best model
    print(f"  [2/5] Residuals plot for {best_model_name}...")
    plot_residuals(results, best_model_name, save_path=figures_path / '02_residuals.png')

    # 3. Predictions vs Actual for best model
    print(f"  [3/5] Predictions vs Actual for {best_model_name}...")
    plot_predictions_vs_actual(results, best_model_name, y_test,
                               save_path=figures_path / '03_predictions_vs_actual.png')

    # 4. Feature importance (Random Forest)
    print("  [4/5] Feature importance plot...")
    with open(Path(models_dir) / 'feature_importances.json', 'r') as f:
        feature_imp_dict = json.load(f)

    feature_importances = np.array([feature_imp_dict[name] for name in feature_names])
    plot_feature_importance(feature_names, feature_importances, top_n=10,
                           save_path=figures_path / '04_feature_importance.png')

    # 5. Error distribution for best model
    print(f"  [5/5] Error distribution for {best_model_name}...")
    plot_error_distribution(results, best_model_name, y_test,
                           save_path=figures_path / '05_error_distribution.png')

    # Calculate and display error statistics for best model
    print("\n" + "=" * 60)
    print(f"ERROR STATISTICS FOR {best_model_name.upper()}")
    print("=" * 60)

    res = results[best_model_name]
    test_errors = np.abs(res['test_residuals'])
    test_pct_errors = (test_errors / y_test) * 100

    print(f"Median Error: {np.median(test_pct_errors):.2f}%")
    print(f"Mean Error: {np.mean(test_pct_errors):.2f}%")
    print(f"Samples with Error < 10%: {(test_pct_errors < 10).sum() / len(test_pct_errors) * 100:.1f}%")
    print(f"Samples with Error < 20%: {(test_pct_errors < 20).sum() / len(test_pct_errors) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All results saved to: {results_path}")
    print(f"Figures saved to: {figures_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_pipeline()
