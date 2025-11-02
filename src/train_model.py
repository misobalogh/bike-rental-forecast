"""
Model training pipeline with cross-validation for bike rental forecasting.
Trains multiple models and saves the best performing one.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import json

from utils import RANDOM_SEED, evaluate_model


def train_decision_tree(X_train, y_train, max_depth_range=(1, 100)):
    """
    Train Decision Tree with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training target
        max_depth_range: Range of max_depth values to try

    Returns:
        tuple: (trained model, best max_depth)
    """
    print("\n[1/4] Training Decision Tree...")

    best = {
        "r2": -np.inf,
        "max_depth": None,
    }

    # Simple grid search for max_depth
    for depth in range(max_depth_range[0], max_depth_range[1]):
        model = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_SEED)

        # Use cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='r2', n_jobs=-1)
        mean_r2 = cv_scores.mean()

        if mean_r2 > best["r2"]:
            best["r2"] = mean_r2
            best["max_depth"] = depth

    print(f"  Best max_depth: {best['max_depth']} (CV R2: {best['r2']:.4f})")

    # Train final model with best parameters
    final_model = DecisionTreeRegressor(max_depth=best["max_depth"], random_state=RANDOM_SEED)
    final_model.fit(X_train, y_train)

    return final_model, best["max_depth"]


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees

    Returns:
        tuple: (trained model, feature importances)
    """
    print("\n[2/4] Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    print(f"  CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    return model, feature_importances


def train_svm(X_train, y_train, C=100):
    """
    Train Support Vector Machine model.

    Args:
        X_train: Training features
        y_train: Training target
        C: Regularization parameter

    Returns:
        trained model
    """
    print("\n[3/4] Training SVM...")

    model = SVR(kernel='rbf', C=C)

    # Cross-validation (using smaller CV for SVM as it's slower)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3,
                                scoring='r2', n_jobs=-1)
    print(f"  CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)

    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1):
    """
    Train Gradient Boosting model.

    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting stages
        learning_rate: Learning rate

    Returns:
        trained model
    """
    print("\n[4/4] Training Gradient Boosting...")

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    print(f"  CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)

    return model


def compare_models(models_dict, X_train, X_test, y_train, y_test):
    """
    Compare all trained models and select the best one.

    Args:
        models_dict: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        tuple: (results dictionary, best model name)
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = {}
    best_model_name = None
    best_test_r2 = -np.inf

    for name, model in models_dict.items():
        eval_results = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = eval_results

        print(f"\n{name}:")
        print(f"  Train R2: {eval_results['train_r2']:.4f} | RMSE: {eval_results['train_rmse']:.2f}")
        print(f"  Test R2:  {eval_results['test_r2']:.4f} | RMSE: {eval_results['test_rmse']:.2f}")

        if eval_results['test_r2'] > best_test_r2:
            best_test_r2 = eval_results['test_r2']
            best_model_name = name

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name} (Test R2: {best_test_r2:.4f})")
    print("=" * 60)

    return results, best_model_name


def train_pipeline(data_path='../data/processed_data.pkl', models_dir='../models'):
    """
    Complete training pipeline.

    Args:
        data_path: Path to processed data
        models_dir: Directory to save trained models

    Returns:
        dict: Training results and models
    """
    print("=" * 60)
    print("BIKE RENTAL MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load processed data
    print("\nLoading processed data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Train models
    print("\nTraining models with cross-validation...")

    dt_model, best_depth = train_decision_tree(X_train, y_train)
    rf_model, feature_importances = train_random_forest(X_train, y_train, n_estimators=100)
    svm_model = train_svm(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=100)

    # Store all models
    models = {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'SVM': svm_model,
        'Gradient Boosting': gb_model
    }

    # Compare models
    results, best_model_name = compare_models(models, X_train, X_test, y_train, y_test)

    # Save models and results
    output_path = Path(models_dir)
    output_path.mkdir(exist_ok=True)

    # Save all models
    for name, model in models.items():
        model_filename = name.lower().replace(' ', '_') + '.pkl'
        with open(output_path / model_filename, 'wb') as f:
            pickle.dump(model, f)

    # Save best model separately
    with open(output_path / 'best_model.pkl', 'wb') as f:
        pickle.dump(models[best_model_name], f)

    # Save results as JSON
    results_json = {}
    for name, res in results.items():
        results_json[name] = {
            'train_r2': float(res['train_r2']),
            'train_rmse': float(res['train_rmse']),
            'train_mae': float(res['train_mae']),
            'test_r2': float(res['test_r2']),
            'test_rmse': float(res['test_rmse']),
            'test_mae': float(res['test_mae'])
        }

    results_json['best_model'] = best_model_name
    results_json['feature_names'] = feature_names
    results_json['dt_best_depth'] = int(best_depth)

    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save feature importances
    feature_imp_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importances)}
    feature_imp_sorted = dict(sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True))

    with open(output_path / 'feature_importances.json', 'w') as f:
        json.dump(feature_imp_sorted, f, indent=2)

    print(f"\nModels saved to: {output_path}")
    print(f"Training results saved to: {output_path / 'training_results.json'}")

    return {
        'models': models,
        'results': results,
        'best_model_name': best_model_name,
        'feature_importances': feature_importances,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    train_pipeline()
