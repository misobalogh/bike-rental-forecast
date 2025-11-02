"""
Utility functions for bike rental forecasting project.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Random seed for reproducibility
RANDOM_SEED = 42


def count_rows_with_missing_values(df):
    """
    Count rows with missing values in a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        tuple: (number of rows with missing values, percentage)
    """
    missing_value_rows = df.isnull().any(axis=1).sum()
    percentage_missing = (missing_value_rows / len(df)) * 100
    return missing_value_rows, percentage_missing


def count_missing_percentage_in_columns(df):
    """
    Calculate percentage of missing values per column.

    Args:
        df: pandas DataFrame

    Returns:
        pd.Series: Missing value percentage for each column
    """
    missing_percentage = df.isnull().mean() * 100
    return missing_percentage.round(2)


def get_IQR_bounds(df, column):
    """
    Calculate IQR bounds for outlier detection.

    Args:
        df: pandas DataFrame
        column: column name

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound


def analyze_outliers(df, column):
    """
    Analyze outliers in a specific column.

    Args:
        df: pandas DataFrame
        column: column name

    Returns:
        dict: Dictionary with outlier statistics
    """
    lower_bound, upper_bound = get_IQR_bounds(df, column)

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return {
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'min_value': df[column].min(),
        'max_value': df[column].max(),
        'mean': df[column].mean(),
        'median': df[column].median()
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a trained model on train and test sets.

    Args:
        model: trained model
        X_train: training features
        X_test: test features
        y_train: training target
        y_test: test target

    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals
    }
