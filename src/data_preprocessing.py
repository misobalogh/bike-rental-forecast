"""
Data preprocessing pipeline for bike rental forecasting.
Handles data loading, cleaning, encoding, and outlier removal.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from utils import RANDOM_SEED, count_rows_with_missing_values, analyze_outliers


# Data validation rules
RULES_FLOAT = {
    'temperature': (-40, 40),
    'humidity': (0, 100),
    'windspeed': (0, 110),
}

RULES_INT = {
    'month': (1, 12),
    'hour': (0, 23),
    'holiday': (0, 1),
    'weekday': (0, 6),
    'workingday': (0, 1),
    'count': (0, None),
}


def load_data(file_path):
    """
    Load the bike rental dataset.

    Args:
        file_path: Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path, delimiter=';')
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def remove_unnecessary_columns(df):
    """
    Remove columns that don't provide useful information for modeling.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with columns removed
    """
    columns_to_drop = ['instant', 'date']
    df_clean = df.drop(columns=columns_to_drop)
    print(f"Removed columns: {columns_to_drop}")
    return df_clean


def handle_missing_values(df):
    """
    Handle missing values by removing rows with NaN.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    missing_rows, missing_percentage = count_rows_with_missing_values(df)
    print(f"Rows with missing values: {missing_rows} ({missing_percentage:.2f}%)")

    df_clean = df.dropna()
    print(f"After removing missing values: {df_clean.shape[0]} rows")
    return df_clean


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    initial_rows = df.shape[0]
    df_clean = df.drop_duplicates()
    removed = initial_rows - df_clean.shape[0]
    print(f"Removed {removed} duplicate rows")
    return df_clean


def validate_data_rules(df):
    """
    Validate data against defined rules and remove invalid rows.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with invalid rows removed
    """
    len_df = df.shape[0]
    df_clean = df.copy()

    # Validate float columns
    for column, (min_val, max_val) in RULES_FLOAT.items():
        df_clean[column] = df_clean[column].astype(np.float64)
        if max_val is not None:
            df_clean = df_clean[(df_clean[column] >= min_val) & (df_clean[column] <= max_val)]
        else:
            df_clean = df_clean[df_clean[column] >= min_val]

    count_float_violations = len_df - df_clean.shape[0]
    print(f"Rows removed due to float rules violations: {count_float_violations}")

    # Validate integer columns
    len_df = df_clean.shape[0]
    for column, (min_val, max_val) in RULES_INT.items():
        df_clean[column] = df_clean[column].astype(np.int64)
        if max_val is not None:
            df_clean = df_clean[(df_clean[column] >= min_val) & (df_clean[column] <= max_val)]
        else:
            df_clean = df_clean[df_clean[column] >= min_val]

    count_int_violations = len_df - df_clean.shape[0]
    print(f"Rows removed due to int rules violations: {count_int_violations}")

    return df_clean


def encode_categorical_variables(df):
    """
    Encode categorical variables (weather) into numeric format.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()

    weather_mapping = {
        'clear': 0,
        'cloudy': 1,
        'light rain/snow': 2,
        'heavy rain/snow': 3
    }
    df_encoded['weather'] = df_encoded['weather'].map(weather_mapping)
    print("Encoded categorical variable: weather")

    return df_encoded


def remove_outliers(df, columns_to_check):
    """
    Remove outliers using IQR method.

    Args:
        df: Input DataFrame
        columns_to_check: List of column names to check for outliers

    Returns:
        tuple: (DataFrame without outliers, outlier analysis dict)
    """
    outlier_analysis = {}

    for col in columns_to_check:
        if col in df.columns:
            analysis = analyze_outliers(df, col)
            outlier_analysis[col] = analysis
            print(f"\n{col.upper()}: {analysis['outlier_count']} outliers ({analysis['outlier_percentage']:.2f}%)")

    # Remove outliers
    df_len = df.shape[0]
    df_clean = df.copy()

    for col, analysis in outlier_analysis.items():
        lower_bound = analysis['lower_bound']
        upper_bound = analysis['upper_bound']
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    print(f"\nTotal rows removed due to outliers: {df_len - df_clean.shape[0]}")
    print(f"Final dataset shape: {df_clean.shape}")

    return df_clean, outlier_analysis


def prepare_train_test_split(df, test_size=0.2):
    """
    Split data into train and test sets and scale features.

    Args:
        df: Input DataFrame (should not contain target column)
        test_size: Proportion of test set

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # Separate features and target
    X = df.drop(columns=['count'])
    y = df['count']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    print("\nTrain-test split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def preprocess_pipeline(file_path, output_dir='data'):
    """
    Complete preprocessing pipeline.

    Args:
        file_path: Path to raw data CSV
        output_dir: Directory to save processed data

    Returns:
        tuple: Processed data and metadata
    """
    print("=" * 60)
    print("BIKE RENTAL DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load data
    print("\n[1/8] Loading data...")
    df = load_data(file_path)

    # Remove unnecessary columns
    print("\n[2/8] Removing unnecessary columns...")
    df = remove_unnecessary_columns(df)

    # Handle missing values
    print("\n[3/8] Handling missing values...")
    df = handle_missing_values(df)

    # Remove duplicates
    print("\n[4/8] Removing duplicates...")
    df = remove_duplicates(df)

    # Validate data rules
    print("\n[5/8] Validating data rules...")
    df = validate_data_rules(df)

    # Encode categorical variables
    print("\n[6/8] Encoding categorical variables...")
    df = encode_categorical_variables(df)

    # Remove outliers
    print("\n[7/8] Removing outliers...")
    cols_for_outlier_check = ['temperature', 'humidity', 'windspeed']
    df_clean, outlier_analysis = remove_outliers(df, cols_for_outlier_check)

    # Train-test split and scaling
    print("\n[8/8] Preparing train-test split and scaling...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = prepare_train_test_split(df_clean)

    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save as pickle for easy loading
    data_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'df_clean': df_clean
    }

    with open(output_path / 'processed_data.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"Processed data saved to: {output_path / 'processed_data.pkl'}")
    print(f"{'=' * 60}")

    return data_dict


if __name__ == "__main__":
    # Run preprocessing pipeline
    data_path = "../z2_data_1y.csv"
    preprocess_pipeline(data_path, output_dir="../data")
