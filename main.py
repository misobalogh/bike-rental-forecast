"""
Main pipeline script for bike rental forecasting.
Runs the complete ML workflow: preprocessing, training, and evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import preprocess_pipeline
from train_model import train_pipeline
from evaluate_model import evaluate_pipeline


def run_complete_pipeline():
    """
    Run the complete ML pipeline from data preprocessing to evaluation.
    """
    print("BIKE RENTAL FORECASTING PIPELINE")

    try:
        # Step 1: Data Preprocessing
        print("STEP 1: DATA PREPROCESSING")
        preprocess_pipeline('z2_data_1y.csv', output_dir='data')

        # Step 2: Model Training
        print("STEP 2: MODEL TRAINING")
        train_pipeline(data_path='data/processed_data.pkl', models_dir='models')

        # Step 3: Model Evaluation
        print("STEP 3: MODEL EVALUATION")
        evaluate_pipeline(data_path='data/processed_data.pkl',
                         models_dir='models',
                         results_dir='results')

        print("\nResults:")
        print("  - Processed data: data/processed_data.pkl")
        print("  - Trained models: models/")
        print("  - Evaluation metrics: results/metrics_table.csv")
        print("  - Figures: results/figures/")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    run_complete_pipeline()
