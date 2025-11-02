# Bike Rental Demand Forecasting

A machine learning project that predicts hourly bike rental demand using regression models and comprehensive data analysis.

## Project Overview

This project demonstrates a complete ML workflow for forecasting bike rental demand based on temporal and weather features. Multiple regression models are trained and compared, with the best model achieving **R² 0.938** on the test set.

For detailed analysis, visualizations, and explanations, refer to the [project report](docs/report_en.md) and the Jupyter notebook `bike_rental.ipynb`.

## Objective

Predict the number of bike rentals per hour based on:
- **Temporal features**: Month, hour, day of week, working day status
- **Weather conditions**: Temperature, humidity, wind speed, weather type
- **Special days**: Holidays

## Dataset

Slightly modified UCI Bike Sharing Dataset.

- **Source**: 1-year hourly bike rental data (2012)
- **Size**: ~8,525 hourly records after preprocessing
- **Target Variable**: `count` (number of bike rentals)
- **Features**: 9 input features after preprocessing

### Data Preprocessing

- Removed unnecessary columns (instant, date)
- Handled missing values (<1% of data)
- Applied business rule validation for temperature, humidity, windspeed ranges
- Encoded categorical weather variable (clear, cloudy, rain/snow)
- Removed outliers using IQR method
- Train-test split: 80/20 with stratification
- Feature scaling using StandardScaler

## Models & Performance

| Model | Test R² | Test RMSE |
|-|-|-|
| **Neural Network** | **0.938** | **53.87** |
| **Random Forest** | **0.925** | **57.00** |
| Decision Tree | 0.889 | 69.17 |
| SVM (RBF) | 0.554 | 138.77 |

**Best Model**: Neural Network with Adam optimizer and early stopping

### Key Findings

- **Most important features**: Hour (70%), temperature (13%), and workingday (6%) account for >89% of prediction importance
- **Peak demand hours**: 8 AM and 5-6 PM (commute times) on workdays
- **Seasonal patterns**: Higher rentals in warm months (April-September), highest in September
- **Weather impact**: Clear weather drives higher demand vs unfavorable weather
- **Model performance**: Best model explains 93.8% of variability in bike rental demand


## Project Structure

```
bike-rental-forecast/
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and feature engineering
│   ├── train_model.py            # Model training with cross-validation
│   ├── evaluate_model.py         # Model evaluation and visualization
│   └── utils.py                  # Utility functions│
│
├── docs/                      # Documentation and results
│   ├── report_en.md
│   └── figures/                  # Generated plots
│
├── trees.ipynb                   # Main analysis notebook
├── z2_data_1y.csv               # Raw dataset
├── main.py                      # Pipeline runner script
├── pyproject.toml               # Project dependencies
└── README.md

```

## Usage

### Option 1: Interactive Notebook

Open `bike_rental.ipynb` in Jupyter for step-by-step exploration with detailed markdown explanations.

### Option 2: Run Complete Pipeline

```bash
python main.py
```

This executes the entire workflow:
1. Data preprocessing
2. Model training with cross-validation
3. Model evaluation with visualizations

### Option 3: Run Individual Steps

```bash
# Data preprocessing
cd src
python data_preprocessing.py

# Model training
python train_model.py

# Model evaluation
python evaluate_model.py
```

## Model Insights

### Feature Importance (Random Forest)
1. **Hour** (70%): Most predictive - captures commute patterns
2. **Temperature** (13%): Strong positive correlation with rentals
3. **Working day** (6%): Differentiates weekday/weekend patterns
4. **Weather** (low): Clear weather drives higher demand

### Residual Analysis
- Residuals are approximately normally distributed
- No systematic bias in predictions
- Slightly higher errors for extreme values (very low/high demand)
