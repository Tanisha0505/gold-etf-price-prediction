# Gold Price Prediction

Machine learning project for predicting gold ETF price movement using the Kaggle **Gold ETF Price Prediction Dataset**.

## Dataset

Kaggle dataset:
https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset

The main file is `FINAL_USO.csv`, which contains gold ETF prices plus market, currency, oil, bond, and precious-metal indicators.

Place the downloaded CSV here:

```text
data/raw/FINAL_USO.csv
```

## Project Structure

```text
gold-price-prediction/
├── data/
│   ├── raw/                 # Original Kaggle data
│   └── processed/           # Cleaned modeling datasets
├── models/                  # Optional saved model artifacts
├── notebooks/               # EDA, model training, and evaluation
│   ├── 01_data_loading_and_eda.ipynb
│   └── 02_model_training_and_model_evaluation.ipynb
├── reports/
│   └── figures/             # Charts and evaluation plots
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Goal

Build regression models that predict the next day's gold ETF adjusted closing price:

```text
target_next_price = next day's adjusted closing price
```

The project compares Multiple Linear Regression, PCA + Linear Regression, and Random Forest Regressor.

## Notebook Workflow

After placing `FINAL_USO.csv` in `data/raw/`, start with:

```text
notebooks/01_data_loading_and_eda.ipynb
notebooks/02_model_training_and_model_evaluation.ipynb
```

The first notebook handles data loading, EDA, target creation, feature selection, and processed data export. The second notebook handles model training, tuning, final test evaluation, and model comparison.
