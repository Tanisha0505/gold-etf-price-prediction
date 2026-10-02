# Gold ETF Price Prediction

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
├── app.py                  # Streamlit prediction app
├── data/
│   ├── raw/                 # Original Kaggle data
│   └── processed/           # Cleaned modeling datasets
├── notebooks/               # EDA, model training, and evaluation
│   ├── 01_data_loading_and_eda.ipynb
│   └── 02_model_training_and_model_evaluation.ipynb
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To run the notebooks, open them in Jupyter Notebook, JupyterLab, or VS Code after installing the requirements.

## Run the Streamlit App

After running the notebooks or placing the processed files in `data/processed/`, start the app with:

```bash
streamlit run app.py
```

The app loads `data/processed/gold_eda_data.csv` and `data/processed/selected_features.csv`, trains the final models, and allows next-day Gold ETF adjusted close prediction using the selected features.

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
