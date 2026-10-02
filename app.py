from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "gold_eda_data.csv"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "FINAL_USO.csv"
SELECTED_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "selected_features.csv"
TARGET_COLUMN = "target_next_price"
DATE_COLUMN = "Date"


st.set_page_config(
    page_title="Gold ETF Price Prediction",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .app-header {
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        padding: 20px 22px;
        background: linear-gradient(135deg, #ffffff 0%, #eef6f7 100%);
        margin-bottom: 18px;
    }
    .app-header h1 {
        margin: 0;
        color: #102a43;
        font-size: 2rem;
        line-height: 1.2;
    }
    .app-header p {
        margin: 8px 0 0 0;
        color: #475467;
        font-size: 1rem;
    }
    .section-card {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        margin-bottom: 14px;
    }
    .section-card h3 {
        margin-top: 0;
        color: #102a43;
        font-size: 1.05rem;
    }
    .info-strip {
        border-left: 4px solid #2e7d7c;
        padding: 10px 12px;
        background: #f2faf9;
        color: #344054;
        border-radius: 4px;
        margin: 10px 0 14px 0;
    }
    .small-muted {
        color: #667085;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_selected_features() -> List[str]:
    if not SELECTED_FEATURES_PATH.exists():
        st.error(f"Missing selected features file: {SELECTED_FEATURES_PATH}")
        st.stop()

    features_df = pd.read_csv(SELECTED_FEATURES_PATH)
    if "feature" not in features_df.columns:
        st.error("selected_features.csv must contain a column named 'feature'.")
        st.stop()

    return features_df["feature"].dropna().tolist()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    elif RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        df[TARGET_COLUMN] = df["Adj Close"].shift(-1)
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    else:
        st.error(
            "Dataset not found. Add gold_eda_data.csv to data/processed/ "
            "or FINAL_USO.csv to data/raw/."
        )
        st.stop()

    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    return df


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)

    return {
        "MAE": mae,
        "MAE %": (mae / mean_actual) * 100,
        "RMSE": rmse,
        "RMSE %": (rmse / mean_actual) * 100,
        "R2": r2_score(y_true, y_pred),
    }


@st.cache_resource
def train_models(df: pd.DataFrame, selected_features: Tuple[str, ...]):
    selected_features = list(selected_features)
    required_columns = selected_features + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error("The dataset is missing these required columns:")
        st.write(missing_columns)
        st.stop()

    modeling_df = df.dropna(subset=required_columns).reset_index(drop=True)
    X = modeling_df[selected_features]
    y = modeling_df[TARGET_COLUMN]

    train_end = int(len(modeling_df) * 0.70)
    val_end = int(len(modeling_df) * 0.85)

    X_train_val = X.iloc[:val_end]
    y_train_val = y.iloc[:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    models = {
        "PCA + Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=11)),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    predictions = {}
    for model_name, model in models.items():
        model.fit(X_train_val, y_train_val)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results.append({"Model": model_name, **metrics})
        predictions[model_name] = y_pred

    results_df = pd.DataFrame(results)

    return {
        "models": models,
        "results": results_df,
        "modeling_df": modeling_df,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "train_rows": train_end,
        "validation_rows": val_end - train_end,
        "test_rows": len(modeling_df) - val_end,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def metric_card(title: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-muted">{title}</div>
            <div style="font-size:1.55rem;font-weight:700;color:#102a43;margin-top:4px;">{value}</div>
            <div class="small-muted">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


df = load_dataset()
selected_features = load_selected_features()
training_artifacts = train_models(df, tuple(selected_features))
results_df = training_artifacts["results"]

with st.sidebar:
    st.header("Project Details")
    st.write("Gold ETF next-day adjusted close prediction using regression models.")
    st.divider()
    st.metric("Selected Features", len(selected_features))
    st.metric("Rows Used", f"{len(training_artifacts['modeling_df']):,}")
    st.metric("Final Models", 2)
    st.divider()
    st.write("Data files used:")
    st.code("data/processed/gold_eda_data.csv")
    st.code("data/processed/selected_features.csv")

st.markdown(
    """
    <div class="app-header">
        <h1>Gold ETF Price Prediction</h1>
        <p>Interactive machine learning dashboard for predicting the next-day adjusted closing price using selected financial indicators.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_1, summary_2, summary_3, summary_4 = st.columns(4)
with summary_1:
    metric_card("Rows Used", f"{len(training_artifacts['modeling_df']):,}", "after target creation")
with summary_2:
    metric_card("Selected Features", str(len(selected_features)), "correlation based")
with summary_3:
    metric_card(
        "Train + Validation",
        f"{training_artifacts['train_rows'] + training_artifacts['validation_rows']:,}",
        "used for final fitting",
    )
with summary_4:
    metric_card("Test Rows", f"{training_artifacts['test_rows']:,}", "final evaluation")

st.divider()

left_panel, right_panel = st.columns([0.9, 1.1])

with left_panel:
    st.subheader("Live Prediction")
    st.markdown(
        '<div class="info-strip">Choose a model, edit the feature values, and the prediction updates automatically.</div>',
        unsafe_allow_html=True,
    )
    model_name = st.radio(
        "Model",
        results_df["Model"].tolist(),
        index=0,
        horizontal=True,
        help="Choose one of the final models tested in the project.",
    )

    reference_options = {
        "Latest available row": -1,
        "First test row": 0,
    }
    reference_label = st.selectbox(
        "Start input values from",
        list(reference_options.keys()),
        help="The fields below are pre-filled from real dataset rows.",
    )

    modeling_df = training_artifacts["modeling_df"]
    if reference_label == "Latest available row":
        base_row = modeling_df.iloc[reference_options[reference_label]]
    else:
        base_row = training_artifacts["X_test"].iloc[reference_options[reference_label]]

    default_input_df = pd.DataFrame(
        {
            "Feature": selected_features,
            "Value": [float(base_row[feature]) for feature in selected_features],
        }
    )

    edited_input_df = st.data_editor(
        default_input_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Feature"],
        column_config={
            "Feature": st.column_config.TextColumn("Selected Feature"),
            "Value": st.column_config.NumberColumn(
                "User Entered Value",
                format="%.6f",
            ),
        },
        key=f"live_input_{reference_label}_{model_name}",
    )

    input_values = dict(
        zip(edited_input_df["Feature"], edited_input_df["Value"])
    )
    input_df = pd.DataFrame([[input_values[feature] for feature in selected_features]], columns=selected_features)
    prediction = training_artifacts["models"][model_name].predict(input_df)[0]

    st.metric("Predicted Next-Day Adjusted Close", f"{prediction:.4f}")
    st.markdown(
        "<p class='small-muted'>The output is a price prediction in the same unit as the dataset's adjusted close value.</p>",
        unsafe_allow_html=True,
    )

with right_panel:
    st.subheader("Final Test Results")
    display_results = results_df.copy()
    for column in ["MAE", "MAE %", "RMSE", "RMSE %", "R2"]:
        display_results[column] = display_results[column].map(format_metric)
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    best_model = results_df.sort_values(["RMSE", "MAE"], ascending=True).iloc[0]
    st.success(
        f"Best test model: {best_model['Model']} "
        f"(RMSE: {best_model['RMSE']:.4f}, R2: {best_model['R2']:.4f})"
    )

st.divider()

tab_overview, tab_features, tab_predictions = st.tabs(
    ["Overview", "Selected Features", "Prediction Check"]
)

with tab_overview:
    st.subheader("Workflow Summary")
    st.write(
        "The project predicts the next-day Gold ETF adjusted closing price. "
        "The dataset was cleaned, a next-day regression target was created, "
        "56 strongly correlated features were selected, and final models were "
        "trained using a chronological train-validation-test split."
    )
    st.write(
        "The app retrains the final models from the processed project data instead of "
        "loading a saved model file, matching the notebook decision not to store model artifacts."
    )

with tab_features:
    st.subheader("56 Features Used for Model Training")
    feature_table = pd.DataFrame(
        {
            "Selected Feature": selected_features,
            "Correlation Selection Rule": ["|corr with target_next_price| >= 0.5"] * len(selected_features),
        }
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

with tab_predictions:
    st.subheader("Actual vs Predicted on Test Set")
    test_display = pd.DataFrame(
        {
            "Actual": training_artifacts["y_test"].values,
            "Predicted": training_artifacts["predictions"][model_name],
        }
    )
    test_display["Error"] = test_display["Actual"] - test_display["Predicted"]
    st.line_chart(test_display[["Actual", "Predicted"]])
    st.dataframe(test_display.head(20), use_container_width=True)
