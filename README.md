# MSIS 522 Homework 1: End-to-End Data Science Workflow

This project uses the historical `Plane Crashes.csv` dataset to predict whether a crash had at least one survivor. The repository includes descriptive analytics, predictive modeling, neural-network tuning, SHAP explainability, saved model artifacts, and a Streamlit app that presents the full workflow in four tabs.

Deployed app: [https://plane-crash-data.streamlit.app](https://plane-crash-data.streamlit.app)

## Project structure

- `src/train_pipeline.py`: builds the full pipeline, generates plots, tunes models, runs SHAP, and saves artifacts.
- `app.py`: Streamlit app with:
  - Executive Summary
  - Descriptive Analytics
  - Model Performance
  - Explainability & Interactive Prediction
- `artifacts/models/`: saved Logistic Regression, Decision Tree, Random Forest, LightGBM, and PyTorch MLP artifacts.
- `artifacts/plots/`: EDA plots, ROC curves, model comparison plot, decision tree visual, SHAP plots, MLP history, and bonus tuning visualization.
- `artifacts/data/`: model comparison table, MLP training history, and neural-network tuning results.
- `artifacts/reports/`: metadata and text payloads used by the app.

## Problem setup

- Target: `Survivors`
  - `1` = at least one survivor
  - `0` = no survivors
- Leakage control:
  - direct fatality counts were excluded from the feature set
  - the model uses occupancy, aircraft age, phase of flight, crash site, region, cause, and grouped categorical features instead
- Models included:
  - Logistic Regression baseline
  - Decision Tree with 5-fold `GridSearchCV`
  - Random Forest with 5-fold `GridSearchCV`
  - LightGBM with 5-fold `GridSearchCV`
  - PyTorch MLP neural network
  - Bonus: PyTorch MLP hyperparameter tuning over hidden sizes, learning rates, and dropout rates

## Current results

On the held-out test set:

- Best overall model: `MLP Neural Network`
  - F1: `0.765`
  - ROC-AUC: `0.785`
- Best tree-based model: `LightGBM`
  - F1: `0.757`
  - ROC-AUC: `0.813`

The app uses LightGBM for SHAP because it is the strongest tree-based model, while the tuned MLP is the strongest overall predictive model.

## How to run locally

If you only want to open the app with the saved artifacts already in the repo, you do not need to retrain first. Install dependencies and run Streamlit.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

### 4. Optional: regenerate all artifacts

Use unbuffered Python so training progress prints in real time:

```bash
python -u src/train_pipeline.py
```

This recreates:

- all plots in `artifacts/plots/`
- all saved models in `artifacts/models/`
- model comparison and neural-network tuning outputs in `artifacts/data/`
- report payloads in `artifacts/reports/`

## Streamlit app behavior

- The app loads saved artifacts and does not retrain models on startup.
- The `Model Performance` tab shows the model comparison table, ROC curves, hyperparameters, decision tree visual, and MLP training history.
- The sidebar contains the bonus neural-network tuning explorer:
  - sliders for hidden layer sizes, learning rate, and dropout
  - a heatmap of validation F1
  - a small table showing the best tuning configuration
- The `Explainability & Interactive Prediction` tab supports live prediction. If the user selects the MLP model for prediction, the app uses the saved PyTorch MLP artifact.

## Deployed Streamlit App in Community Cloud

Current deployed app URL:

- [https://plane-crash-data.streamlit.app](https://plane-crash-data.streamlit.app)

