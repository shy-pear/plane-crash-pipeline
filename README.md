# MSIS 522 Homework 1: End-to-End Data Science Workflow

This project uses the historical `Plane Crashes.csv` dataset to predict whether a crash had at least one survivor. The repository includes descriptive analytics, predictive modeling, SHAP explainability, pre-trained saved models, and a Streamlit app that presents the full workflow in four tabs.

## Project structure

- `src/train_pipeline.py`: end-to-end training script that engineers features, creates plots, tunes models, runs SHAP, and saves artifacts.
- `app.py`: Streamlit app with the required tabs:
  - Executive Summary
  - Descriptive Analytics
  - Model Performance
  - Explainability & Interactive Prediction
- `artifacts/models/`: saved models for Logistic Regression, Decision Tree, Random Forest, LightGBM, and the MLP section.
- `artifacts/plots/`: saved EDA plots, ROC curves, model comparison plot, decision tree visual, and SHAP plots.
- `artifacts/data/`: model comparison table and MLP training history.
- `artifacts/reports/`: metadata and text payloads used by the app.

## Modeling approach

- Target: `Survivors` mapped to a binary label:
  - `1` = at least one survivor
  - `0` = no survivors
- Leakage control:
  - direct fatality counts were excluded from the feature set
  - the model uses crash context, occupancy, aircraft age, and grouped categorical features instead
- Models included:
  - Logistic Regression baseline
  - Decision Tree with 5-fold `GridSearchCV`
  - Random Forest with 5-fold `GridSearchCV`
  - LightGBM with 5-fold `GridSearchCV`
  - MLP neural network section

## Current results

On the held-out test set, the best model is LightGBM with approximately:

- F1: `0.757`
- ROC-AUC: `0.813`

The best tree-based model is also LightGBM, which is the model used for SHAP analysis.

## How to run locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Regenerate all artifacts

```bash
python3 src/train_pipeline.py
```

This will recreate:

- all plots in `artifacts/plots/`
- all saved models in `artifacts/models/`
- the comparison table and history files in `artifacts/data/`
- the report metadata used by the app in `artifacts/reports/`

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

## Important note about the neural-network requirement

The current environment did not have TensorFlow, Keras, or PyTorch installed. To keep the project runnable, the MLP section currently uses a fallback implementation based on `sklearn.neural_network.MLPClassifier`, and the app clearly notes that in the model-performance tab.

If you want the neural-network section to match the homework wording as strictly as possible, install one of these before rerunning `src/train_pipeline.py`:

```bash
pip install tensorflow-cpu
```

or

```bash
pip install torch
```

Then update the MLP training block in `src/train_pipeline.py` to use that framework instead of the fallback.

## What you still need to do on your end

### 1. Put this into a Git repository and push it to GitHub

If this folder is not already a repo:

```bash
git init
git add .
git commit -m "Complete MSIS 522 homework 1 project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy the Streamlit app

The simplest path is Streamlit Community Cloud:

1. Push this project to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repo and branch.
5. Set the main file path to `app.py`.
6. Deploy the app.

After deployment, open the public URL in an incognito window to confirm it works for people who are not on your machine.

## Submission checklist

Before submitting on Canvas, make sure you have:

- a GitHub repository link
- the deployed Streamlit app link
- saved model files included in the repo
- `requirements.txt`
- `README.md`
- the notebook or Python scripts with the full analysis

## Suggested final polish before submission

- Review the executive summary text and make sure it sounds like your own voice.
- Check every plot caption in the app and adjust wording if you want it to sound more personal.
- If your instructor is strict about the neural-network framework requirement, install TensorFlow or PyTorch and swap out the fallback MLP implementation before you submit.
