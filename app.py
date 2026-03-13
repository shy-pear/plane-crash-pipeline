from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ARTIFACTS / "reports"
MODELS = ARTIFACTS / "models"
PLOTS = ARTIFACTS / "plots"
DATA = ARTIFACTS / "data"


st.set_page_config(
    page_title="Plane Crash Survivability Workflow",
    layout="wide",
)


@st.cache_data
def load_report_payload() -> dict[str, Any]:
    return json.loads((REPORTS / "report_payload.json").read_text())


@st.cache_data
def load_metadata() -> dict[str, Any]:
    return json.loads((REPORTS / "metadata.json").read_text())


@st.cache_data
def load_input_config() -> dict[str, Any]:
    return json.loads((REPORTS / "input_config.json").read_text())


@st.cache_data
def load_model_table() -> pd.DataFrame:
    return pd.read_csv(DATA / "model_comparison.csv")


@st.cache_data
def load_mlp_tuning_table() -> pd.DataFrame | None:
    path = DATA / "mlp_tuning_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for model_path in sorted(MODELS.glob("*.joblib")):
        if model_path.name == "shap_support.joblib":
            continue
        payload = joblib.load(model_path)
        bundles[payload["model_name"]] = payload
    return bundles


@st.cache_resource
def get_tree_explainer(model_name: str) -> tuple[Any, list[str], Any]:
    bundle = load_model_bundles()[model_name]
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(estimator)
    return explainer, feature_names, pipeline


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x: Any) -> Any:
        return self.network(x)


def predict_with_torch_bundle(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[int, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required to use the saved MLP model in this app.")

    preprocessor = bundle["preprocessor"]
    transformed = np.asarray(preprocessor.transform(features), dtype=np.float32)
    architecture = bundle["architecture"]
    model = TorchMLP(
        input_dim=int(architecture["input_dim"]),
        hidden_sizes=tuple(architecture["hidden_sizes"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(transformed)).squeeze(1)
        probability = float(torch.sigmoid(logits).cpu().numpy()[0])
    prediction = int(probability >= 0.5)
    return prediction, probability


def predict_with_bundle(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[int, float]:
    if bundle["bundle_type"] == "pipeline":
        model = bundle["pipeline"]
        probability = float(model.predict_proba(features)[0, 1])
        prediction = int(model.predict(features)[0])
        return prediction, probability

    if bundle["bundle_type"] == "torch_model":
        return predict_with_torch_bundle(bundle, features)

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    transformed = preprocessor.transform(features)
    probability = float(model.predict_proba(transformed)[0, 1])
    prediction = int(model.predict(transformed)[0])
    return prediction, probability


def build_feature_row(config: dict[str, Any], overrides: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for feature, details in config.items():
        row[feature] = overrides.get(feature, details["default"])

    row["quarter"] = int(np.clip(np.ceil(float(row["month"]) / 3), 1, 4))
    row["total_on_board"] = float(row["crew_on_board"]) + float(row["pax_on_board"])
    if row["total_on_board"] > 0:
        row["crew_share"] = float(row["crew_on_board"]) / float(row["total_on_board"])
    else:
        row["crew_share"] = config["crew_share"]["default"]

    return pd.DataFrame([row])


def create_waterfall_figure(model_name: str, feature_row: pd.DataFrame) -> plt.Figure:
    explainer, feature_names, pipeline = get_tree_explainer(model_name)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_row)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])[0]
        base_value = float(np.asarray(explainer.expected_value)[1])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            values = shap_array[0, :, 1]
            base_value = float(np.asarray(explainer.expected_value)[1])
        else:
            values = shap_array[0]
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=np.asarray(transformed)[0],
        feature_names=feature_names,
    )

    plt.close("all")
    shap.plots.waterfall(explanation, show=False, max_display=12)
    fig = plt.gcf()
    return fig


def image(path: str, caption: str) -> None:
    st.image(str(ROOT / path), use_container_width=True)
    st.caption(caption)


payload = load_report_payload()
metadata = load_metadata()
input_config = load_input_config()
model_table = load_model_table().sort_values("f1", ascending=False)
mlp_tuning_table = load_mlp_tuning_table()
model_bundles = load_model_bundles()


if mlp_tuning_table is not None:
    st.sidebar.header("NN Tuning Bonus")
    st.sidebar.write(
        "Explore the precomputed PyTorch MLP grid search over hidden sizes, learning rates, and dropout rates."
    )

    hidden_options = mlp_tuning_table["hidden_layer_sizes"].drop_duplicates().tolist()
    lr_options = sorted(mlp_tuning_table["learning_rate"].drop_duplicates().tolist())
    dropout_options = sorted(mlp_tuning_table["dropout"].drop_duplicates().tolist())

    selected_hidden = st.sidebar.selectbox("Hidden layers", hidden_options)
    selected_lr = st.sidebar.selectbox("Learning rate", lr_options)
    selected_dropout = st.sidebar.selectbox("Dropout", dropout_options)

    selected_row = mlp_tuning_table[
        (mlp_tuning_table["hidden_layer_sizes"] == selected_hidden)
        & (mlp_tuning_table["learning_rate"] == selected_lr)
        & (mlp_tuning_table["dropout"] == selected_dropout)
    ].iloc[0]

    best_row = mlp_tuning_table.sort_values("val_f1", ascending=False).iloc[0]

    st.sidebar.metric("Validation F1", f"{selected_row['val_f1']:.3f}")
    st.sidebar.metric("Validation Accuracy", f"{selected_row['val_accuracy']:.3f}")
    st.sidebar.metric("Final Loss", f"{selected_row['final_loss']:.3f}")

    st.sidebar.write("Best configuration")
    best_config_table = pd.DataFrame(
        [
            {
                "hidden_layers": best_row["hidden_layer_sizes"],
                "learning_rate": best_row["learning_rate"],
                "dropout": best_row["dropout"],
                "val_f1": round(float(best_row["val_f1"]), 3),
            }
        ]
    )
    st.sidebar.dataframe(best_config_table, use_container_width=True, hide_index=True)

    heatmap_df = mlp_tuning_table[mlp_tuning_table["dropout"] == selected_dropout].copy()
    heatmap_df["learning_rate"] = heatmap_df["learning_rate"].astype(str)
    heatmap_pivot = heatmap_df.pivot(
        index="hidden_layer_sizes",
        columns="learning_rate",
        values="val_f1",
    )
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_title(f"Validation F1 at dropout={selected_dropout}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Hidden layers")
    st.sidebar.pyplot(fig, clear_figure=True)


st.title("End-to-End Data Science Workflow: Plane Crash Survivability")
st.write(
    "This Streamlit app presents the full workflow for predicting whether a historical plane crash record had at least one survivor. "
    "All models are pre-trained and loaded from disk, so the app focuses on storytelling, evaluation, explainability, and interactive prediction."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)


with tab1:
    st.subheader("Dataset and Prediction Task")
    st.write(payload["executive_summary"]["dataset_description"])

    st.subheader("Why This Matters")
    st.write(payload["executive_summary"]["why_it_matters"])

    st.subheader("Approach and Key Findings")
    st.write(payload["executive_summary"]["approach_findings"])

    cols = st.columns(4)
    summary = payload["dataset_summary"]
    best_row = model_table.iloc[0]
    cols[0].metric("Rows modeled", f"{summary['rows']:,}")
    cols[1].metric("Features used", f"{summary['modeling_features']}")
    cols[2].metric("Best model", best_row["model_name"])
    cols[3].metric("Best F1", f"{best_row['f1']:.3f}")


with tab2:
    st.subheader("Target Distribution")
    image("artifacts/plots/target_distribution.png", payload["plot_captions"]["target_distribution"])

    st.subheader("Feature Distributions and Relationships")
    image("artifacts/plots/occupancy_vs_survival.png", payload["plot_captions"]["occupancy_vs_survival"])
    image("artifacts/plots/survival_by_cause.png", payload["plot_captions"]["survival_by_cause"])
    image("artifacts/plots/phase_site_heatmap.png", payload["plot_captions"]["phase_site_heatmap"])
    image("artifacts/plots/survival_by_region.png", payload["plot_captions"]["survival_by_region"])

    st.subheader("Correlation Heatmap")
    image("artifacts/plots/correlation_heatmap.png", payload["plot_captions"]["correlation_heatmap"])


with tab3:
    st.subheader("Model Comparison Summary")
    display_table = model_table[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        display_table[metric] = display_table[metric].map(lambda value: f"{value:.3f}")
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.write(payload["model_summary"]["best_model_paragraph"])

    st.image(str(PLOTS / "model_comparison_f1.png"), use_container_width=True)
    st.image(str(PLOTS / "roc_curves_combined.png"), use_container_width=True)

    st.subheader("Per-Model Metrics, ROC Curves, and Hyperparameters")
    for _, row in model_table.iterrows():
        st.markdown(f"### {row['model_name']}")
        metrics_cols = st.columns(5)
        metrics_cols[0].metric("Accuracy", f"{row['accuracy']:.3f}")
        metrics_cols[1].metric("Precision", f"{row['precision']:.3f}")
        metrics_cols[2].metric("Recall", f"{row['recall']:.3f}")
        metrics_cols[3].metric("F1", f"{row['f1']:.3f}")
        metrics_cols[4].metric("ROC-AUC", f"{row['roc_auc']:.3f}")
        st.image(str(ROOT / row["roc_path"]), use_container_width=True)
        st.json(ast.literal_eval(row["best_params"]))
        if isinstance(row.get("notes"), str) and row["notes"] == row["notes"]:
            st.info(row["notes"])

    st.subheader("Additional Model Artifact")
    st.image(str(PLOTS / "decision_tree_visual.png"), use_container_width=True)
    if (PLOTS / "mlp_training_history.png").exists():
        st.image(str(PLOTS / "mlp_training_history.png"), use_container_width=True)


with tab4:
    st.subheader("Global Explainability")
    st.write(payload["shap_summary"]["interpretation"])
    st.image(str(PLOTS / "shap_summary.png"), use_container_width=True)
    st.image(str(PLOTS / "shap_bar.png"), use_container_width=True)

    st.subheader("Interactive Prediction")
    model_choice = st.selectbox("Choose a model for prediction", model_table["model_name"].tolist())

    input_left, input_right = st.columns(2)
    overrides: dict[str, Any] = {}
    with input_left:
        overrides["year"] = st.slider("Crash year", int(input_config["year"]["min"]), int(input_config["year"]["max"]), int(input_config["year"]["default"]))
        overrides["month"] = st.slider("Month", 1, 12, int(input_config["month"]["default"]))
        overrides["crew_on_board"] = st.slider(
            "Crew on board",
            int(input_config["crew_on_board"]["min"]),
            int(input_config["crew_on_board"]["max"]),
            int(input_config["crew_on_board"]["default"]),
        )
        overrides["pax_on_board"] = st.slider(
            "Passengers on board",
            int(input_config["pax_on_board"]["min"]),
            int(input_config["pax_on_board"]["max"]),
            int(input_config["pax_on_board"]["default"]),
        )
        overrides["aircraft_age"] = st.slider(
            "Aircraft age",
            int(input_config["aircraft_age"]["min"]),
            int(input_config["aircraft_age"]["max"]),
            int(input_config["aircraft_age"]["default"]),
        )
        overrides["Flight phase"] = st.selectbox("Flight phase", input_config["Flight phase"]["options"])
    with input_right:
        overrides["Flight type"] = st.selectbox("Flight type", input_config["Flight type"]["options"])
        overrides["Crash site"] = st.selectbox("Crash site", input_config["Crash site"]["options"])
        overrides["Region"] = st.selectbox("Region", input_config["Region"]["options"])
        overrides["Crash cause"] = st.selectbox("Crash cause", input_config["Crash cause"]["options"])
        overrides["operator_category"] = st.selectbox("Operator category", input_config["operator_category"]["options"])
        overrides["aircraft_family"] = st.selectbox("Aircraft family", input_config["aircraft_family"]["options"])

    feature_row = build_feature_row(input_config, overrides)
    prediction, probability = predict_with_bundle(model_bundles[model_choice], feature_row)

    result_cols = st.columns(2)
    result_cols[0].metric("Predicted class", "At least one survivor" if prediction == 1 else "No survivors")
    result_cols[1].metric("Predicted probability of survival", f"{probability:.1%}")

    st.write(
        "The app lets you choose the predictive model, but SHAP waterfall explanations are only available for tree-based models. "
        "If you pick Logistic Regression or the MLP, the waterfall below falls back to the best tree-based model so the explanation remains meaningful and assignment-compliant."
    )

    tree_models = {"Decision Tree", "Random Forest", "LightGBM"}
    shap_model_name = model_choice if model_choice in tree_models else metadata["best_tree_model"]
    waterfall_fig = create_waterfall_figure(shap_model_name, feature_row)
    st.pyplot(waterfall_fig, clear_figure=True)
    st.caption(f"Waterfall explanation generated with {shap_model_name}.")
