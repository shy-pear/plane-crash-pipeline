from __future__ import annotations

import inspect
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Plane Crashes.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DATA_DIR = ARTIFACTS_DIR / "data"


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    best_params: dict[str, Any]
    model_path: str
    roc_path: str | None = None
    training_history_path: str | None = None
    notes: str | None = None


def ensure_dirs() -> None:
    for directory in [ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def make_one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def bucket_top_categories(series: pd.Series, top_n: int) -> pd.Series:
    clean = series.map(normalize_text)
    top_values = clean.value_counts().head(top_n).index
    return clean.where(clean.isin(top_values), "Other")


def infer_operator_category(operator: str) -> str:
    text = normalize_text(operator).lower()
    if text == "unknown":
        return "Unknown"
    military_markers = [
        "air force",
        "raf",
        "army",
        "navy",
        "military",
        "marine",
        "signal corps",
    ]
    cargo_markers = ["cargo", "freight"]
    charter_markers = ["charter", "air taxi"]
    if any(marker in text for marker in military_markers):
        return "Military"
    if any(marker in text for marker in cargo_markers):
        return "Cargo"
    if any(marker in text for marker in charter_markers):
        return "Charter"
    if any(marker in text for marker in ["airlines", "airways", "air line", "air ", "fly", "aviation"]):
        return "Commercial"
    return "Other"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.loc[data["Date"].notna()].copy()
    data["target"] = data["Survivors"].map({"Yes": 1, "No": 0})
    data = data.loc[data["target"].notna()].copy()
    data["target"] = data["target"].astype(int)

    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["quarter"] = data["Date"].dt.quarter
    data["day_of_week"] = data["Date"].dt.day_name()
    data["crew_on_board"] = data["Crew on board"]
    data["pax_on_board"] = data["Pax on board"]
    data["total_on_board"] = data["crew_on_board"].fillna(0) + data["pax_on_board"].fillna(0)
    data["crew_share"] = np.where(
        data["total_on_board"] > 0,
        data["crew_on_board"].fillna(0) / data["total_on_board"],
        np.nan,
    )

    data["YOM"] = pd.to_numeric(data["YOM"], errors="coerce")
    data.loc[(data["YOM"] < 1903) | (data["YOM"] > data["year"]), "YOM"] = np.nan
    data["aircraft_age"] = data["year"] - data["YOM"]
    data["aircraft_family"] = data["Aircraft"].map(normalize_text).str.split().str[:2].str.join(" ")
    data["aircraft_family"] = bucket_top_categories(data["aircraft_family"], top_n=20)
    data["country_group"] = bucket_top_categories(data["Country"], top_n=12)
    data["operator_category"] = data["Operator"].map(infer_operator_category)

    categorical_fill_cols = [
        "Flight phase",
        "Flight type",
        "Crash site",
        "Region",
        "Crash cause",
        "day_of_week",
        "country_group",
        "operator_category",
        "aircraft_family",
    ]
    for col in categorical_fill_cols:
        data[col] = data[col].map(normalize_text)

    return data


def build_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline(categorical_steps), categorical_features),
        ],
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def plot_and_save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_target_distribution(data: pd.DataFrame) -> dict[str, Any]:
    counts = data["target"].map({0: "No survivors", 1: "At least one survivor"}).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#9b2226", "#2a9d8f"], legend=False, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_ylabel("Crash records")
    ax.set_xlabel("")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 120, f"{value:,}", ha="center", fontsize=10)
    plot_and_save(fig, PLOTS_DIR / "target_distribution.png")
    return {
        "class_counts": counts.to_dict(),
        "survival_rate": float(data["target"].mean()),
    }


def save_descriptive_plots(data: pd.DataFrame) -> dict[str, Any]:
    medians = data.groupby("target")["total_on_board"].median().to_dict()

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=data,
        x=data["target"].map({0: "No survivors", 1: "At least one survivor"}),
        y="total_on_board",
        hue=data["target"].map({0: "No survivors", 1: "At least one survivor"}),
        palette=["#9b2226", "#2a9d8f"],
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Occupancy by Survival Outcome")
    ax1.set_xlabel("")
    ax1.set_ylabel("Total on board")
    plot_and_save(fig1, PLOTS_DIR / "occupancy_vs_survival.png")

    cause_summary = (
        data.groupby("Crash cause")
        .agg(crashes=("target", "size"), survival_rate=("target", "mean"))
        .sort_values("crashes", ascending=False)
        .head(8)
        .reset_index()
    )
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=cause_summary, x="Crash cause", y="survival_rate", hue="Crash cause", palette="crest", legend=False, ax=ax2)
    ax2.set_title("Survival Rate by Crash Cause")
    ax2.set_xlabel("")
    ax2.set_ylabel("Share with survivors")
    ax2.tick_params(axis="x", rotation=25)
    plot_and_save(fig2, PLOTS_DIR / "survival_by_cause.png")

    heatmap_data = (
        data.pivot_table(
            index="Flight phase",
            columns="Crash site",
            values="target",
            aggfunc="mean",
        )
        .fillna(0)
    )
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax3)
    ax3.set_title("Survival Rate by Flight Phase and Crash Site")
    plot_and_save(fig3, PLOTS_DIR / "phase_site_heatmap.png")

    region_summary = (
        data.groupby("Region")
        .agg(crashes=("target", "size"), survival_rate=("target", "mean"))
        .sort_values("crashes", ascending=False)
        .head(8)
        .reset_index()
    )
    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=region_summary, x="Region", y="survival_rate", hue="Region", palette="viridis", legend=False, ax=ax4)
    ax4.set_title("Survival Rate by Region")
    ax4.set_xlabel("")
    ax4.set_ylabel("Share with survivors")
    ax4.tick_params(axis="x", rotation=25)
    plot_and_save(fig4, PLOTS_DIR / "survival_by_region.png")

    corr_cols = ["year", "month", "crew_on_board", "pax_on_board", "total_on_board", "crew_share", "aircraft_age", "target"]
    corr = data[corr_cols].corr(numeric_only=True)
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax5)
    ax5.set_title("Correlation Heatmap")
    plot_and_save(fig5, PLOTS_DIR / "correlation_heatmap.png")

    corr_pairs = (
        corr.where(~np.eye(len(corr), dtype=bool))
        .stack()
        .sort_values(key=np.abs, ascending=False)
    )
    strongest_corr = corr_pairs.index[0]
    strongest_value = corr_pairs.iloc[0]

    return {
        "median_total_on_board_no_survivors": float(medians.get(0, math.nan)),
        "median_total_on_board_with_survivors": float(medians.get(1, math.nan)),
        "top_cause_survival": cause_summary.iloc[0].to_dict(),
        "top_region_survival": region_summary.sort_values("survival_rate", ascending=False).iloc[0].to_dict(),
        "strongest_corr_pair": [str(strongest_corr[0]), str(strongest_corr[1])],
        "strongest_corr_value": float(strongest_value),
        "heatmap_shape": [int(heatmap_data.shape[0]), int(heatmap_data.shape[1])],
    }


def create_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", color="#2a9d8f")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plot_and_save(fig, path)


def create_combined_roc_plot(y_true: np.ndarray, roc_payload: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = sns.color_palette("Set2", len(roc_payload))
    for idx, (model_name, probs) in enumerate(roc_payload.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})", color=palette[idx])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves Across Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plot_and_save(fig, PLOTS_DIR / "roc_curves_combined.png")


def persist_model_bundle(path: Path, payload: dict[str, Any]) -> None:
    joblib.dump(payload, path)


def train_grid_model(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[ModelResult, np.ndarray, Any]:
    pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=0,
    )
    print(f"Running grid search for {model_name}...", flush=True)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)

    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    payload = {
        "bundle_type": "pipeline",
        "model_name": model_name,
        "pipeline": best_model,
        "best_params": grid.best_params_,
        "selected_features": list(X_train.columns),
    }
    persist_model_bundle(model_path, payload)

    roc_path = PLOTS_DIR / f"roc_{model_name.lower().replace(' ', '_')}.png"
    create_roc_plot(y_test.to_numpy(), y_prob, model_name, roc_path)

    return (
        ModelResult(
            model_name=model_name,
            best_params=grid.best_params_,
            model_path=str(model_path.relative_to(ROOT)),
            roc_path=str(roc_path.relative_to(ROOT)),
            **metrics,
        ),
        y_prob,
        best_model,
    )


def train_logistic_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[ModelResult, np.ndarray]:
    pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)

    model_path = MODELS_DIR / "logistic_regression.joblib"
    payload = {
        "bundle_type": "pipeline",
        "model_name": "Logistic Regression",
        "pipeline": pipeline,
        "best_params": {"class_weight": "balanced", "max_iter": 2000},
        "selected_features": list(X_train.columns),
    }
    persist_model_bundle(model_path, payload)

    roc_path = PLOTS_DIR / "roc_logistic_regression.png"
    create_roc_plot(y_test.to_numpy(), y_prob, "Logistic Regression", roc_path)

    return (
        ModelResult(
            model_name="Logistic Regression",
            best_params={"class_weight": "balanced", "max_iter": 2000},
            model_path=str(model_path.relative_to(ROOT)),
            roc_path=str(roc_path.relative_to(ROOT)),
            **metrics,
        ),
        y_prob,
    )


def train_mlp_fallback(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[ModelResult, np.ndarray]:
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    fitted_preprocessor = clone(preprocessor)
    X_train_proc = fitted_preprocessor.fit_transform(X_train_sub)
    X_val_proc = fitted_preprocessor.transform(X_val)
    X_test_proc = fitted_preprocessor.transform(X_test)

    epochs = 40
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        batch_size=256,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    history_rows: list[dict[str, float]] = []
    classes = np.array([0, 1])
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            mlp.partial_fit(X_train_proc, y_train_sub, classes=classes)
        else:
            mlp.partial_fit(X_train_proc, y_train_sub)
        val_prob = mlp.predict_proba(X_val_proc)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        history_rows.append(
            {
                "epoch": epoch,
                "loss": float(mlp.loss_),
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
                "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
            }
        )

    y_prob = mlp.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)

    history_df = pd.DataFrame(history_rows)
    history_path = DATA_DIR / "mlp_training_history.csv"
    history_df.to_csv(history_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.lineplot(data=history_df, x="epoch", y="loss", ax=axes[0], color="#e76f51")
    axes[0].set_title("MLP Training Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_f1", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("MLP Validation F1")
    plot_and_save(fig, PLOTS_DIR / "mlp_training_history.png")

    model_path = MODELS_DIR / "mlp_classifier.joblib"
    payload = {
        "bundle_type": "preprocessed_model",
        "model_name": "MLP Neural Network",
        "preprocessor": fitted_preprocessor,
        "model": mlp,
        "best_params": {
            "hidden_layer_sizes": [128, 128],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate_init": 0.001,
            "epochs": epochs,
        },
        "selected_features": list(X_train.columns),
        "notes": "Fallback implementation uses sklearn.neural_network.MLPClassifier because TensorFlow/PyTorch was unavailable in this environment.",
    }
    persist_model_bundle(model_path, payload)

    roc_path = PLOTS_DIR / "roc_mlp_neural_network.png"
    create_roc_plot(y_test.to_numpy(), y_prob, "MLP Neural Network", roc_path)

    return (
        ModelResult(
            model_name="MLP Neural Network",
            best_params=payload["best_params"],
            model_path=str(model_path.relative_to(ROOT)),
            roc_path=str(roc_path.relative_to(ROOT)),
            training_history_path=str((PLOTS_DIR / "mlp_training_history.png").relative_to(ROOT)),
            notes=payload["notes"],
            **metrics,
        ),
        y_prob,
    )


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


def train_torch_mlp(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[ModelResult, np.ndarray]:
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    fitted_preprocessor = clone(preprocessor)
    X_train_proc = np.asarray(fitted_preprocessor.fit_transform(X_train_sub), dtype=np.float32)
    X_val_proc = np.asarray(fitted_preprocessor.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(fitted_preprocessor.transform(X_test), dtype=np.float32)

    y_train_np = y_train_sub.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_np = y_val.to_numpy(dtype=np.float32)

    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(1)
    input_dim = X_train_proc.shape[1]
    hidden_sizes = (64, 64)
    dropout = 0.2
    learning_rate = 0.001
    epochs = 20
    batch_size = 512

    model = TorchMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    X_train_tensor = torch.from_numpy(X_train_proc)
    y_train_tensor = torch.from_numpy(y_train_np)
    X_val_tensor = torch.from_numpy(X_val_proc)
    X_test_tensor = torch.from_numpy(X_test_proc)

    history_rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(X_train_tensor.size(0))
        for start_idx in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            batch_x = X_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor).squeeze(1)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_pred = (val_prob >= 0.5).astype(int)
        history_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(X_train_proc),
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
                "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"PyTorch MLP epoch {epoch}/{epochs} "
                f"- loss={history_rows[-1]['loss']:.4f} "
                f"- val_f1={history_rows[-1]['val_f1']:.4f}",
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor).squeeze(1)
        y_prob = torch.sigmoid(test_logits).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)

    history_df = pd.DataFrame(history_rows)
    history_path = DATA_DIR / "mlp_training_history.csv"
    history_df.to_csv(history_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.lineplot(data=history_df, x="epoch", y="loss", ax=axes[0], color="#e76f51")
    axes[0].set_title("MLP Training Loss")
    sns.lineplot(data=history_df, x="epoch", y="val_f1", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("MLP Validation F1")
    plot_and_save(fig, PLOTS_DIR / "mlp_training_history.png")

    model_path = MODELS_DIR / "mlp_classifier.joblib"
    payload = {
        "bundle_type": "torch_model",
        "framework": "pytorch",
        "model_name": "MLP Neural Network",
        "preprocessor": fitted_preprocessor,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": {
            "input_dim": input_dim,
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
        },
        "best_params": {
            "hidden_layer_sizes": list(hidden_sizes),
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "selected_features": list(X_train.columns),
        "notes": "PyTorch MLP with two hidden layers, ReLU activations, dropout regularization, and Adam optimization.",
    }
    persist_model_bundle(model_path, payload)

    roc_path = PLOTS_DIR / "roc_mlp_neural_network.png"
    create_roc_plot(y_test.to_numpy(), y_prob, "MLP Neural Network", roc_path)

    return (
        ModelResult(
            model_name="MLP Neural Network",
            best_params=payload["best_params"],
            model_path=str(model_path.relative_to(ROOT)),
            roc_path=str(roc_path.relative_to(ROOT)),
            training_history_path=str((PLOTS_DIR / "mlp_training_history.png").relative_to(ROOT)),
            notes=payload["notes"],
            **metrics,
        ),
        y_prob,
    )


def save_tree_visual(best_tree_pipeline: Pipeline) -> None:
    model = best_tree_pipeline.named_steps["model"]
    preprocessor = best_tree_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No survivors", "At least one survivor"],
        filled=True,
        max_depth=3,
        ax=ax,
        fontsize=8,
    )
    ax.set_title("Decision Tree (top 3 levels)")
    plot_and_save(fig, PLOTS_DIR / "decision_tree_visual.png")


def make_input_config(data: pd.DataFrame, selected_features: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for feature in selected_features:
        if pd.api.types.is_numeric_dtype(data[feature]):
            series = data[feature].dropna()
            config[feature] = {
                "kind": "numeric",
                "default": float(series.median()),
                "min": float(series.quantile(0.05)),
                "max": float(series.quantile(0.95)),
            }
        else:
            top_values = data[feature].astype(str).value_counts().head(10).index.tolist()
            default = top_values[0] if top_values else "Unknown"
            config[feature] = {
                "kind": "categorical",
                "default": default,
                "options": top_values,
            }
    return config


def run_shap_analysis(
    model_name: str,
    fitted_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    background = preprocessor.transform(X_train.sample(n=min(800, len(X_train)), random_state=RANDOM_STATE))
    test_sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
    test_transformed = preprocessor.transform(test_sample)

    explainer = shap.TreeExplainer(estimator)
    print(f"Running SHAP analysis for {model_name}...", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
        )
        shap_values = explainer.shap_values(test_transformed)
    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[1])
        expected_value = float(explainer.expected_value[1])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_matrix = shap_array[:, :, 1]
            expected_value = float(np.asarray(explainer.expected_value)[1])
        else:
            shap_matrix = shap_array
            expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
    importance = shap_df.abs().mean().sort_values(ascending=False)
    top_features = importance.head(10)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, test_transformed, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot: {model_name}")
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_matrix, test_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance: {model_name}")
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    waterfall_index = int(np.argmax(np.abs(shap_matrix).sum(axis=1)))
    waterfall_explanation = shap.Explanation(
        values=shap_matrix[waterfall_index],
        base_values=expected_value,
        data=np.asarray(test_transformed)[waterfall_index],
        feature_names=feature_names,
    )
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(waterfall_explanation, show=False, max_display=12)
    plt.savefig(PLOTS_DIR / "shap_waterfall_reference.png", dpi=200, bbox_inches="tight")
    plt.close()

    explainer_path = MODELS_DIR / "shap_support.joblib"
    persist_model_bundle(
        explainer_path,
        {
            "model_name": model_name,
            "feature_names": list(feature_names),
            "background_transformed": np.asarray(background),
            "reference_transformed": np.asarray(test_transformed),
            "expected_value": expected_value,
        },
    )

    return {
        "best_tree_model": model_name,
        "top_shap_features": top_features.index.tolist(),
        "top_shap_values": [float(value) for value in top_features.values],
        "shap_support_path": str(explainer_path.relative_to(ROOT)),
    }


def build_report_payload(
    data: pd.DataFrame,
    target_summary: dict[str, Any],
    descriptive_summary: dict[str, Any],
    model_results: list[ModelResult],
    shap_summary: dict[str, Any],
    selected_features: list[str],
) -> dict[str, Any]:
    best_overall = max(model_results, key=lambda item: item.f1)
    best_tree_name = shap_summary["best_tree_model"]
    rows = len(data)
    columns = len(selected_features)
    numeric_count = sum(pd.api.types.is_numeric_dtype(data[col]) for col in selected_features)
    categorical_count = columns - numeric_count

    top_cause = descriptive_summary["top_cause_survival"]
    top_region = descriptive_summary["top_region_survival"]

    return {
        "dataset_summary": {
            "rows": rows,
            "modeling_features": columns,
            "numeric_features": numeric_count,
            "categorical_features": categorical_count,
            "target_name": "Survivors",
            "target_positive_label": "At least one survivor",
        },
        "executive_summary": {
            "dataset_description": (
                f"This project uses {rows:,} historical airplane crash records from the `Plane Crashes.csv` dataset. "
                "Each row represents one crash event and includes operational context such as flight phase, flight type, crash site, region, likely cause, aircraft family, and how many crew and passengers were on board. "
                "The prediction target is whether the crash had at least one survivor, turning the dataset into a binary classification problem that stakeholders can understand immediately."
            ),
            "why_it_matters": (
                "Although survival after an air crash is an extreme and tragic outcome, the broader analytics question is meaningful: which operating conditions and crash contexts are associated with any chance of survival? "
                "A model like this can help safety analysts identify patterns tied to survivability, communicate risk more clearly, and prioritize deeper investigation into the conditions that appear most dangerous."
            ),
            "approach_findings": (
                f"I engineered a leakage-safe feature set by excluding direct fatality counts and instead modeling survivability from contextual signals such as occupancy, aircraft age, flight phase, crash site, region, and crash cause. "
                f"I then compared Logistic Regression, Decision Tree, Random Forest, LightGBM, and an MLP neural network with a fixed random seed and a held-out 30 percent test set. "
                f"The strongest test-set F1 score came from {best_overall.model_name} at {best_overall.f1:.3f}, with ROC-AUC of {best_overall.roc_auc:.3f}. "
                f"For explainability, I used {best_tree_name} because it was the best-performing tree-based model and supports a detailed SHAP analysis."
            ),
        },
        "plot_captions": {
            "target_distribution": (
                f"The target is reasonably balanced: {target_summary['class_counts']['At least one survivor']:,} crashes had at least one survivor, while "
                f"{target_summary['class_counts']['No survivors']:,} had none. That balance means F1 and ROC-AUC are still more informative than accuracy alone, but we do not face an extreme class-imbalance problem."
            ),
            "occupancy_vs_survival": (
                f"Crashes with no survivors tend to involve more people on board, with a median of {descriptive_summary['median_total_on_board_no_survivors']:.1f} versus "
                f"{descriptive_summary['median_total_on_board_with_survivors']:.1f} when at least one person survives. This does not prove causality, but it suggests that higher-occupancy events may be harder to escape or may correlate with more severe crash contexts."
            ),
            "survival_by_cause": (
                f"Different crash-cause categories show noticeably different survival patterns. The most common high-survival category in this top slice is {top_cause['Crash cause']}, "
                f"where roughly {top_cause['survival_rate']:.1%} of records still had at least one survivor, highlighting how cause classification may capture meaningful differences in crash severity."
            ),
            "phase_site_heatmap": (
                "Survival rates vary meaningfully when flight phase and crash site are considered together rather than in isolation. "
                "This kind of interaction view is useful because it shows that the same phase can have very different outcomes depending on whether the crash happened near an airport, over water, or in a more remote setting."
            ),
            "survival_by_region": (
                f"Regional differences appear in the historical data, although they likely reflect a mix of reporting practices, aircraft usage patterns, and geography rather than a simple regional effect. "
                f"In this subset, {top_region['Region']} shows one of the highest observed survival rates among the busiest regions in the dataset."
            ),
            "correlation_heatmap": (
                f"The strongest numeric correlation in the engineered feature set is between {descriptive_summary['strongest_corr_pair'][0]} and {descriptive_summary['strongest_corr_pair'][1]} "
                f"at {descriptive_summary['strongest_corr_value']:.2f}. This matters for modeling because highly related occupancy variables can reinforce similar signals, especially for linear models."
            ),
        },
        "model_summary": {
            "best_model_paragraph": (
                f"{best_overall.model_name} delivered the best F1 score on the held-out test set, which suggests it captured the nonlinear relationships in the crash context better than the simpler baseline models. "
                "The main trade-off is interpretability: Logistic Regression and the Decision Tree are easier to explain quickly, but the ensemble methods produced stronger predictive performance."
            )
        },
        "shap_summary": {
            "interpretation": (
                f"The SHAP analysis shows that the most influential features in {best_tree_name} are {', '.join(shap_summary['top_shap_features'][:5])}. "
                "Positive and negative SHAP values reveal not just which variables matter, but which directions of those variables move a prediction toward or away from survivability. "
                "For a decision-maker, this is useful because it turns the model from a black box into a ranked explanation of the conditions most associated with survival."
            )
        },
        "model_results": [asdict(result) for result in model_results],
    }


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_dirs()

    raw_df = pd.read_csv(DATA_PATH)
    data = engineer_features(raw_df)

    selected_features = [
        "year",
        "month",
        "quarter",
        "crew_on_board",
        "pax_on_board",
        "total_on_board",
        "crew_share",
        "aircraft_age",
        "Flight phase",
        "Flight type",
        "Crash site",
        "Region",
        "Crash cause",
        "day_of_week",
        "country_group",
        "operator_category",
        "aircraft_family",
    ]

    X = data[selected_features].copy()
    y = data["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    numeric_features = [column for column in selected_features if pd.api.types.is_numeric_dtype(X[column])]
    categorical_features = [column for column in selected_features if column not in numeric_features]

    linear_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    tree_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    target_summary = save_target_distribution(data)
    descriptive_summary = save_descriptive_plots(data)

    results: list[ModelResult] = []
    roc_payload: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    logistic_result, logistic_prob = train_logistic_regression(linear_preprocessor, X_train, y_train, X_test, y_test)
    print("Finished Logistic Regression.", flush=True)
    results.append(logistic_result)
    roc_payload[logistic_result.model_name] = logistic_prob

    decision_result, decision_prob, decision_model = train_grid_model(
        model_name="Decision Tree",
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid={
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_leaf": [5, 10, 20, 50],
        },
        preprocessor=tree_preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results.append(decision_result)
    roc_payload[decision_result.model_name] = decision_prob
    fitted_models[decision_result.model_name] = decision_model
    save_tree_visual(decision_model)

    rf_result, rf_prob, rf_model = train_grid_model(
        model_name="Random Forest",
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        param_grid={
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 8],
        },
        preprocessor=tree_preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results.append(rf_result)
    roc_payload[rf_result.model_name] = rf_prob
    fitted_models[rf_result.model_name] = rf_model

    lgb_result, lgb_prob, lgb_model = train_grid_model(
        model_name="LightGBM",
        estimator=lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
        ),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__min_child_samples": [20, 50],
        },
        preprocessor=tree_preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results.append(lgb_result)
    roc_payload[lgb_result.model_name] = lgb_prob
    fitted_models[lgb_result.model_name] = lgb_model

    if torch is not None:
        print("Training PyTorch MLP...", flush=True)
        mlp_result, mlp_prob = train_torch_mlp(linear_preprocessor, X_train, y_train, X_test, y_test)
        print("Finished PyTorch MLP.", flush=True)
    else:
        print("PyTorch not available; using sklearn MLP fallback...", flush=True)
        mlp_result, mlp_prob = train_mlp_fallback(linear_preprocessor, X_train, y_train, X_test, y_test)
    results.append(mlp_result)
    roc_payload[mlp_result.model_name] = mlp_prob

    comparison_df = pd.DataFrame([asdict(result) for result in results]).sort_values("f1", ascending=False)
    comparison_df.to_csv(DATA_DIR / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=comparison_df, x="model_name", y="f1", hue="model_name", palette="mako", legend=False, ax=ax)
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.tick_params(axis="x", rotation=20)
    plot_and_save(fig, PLOTS_DIR / "model_comparison_f1.png")

    create_combined_roc_plot(y_test.to_numpy(), roc_payload)

    best_tree_result = max(
        [item for item in results if item.model_name in {"Decision Tree", "Random Forest", "LightGBM"}],
        key=lambda item: item.f1,
    )
    shap_summary = run_shap_analysis(best_tree_result.model_name, fitted_models[best_tree_result.model_name], X_train, X_test)
    print("Finished SHAP analysis.", flush=True)

    input_config = make_input_config(data, selected_features)
    save_json(REPORTS_DIR / "input_config.json", input_config)

    report_payload = build_report_payload(
        data=data,
        target_summary=target_summary,
        descriptive_summary=descriptive_summary,
        model_results=results,
        shap_summary=shap_summary,
        selected_features=selected_features,
    )
    save_json(REPORTS_DIR / "report_payload.json", report_payload)

    metadata = {
        "random_state": RANDOM_STATE,
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_overall_model": comparison_df.iloc[0]["model_name"],
        "best_tree_model": best_tree_result.model_name,
        "target_distribution": target_summary,
        "shap_summary": shap_summary,
        "mlp_note": mlp_result.notes,
    }
    save_json(REPORTS_DIR / "metadata.json", metadata)

    print("Training complete.", flush=True)
    print(f"Best overall model: {comparison_df.iloc[0]['model_name']} (F1={comparison_df.iloc[0]['f1']:.3f})", flush=True)
    print(f"Best tree-based model: {best_tree_result.model_name}", flush=True)


if __name__ == "__main__":
    main()
