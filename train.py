# src/train.py
# ─────────────────────────────────────────────────────────────
# Full training pipeline with MLflow experiment tracking.
#
# What MLflow gives us:
#   - Every run is logged: params, metrics, artifacts, model
#   - You can compare runs in a UI (mlflow ui)
#   - The best model gets registered in the MLflow Model Registry
#   - FastAPI loads the model FROM the registry — no manual file copying
# ─────────────────────────────────────────────────────────────

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — needed in CI/CD (no display)
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack, save_npz
import xgboost as xgb

from preprocess import build_combined_feature   # our shared utility

# ── CONFIG ────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
TFIDF_MAX_FEAT  = 10_000
EXPERIMENT_NAME = "personalized-medicine"
MODEL_NAME      = "cancer-classifier"          # name in MLflow Model Registry

np.random.seed(RANDOM_STATE)


def load_data(variants_path: str, text_path: str) -> pd.DataFrame:
    variants = pd.read_csv(variants_path)
    text_df  = pd.read_csv(text_path, sep=r"\|\|", engine="python",
                           skiprows=1, names=["ID", "Text"])
    df = variants.merge(text_df, on="ID")
    df["Text"] = df["Text"].fillna("")
    return df


def build_features(df: pd.DataFrame):
    """
    Build feature matrix and return fitted vectorizers for inference.
    Returns X (sparse matrix), y (array), and dict of fitted transformers.
    """
    df["combined"] = df.apply(
        lambda r: build_combined_feature(r["Gene"], r["Variation"], r["Text"]),
        axis=1
    )

    tfidf      = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1, 2),
                                 min_df=3, sublinear_tf=True)
    gene_tfidf = TfidfVectorizer(max_features=500)
    var_tfidf  = TfidfVectorizer(max_features=500)

    X_text = tfidf.fit_transform(df["combined"])
    X_gene = gene_tfidf.fit_transform(df["Gene"].str.lower())
    X_var  = var_tfidf.fit_transform(df["Variation"].str.lower())

    X = hstack([X_text, X_gene, X_var])
    y = df["Class"].values - 1    # 0-indexed for XGBoost

    transformers = {
        "tfidf": tfidf,
        "gene_tfidf": gene_tfidf,
        "var_tfidf": var_tfidf,
    }
    return X, y, transformers


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[f"C{i+1}" for i in range(9)])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = f"/tmp/cm_{model_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_and_log(model, model_name: str, params: dict,
                  X_train, X_test, y_train, y_test,
                  transformers: dict, is_best: bool = False):
    """
    Train one model, log everything to MLflow, optionally register it.
    """
    with mlflow.start_run(run_name=model_name):
        # ── Log hyperparameters ──────────────────────────────
        mlflow.log_params(params)
        mlflow.log_param("tfidf_max_features", TFIDF_MAX_FEAT)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # ── Train ────────────────────────────────────────────
        print(f"\n  ▶ Training {model_name}...")
        model.fit(X_train, y_train)

        # ── Metrics ──────────────────────────────────────────
        y_train_proba = model.predict_proba(X_train)
        y_test_proba  = model.predict_proba(X_test)
        y_pred        = model.predict(X_test)

        train_logloss = log_loss(y_train, y_train_proba)
        test_logloss  = log_loss(y_test,  y_test_proba)

        mlflow.log_metric("train_log_loss", train_logloss)
        mlflow.log_metric("test_log_loss",  test_logloss)
        mlflow.log_metric("overfit_gap", test_logloss - train_logloss)

        print(f"    Train Log Loss: {train_logloss:.4f}")
        print(f"    Test  Log Loss: {test_logloss:.4f}")

        # ── Artifacts: confusion matrix plot ─────────────────
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ── Log classification report as text artifact ────────
        report = classification_report(
            y_test, y_pred,
            target_names=[f"Class {i+1}" for i in range(9)]
        )
        report_path = f"/tmp/report_{model_name.replace(' ', '_')}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # ── Log vectorizers as artifacts ──────────────────────
        # WHY: The model is USELESS without the same transformers used at train time.
        # Always version them together.
        for name, vec in transformers.items():
            vec_path = f"/tmp/{name}.joblib"
            joblib.dump(vec, vec_path)
            mlflow.log_artifact(vec_path, artifact_path="transformers")

        # ── Log the model ─────────────────────────────────────
        # Infer signature so MLflow knows input/output schema
        sample_input  = X_test[:5]
        sample_output = model.predict_proba(sample_input)

        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(
                model, artifact_path="model",
                registered_model_name=MODEL_NAME if is_best else None,
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                registered_model_name=MODEL_NAME if is_best else None,
            )

        run_id = mlflow.active_run().info.run_id
        print(f"    [✓] MLflow run logged: {run_id}")

    return test_logloss, run_id


def main(variants_path: str, text_path: str):
    # ── Setup MLflow experiment ───────────────────────────────
    # MLFLOW_TRACKING_URI can be set as env var to point to a remote server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  MLflow tracking URI : {tracking_uri}")
    print(f"  Experiment          : {EXPERIMENT_NAME}")

    # ── Load & Feature Engineering ───────────────────────────
    print("\n  Loading data...")
    df = load_data(variants_path, text_path)
    print(f"  Dataset shape: {df.shape}")

    X, y, transformers = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── Define Models + Their Params ─────────────────────────
    experiments = [
        (
            LogisticRegression(C=1.0, max_iter=500, solver="saga",
                               multi_class="multinomial", random_state=RANDOM_STATE, n_jobs=-1),
            "Logistic Regression",
            {"C": 1.0, "max_iter": 500, "solver": "saga"},
        ),
        (
            RandomForestClassifier(n_estimators=200, max_depth=20,
                                   min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
            "Random Forest",
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2},
        ),
        (
            xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric="mlogloss",
                               random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost",
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8},
        ),
    ]

    # ── Train all, find best ──────────────────────────────────
    run_results = []
    for model, name, params in experiments:
        logloss, run_id = train_and_log(
            model, name, params,
            X_train, X_test, y_train, y_test,
            transformers, is_best=False
        )
        run_results.append((logloss, name, model, params, run_id))

    # ── Re-register best model ────────────────────────────────
    best_logloss, best_name, best_model, best_params, _ = min(run_results, key=lambda x: x[0])
    print(f"\n Best Model: {best_name} (Log Loss: {best_logloss:.4f})")
    print(f"  Registering '{MODEL_NAME}' in MLflow Model Registry...")

    train_and_log(
        best_model, f"{best_name} [BEST]", best_params,
        X_train, X_test, y_train, y_test,
        transformers, is_best=True
    )

    print(f"\n  Run 'mlflow ui' to explore all experiments in the browser.")
    print(f"  Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cancer classifier with MLflow tracking")
    parser.add_argument("--variants", default="training_variants.csv")
    parser.add_argument("--text",     default="training_text.csv")
    args = parser.parse_args()
    main(args.variants, args.text)
