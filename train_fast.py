# train_fast.py
# ─────────────────────────────────────────────────────────────
# Quick verification script — trains ONLY Logistic Regression
# (our known best model) and registers it, skipping Random
# Forest and XGBoost to save time while we verify the Docker
# artifact-serving fix actually works.
#
# Once confirmed working, you can go back to the full train.py
# for a "real" comparison run later if you want that in your
# portfolio history.
# ─────────────────────────────────────────────────────────────

import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import mlflow

from train import (
    load_data, build_features, train_and_log,
    RANDOM_STATE, TEST_SIZE, EXPERIMENT_NAME, MODEL_NAME
)


def main(variants_path: str, text_path: str):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  MLflow tracking URI : {tracking_uri}")
    print(f"  Experiment          : {EXPERIMENT_NAME}")

    print("\n  Loading data...")
    df = load_data(variants_path, text_path)
    print(f"  Dataset shape: {df.shape}")

    X, y, transformers = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(
        C=1.0, max_iter=500, solver="saga",
        random_state=RANDOM_STATE
    )
    params = {"C": 1.0, "max_iter": 500, "solver": "saga"}

    logloss, run_id = train_and_log(
        model, "Logistic Regression [BEST]", params,
        X_train, X_test, y_train, y_test,
        transformers, is_best=True   # registers it immediately
    )

    print(f"\n  Done ✓  (test_log_loss={logloss:.4f}, run_id={run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="data/training_variants.csv")
    parser.add_argument("--text", default="data/training_text.csv")
    args = parser.parse_args()
    main(args.variants, args.text)