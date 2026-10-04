# src/churn/train.py
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.churn.data import load_clean, TARGET_COL
from src.churn.features import build_pipeline, ALL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def run_experiment(name: str, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return pipeline, metrics


def main():
    mlflow.set_experiment("churn-prediction")

    df = load_clean()
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # Baseline: simple, interpretable, the number everything else must beat
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    results["logistic_regression"] = run_experiment(
        "logistic_regression", lr, X_train, X_test, y_train, y_test
    )

    # Stronger model: usually wins on imbalanced tabular data like this
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    results["gradient_boosting"] = run_experiment(
        "gradient_boosting", gb, X_train, X_test, y_train, y_test
    )

    # Pick the best by AUC (sensible for an imbalanced target), save it
    best_name = max(results, key=lambda k: results[k][1]["auc"])
    best_pipeline, best_metrics = results[best_name]
    print(f"\nBest model: {best_name} (AUC={best_metrics['auc']:.4f})")

    import joblib
    joblib.dump(best_pipeline, MODEL_DIR / "model.joblib")
    print(f"Saved to {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
