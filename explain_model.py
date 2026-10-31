"""
Explainable ML Classifier
-------------------------
Trains two models on the Breast Cancer Wisconsin dataset and explains them:

  * Logistic Regression  -> LOCAL explanation (why this one prediction?)
  * Random Forest        -> GLOBAL explanation (which features matter overall?)

No API keys or downloads needed: the dataset ships with scikit-learn.

Usage:
    python explain_model.py
    python explain_model.py --sample-index 5 --output-dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works on machines without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainable ML classifier demo")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the test sample to explain (default: 0)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top features to show in the plots (default: 10)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Folder for plots and metrics (default: outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def load_data(seed: int):
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test, list(data.target_names)


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }


def plot_global_importance(rf, X_test, y_test, top_k, seed, path: Path) -> pd.Series:
    """Permutation importance: how much does the log-loss get worse if a feature is shuffled?

    Log-loss is used instead of accuracy because it is more sensitive: accuracy only
    changes when a prediction flips, log-loss changes whenever confidence shifts.
    """
    result = permutation_importance(
        rf, X_test, y_test, scoring="neg_log_loss",
        n_repeats=20, random_state=seed, n_jobs=1,
    )
    importance = pd.Series(result.importances_mean, index=X_test.columns)
    top = importance.sort_values(ascending=False).head(top_k)
    errors = pd.Series(result.importances_std, index=X_test.columns)[top.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], xerr=errors.values[::-1], color="#2a7de1")
    ax.set_xlabel("Increase in log-loss when feature is shuffled (higher = more important)")
    ax.set_title(f"Global explanation: top {top_k} features (Random Forest)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return top


def explain_one_sample(lr_pipeline, X_test, idx, class_names, top_k, path: Path) -> dict:
    """For a linear model, contribution = coefficient x standardized feature value."""
    scaler = lr_pipeline.named_steps["standardscaler"]
    clf = lr_pipeline.named_steps["logisticregression"]

    sample = X_test.iloc[[idx]]
    scaled = scaler.transform(sample)[0]
    contributions = pd.Series(clf.coef_[0] * scaled, index=X_test.columns)

    proba = float(lr_pipeline.predict_proba(sample)[0, 1])
    predicted = class_names[int(proba >= 0.5)]

    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_k)
    colors = ["#2e9e5b" if v > 0 else "#d64545" for v in top.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Push toward '{class_names[1]}' (green) or '{class_names[0]}' (red)")
    ax.set_title(f"Local explanation: sample #{idx} -> predicted '{predicted}' ({proba:.1%} '{class_names[1]}')")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "sample_index": idx,
        "predicted_class": predicted,
        "probability_of_" + class_names[1]: round(proba, 4),
        "intercept": round(float(clf.intercept_[0]), 4),
        "top_contributions": {k: round(float(v), 4) for k, v in top.items()},
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, class_names = load_data(args.seed)
    if not 0 <= args.sample_index < len(X_test):
        raise SystemExit(f"--sample-index must be between 0 and {len(X_test) - 1}")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    metrics = {
        "logistic_regression": evaluate(lr, X_test, y_test),
        "random_forest": evaluate(rf, X_test, y_test),
    }
    print("Model performance on the held-out test set")
    print(pd.DataFrame(metrics).T.to_string())

    top_global = plot_global_importance(rf, X_test, y_test, args.top_k, args.seed,
                                        out / "global_importance.png")
    print(f"\nTop {args.top_k} global features (Random Forest):")
    print(top_global.round(4).to_string())

    local = explain_one_sample(lr, X_test, args.sample_index, class_names, args.top_k,
                               out / "local_explanation.png")
    true_label = class_names[int(y_test.iloc[args.sample_index])]
    print(f"\nSample #{args.sample_index}: predicted '{local['predicted_class']}', true label '{true_label}'")

    report = {"metrics": metrics, "global_top_features": top_global.round(4).to_dict(),
              "local_explanation": local, "true_label": true_label}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved plots and report.json to: {out.resolve()}")


if __name__ == "__main__":
    main()
