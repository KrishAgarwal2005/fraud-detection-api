"""Train and compare fraud models, pick the best one, and save it for the API.

Run from the project folder:
    python -m fraud.train --data data/creditcard.csv

Steps:
1. Sort transactions by time and split them into train (first 64%),
   validation (next 16%) and test (last 20%). Splitting by time, not at random,
   is closer to real life: the model is always tested on transactions that
   happened after the ones it learned from.
2. Train four models. Fraud is only 0.17% of rows, so each model is told to
   care about the rare class, either with class weights or with SMOTE.
   SMOTE is applied to the training data only, inside the pipeline.
3. Pick the decision threshold on the validation set using a simple cost rule:
   a missed fraud costs COST_MISS times as much as a false alarm.
4. Choose the model with the best validation PR-AUC, report everything on the
   untouched test set, and save the winner.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud.features import FEATURES, add_features

RANDOM_STATE = 42


def time_split(df: pd.DataFrame):
    """Split by time order: 64% train, 16% validation, 20% test."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.64), int(n * 0.80)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def build_models(n_negative: int, n_positive: int) -> dict:
    """The four models we compare."""
    return {
        "Logistic Regression (class weights)": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "Logistic Regression (SMOTE)": ImbPipeline([
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),  # runs on training data only
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Random Forest (class weights)": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_negative / n_positive,  # how much more a fraud counts
            eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def pick_threshold(y_true, proba, cost_miss: float) -> float:
    """Return the threshold with the lowest total cost on this data.

    cost = (missed frauds x cost_miss) + (false alarms x 1)
    """
    best_t, best_cost = 0.5, float("inf")
    # Steps of 0.01, plus finer steps near 1.0 (weighted models give high scores).
    candidates = np.concatenate([np.arange(0.01, 0.99, 0.01), np.arange(0.990, 1.0, 0.001)])
    for t in candidates:
        pred = proba >= t
        missed = int(((pred == 0) & (y_true == 1)).sum())
        false_alarms = int(((pred == 1) & (y_true == 0)).sum())
        cost = missed * cost_miss + false_alarms
        if cost < best_cost:
            best_t, best_cost = float(round(t, 3)), cost
    return best_t


def scores(y_true, proba, threshold: float) -> dict:
    """All the numbers we report for one model at one threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--cost-miss", type=float, default=10.0,
                        help="how many false alarms one missed fraud is worth")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train, val, test = time_split(df)
    X_train, y_train = add_features(train), train["Class"].to_numpy()
    X_val, y_val = add_features(val), val["Class"].to_numpy()
    X_test, y_test = add_features(test), test["Class"].to_numpy()

    split_info = {name: {"rows": len(part), "frauds": int(part["Class"].sum())}
                  for name, part in [("train", train), ("validation", val), ("test", test)]}
    print("Split:", split_info)
    # Why accuracy is useless here: predicting "not fraud" every time is already this accurate.
    always_no = round(1 - y_test.mean(), 4)
    print(f"Accuracy of always saying 'not fraud' on test: {always_no:.2%}")

    n_pos = int(y_train.sum())
    models = build_models(len(y_train) - n_pos, n_pos)
    results = {}
    fitted = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        threshold = pick_threshold(y_val, val_proba, args.cost_miss)
        results[name] = {
            "validation_pr_auc": round(average_precision_score(y_val, val_proba), 4),
            "test_at_0.5": scores(y_test, test_proba, 0.5),
            "test_at_chosen_threshold": scores(y_test, test_proba, threshold),
        }
        fitted[name] = (model, threshold)
        print(f"  validation PR-AUC {results[name]['validation_pr_auc']}, "
              f"test at threshold {threshold}: {results[name]['test_at_chosen_threshold']}")

    best = max(results, key=lambda n: results[n]["validation_pr_auc"])
    model, threshold = fitted[best]
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES,
                 "model_name": best, "version": version, "cost_miss": args.cost_miss},
                args.model_out)
    print(f"Best model: {best} (threshold {threshold}), saved to {args.model_out}")

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    summary = {"split": split_info, "accuracy_if_always_not_fraud": always_no,
               "cost_miss": args.cost_miss, "best_model": best, "models": results}
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Model | Val PR-AUC | Test PR-AUC | Threshold | Precision | Recall | F1 | Frauds caught | False alarms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        t = r["test_at_chosen_threshold"]
        lines.append(f"| {name} | {r['validation_pr_auc']} | {t['pr_auc']} | {t['threshold']} | "
                     f"{t['precision']} | {t['recall']} | {t['f1']} | {t['tp']} of {t['tp'] + t['fn']} | {t['fp']} |")
    (reports / "model_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
