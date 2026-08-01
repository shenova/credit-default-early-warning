"""
Trains and compares two models on the early-warning task:
  1. LightGBM  - gradient-boosted trees, the industry-standard baseline
                 for tabular credit risk models
  2. MLP       - a small neural net, to demonstrate the deep-learning
                 comparison point (swap in a PyTorch LSTM over the raw
                 monthly panel for a fuller sequence model — see README)

Saves: trained LightGBM model, SHAP values, metrics, and a scored
dataset (predictions + SHAP top-reasons per borrower) that the
dashboard and the LLM narrative layer both consume.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

FEATURE_COLS = None  # set at runtime


def load_dataset():
    df = pd.read_csv("data/model_dataset.csv")
    y = df["defaulted"]
    X = df.drop(columns=["defaulted", "borrower_id"])
    return df, X, y


def main():
    df, X, y = load_dataset()
    global FEATURE_COLS
    FEATURE_COLS = list(X.columns)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df["borrower_id"], test_size=0.25, random_state=42, stratify=y
    )

    # --- Model 1: LightGBM (primary model) ---
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]

    # --- Model 2: MLP (neural net comparison) ---
    scaler = StandardScaler().fit(X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
        max_iter=500, random_state=42,
    )
    mlp.fit(scaler.transform(X_train), y_train)
    mlp_proba = mlp.predict_proba(scaler.transform(X_test))[:, 1]

    metrics = {
        "lightgbm": {
            "roc_auc": round(roc_auc_score(y_test, lgbm_proba), 4),
            "pr_auc": round(average_precision_score(y_test, lgbm_proba), 4),
        },
        "mlp": {
            "roc_auc": round(roc_auc_score(y_test, mlp_proba), 4),
            "pr_auc": round(average_precision_score(y_test, mlp_proba), 4),
        },
    }
    print(json.dumps(metrics, indent=2))

    fpr, tpr, _ = roc_curve(y_test, lgbm_proba)
    roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_curve_df.to_csv("outputs/roc_curve.csv", index=False)

    # --- SHAP explainability (on the full dataset, for the dashboard) ---
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # older shap API returns list for binary
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLS])
    shap_df["borrower_id"] = df["borrower_id"].values

    full_proba = lgbm.predict_proba(X)[:, 1]
    scored = df[["borrower_id"] + FEATURE_COLS + ["defaulted"]].copy()
    scored["risk_score"] = full_proba
    scored["risk_tier"] = pd.cut(
        scored["risk_score"], bins=[-0.01, 0.1, 0.3, 1.0],
        labels=["Low", "Watch", "High"]
    )

    scored = scored.merge(shap_df, on="borrower_id")

    # top 3 SHAP reasons per borrower (feature name + contribution), for narratives
    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]

    def top_reasons(row):
        vals = row[shap_cols].astype(float)
        top = vals.reindex(vals.abs().sort_values(ascending=False).index[:3])
        return json.dumps([
            {"feature": k.replace("shap_", ""), "shap_value": round(v, 4),
             "feature_value": round(float(row[k.replace('shap_', '')]), 4)}
            for k, v in top.items()
        ])

    scored["top_reasons"] = scored.apply(top_reasons, axis=1)
    scored.drop(columns=shap_cols).to_csv("outputs/scored_borrowers.csv", index=False)

    joblib.dump(lgbm, "models/lgbm_model.joblib")
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {len(scored)} scored borrowers to outputs/scored_borrowers.csv")
    print(f"Risk tier distribution:\n{scored['risk_tier'].value_counts()}")


if __name__ == "__main__":
    main()
