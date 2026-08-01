"""
Trains and compares LightGBM (primary) vs. an MLP (neural net comparison)
on the real UCI credit default dataset, with SHAP explainability — same
architecture as the synthetic version, run on real data this time.
"""
import json

import joblib
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def load_dataset():
    df = pd.read_csv("data/real_model_dataset.csv")
    y = df["defaulted"]
    X = df.drop(columns=["defaulted", "borrower_id"])
    return df, X, y


def main():
    df, X, y = load_dataset()
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lgbm = LGBMClassifier(
        n_estimators=400, learning_rate=0.04, num_leaves=25,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]

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
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("outputs/real_roc_curve.csv", index=False)

    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["borrower_id"] = df["borrower_id"].values

    full_proba = lgbm.predict_proba(X)[:, 1]
    scored = df[["borrower_id"] + feature_cols + ["defaulted"]].copy()
    scored["risk_score"] = full_proba
    low_cut, high_cut = scored["risk_score"].quantile([0.60, 0.90])
    scored["risk_tier"] = pd.cut(
        scored["risk_score"], bins=[-0.01, low_cut, high_cut, 1.0],
        labels=["Low", "Watch", "High"]
    )
    scored = scored.merge(shap_df, on="borrower_id")

    shap_cols = [f"shap_{c}" for c in feature_cols]

    def top_reasons(row):
        vals = row[shap_cols].astype(float)
        top = vals.reindex(vals.abs().sort_values(ascending=False).index[:3])
        return json.dumps([
            {"feature": k.replace("shap_", ""), "shap_value": round(v, 4),
             "feature_value": round(float(row[k.replace('shap_', '')]), 4)}
            for k, v in top.items()
        ])

    scored["top_reasons"] = scored.apply(top_reasons, axis=1)
    scored.drop(columns=shap_cols).to_csv("outputs/real_scored_borrowers.csv", index=False)

    joblib.dump(lgbm, "models/lgbm_real_model.joblib")
    with open("outputs/real_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {len(scored)} scored borrowers")
    print(scored["risk_tier"].value_counts())


if __name__ == "__main__":
    main()
