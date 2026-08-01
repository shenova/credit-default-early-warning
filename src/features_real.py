"""
Feature engineering for the real UCI dataset. Unlike the synthetic version,
we don't need to hold out a prediction month — the target ("defaulted") is
already "did this account default the month AFTER the 6 months of observed
history," so we use the full 6-month panel (month 0-5) as the early-warning
window.

Snapshot = most recent month (month 5, September)
Trend     = recent 3 months (3,4,5) vs. prior 3 months (0,1,2)
"""
import numpy as np
import pandas as pd

TREND_WINDOW = 3


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    snap = (
        panel[panel["month"] == 5]
        .set_index("borrower_id")[
            ["credit_limit", "balance", "utilization", "pay_status",
             "is_late", "age", "sex", "education", "marriage"]
        ]
        .add_suffix("_now")
    )

    recent = panel[panel["month"] >= 6 - TREND_WINDOW]
    prior = panel[panel["month"] < 6 - TREND_WINDOW]

    recent_avg = recent.groupby("borrower_id")[["utilization", "pay_status", "is_late"]].mean()
    prior_avg = prior.groupby("borrower_id")[["utilization", "pay_status", "is_late"]].mean()
    trend = (recent_avg - prior_avg).add_suffix("_trend_3m")

    late_count = panel.groupby("borrower_id")["is_late"].sum().rename("late_count_6m")
    max_pay_status = panel.groupby("borrower_id")["pay_status"].max().rename("max_pay_status_6m")
    util_volatility = panel.groupby("borrower_id")["utilization"].std().rename("utilization_volatility")
    avg_payment = panel.groupby("borrower_id")["payment_amount"].mean().rename("avg_payment_amount")

    features = (
        snap.join(trend, how="left")
        .join(late_count, how="left")
        .join(max_pay_status, how="left")
        .join(util_volatility, how="left")
        .join(avg_payment, how="left")
    )
    features = features.fillna(0.0)

    features["headroom"] = 1 - features["utilization_now"]
    features["payment_to_balance_ratio"] = features["avg_payment_amount"] / (
        features["balance_now"].abs() + 1.0
    )

    return features.reset_index()


def main():
    panel = pd.read_csv("data/real_panel.csv")
    borrowers = pd.read_csv("data/real_borrowers.csv")

    feats = build_features(panel)
    dataset = feats.merge(borrowers, on="borrower_id")
    dataset.to_csv("data/real_model_dataset.csv", index=False)
    print(f"Feature table: {dataset.shape[0]} rows x {dataset.shape[1]} cols")
    print(dataset["defaulted"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
