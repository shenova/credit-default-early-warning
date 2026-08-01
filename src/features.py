"""
Turns the raw month-by-month panel into one row per borrower with:
  - a snapshot of their most recent state (as of month 9, so the model is
    predicting a month-12 default using only data through month 9 — this
    is what makes it an "early warning" model rather than a lagging one)
  - trend features (deltas/slopes over the prior 3 months) that capture
    DIRECTION of change, which is the actual early-warning signal.
    A borrower at 70% utilization who was steady is very different from
    one who was at 40% three months ago and is climbing fast.
"""
import numpy as np
import pandas as pd

CUTOFF_MONTH = 8  # use months 0-8 (9 months) to predict month-12 outcome
TREND_WINDOW = 3   # months 6,7,8 vs months 3,4,5 for trend calc


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    hist = panel[panel["month"] <= CUTOFF_MONTH].copy()

    snap = (
        hist[hist["month"] == CUTOFF_MONTH]
        .set_index("borrower_id")[
            ["income", "credit_limit", "balance", "utilization",
             "credit_score", "age", "late_streak", "days_late"]
        ]
        .add_suffix("_now")
    )

    recent = hist[hist["month"].between(CUTOFF_MONTH - TREND_WINDOW + 1, CUTOFF_MONTH)]
    prior = hist[hist["month"].between(CUTOFF_MONTH - 2 * TREND_WINDOW + 1, CUTOFF_MONTH - TREND_WINDOW)]

    recent_avg = recent.groupby("borrower_id")[["utilization", "credit_score", "is_late"]].mean()
    prior_avg = prior.groupby("borrower_id")[["utilization", "credit_score", "is_late"]].mean()

    trend = (recent_avg - prior_avg).add_suffix("_trend_3m")

    late_count = hist.groupby("borrower_id")["is_late"].sum().rename("late_count_9m")
    max_days_late = hist.groupby("borrower_id")["days_late"].max().rename("max_days_late_9m")
    util_volatility = hist.groupby("borrower_id")["utilization"].std().rename("utilization_volatility")

    features = (
        snap.join(trend, how="left")
        .join(late_count, how="left")
        .join(max_days_late, how="left")
        .join(util_volatility, how="left")
    )
    features = features.fillna(0.0)

    # a few explicit ratio/derived features that are easy to explain in a narrative
    features["dti_proxy"] = features["balance_now"] / (features["income_now"] / 12 + 1e-6)
    features["headroom"] = 1 - features["utilization_now"]

    return features.reset_index()


def main():
    panel = pd.read_csv("data/credit_panel.csv")
    borrowers = pd.read_csv("data/borrowers.csv")

    feats = build_features(panel)
    dataset = feats.merge(borrowers, on="borrower_id")
    dataset.to_csv("data/model_dataset.csv", index=False)
    print(f"Feature table: {dataset.shape[0]} rows x {dataset.shape[1]} cols")
    print(dataset["defaulted"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
