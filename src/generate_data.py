"""
Generates a synthetic but realistically-structured panel dataset of borrower
credit behavior over time. Each borrower has 12 months of history: balance,
utilization, payments, delinquency flags, etc. A subset of borrowers are
seeded with a "deteriorating" trajectory (rising utilization, slipping
payments) that leads to default in month 12 — this is the signal an
early-warning model should learn to catch *before* the hard default happens.

Why synthetic data: this project is designed to run anywhere without needing
to download an external dataset. Swap in a real dataset (e.g. Kaggle's
"Home Credit Default Risk" or Lending Club) by replacing this script's
output with the same schema (see README) — everything downstream
(features, model, dashboard) works unchanged.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BORROWERS = 4000
N_MONTHS = 12
DEFAULT_RATE_TARGET = 0.11  # realistic-ish retail credit default rate


def make_borrower(borrower_id: int):
    """Simulate one borrower's 12-month panel + whether they end in default."""
    # static attributes
    income = RNG.lognormal(mean=10.8, sigma=0.4)          # ~ $45k median
    credit_limit = np.clip(RNG.normal(8000, 4000), 500, 40000)
    base_score = int(np.clip(RNG.normal(680, 60), 300, 850))
    age = int(np.clip(RNG.normal(38, 12), 21, 75))

    # decide trajectory type
    # 'stable' = normal noisy behavior, 'deteriorating' = rising risk signal
    is_deteriorating = RNG.random() < DEFAULT_RATE_TARGET * 1.6
    drift = RNG.uniform(0.03, 0.09) if is_deteriorating else RNG.uniform(-0.01, 0.01)

    balance = credit_limit * RNG.uniform(0.1, 0.35)
    score = base_score
    rows = []
    late_streak = 0
    for m in range(N_MONTHS):
        # utilization drifts up for deteriorating borrowers, noisy otherwise
        util_shock = RNG.normal(0, 0.02)
        balance = np.clip(balance * (1 + drift + util_shock), 0, credit_limit * 1.1)
        utilization = balance / credit_limit

        # payment behavior worsens as utilization climbs, for deteriorating group
        late_prob = 0.03 + (0.35 if is_deteriorating and utilization > 0.55 else 0)
        is_late = RNG.random() < late_prob
        days_late = int(RNG.integers(5, 60)) if is_late else 0
        late_streak = late_streak + 1 if is_late else 0

        min_due = balance * 0.02
        payment = min_due * RNG.uniform(0.5, 1.0) if is_late else min_due * RNG.uniform(1.0, 3.0)

        # score reacts to behavior with a lag
        score += RNG.normal(0, 3) - (6 if is_late else -1) - (utilization > 0.7) * 2
        score = int(np.clip(score, 300, 850))

        rows.append(dict(
            borrower_id=borrower_id, month=m, income=round(income, 2),
            credit_limit=round(credit_limit, 2), balance=round(balance, 2),
            utilization=round(utilization, 4), payment_amount=round(payment, 2),
            min_due=round(min_due, 2), is_late=int(is_late), days_late=days_late,
            late_streak=late_streak, credit_score=score, age=age,
        ))

    # default label: deteriorating borrowers with high late_streak by month 12 default
    final = rows[-1]
    default_prob = 0.02
    if is_deteriorating:
        default_prob = 0.35 + 0.12 * final["late_streak"] + 0.3 * (final["utilization"] > 0.7)
    defaulted = int(RNG.random() < np.clip(default_prob, 0, 0.95))

    return rows, defaulted


def main():
    panel_rows = []
    labels = []
    for bid in range(N_BORROWERS):
        rows, defaulted = make_borrower(bid)
        panel_rows.extend(rows)
        labels.append(dict(borrower_id=bid, defaulted=defaulted))

    panel = pd.DataFrame(panel_rows)
    borrowers = pd.DataFrame(labels)

    panel.to_csv("data/credit_panel.csv", index=False)
    borrowers.to_csv("data/borrowers.csv", index=False)

    rate = borrowers["defaulted"].mean()
    print(f"Generated {N_BORROWERS} borrowers x {N_MONTHS} months.")
    print(f"Default rate: {rate:.2%}")


if __name__ == "__main__":
    main()
