"""
Converts a borrower's model-driven risk score + SHAP top-reasons into a
plain-English underwriting narrative, using the Claude API.

This is the "make the black box explainable" layer — same idea as an
LLM-as-judge, applied to risk explanation instead of text evaluation.

Requires: ANTHROPIC_API_KEY environment variable.
Run: python src/narrative.py            # narrates the top 15 High-risk borrowers
     python src/narrative.py --id 42    # narrates one specific borrower
"""
import argparse
import json
import os

import pandas as pd

FEATURE_DESCRIPTIONS = {
    "utilization_now": "current credit utilization",
    "utilization_trend_3m": "change in utilization over the last 3 months",
    "pay_status_now": "current repayment status (months delinquent)",
    "pay_status_trend_3m": "change in repayment status over the last 3 months",
    "is_late_now": "currently in a late-payment state",
    "is_late_trend_3m": "change in late-payment frequency",
    "late_count_6m": "number of late months in the last 6 months",
    "max_pay_status_6m": "worst delinquency (months late) in the last 6 months",
    "utilization_volatility": "volatility of utilization month to month",
    "avg_payment_amount": "average monthly payment amount",
    "payment_to_balance_ratio": "ratio of payments made to balance owed",
    "headroom": "unused credit headroom",
    "balance_now": "current balance",
    "credit_limit_now": "credit limit",
    "age_now": "age",
    "sex_now": "sex",
    "education_now": "education level",
    "marriage_now": "marital status",
}


def build_prompt(borrower_row: pd.Series) -> str:
    reasons = json.loads(borrower_row["top_reasons"])
    reason_lines = []
    for r in reasons:
        label = FEATURE_DESCRIPTIONS.get(r["feature"], r["feature"])
        direction = "increasing" if r["shap_value"] > 0 else "decreasing"
        reason_lines.append(
            f"- {label}: current value {r['feature_value']}, "
            f"{direction} predicted risk (SHAP contribution {r['shap_value']})"
        )
    reasons_text = "\n".join(reason_lines)

    return f"""You are a credit risk analyst assistant. Write a 2-3 sentence plain-English
early-warning narrative for an underwriter reviewing this account. Be concrete and cite
the actual numbers. Do not recommend an action (approve/deny) — just explain the risk
picture. Avoid hedging language like "may" repeated more than once.

Borrower risk score: {borrower_row['risk_score']:.1%} probability of default
Risk tier: {borrower_row['risk_tier']}

Top contributing factors (from SHAP model explanation):
{reasons_text}
"""


def template_narrative(borrower_row: pd.Series) -> str:
    """Fallback narrative with no API call needed — used by the dashboard by
    default so it runs out-of-the-box. Swap for call_claude() once you have
    an ANTHROPIC_API_KEY to see the LLM-generated version."""
    reasons = json.loads(borrower_row["top_reasons"])
    top = reasons[0]
    label = FEATURE_DESCRIPTIONS.get(top["feature"], top["feature"])
    other_labels = [FEATURE_DESCRIPTIONS.get(r["feature"], r["feature"]) for r in reasons[1:]]
    return (
        f"This account is flagged {borrower_row['risk_tier']} risk at "
        f"{borrower_row['risk_score']:.1%} predicted probability of default, "
        f"driven primarily by {label} (current value {top['feature_value']}). "
        f"Secondary contributors: {', '.join(other_labels)}."
    )


def call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=None, help="Narrate one borrower_id")
    parser.add_argument("--n", type=int, default=15, help="How many top-risk borrowers to narrate")
    args = parser.parse_args()

    scored = pd.read_csv("outputs/real_scored_borrowers.csv")

    if args.id is not None:
        targets = scored[scored["borrower_id"] == args.id]
    else:
        targets = scored[scored["risk_tier"] == "High"].sort_values(
            "risk_score", ascending=False
        ).head(args.n)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — printing prompts only (no API call).\n")
        for _, row in targets.iterrows():
            print(f"=== Borrower {row['borrower_id']} ===")
            print(build_prompt(row))
            print()
        return

    narratives = {}
    for _, row in targets.iterrows():
        prompt = build_prompt(row)
        text = call_claude(prompt)
        narratives[int(row["borrower_id"])] = text
        print(f"=== Borrower {row['borrower_id']} ({row['risk_score']:.1%} risk) ===")
        print(text)
        print()

    with open("outputs/narratives.json", "w") as f:
        json.dump(narratives, f, indent=2)


if __name__ == "__main__":
    main()
