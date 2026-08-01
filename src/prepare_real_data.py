"""
Loads the real UCI "Default of Credit Card Clients" dataset (30,000 Taiwanese
credit card accounts, Apr-Sep 2005) and reshapes it from wide format
(one row per client, 6 months of columns) into a long panel format
(one row per client-month) — the same shape our early-warning pipeline expects.

Source: UCI Machine Learning Repository / I-Cheng Yeh, "Default of Credit
Card Clients Dataset," mirrored on Kaggle (uciml/default-of-credit-card-
clients-dataset) and pulled here via a GitHub mirror
(Navneet2409/credit-card-default-prediction). See README for citation.

Original columns per client:
  PAY_0..PAY_6      repayment status per month (-2/-1 = paid/no balance, 
                     0 = revolving, 1-9 = months delinquent), Sep->Apr
  BILL_AMT1..6       bill statement amount per month, Sep->Apr
  PAY_AMT1..6        amount actually paid per month, Sep->Apr
  LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE   static attributes
  default payment next month                 target (defaulted in Oct 2005)

We reshape the 6 months (Apr=month 0 ... Sep=month 5) into a panel so the
SAME feature-engineering / trend logic used in the synthetic version applies
directly to real data: use months 0-5 behavior to predict the (already
given) October default outcome.
"""
import numpy as np
import pandas as pd

# NOTE: PAY_0 is actually September (not "month 0"), PAY_2..PAY_6 are Aug->Apr.
# Chronological order oldest -> newest: PAY_6(Apr), PAY_5(May), PAY_4(Jun),
# PAY_3(Jul), PAY_2(Aug), PAY_0(Sep)
MONTH_ORDER = [6, 5, 4, 3, 2, 0]  # oldest to newest, matches original column suffixes


def main():
    raw = pd.read_csv("data/uci_credit_default_raw.csv", skiprows=1)
    raw = raw.rename(columns={"default payment next month": "defaulted"})

    panel_rows = []
    for _, r in raw.iterrows():
        for month_idx, suffix in enumerate(MONTH_ORDER):
            pay_status = r[f"PAY_{suffix}"]
            bill_amt = r[f"BILL_AMT{month_idx + 1}"]
            pay_amt = r[f"PAY_AMT{month_idx + 1}"]
            panel_rows.append(dict(
                borrower_id=int(r["ID"]),
                month=month_idx,
                credit_limit=float(r["LIMIT_BAL"]),
                balance=float(bill_amt),
                payment_amount=float(pay_amt),
                pay_status=int(pay_status),          # -2..-1 ok, 0 revolving, 1+ = months delinquent
                is_late=int(pay_status >= 1),
                days_late_proxy=int(max(pay_status, 0)) * 30,  # months delinquent -> rough day proxy
                age=int(r["AGE"]),
                sex=int(r["SEX"]),
                education=int(r["EDUCATION"]),
                marriage=int(r["MARRIAGE"]),
            ))

    panel = pd.DataFrame(panel_rows)
    # utilization can be negative/>1 in raw data (overpayment, disputed charges) — clip for modeling sanity
    panel["utilization"] = (panel["balance"] / panel["credit_limit"]).clip(-0.5, 2.0)

    borrowers = raw[["ID", "defaulted"]].rename(columns={"ID": "borrower_id"})

    panel.to_csv("data/real_panel.csv", index=False)
    borrowers.to_csv("data/real_borrowers.csv", index=False)

    print(f"Panel: {panel.shape[0]} rows ({raw.shape[0]} borrowers x 6 months)")
    print(f"Default rate: {borrowers['defaulted'].mean():.2%}")


if __name__ == "__main__":
    main()
