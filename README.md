# 🏦 Credit Early-Warning System

**ML risk scoring + SHAP explainability + LLM-generated underwriting narratives**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-success)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A machine learning system that predicts consumer credit default risk and explains
**why** in plain English — combining a gradient-boosted risk model, SHAP-based
explainability, and Claude-generated underwriting narratives, all surfaced through
an interactive dashboard.

> Trained and validated on the real **UCI Default of Credit Card Clients dataset**
> (30,000 real accounts) — **0.776 ROC AUC**, in line with published benchmarks.

<!-- 📸 Add a dashboard screenshot here once you have one:
![Dashboard screenshot](docs/dashboard_screenshot.png)
-->

---

## Table of Contents
- [Why This Project](#why-this-project)
- [Data Source](#data-source)
- [Architecture](#architecture)
- [Setup](#setup)
- [Results](#results)
- [Dashboard](#dashboard)
- [Limitations](#limitations)
- [Extensions](#extensions)
- [Resume Framing](#resume-framing)

---

## Why This Project

Most public credit-risk portfolio projects stop at "trained a model, got an AUC
score." This one adds two things that map directly to what a real credit risk /
retail banking team actually needs:

1. **Trend features, not just a snapshot** — utilization drift and repayment-status
   change over the last 3 months vs. the prior 3, on top of the current snapshot.
   A borrower steady at 70% utilization is a very different risk than one who was
   at 40% three months ago and is climbing.
2. **An explanation layer** — SHAP feature attributions are converted into
   underwriter-readable risk narratives via the Claude API, so the output isn't
   just a score — it's something a human can act on.

## Data Source

This project runs on the real **[UCI "Default of Credit Card Clients" dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**
— 30,000 real Taiwanese credit card accounts, April–September 2005, 25
attributes, no missing values. It's one of the most widely used public datasets
for credit risk modeling (credited to I-Cheng Yeh; also mirrored on Kaggle as
[`uciml/default-of-credit-card-clients-dataset`](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)).

The target (`default payment next month`) is whether the account defaulted in
October 2005 — a genuine early-warning setup: 6 months of repayment behavior
(Apr–Sep) predicting the following month's outcome.

`data/uci_credit_default_raw.csv` is included in this repo as the original
wide-format file.

## Architecture

```
src/prepare_real_data.py → reshapes the raw UCI wide-format CSV into a monthly panel
src/features_real.py     → snapshot + 3-month trend feature engineering
src/train_model_real.py  → LightGBM (primary) + MLP (DL comparison) + SHAP
src/narrative.py         → SHAP → plain-English risk narrative (template or Claude API)
app/dashboard.py         → Streamlit: portfolio view, borrower drill-down, model performance
```

`src/generate_data.py` / `src/features.py` / `src/train_model.py` are a
synthetic-data version of the same pipeline, useful for a quick demo with zero
data-prep step.

## Setup

```bash
git clone <your-repo-url>
cd credit-early-warning-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python src/prepare_real_data.py   # reshapes raw CSV into a monthly panel
python src/features_real.py       # builds trend/snapshot features
python src/train_model_real.py    # trains models, computes SHAP, scores every account

streamlit run app/dashboard.py    # opens the dashboard at localhost:8501
```

**Optional — LLM-generated narratives** (dashboard works fine without this,
using a template fallback):
```bash
export ANTHROPIC_API_KEY=your_key_here
python src/narrative.py --n 15       # narrates the 15 highest-risk accounts
python src/narrative.py --id 42      # narrates one specific borrower
```

## Results

| Model | ROC AUC | PR AUC |
|---|---|---|
| **LightGBM** | **0.776** | **0.562** |
| MLP (neural net) | 0.730 | 0.481 |

LightGBM beats the neural net, consistent with the broader finding that
gradient-boosted trees tend to outperform neural nets on small/moderate
structured tabular data. 0.776 ROC AUC is in line with published benchmarks
on this dataset (most public work lands ~0.75–0.78) — a well-studied dataset,
not a case of a novel edge.

## Dashboard

Three tabs, built in Streamlit + Plotly:

- **Portfolio Overview** — risk-tier distribution, risk-vs-utilization scatter,
  a utilization × repayment-status risk heatmap
- **Borrower Drill-Down** — individual risk score, utilization/repayment
  trajectory over time, top SHAP-driven risk factors, and a plain-English
  risk narrative
- **Model Performance** — ROC curve and LightGBM vs. MLP comparison

## Limitations

Worth knowing before discussing this in an interview:

- **Dataset is from 2005 and Taiwan-specific** — repayment behavior and
  macroeconomic context don't necessarily generalize to a present-day retail
  bank elsewhere. This is a methodology demo, not a production-ready model.
- **No demographic fairness audit performed** — `SEX`, `EDUCATION`, `MARRIAGE`
  are used as model features here for simplicity; a real deployment would need
  a disparate-impact review before using demographic attributes this directly.
- **6-month trend window is short** — real early-warning systems typically use
  12–24 months of history where available.

## Extensions

Good "what would you do next" answers for an interview:

- **Sequence model** — replace hand-engineered trend features with a PyTorch
  LSTM or Temporal Fusion Transformer trained directly on the raw monthly panel
- **Calibration** — add isotonic/Platt calibration so `risk_score` reads as a
  true probability, not just a ranking score
- **Fairness audit** — slice model performance and SHAP drivers by demographic
  attributes to check for disparate impact, and consider a behavioral-only
  feature set for production
- **Cost-sensitive thresholds** — tune risk-tier cutoffs against an actual cost
  matrix (missed default vs. false flag) instead of percentile-based bands


*Built by Shenova Davis · [LinkedIn](https://linkedin.com/in/shenova-davis) · [GitHub](https://github.com/shenova)*
