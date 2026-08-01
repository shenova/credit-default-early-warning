# Credit Early-Warning System
### ML risk scoring + SHAP explainability + LLM-generated underwriting narratives

A portfolio project that predicts consumer credit default and explains **why**
in plain English — the same "make black-box model output interpretable"
problem as an LLM-as-judge system, applied to lending instead of text.

## Data source
This project runs on the real **UCI "Default of Credit Card Clients" dataset**
— 30,000 real Taiwanese credit card accounts, April-September 2005, 25
attributes, no missing values. It's one of the most widely used public
datasets for credit risk modeling (originally UCI Machine Learning
Repository, credited to I-Cheng Yeh; also mirrored on Kaggle as
`uciml/default-of-credit-card-clients-dataset`).

The target (`default payment next month`) is whether the account defaulted
in October 2005, so this is a genuine early-warning setup: 6 months of
repayment behavior (Apr-Sep) predicting the following month's outcome.

## Why this project
Most public credit-risk portfolio projects stop at "trained XGBoost on this
dataset, got some AUC." This one adds two things that map directly to what a
real credit risk / retail banking team needs:

1. **Trend features, not just a snapshot** — utilization drift and
   repayment-status trend over the last 3 months vs. the prior 3, on top of
   the current snapshot. A borrower steady at 70% utilization is a different
   risk than one who was at 40% three months ago and is climbing.
2. **An explanation layer** — SHAP feature attributions are converted into
   underwriter-readable narratives via Claude, so the output isn't just a
   score, it's something a human can act on.

## Architecture
```
src/prepare_real_data.py → reshapes the raw UCI wide-format CSV into a monthly panel
src/features_real.py     → snapshot + 3-month trend feature engineering
src/train_model_real.py  → LightGBM (primary) + MLP (DL comparison) + SHAP
src/narrative.py         → SHAP → plain-English risk narrative (template or Claude API)
app/dashboard.py         → Streamlit: portfolio view, borrower drill-down, model performance
```

`src/generate_data.py` / `src/features.py` / `src/train_model.py` are the
original synthetic-data versions of this pipeline — kept in case you want a
version that runs with zero data-download step, e.g. for a quick demo.

## Setup
```bash
 
```

To generate LLM narratives (optional — dashboard works without this using a
template fallback):
```bash
export ANTHROPIC_API_KEY=your_key_here
python src/narrative.py --n 15       # narrates the 15 highest-risk accounts
python src/narrative.py --id 42      # narrates one borrower
```

## Results (real data)
| Model | ROC AUC | PR AUC |
|---|---|---|
| LightGBM | 0.776 | 0.562 |
| MLP (neural net) | 0.730 | 0.481 |

0.776 ROC AUC is in line with published results on this dataset (most public
benchmarks land around 0.75-0.78), which is worth knowing so you don't
oversell the number — this is a well-studied dataset, not a case where
you've discovered a novel edge.

LightGBM beats the neural net here, consistent with the broader finding that
gradient-boosted trees tend to outperform neural nets on small/moderate
structured tabular data — a good discussion point in interviews.

## Honest limitations (know these before an interview)
- **Dataset is from 2005 and Taiwan-specific** — repayment behavior, credit
  norms, and macroeconomic context don't necessarily generalize to, say, a US
  retail bank in 2026. Frame this as a methodology demo, not a
  production-ready model.
- **No demographic fairness audit performed** — SEX, EDUCATION, MARRIAGE are
  in the raw data and were used as model features here for simplicity; a
  real deployment would need a disparate-impact review before using
  demographic attributes this directly.
- **"6 months of trend" is a short window** — real early-warning systems
  typically use 12-24 months of history where available.

## Extensions (good "what would you do next" answers in an interview)
- **Sequence model**: replace the trend-feature approach with a PyTorch LSTM
  or Temporal Fusion Transformer trained directly on the raw monthly panel
  (`data/real_panel.csv`) instead of hand-engineered trend features.
- **Calibration**: add isotonic/Platt calibration so `risk_score` can be read
  as a true probability, not just a ranking score.
- **Fairness audit**: slice model performance and SHAP drivers by SEX/
  EDUCATION/MARRIAGE to check for disparate impact, and consider dropping
  demographic features from the model entirely in favor of behavioral-only
  signals — standard practice before any credit model goes into production.
- **Cost-sensitive threshold**: tune the High/Watch/Low cutoffs against an
  actual cost matrix (cost of a missed default vs. cost of a false flag)
  instead of percentile-based bands.

## Resume framing
> Built an end-to-end credit early-warning system (LightGBM + SHAP + LLM
> explainability layer) on the UCI Default of Credit Card Clients dataset
> (30K real accounts), achieving 0.776 ROC AUC — in line with published
> benchmarks — using 3-month behavioral trend features to flag at-risk
> accounts before default; added an LLM layer generating underwriter-ready
> risk narratives via the Claude API, visualized through an interactive
> Streamlit dashboard with portfolio and account-level views.
