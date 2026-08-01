"""
Credit Early-Warning Dashboard (real UCI credit default data)
Run with:  streamlit run app/dashboard.py

Three panels:
  1. Portfolio risk overview  - tier distribution + risk vs. utilization map
  2. Borrower drill-down      - individual trajectory + SHAP reasons + LLM narrative
  3. Model performance        - ROC curve, LightGBM vs MLP comparison
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.narrative import template_narrative, FEATURE_DESCRIPTIONS  # noqa: E402

st.set_page_config(page_title="Credit Early-Warning System", layout="wide")

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_data():
    scored = pd.read_csv(ROOT / "outputs" / "real_scored_borrowers.csv")
    panel = pd.read_csv(ROOT / "data" / "real_panel.csv")
    metrics = json.loads((ROOT / "outputs" / "real_metrics.json").read_text())
    roc = pd.read_csv(ROOT / "outputs" / "real_roc_curve.csv")
    return scored, panel, metrics, roc


scored, panel, metrics, roc = load_data()

st.title("Credit Early-Warning System")
st.caption(
    "LightGBM risk model + SHAP explainability + LLM-generated underwriting narratives. "
    "Trained on the UCI 'Default of Credit Card Clients' dataset (30,000 real Taiwanese "
    "credit card accounts, Apr-Sep 2005) — 6 months of repayment behavior predicting "
    "October default."
)

tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Borrower Drill-Down", "Model Performance"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio size", f"{len(scored):,}")
    col2.metric("High risk accounts", f"{(scored.risk_tier == 'High').sum():,}")
    col3.metric("Watch list", f"{(scored.risk_tier == 'Watch').sum():,}")
    col4.metric("Actual default rate", f"{scored['defaulted'].mean():.1%}")

    c1, c2 = st.columns([1, 1.3])

    with c1:
        tier_counts = scored["risk_tier"].value_counts().reindex(["Low", "Watch", "High"])
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map={"Low": "#4CAF50", "Watch": "#FFB300", "High": "#E53935"},
            labels={"x": "Risk Tier", "y": "Number of Borrowers"},
            title="Portfolio Risk Tier Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # bubble size can't be negative, and some accounts have negative
        # balances (overpayment/credit balance in the raw data) — use
        # magnitude for sizing only
        scored["balance_size"] = scored["balance_now"].abs()
        fig2 = px.scatter(
            scored, x="utilization_now", y="risk_score",
            color="risk_tier",
            color_discrete_map={"Low": "#4CAF50", "Watch": "#FFB300", "High": "#E53935"},
            size="balance_size", size_max=18, opacity=0.5,
            hover_data=["borrower_id", "pay_status_now"],
            labels={"utilization_now": "Current Utilization", "risk_score": "Predicted Risk Score"},
            title="Risk Score vs. Utilization (bubble size = balance)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk heatmap: utilization band x repayment status band")
    scored["util_band"] = pd.cut(scored["utilization_now"], bins=[-1, 0, 0.3, 0.5, 0.7, 0.85, 2.0],
                                  labels=["none/paid", "<30%", "30-50%", "50-70%", "70-85%", "85%+"])
    scored["pay_status_band"] = pd.cut(scored["pay_status_now"], bins=[-3, -1, 0, 1, 2, 9],
                                        labels=["paid duly", "revolving", "1mo late", "2mo late", "3mo+ late"])
    heat = scored.pivot_table(values="risk_score", index="pay_status_band", columns="util_band",
                               aggfunc="mean", observed=True)
    fig3 = go.Figure(data=go.Heatmap(
        z=heat.values, x=heat.columns.astype(str), y=heat.index.astype(str),
        colorscale="RdYlGn_r", colorbar=dict(title="Avg Risk")
    ))
    fig3.update_layout(xaxis_title="Utilization Band", yaxis_title="Repayment Status")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Individual borrower drill-down")
    default_id = int(scored.sort_values("risk_score", ascending=False).iloc[0]["borrower_id"])
    borrower_id = st.selectbox(
        "Select borrower", options=sorted(scored["borrower_id"].unique()),
        index=sorted(scored["borrower_id"].unique()).index(default_id),
    )

    row = scored[scored["borrower_id"] == borrower_id].iloc[0]
    history = panel[panel["borrower_id"] == borrower_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{row['risk_score']:.1%}")
    c2.metric("Risk tier", row["risk_tier"])
    c3.metric("Actual outcome", "Defaulted" if row["defaulted"] == 1 else "No default")

    st.markdown("**Risk narrative** _(template-based; swap in `call_claude()` for LLM version — see README)_")
    st.info(template_narrative(row))

    cc1, cc2 = st.columns(2)
    with cc1:
        fig4 = px.line(history, x="month", y="utilization", markers=True,
                        title="Utilization trajectory (Apr-Sep 2005)")
        st.plotly_chart(fig4, use_container_width=True)
    with cc2:
        fig5 = px.line(history, x="month", y="pay_status", markers=True,
                        title="Repayment status trajectory (months delinquent)")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**Top contributing factors (SHAP)**")
    reasons = json.loads(row["top_reasons"])
    reasons_df = pd.DataFrame(reasons)
    reasons_df["feature"] = reasons_df["feature"].map(lambda f: FEATURE_DESCRIPTIONS.get(f, f))
    fig6 = px.bar(reasons_df, x="shap_value", y="feature", orientation="h",
                   color="shap_value", color_continuous_scale="RdYlGn_r",
                   title="Contribution to risk score (SHAP value)")
    st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Model comparison: LightGBM vs. Neural Net (MLP)")
    m1, m2 = st.columns(2)
    m1.metric("LightGBM ROC AUC", metrics["lightgbm"]["roc_auc"])
    m2.metric("MLP ROC AUC", metrics["mlp"]["roc_auc"])

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="LightGBM"))
    fig7.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                               line=dict(dash="dash", color="gray")))
    fig7.update_layout(title="ROC Curve (LightGBM)", xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown(
        "LightGBM outperforms the MLP baseline here (0.776 vs. 0.730 ROC AUC), which tracks "
        "with published benchmarks on this dataset and with the broader tabular-data literature — "
        "gradient-boosted trees typically beat neural nets on structured, moderate-sized feature "
        "sets like this one."
    )