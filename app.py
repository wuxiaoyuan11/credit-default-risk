from __future__ import annotations

from pathlib import Path
import math
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


st.set_page_config(
    page_title="Credit Default Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .insight-box {
        background: #eef5ff;
        border-left: 5px solid #2f6fbd;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 1.3rem 0;
        color: #17324d;
        font-size: 1.02rem;
        line-height: 1.55;
    }
    .section-note {
        color: #5f6673;
        font-size: 0.96rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(REPORTS / "model_metrics.csv"),
        "decile": pd.read_csv(REPORTS / "decile_lift_gains.csv"),
        "thresholds": pd.read_csv(REPORTS / "threshold_business_decisions.csv"),
        "test_scores": pd.read_csv(REPORTS / "test_scores.csv"),
        "scorecard_compare": pd.read_csv(REPORTS / "scorecard_vs_ml_comparison.csv"),
        "scorecard_points": pd.read_csv(REPORTS / "scorecard_points.csv"),
        "scorecard_params": pd.read_csv(REPORTS / "scorecard_scaling_params.csv"),
        "scorecard_scores": pd.read_csv(REPORTS / "scorecard_test_scores.csv"),
        "shap": pd.read_csv(REPORTS / "shap_importance.csv"),
        "iv": pd.read_csv(REPORTS / "iv_summary.csv"),
        "calibration": pd.read_csv(REPORTS / "calibration_table.csv"),
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def insight(text: str) -> None:
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


FEATURE_INFO = {
    "total_past_due_events": {
        "label": "Total Past-Due Events",
        "source": "Engineered",
        "meaning": "Combined count of 30-59, 60-89, and 90+ day delinquency events.",
    },
    "has_past_due": {
        "label": "Has Any Past-Due History",
        "source": "Engineered",
        "meaning": "Flag indicating whether the borrower has any prior delinquency event.",
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Revolving Credit Utilization",
        "source": "Raw",
        "meaning": "Share of unsecured revolving credit lines currently used.",
    },
    "credit_utilization": {
        "label": "Credit Utilization",
        "source": "Raw alias",
        "meaning": "Readable alias of the raw revolving credit utilization field.",
    },
    "high_credit_utilization": {
        "label": "High Credit Utilization Flag",
        "source": "Engineered",
        "meaning": "Flag for utilization above 80%.",
    },
    "age": {
        "label": "Age",
        "source": "Raw",
        "meaning": "Borrower age.",
    },
    "age_band": {
        "label": "Age Band",
        "source": "Engineered",
        "meaning": "Grouped age segment.",
    },
    "MonthlyIncome": {
        "label": "Monthly Income",
        "source": "Raw",
        "meaning": "Borrower reported monthly income.",
    },
    "DebtRatio": {
        "label": "Debt Ratio",
        "source": "Raw",
        "meaning": "Debt burden ratio provided in the dataset.",
    },
    "debt_to_income_ratio": {
        "label": "Debt-to-Income Proxy",
        "source": "Engineered",
        "meaning": "Proxy based on the dataset debt ratio field.",
    },
    "estimated_monthly_debt": {
        "label": "Estimated Monthly Debt",
        "source": "Engineered",
        "meaning": "DebtRatio multiplied by filled monthly income.",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines and Loans",
        "source": "Raw",
        "meaning": "Number of active credit lines and loans.",
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real Estate Loans or Lines",
        "source": "Raw",
        "meaning": "Number of mortgage or real-estate-backed credit lines.",
    },
    "NumberOfTimes90DaysLate": {
        "label": "90+ Days Late Count",
        "source": "Raw",
        "meaning": "Number of times borrower was 90+ days past due.",
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "30-59 Days Late Count",
        "source": "Raw",
        "meaning": "Number of 30-59 day past-due events.",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "60-89 Days Late Count",
        "source": "Raw",
        "meaning": "Number of 60-89 day past-due events.",
    },
    "income_band": {
        "label": "Income Band",
        "source": "Engineered",
        "meaning": "Income bucket derived from monthly income.",
    },
    "monthly_income_missing": {
        "label": "Monthly Income Missing Flag",
        "source": "Engineered",
        "meaning": "Flag indicating missing monthly income.",
    },
}


def feature_label(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("label", feature)


def feature_source(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("source", "Raw / model field")


def feature_meaning(feature: str) -> str:
    return FEATURE_INFO.get(feature, {}).get("meaning", "Model input feature.")


def add_feature_display(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    df = df.copy()
    df["feature_display"] = df[feature_col].map(feature_label)
    df["source"] = df[feature_col].map(feature_source)
    df["meaning"] = df[feature_col].map(feature_meaning)
    return df


def format_input_value(value) -> str:
    if value == "Missing" or pd.isna(value):
        return "Missing"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value.is_integer():
        return f"{numeric_value:,.0f}"
    return f"{numeric_value:,.2f}"


def threshold_metrics(scores: pd.DataFrame, threshold: float) -> dict[str, float]:
    approved = scores["risk_score"] < threshold
    rejected = ~approved
    total_bads = scores["actual_default"].sum()
    return {
        "approval_rate": approved.mean(),
        "rejection_rate": rejected.mean(),
        "approved_default_rate": scores.loc[approved, "actual_default"].mean()
        if approved.any()
        else 0.0,
        "rejected_default_rate": scores.loc[rejected, "actual_default"].mean()
        if rejected.any()
        else 0.0,
        "bad_capture_rate": scores.loc[rejected, "actual_default"].sum() / total_bads
        if total_bads
        else 0.0,
        "approved_count": int(approved.sum()),
        "rejected_count": int(rejected.sum()),
    }


def top_risk_metrics(scores: pd.DataFrame, top_share: int) -> dict[str, float]:
    sorted_scores = scores.sort_values("risk_score", ascending=False)
    n = max(1, int(len(sorted_scores) * top_share / 100))
    selected = sorted_scores.head(n)
    total_bads = sorted_scores["actual_default"].sum()
    base_default_rate = sorted_scores["actual_default"].mean()
    default_rate = selected["actual_default"].mean()
    return {
        "customers": n,
        "customer_share": n / len(sorted_scores),
        "default_rate": default_rate,
        "bad_capture_rate": selected["actual_default"].sum() / total_bads
        if total_bads
        else 0.0,
        "lift": default_rate / base_default_rate if base_default_rate else 0.0,
    }


def parse_interval(text: str) -> tuple[float, float, bool, bool] | None:
    match = re.match(r"^([\\(\\[])([-\\d\\.]+),\\s*([-\\d\\.]+)([\\)\\]])$", text)
    if not match:
        return None
    left_symbol, left, right, right_symbol = match.groups()
    return (
        float(left),
        float(right),
        left_symbol == "[",
        right_symbol == "]",
    )


def value_in_interval(value: float, interval: tuple[float, float, bool, bool]) -> bool:
    left, right, include_left, include_right = interval
    left_ok = value >= left if include_left else value > left
    right_ok = value <= right if include_right else value < right
    return left_ok and right_ok


def find_scorecard_bin(points: pd.DataFrame, feature: str, value) -> pd.Series:
    feature_bins = points[points["feature"] == feature].copy()
    if value is None:
        missing = feature_bins[feature_bins["bin"].astype(str) == "Missing"]
        return missing.iloc[0] if not missing.empty else feature_bins.iloc[0]

    for _, row in feature_bins.iterrows():
        bin_text = str(row["bin"])
        interval = parse_interval(bin_text)
        if interval is not None and value_in_interval(float(value), interval):
            return row
        if bin_text == str(value) or bin_text == str(float(value)):
            return row
    return feature_bins.iloc[-1]


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def scorecard_simulation(
    points: pd.DataFrame,
    params: pd.Series,
    borrower: dict[str, float | int | None],
) -> tuple[pd.DataFrame, float, float, str, str]:
    rows = []
    total_points = float(params["score_offset"])
    for feature, value in borrower.items():
        row = find_scorecard_bin(points, feature, value)
        points_value = float(row["points"])
        total_points += points_value
        rows.append(
            {
                "feature": feature,
                "input_value": "Missing" if value is None else value,
                "matched_bin": row["bin"],
                "bin_default_rate": row["event_rate"],
                "woe": row["woe"],
                "points": points_value,
            }
        )

    logit = float(params["base_logit"]) + (
        float(params["base_score"]) - total_points
    ) / float(params["factor"])
    pd_estimate = sigmoid(logit)
    if total_points >= 660:
        band = "Low risk"
        action = "Auto-approve candidate"
    elif total_points >= 560:
        band = "Medium risk"
        action = "Manual review / additional verification"
    else:
        band = "High risk"
        action = "Reject, reduce exposure, or require stronger terms"
    return pd.DataFrame(rows), total_points, pd_estimate, band, action


data = load_data()
metrics = data["metrics"]
decile = data["decile"]
scores = data["test_scores"]
scorecard_compare = data["scorecard_compare"]
scorecard_points = data["scorecard_points"]
scorecard_params = data["scorecard_params"].iloc[0]
scorecard_scores = data["scorecard_scores"]
shap = data["shap"]
iv = data["iv"]
calibration = data["calibration"]

shap_display = add_feature_display(shap)
iv_display = add_feature_display(iv)

best = metrics.iloc[0]
top_decile = decile.iloc[0]

st.title("Credit Default Risk Analytics Dashboard")
st.caption(
    "Interactive dashboard for credit default prediction, approval threshold trade-offs, risk segmentation, and WOE scorecard demonstration."
)

tab_overview, tab_threshold, tab_segmentation, tab_scorecard, tab_limitations = st.tabs(
    [
        "1. Executive Overview",
        "2. Approval Threshold",
        "3. Risk Segmentation",
        "4. Scorecard Demo",
        "5. Limitations & Next Steps",
    ]
)

with tab_overview:
    st.subheader("Executive Overview")
    insight(
        "Business takeaway: the champion model can prioritize risk effectively. "
        f"The highest-risk 10% of borrowers captures {pct(top_decile['cumulative_bad_capture_rate'])} "
        f"of observed defaults with {top_decile['lift']:.2f}x lift versus the portfolio average."
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Champion Model", "Hist GB")
    with k2:
        metric_card("AUC", f"{best['auc']:.3f}")
    with k3:
        metric_card("KS", f"{best['ks']:.3f}")
    with k4:
        metric_card("Top-Decile Lift", f"{top_decile['lift']:.2f}x")
    with k5:
        metric_card("Top 10% Bad Capture", pct(top_decile["cumulative_bad_capture_rate"]))

    st.markdown("#### Model Performance and Role")
    model_roles = {
        "hist_gradient_boosting": "ML champion; strongest ranking performance",
        "random_forest": "Tree-based nonlinear baseline",
        "logistic_regression": "Simple linear baseline",
    }
    model_display_names = {
        "hist_gradient_boosting": "Hist Gradient Boosting",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }
    comparison_table = metrics.copy()
    comparison_table["model_name"] = comparison_table["model"].map(model_display_names)
    comparison_table["role"] = comparison_table["model"].map(model_roles)
    scorecard_row = scorecard_compare[
        scorecard_compare["model"] == "woe_logistic_scorecard"
    ].iloc[0]
    scorecard_display = pd.DataFrame(
        [
            {
                "model_name": "WOE Logistic Scorecard",
                "role": "Traditional interpretable benchmark",
                "auc": scorecard_row["auc"],
                "ks": scorecard_row["ks"],
            }
        ]
    )
    comparison_table = pd.concat(
        [
            comparison_table[["model_name", "role", "auc", "ks"]],
            scorecard_display,
        ],
        ignore_index=True,
    )
    st.dataframe(
        comparison_table.style.format({"auc": "{:.4f}", "ks": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The WOE scorecard is included here as an interpretable benchmark, not as a separate champion candidate. "
        "It helps show the trade-off between policy transparency and predictive performance."
    )

    c3, c4 = st.columns([1, 1])
    with c3:
        fig = px.bar(
            shap_display.head(8).sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature_display",
            color="source",
            orientation="h",
            title="What Drives the ML Model's Risk Ranking",
            labels={
                "mean_abs_shap": "Mean absolute SHAP",
                "feature_display": "Risk driver",
                "source": "Source",
            },
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.bar(
            iv_display.head(8).sort_values("iv"),
            x="iv",
            y="feature_display",
            color="source",
            orientation="h",
            title="What Traditional Credit Screening Flags as Predictive",
            labels={
                "iv": "Information Value",
                "feature_display": "Risk variable",
                "source": "Source",
            },
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Feature dictionary for the risk drivers shown above"):
        st.markdown("##### Source type")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Raw**  \nOriginal field from the Kaggle dataset.")
        with s2:
            st.markdown("**Raw alias**  \nRenamed version of a raw field for readability.")
        with s3:
            st.markdown("**Engineered**  \nFeature created through aggregation, binning, flags, or transformations.")

        st.markdown("##### Column guide")
        st.markdown(
            """
            - **Technical field**: the exact column name used in the dataset or model code.
            - **Business label**: a cleaner name used in charts and business discussion.
            - **Source**: whether the feature is raw, renamed, or engineered.
            - **Meaning**: what the variable represents in credit-risk terms.
            """
        )
        feature_dictionary = pd.concat(
            [
                shap_display.head(8)[["feature", "feature_display", "source", "meaning"]],
                iv_display.head(8)[["feature", "feature_display", "source", "meaning"]],
            ],
            ignore_index=True,
        ).drop_duplicates("feature")
        feature_dictionary = feature_dictionary.rename(
            columns={
                "feature": "Technical field",
                "feature_display": "Business label",
                "source": "Source",
                "meaning": "Meaning",
            }
        )
        st.dataframe(
            feature_dictionary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Technical field": st.column_config.TextColumn(
                    "Technical field",
                    help="Exact feature name used in the dataset or model code.",
                    width="medium",
                ),
                "Business label": st.column_config.TextColumn(
                    "Business label",
                    help="Cleaner label used for dashboard charts and stakeholder discussion.",
                    width="medium",
                ),
                "Source": st.column_config.TextColumn(
                    "Source",
                    help="Raw, Raw alias, or Engineered.",
                    width="small",
                ),
                "Meaning": st.column_config.TextColumn(
                    "Meaning",
                    help="Plain-English credit-risk interpretation.",
                    width="large",
                ),
            },
        )

with tab_threshold:
    st.subheader("Approval Policy Trade-off")
    insight(
        "Business takeaway: the approval threshold controls the trade-off between growth and risk. "
        "A stricter threshold sends fewer borrowers to approval and captures more future defaulters for rejection or manual review."
    )
    st.write(
        "Move the threshold to see how approval volume and default-risk control change. Borrowers below the threshold are approved; borrowers above it are rejected or routed to manual review."
    )
    threshold = st.slider(
        "Approval threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.30,
        step=0.01,
        help="Lower threshold = more conservative policy. Higher threshold = more approvals but more risk accepted.",
    )
    tm = threshold_metrics(scores, threshold)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Approval Rate", pct(tm["approval_rate"]))
    c2.metric("Rejection / Review Rate", pct(tm["rejection_rate"]))
    c3.metric("Approved Default Rate", pct(tm["approved_default_rate"]))
    c4.metric("Rejected Default Rate", pct(tm["rejected_default_rate"]))
    c5.metric("Bad Capture Rate", pct(tm["bad_capture_rate"]))

    summary = pd.DataFrame(
        [
            {
                "group": "Approved",
                "customers": tm["approved_count"],
                "share": tm["approval_rate"],
                "default_rate": tm["approved_default_rate"],
            },
            {
                "group": "Rejected / Manual Review",
                "customers": tm["rejected_count"],
                "share": tm["rejection_rate"],
                "default_rate": tm["rejected_default_rate"],
            },
        ]
    )
    fig = px.bar(
        summary,
        x="group",
        y="customers",
        color="default_rate",
        color_continuous_scale="Reds",
        text="customers",
        title="Approval Volume and Risk Concentration",
        labels={
            "group": "Decision group",
            "customers": "Borrowers",
            "default_rate": "Observed default rate",
        },
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Policy Scenario Comparison")
    st.caption(
        "Use this table to compare candidate approval policies. A lower threshold is stricter: it approves fewer borrowers but captures more future defaulters in the rejected/manual-review group."
    )
    threshold_display = data["thresholds"].rename(
        columns={
            "approval_threshold": "Approval threshold",
            "approval_rate": "Approved borrowers",
            "rejection_rate": "Rejected / manual review",
            "approved_default_rate": "Default rate among approved",
            "rejected_default_rate": "Default rate among rejected",
            "bad_capture_rate": "Defaults captured by rejection",
        }
    )
    st.dataframe(
        threshold_display.style.format(
            {
                "Approval threshold": "{:.1f}",
                "Approved borrowers": "{:.2%}",
                "Rejected / manual review": "{:.2%}",
                "Default rate among approved": "{:.2%}",
                "Default rate among rejected": "{:.2%}",
                "Defaults captured by rejection": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Reading example: at threshold 0.3, the model would approve 94.25% of borrowers and route 5.75% to rejection/manual review. That small reviewed group contains 39.65% of all observed defaults."
    )

with tab_segmentation:
    st.subheader("Risk Segmentation and Review Prioritization")
    insight(
        "Business takeaway: risk segmentation turns model scores into a review queue. "
        f"Reviewing the top-risk 10% captures {pct(top_decile['cumulative_bad_capture_rate'])} "
        f"of observed defaults with {top_decile['lift']:.2f}x lift."
    )
    st.write(
        "Select the share of highest-risk borrowers to review. This shows how much default risk can be captured with limited review capacity."
    )
    top_share = st.slider(
        "Highest-risk population to review",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        format="%d%%",
    )
    rm = top_risk_metrics(scores, top_share)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers Reviewed", f"{rm['customers']:,}")
    c2.metric("Observed Default Rate", pct(rm["default_rate"]))
    c3.metric("Bad Capture Rate", pct(rm["bad_capture_rate"]))
    c4.metric("Lift vs Portfolio", f"{rm['lift']:.2f}x")

    c5, c6 = st.columns([1, 1])
    with c5:
        fig = px.line(
            decile,
            x="risk_decile",
            y="default_rate",
            markers=True,
            title="Observed Default Rate by Ranked Risk Group",
        )
        fig.update_xaxes(autorange="reversed", title="Risk decile (1 = highest risk)")
        fig.update_yaxes(tickformat=".0%", title="Observed default rate")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This chart checks whether the model ranks risk correctly. Decile 1 is the riskiest 10% of borrowers; its default rate should be much higher than later deciles."
        )
    with c6:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=decile["cumulative_customer_share"],
                y=decile["cumulative_bad_capture_rate"],
                mode="lines+markers",
                name="Model",
            )
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
        )
        fig.update_layout(
            title="Default Capture by Review Capacity",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            xaxis_title="Cumulative customer share",
            yaxis_title="Cumulative bad capture rate",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This chart shows review efficiency. The farther the model line sits above the random line, the more defaults the model finds with the same review capacity."
        )

    st.markdown("#### Risk Ranking Table")
    st.caption(
        "Each row is 10% of borrowers sorted by model risk score. This table turns the model ranking into an operational review queue."
    )
    decile_display = decile[
        [
            "risk_decile",
            "customers",
            "bads",
            "default_rate",
            "cumulative_bad_capture_rate",
            "lift",
            "cumulative_lift",
        ]
    ].rename(
        columns={
            "risk_decile": "Risk group",
            "customers": "Borrowers",
            "bads": "Observed defaults",
            "default_rate": "Default rate",
            "cumulative_bad_capture_rate": "Cumulative defaults captured",
            "lift": "Lift vs average",
            "cumulative_lift": "Cumulative lift",
        }
    )
    st.dataframe(
        decile_display.style.format(
            {
                "Default rate": "{:.2%}",
                "Cumulative defaults captured": "{:.2%}",
                "Lift vs average": "{:.2f}x",
                "Cumulative lift": "{:.2f}x",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Reading example: the highest-risk 10% group has a 36.90% observed default rate and captures 55.21% of all defaults, which is 5.52x the portfolio average."
    )

with tab_scorecard:
    st.subheader("Borrower-Level Scorecard Explanation")
    insight(
        "Business takeaway: the scorecard translates borrower attributes into transparent points. "
        "Higher scores indicate lower estimated default risk, while negative point contributions show the main risk concerns."
    )
    st.write(
        "This simulator uses the simplified WOE Logistic Scorecard. Higher score means lower default risk. It is designed for interpretability, not as a production approval engine."
    )

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("#### Borrower Inputs")
        past_due = st.number_input("Total past-due events", min_value=0, max_value=6, value=0, step=1)
        util = st.slider("Credit utilization", min_value=0.0, max_value=1.2, value=0.35, step=0.01)
        age = st.slider("Age", min_value=24, max_value=87, value=45, step=1)
        income_missing = st.checkbox("Monthly income missing", value=False)
        income = None
        if not income_missing:
            income = st.number_input("Monthly income", min_value=0, max_value=25000, value=6000, step=500)
        real_estate = st.number_input("Real estate loans or lines", min_value=0, max_value=4, value=1, step=1)
        debt_ratio = st.slider("Debt-to-income proxy / debt ratio", min_value=0.0, max_value=1.5, value=0.45, step=0.01)
        open_lines = st.number_input("Open credit lines and loans", min_value=0, max_value=30, value=8, step=1)

    income_band = None
    if income is None:
        income_band = None
    elif income <= 2500:
        income_band = 1.0
    elif income <= 5000:
        income_band = 2.0
    elif income <= 10000:
        income_band = 3.0
    elif income <= 20000:
        income_band = 4.0
    else:
        income_band = 5.0

    borrower = {
        "total_past_due_events": int(past_due),
        "credit_utilization": float(util),
        "age": int(age),
        "MonthlyIncome": None if income is None else float(income),
        "NumberRealEstateLoansOrLines": int(real_estate),
        "income_band": income_band,
        "debt_to_income_ratio": float(debt_ratio),
        "NumberOfOpenCreditLinesAndLoans": int(open_lines),
    }
    contribution, score, pd_estimate, band, action = scorecard_simulation(
        scorecard_points,
        scorecard_params,
        borrower,
    )

    with right:
        st.markdown("#### Scorecard Output")
        r1, r2, r3 = st.columns(3)
        r1.metric("Credit Score", f"{score:.0f}")
        r2.metric("Estimated PD", pct(pd_estimate))
        r3.metric("Risk Band", band)
        st.success(f"Suggested action: {action}")

        contribution_chart = contribution.copy()
        contribution_chart["Risk driver"] = contribution_chart["feature"].map(feature_label)
        contribution_chart["Contribution type"] = contribution_chart["points"].apply(
            lambda value: "Adds score / lowers risk" if value >= 0 else "Subtracts score / raises risk"
        )
        fig = px.bar(
            contribution_chart.sort_values("points"),
            x="points",
            y="Risk driver",
            orientation="h",
            color="Contribution type",
            color_discrete_map={
                "Adds score / lowers risk": "#2ca25f",
                "Subtracts score / raises risk": "#de2d26",
            },
            title="Score Contribution by Feature",
            labels={"points": "Score points"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Positive points increase the credit score and reduce estimated risk; negative points reduce the score and indicate higher risk."
        )

        risk_increasing = (
            contribution_chart[contribution_chart["points"] < 0]
            .sort_values("points")
            .head(3)[["Risk driver", "points"]]
            .rename(columns={"points": "Score impact"})
        )
        risk_reducing = (
            contribution_chart[contribution_chart["points"] > 0]
            .sort_values("points", ascending=False)
            .head(3)[["Risk driver", "points"]]
            .rename(columns={"points": "Score impact"})
        )
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("##### Main Risk Concerns")
            if risk_increasing.empty:
                st.caption("No negative score contributions for this borrower profile.")
            else:
                st.dataframe(
                    risk_increasing.style.format({"Score impact": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
        with d2:
            st.markdown("##### Main Risk Offsets")
            if risk_reducing.empty:
                st.caption("No positive score contributions for this borrower profile.")
            else:
                st.dataframe(
                    risk_reducing.style.format({"Score impact": "+{:.1f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("#### Matched Scorecard Bins")
    st.caption(
        "This table shows how the selected borrower inputs are translated into scorecard bins, WOE values, and final score points."
    )
    contribution_display = contribution.copy()
    contribution_display["Risk driver"] = contribution_display["feature"].map(feature_label)
    contribution_display["input_value"] = contribution_display["input_value"].map(format_input_value)
    contribution_display = contribution_display.rename(
        columns={
            "input_value": "Input value",
            "matched_bin": "Matched scorecard bin",
            "bin_default_rate": "Bin default rate",
            "woe": "WOE",
            "points": "Score points",
        }
    )[
        [
            "Risk driver",
            "Input value",
            "Matched scorecard bin",
            "Bin default rate",
            "WOE",
            "Score points",
        ]
    ]
    st.dataframe(
        contribution_display.style.format(
            {
                "Bin default rate": "{:.2%}",
                "WOE": "{:.4f}",
                "Score points": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Reading example: if credit utilization falls into a high-risk bin, it receives negative points and lowers the borrower's final credit score."
    )

    st.markdown("#### Score Separation in Test Set")
    scorecard_scores_display = scorecard_scores.copy()
    scorecard_scores_display["Default status"] = scorecard_scores_display["actual_default"].map(
        {0: "Non-default", 1: "Default"}
    )
    fig = px.histogram(
        scorecard_scores_display,
        x="credit_score",
        color="Default status",
        nbins=35,
        barmode="overlay",
        title="Scorecard Credit Score Distribution",
        labels={"credit_score": "Credit score", "count": "Borrowers"},
        color_discrete_map={"Non-default": "#4c78a8", "Default": "#e45756"},
        opacity=0.65,
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The legend separates borrowers who did not default from borrowers who did default. A useful scorecard should push more defaulters toward lower scores and more non-defaulters toward higher scores."
    )

with tab_limitations:
    st.subheader("Limitations & Next Steps")
    insight(
        "Business takeaway: this dashboard summarizes a credit-risk modeling workflow and highlights the validation gaps that would need to be addressed before real-world lending use."
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - **Data representativeness:** results are based on a public historical dataset, not a specific lender's current applicant portfolio.
        - **Observed-outcome bias:** the target is learned from available borrower outcomes and does not fully address applicants without observed repayment behavior.
        - **Business cost not modeled:** approval thresholds are compared by risk metrics, but not yet by profit, expected loss, LGD, or manual-review cost.
        - **Governance checks are limited:** fairness, stability, calibration drift, and production monitoring are outside the current project scope.
        """
    )

    st.markdown("#### Next Steps")
    st.markdown(
        """
        - **Validate on lender-specific data:** compare feature distributions, missingness patterns, and default rates against a real target portfolio.
        - **Define an expected-loss framework:** connect PD scores to LGD, exposure, approval revenue, and manual-review capacity.
        - **Extend model validation:** add stability metrics, calibration tracking, and segment-level performance analysis.
        - **Document decision policy:** specify how scores, thresholds, overrides, and scorecard explanations would be used by a risk team.
        """
    )

st.divider()
st.caption(
    "Portfolio dashboard generated from the project reports. Data source: Kaggle Give Me Some Credit."
)
