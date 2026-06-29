# Credit Default Prediction and Risk Explainability

This project builds an end-to-end credit default risk modeling workflow using the Kaggle **Give Me Some Credit** dataset. The goal is to predict whether a borrower will experience serious delinquency within the next two years and translate model scores into practical credit approval decisions.

The project covers risk exploration, feature engineering, model comparison, a WOE logistic scorecard benchmark, model explainability, score calibration, decile/lift analysis, WOE/IV screening, and approval-threshold analysis.

**Interactive dashboard:** [Credit Default Risk Dashboard](https://credit-default-risk-3hhnmuynvxyr4z5yamarft.streamlit.app/)

## Business Objective

Credit lenders need to assess borrower risk before issuing loans or credit lines. A useful model should not only predict default risk, but also help answer business questions such as:

- Which borrowers are most likely to become seriously delinquent?
- Which risk factors drive the prediction?
- How should approval thresholds be set?
- What is the trade-off between approval rate and default risk?
- Which borrower segments should be routed to manual review?

Target variable:

```text
SeriousDlqin2yrs
```

Definition:

- `0`: no serious delinquency within the next two years
- `1`: serious delinquency within the next two years

## Dataset

Source: Kaggle **Give Me Some Credit**

Training data:

```text
150,000 borrowers
12 original columns
6.684% observed default rate
```

Main fields:

| Field | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Utilization of unsecured credit lines |
| `age` | Borrower age |
| `DebtRatio` | Debt burden ratio |
| `MonthlyIncome` | Monthly income |
| `NumberOfTimes90DaysLate` | Number of 90+ days past-due events |
| `NumberOfOpenCreditLinesAndLoans` | Number of open credit lines and loans |
| `NumberRealEstateLoansOrLines` | Number of real estate loans or lines |
| `NumberOfDependents` | Number of dependents |

## Project Structure

```text
credit-default-risk/
  app.py                         # Interactive Streamlit dashboard
  data/
    raw/                         # Raw Kaggle files
    processed/                   # Cleaned modeling dataset
  notebooks/
    01_credit_default_prediction_risk_explainability.ipynb
  reports/
    figures/                     # EDA and model diagnostic plots
    models/                      # Trained model artifacts
    business_decision_report.md
    model_metrics.csv
    threshold_business_decisions.csv
    risk_segments.csv
  src/
    train_credit_default.py      # Reproducible modeling pipeline
  requirements.txt
```

## Methodology

### 1. Risk EDA

The analysis explores how default risk varies by:

- Credit utilization
- Debt ratio
- Monthly income
- Age
- Historical delinquency counts
- Number of open credit lines
- Number of real estate loans

Generated figures are stored in:

```text
reports/figures/
```

### 2. Data Cleaning

Cleaning steps:

- Removed index-like columns
- Preserved missingness indicators for income and dependents
- Imputed missing numerical values with medians
- Clipped extreme values at the 1st and 99th percentiles
- Created a clean modeling dataset

### 3. Feature Engineering

Interpretable credit risk features were created from the raw borrower attributes:

| Feature | Interpretation |
|---|---|
| `total_past_due_events` | Total historical delinquency events |
| `has_past_due` | Whether the borrower has any past-due history |
| `high_credit_utilization` | Whether utilization exceeds 80% |
| `income_band` | Income bucket |
| `age_band` | Age bucket |
| `estimated_monthly_debt` | Proxy for estimated monthly debt burden |
| `debt_to_income_ratio` | Debt-to-income proxy based on available fields |
| `payment_burden_proxy` | Repayment burden proxy |

Note: the dataset does not contain loan amount, loan purpose, or external bureau scores, so loan-to-income ratio and external credit score aggregation are not implemented in this version.

### 4. Modeling

Three models were trained and compared:

| Model | AUC | KS |
|---|---:|---:|
| Hist Gradient Boosting | 0.8684 | 0.5827 |
| Random Forest | 0.8642 | 0.5823 |
| Logistic Regression | 0.8632 | 0.5716 |

Best model:

```text
Hist Gradient Boosting
```

### 5. Traditional Scorecard Benchmark

In addition to machine learning models, a compact WOE Logistic Scorecard benchmark was built to reflect a more traditional credit-risk modeling workflow.

Scorecard steps:

1. Use IV to identify predictive variables.
2. Select a compact set of non-redundant variables.
3. Bin selected variables.
4. Convert bins into WOE values.
5. Train Logistic Regression on WOE-transformed features.
6. Compare against the machine learning champion model.

Selected scorecard variables:

```text
total_past_due_events
credit_utilization
age
MonthlyIncome
NumberRealEstateLoansOrLines
income_band
debt_to_income_ratio
NumberOfOpenCreditLinesAndLoans
```

Scorecard vs ML champion:

| Model | Role | AUC | KS |
|---|---|---:|---:|
| WOE Logistic Scorecard | Traditional interpretable benchmark | 0.8262 | 0.5192 |
| Hist Gradient Boosting | Machine learning champion | 0.8684 | 0.5827 |

Interpretation:

The scorecard is more transparent and easier to explain to policy stakeholders, while the gradient boosting model provides stronger predictive performance. This comparison highlights the practical trade-off between interpretability and model power.

The scorecard was also scaled into a points-based credit score:

```text
Base score: 600
PDO: 50
Higher score = lower default risk
```

Outputs:

```text
reports/scorecard_points.csv
reports/scorecard_scaling_params.csv
reports/scorecard_test_scores.csv
reports/figures/18_scorecard_score_distribution.png
```

### 6. Explainability

Permutation importance and SHAP were used to identify key risk drivers. The strongest early warning signals were:

- `total_past_due_events`
- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfOpenCreditLinesAndLoans`
- `MonthlyIncome`
- `DebtRatio`
- `estimated_monthly_debt`
- `NumberRealEstateLoansOrLines`

Interpretation:

Borrowers with more past-due events, higher credit utilization, weaker income, and heavier debt burden tend to have higher predicted default risk.

### 7. Decile, Lift, and Gains Analysis

Risk models are often evaluated by how well they rank-order borrowers. The test population was sorted by predicted default risk and split into 10 equal-sized deciles.

Key results:

| Risk Decile | Customer Share | Default Rate | Bad Capture Rate | Lift |
|---:|---:|---:|---:|---:|
| 1, highest risk | 10% | 36.90% | 55.21% | 5.52x |
| 1-2, cumulative | 20% | - | 73.32% | 3.67x |
| 1-3, cumulative | 30% | - | 83.34% | 2.78x |

Interpretation:

- The top 10% highest-risk borrowers contain 55.21% of all observed defaulters.
- The top 20% highest-risk borrowers contain 73.32% of all observed defaulters.
- This confirms that the model is useful for risk ranking, not only probability prediction.

Outputs:

```text
reports/decile_lift_gains.csv
reports/figures/12_default_rate_by_decile.png
reports/figures/13_cumulative_gains.png
```

### 8. WOE / IV Analysis

WOE (Weight of Evidence) and IV (Information Value) are traditional credit scorecard tools used to evaluate variable predictive power.

Top variables by IV:

| Feature | IV | Predictive Power |
|---|---:|---|
| `total_past_due_events` | 1.443 | Very strong |
| `has_past_due` | 1.167 | Very strong |
| `credit_utilization` | 1.057 | Very strong |
| `NumberOfTimes90DaysLate` | 0.884 | Very strong |
| `age` | 0.238 | Medium |

Interpretation:

- Historical delinquency and credit utilization are the strongest predictors.
- Some IV values are very high because multiple engineered variables are derived from the same delinquency signals. In a production scorecard, these would need correlation checks and feature selection to avoid redundant variables.

Outputs:

```text
reports/iv_summary.csv
reports/woe_bins.csv
reports/figures/14_information_value.png
```

### 9. Calibration Analysis

Calibration checks whether predicted probabilities align with observed default rates.

Result:

```text
Brier score: 0.0487
```

The calibration table compares average predicted probability and observed default rate across score bins. The model is reasonably well aligned by risk bucket in this dataset.

Outputs:

```text
reports/calibration_table.csv
reports/figures/15_calibration_plot.png
```

## Approval Threshold Analysis

The model produces a risk score. A lender can convert this score into an approval decision:

```text
risk_score < threshold  -> approve
risk_score >= threshold -> reject or manual review
```

Threshold comparison:

| Approval Threshold | Approval Rate | Approved Default Rate | Bad Capture Rate |
|---:|---:|---:|---:|
| 0.3 | 94.25% | 4.28% | 39.65% |
| 0.5 | 97.87% | 5.52% | 19.10% |
| 0.7 | 99.76% | 6.52% | 2.64% |

Business interpretation:

- A lower threshold is more conservative: it rejects more borrowers and captures more future defaulters.
- A higher threshold increases approval volume but allows more default risk into the approved population.
- The threshold should be selected based on the lender's risk appetite, growth target, and manual review capacity.

## Key Outputs

```text
dashboard/index.html
reports/portfolio_credit_risk_report.md
reports/risk_model_report.md
reports/model_metrics.csv
reports/business_decision_report.md
reports/threshold_business_decisions.csv
reports/risk_segments.csv
reports/permutation_importance.csv
reports/scorecard_benchmark_report.md
reports/scorecard_vs_ml_comparison.csv
reports/scorecard_selected_features.csv
reports/scorecard_woe_bins.csv
reports/scorecard_coefficients.csv
reports/scorecard_points.csv
reports/scorecard_scaling_params.csv
reports/scorecard_test_scores.csv
reports/decile_lift_gains.csv
reports/iv_summary.csv
reports/woe_bins.csv
reports/calibration_table.csv
reports/shap_importance.csv
reports/figures/
```

## Dashboard

A static portfolio dashboard is available at:

```text
dashboard/index.html
```

An interactive Streamlit dashboard is available via:

```bash
streamlit run app.py
```

The dashboards summarize:

- champion model AUC / KS
- top-decile lift and bad capture rate
- WOE Logistic Scorecard vs ML champion comparison
- decile and cumulative gains charts
- SHAP and IV risk drivers
- calibration plot
- scorecard score distribution
- approval threshold trade-off
- interactive approval threshold slider
- interactive top-risk population selector
- scorecard borrower simulator

The dashboard is designed for quick portfolio review; the Markdown reports and notebook provide deeper implementation detail.

## How to Run

Option 1: run the helper script:

```bash
./run.sh
```

Option 2: run manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_credit_default.py
```

Open the portfolio dashboard:

```text
dashboard/index.html
```

Or run the interactive Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

The notebook is included as a detailed analysis artifact, but the README, report, and dashboard are the main portfolio-facing deliverables.

## Current Limitations

- The dataset does not include loan amount, loan purpose, interest rate, application channel, or external bureau score.
- XGBoost is not enabled in the current local environment because macOS requires the OpenMP runtime (`libomp`).
- SHAP is currently used for global model explanation; local single-borrower waterfall explanations are not yet included.
- A simplified WOE Logistic Scorecard and points-based score scaling are implemented, but the scorecard is not production-grade.
- Calibration is evaluated by score bins, but no post-calibration model such as Platt scaling or isotonic regression has been applied yet.
- Monitoring metrics such as PSI are not yet implemented.

## Next Improvements

- Add SHAP local waterfall explanations for selected high-risk borrowers.
- Add population stability index (PSI) for model monitoring.
- Add reject inference discussion and data leakage checks.
- Add out-of-time validation if timestamped data becomes available.
