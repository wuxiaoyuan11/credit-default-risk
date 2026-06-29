# Credit Risk Model Report

## Model Performance

Best model: `hist_gradient_boosting`

                 model    auc     ks
hist_gradient_boosting 0.8684 0.5827
         random_forest 0.8642 0.5823
   logistic_regression 0.8632 0.5716

## Decile / Lift / Gains

- Top 10% highest-risk borrowers have an observed default rate of 36.90%.
- Top 10% highest-risk borrowers capture 55.21% of all observed defaulters.
- Top 20% highest-risk borrowers capture 73.32% of all observed defaulters.
- Top 30% highest-risk borrowers capture 83.34% of all observed defaulters.
- Top-decile lift is 5.52x over the portfolio default rate.

Business use: decile analysis shows whether the model can rank-order risk. This is useful for manual review queues, risk-based pricing, and credit line management.

## WOE / IV Screening

                             feature     iv    predictive_power
               total_past_due_events 1.4428 Suspiciously strong
                        has_past_due 1.1671 Suspiciously strong
                  credit_utilization 1.0568 Suspiciously strong
RevolvingUtilizationOfUnsecuredLines 1.0568 Suspiciously strong
             NumberOfTimes90DaysLate 0.8845 Suspiciously strong
NumberOfTime30-59DaysPastDueNotWorse 0.7470 Suspiciously strong
             high_credit_utilization 0.7350 Suspiciously strong
NumberOfTime60-89DaysPastDueNotWorse 0.6023 Suspiciously strong
                                 age 0.2380              Medium
                            age_band 0.2197              Medium

Business use: IV identifies variables with strong discriminatory power. Very high IV values should be reviewed for redundancy or leakage before production use.

## WOE Logistic Scorecard Benchmark

Selected scorecard variables:

- total_past_due_events
- credit_utilization
- age
- MonthlyIncome
- NumberRealEstateLoansOrLines
- income_band
- debt_to_income_ratio
- NumberOfOpenCreditLinesAndLoans

                 model                                role    auc     ks  auc_gap_vs_champion  ks_gap_vs_champion
woe_logistic_scorecard Traditional interpretable benchmark 0.8262 0.5192              -0.0422             -0.0635
hist_gradient_boosting           Machine learning champion 0.8684 0.5827               0.0000              0.0000

Business use: the WOE logistic scorecard acts as a traditional, transparent benchmark. It is easier to explain to credit policy stakeholders, while the machine learning champion provides stronger flexibility for non-linear risk patterns.

## Calibration

Brier score: 0.0487

Business use: calibration checks whether predicted default probabilities are close to observed default rates. This matters when model scores are used directly as expected loss inputs.

## SHAP Explainability

SHAP analysis completed successfully.

Top SHAP drivers:

                             feature  mean_abs_shap
               total_past_due_events         0.6019
RevolvingUtilizationOfUnsecuredLines         0.5946
                                 age         0.2368
     NumberOfOpenCreditLinesAndLoans         0.1379
                       MonthlyIncome         0.1321
                           DebtRatio         0.0897
              estimated_monthly_debt         0.0869
        NumberRealEstateLoansOrLines         0.0624
             NumberOfTimes90DaysLate         0.0439
NumberOfTime30-59DaysPastDueNotWorse         0.0414

Business use: SHAP explains which variables push model risk predictions higher or lower. This helps with model governance, stakeholder communication, and adverse-action style reasoning.

## Approval Threshold Trade-off

 approval_threshold  approval_rate  rejection_rate  approved_default_rate  rejected_default_rate  bad_capture_rate
             0.3000         0.9425          0.0575                 0.0428                 0.4611            0.3965
             0.5000         0.9787          0.0213                 0.0552                 0.5994            0.1910
             0.7000         0.9976          0.0024                 0.0652                 0.7260            0.0264

Business use: threshold analysis translates risk scores into approval, rejection, or manual review decisions.