# WOE Logistic Scorecard Benchmark

## Objective

Build a compact traditional credit scorecard benchmark using IV-selected variables and WOE transformations, then compare it with the machine learning champion model.

## Selected Variables

- total_past_due_events
- credit_utilization
- age
- MonthlyIncome
- NumberRealEstateLoansOrLines
- income_band
- debt_to_income_ratio
- NumberOfOpenCreditLinesAndLoans

## Model Comparison

                 model                                role    auc     ks  auc_gap_vs_champion  ks_gap_vs_champion
woe_logistic_scorecard Traditional interpretable benchmark 0.8262 0.5192              -0.0422             -0.0635
hist_gradient_boosting           Machine learning champion 0.8684 0.5827               0.0000              0.0000

## Interpretation

The WOE logistic scorecard is more transparent and easier to explain to credit policy stakeholders. The machine learning champion can capture more complex non-linear relationships and interactions. The comparison shows the trade-off between interpretability and predictive performance.

## Output Files

- reports/scorecard_selected_features.csv
- reports/scorecard_woe_bins.csv
- reports/scorecard_coefficients.csv
- reports/scorecard_points.csv
- reports/scorecard_scaling_params.csv
- reports/scorecard_test_scores.csv
- reports/scorecard_vs_ml_comparison.csv
- reports/scorecard_decile.csv
- reports/figures/17_scorecard_coefficients.png
- reports/figures/18_scorecard_score_distribution.png