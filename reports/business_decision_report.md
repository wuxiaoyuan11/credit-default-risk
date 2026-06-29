# Business Decision Report

Best model: `hist_gradient_boosting`

## Threshold trade-off

 approval_threshold  approval_rate  rejection_rate  approved_default_rate  rejected_default_rate  bad_capture_rate
             0.3000         0.9425          0.0575                 0.0428                 0.4611            0.3965
             0.5000         0.9787          0.0213                 0.0552                 0.5994            0.1910
             0.7000         0.9976          0.0024                 0.0652                 0.7260            0.0264

Interpretation: a lower approval threshold rejects more applicants and captures more future defaulters, but it also reduces approval volume. A higher threshold approves more customers, but accepts more default risk.

## High-risk segments

Segments with higher predicted and observed risk are mainly driven by past-due history, high credit utilization, low or missing income, and heavy debt burden proxies.

## Early warning signals

- total_past_due_events
- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfOpenCreditLinesAndLoans
- MonthlyIncome
- DebtRatio
- estimated_monthly_debt
- NumberRealEstateLoansOrLines

## Notes

The Give Me Some Credit dataset does not include loan amount, loan purpose, or external credit bureau score fields, so loan-to-income and external score aggregation are documented as recommended extensions rather than implemented features.