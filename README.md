# Customer Lifetime Value Prediction (APAN5200 PAC)

Predictive Analysis Competition for **APAN5200 Applied Machine Learning I**, Columbia University SPS, M.S. in Applied Analytics (Fall 2026). Hosted on Kaggle.

## Problem

An online retailer wants to identify high-value customers early so it can target promotions, prioritize support, and forecast revenue. The goal is to predict each customer's spending over the next 12 months (`future_spend`) from their demographics, shopping history, device preferences, and engagement metrics.

- **Target:** `future_spend` (continuous, dollars)
- **Metric:** RMSE (lower is better), ranked on a private Kaggle leaderboard
- **Training data:** 40,000 customers with 18 features + target
- **Scoring data:** 10,000 customers without the target

> The dataset is provided for course use only and is **not included** in this repository.

## Approach

| Step | Description | Status |
|---|---|---|
| 0 | Load data | Done |
| 1 | Explore data: missing values, target distribution, correlations, categorical group means | Done |
| 2 | Separate features/target; stratified 70/30 train-test split on binned target (seed 617) | Done |
| 3 | Preprocessing: median imputation (fit on train) + missing-value indicators; `Unknown` level for missing `device_type`; one-hot encoding with `drop_first=True`, test/scoring columns aligned to train | Done |
| 4 | Linear regression baseline | Done |
| 5 | Decision tree tuned with grid search + 5-fold CV | Next |
| 6 | Random forest and gradient boosting, tuned with 5-fold CV | Planned |
| 7 | Refit best model on all training data; predict scoring data; submit to Kaggle | Planned |

## Key Findings from EDA

- `future_spend` is strongly right-skewed: median ~544 vs. mean ~1,109, max ~20,000.
- Strongest numeric predictors: `purchase_frequency` (r = 0.60), `income` (0.40), `average_order_value` (0.34); `recency_days` is negatively related (-0.28).
- `membership_program` separates spending sharply: Basic ~285, Silver ~611, Gold ~1,177, Platinum ~3,126 average.
- Missingness is informative: customers with missing `income` spend more on average (~1,404 vs. ~1,052), while those with missing `return_rate` spend less (~552 vs. ~1,191). Indicator columns preserve this signal.

## Results

| Model | CV RMSE | Test RMSE | Notes |
|---|---|---|---|
| Predict train mean | - | 1,716 | Reference point |
| Linear regression | 1,185 | 1,195 | Test R² = 0.515; underfits; produces negative predictions |

## Repository Structure

```
pac.ipynb      # Main notebook: full workflow from loading to submission
README.md
.gitignore     # Excludes course data (*.csv) and materials (*.pdf)
```

## How to Run

1. Download `analysis_data.csv` and `scoring_data.csv` from the Kaggle competition page.
2. Place them in the same folder as `pac.ipynb`.
3. Run the notebook top to bottom (Python 3.13, pandas, scikit-learn, matplotlib).
