# Bank Customer Churn Prediction — ANN vs. Gradient Boosting

Predicting whether a bank customer will leave (churn), framed as an **imbalanced binary classification** problem. The project benchmarks an **Artificial Neural Network** (TensorFlow/Keras) against **XGBoost** under stratified cross-validation, optimizing for **PR-AUC and recall** rather than raw accuracy. It covers the full workflow: data cleaning, feature engineering, encoding, scaling, model building, threshold tuning, and rigorous evaluation.

**Key result: XGBoost outperforms the ANN on every metric (PR-AUC 0.71 vs 0.67, ROC-AUC 0.87 vs 0.85); threshold tuning lifts churn recall from 44% → 71%.**

## Problem
A bank wants to know which customers are likely to close their accounts so it can intervene before they leave. Using 10,000 customer records (~20% churners), we train a classifier to predict the `Exited` flag (1 = churned, 0 = stayed).

Because only ~20% of customers churn, **accuracy is a misleading objective** — a model predicting "nobody churns" scores ~80%. This project therefore optimizes and reports **PR-AUC, ROC-AUC, precision, and recall**, and treats the precision/recall tradeoff as a business cost decision.

## Dataset
`Churn_Modelling.csv` — 10,000 rows, one bank customer per row.

| Column | Description |
| --- | --- |
| CreditScore | Customer credit score |
| Geography | Country (France / Spain / Germany) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Has a credit card (0/1) |
| IsActiveMember | Active membership (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target** — left the bank (0/1) |

Source: [Churn Modelling — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

## Feature Engineering
Beyond basic one-hot encoding, the notebook derives signal-rich features:
- **Ratio features** — `BalanceSalaryRatio`, `TenureByAge`, `CreditScoreByAge`
- **Flags** — `ZeroBalance`, `IsSenior`, `IsNewCustomer`, `HasBalance_Inactive`
- **Interaction** — `Age_x_Active` (age × active membership) and `Products_x_Active` (products × active membership)
- **Binned age** — one-hot age groups (Young / Adult / MidAge / Senior)
- **`NumOfProducts` as categorical** — captures its non-monotonic relationship with churn
- **One-hot encoding** of `Geography` and `Gender`

This expands the feature set to **25 columns**, all standardized with `StandardScaler` (fit on training data only, to prevent leakage).

**Validation:** 8 of the engineered features rank in the **top-20 importances**, led by the `Products_x_Active` interaction — engaged multi-product customers behave very differently from disengaged ones, a signal the raw columns don't capture on their own.

## Models
**Keras ANN** (`Sequential`):
- Input + 2 hidden layers (ReLU) with **Dropout** regularization
- Sigmoid output · Optimizer `adam` · Loss `binary_crossentropy`
- **EarlyStopping** on validation loss with `restore_best_weights`

**XGBoost** benchmark:
- `scale_pos_weight` set to the neg/pos ratio to handle imbalance
- Evaluated head-to-head against the ANN on identical folds

## Evaluation & Results
Both models evaluated under **stratified 5-fold cross-validation** (preserving the ~20% churn rate per fold), reported as mean ± std:

| Model | PR-AUC | ROC-AUC |
| --- | --- | --- |
| ANN | 0.675 ± 0.019 | 0.849 ± 0.010 |
| **XGBoost** | **0.708 ± 0.014** | **0.868 ± 0.006** |

**Threshold tuning (ANN):** moving the decision threshold from 0.5 → ~0.21 shifts the operating point from **44% recall / 75% precision** to **71% recall / 51% precision** — framing model selection as a retention-cost decision (cost of a wasted offer vs. a lost customer) rather than a single accuracy figure.

**Note on class weighting:** explicit class weighting produced essentially the same precision/recall regime as threshold tuning (identical PR-AUC), confirming that both techniques slide the operating point along a *fixed* precision-recall frontier rather than improving the model's underlying ranking. The ranking ceiling is set by the model and features — which is why switching to XGBoost, not further threshold/weight tuning, was what actually moved the metrics.

## Tech Stack
Python · TensorFlow / Keras · XGBoost · scikit-learn · pandas · NumPy · Matplotlib
