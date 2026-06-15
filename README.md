# Bank Customer Churn Prediction using ANN

Predicting whether a bank customer will leave (churn) using an **Artificial Neural Network** built with TensorFlow/Keras. The project covers the full workflow: data cleaning, feature engineering, encoding, scaling, model building, training with early stopping, and evaluation.

**Test accuracy: ~86%**


## Problem

A bank wants to know which customers are likely to close their accounts so it can act before they leave. Using 10,000 customer records, we train a binary classifier to predict the `Exited` flag (1 = churned, 0 = stayed).

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
- **Flags** — `ZeroBalance`, `IsSenior`
- **Interaction** — `Age_x_Active` (age × active membership)
- **Binned age** — one-hot age groups (Young / Adult / MidAge / Senior)
- **`NumOfProducts` as categorical** — captures its non-monotonic relationship with churn
- **One-hot encoding** of `Geography` and `Gender`

This expands the feature set to **22 columns**, all standardized with `StandardScaler`.

## Model

A `Sequential` Keras ANN:

- Input + 2 hidden layers (ReLU) with **Dropout** regularization
- Sigmoid output for binary classification
- Optimizer: `adam` · Loss: `binary_crossentropy`
- **EarlyStopping** on validation loss to avoid overfitting

## Results

- **Test accuracy: ~86%**
- Training/validation **accuracy and loss curves** plotted
- **Confusion matrix** + accuracy score on the held-out test set


## Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib

