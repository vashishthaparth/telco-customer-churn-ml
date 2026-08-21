# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting whether a telecom customer is likely to churn.

## Problem

Customer churn is an important business problem for telecom companies. The goal of this project is to predict customers who are likely to leave the service so that retention efforts can be targeted appropriately.

## Dataset

The dataset contains 7,043 telecom customers and information about:

- Customer demographics
- Tenure
- Contract type
- Internet and support services
- Payment method
- Monthly charges
- Total charges

Target:

- `Churn = Yes` → customer churned
- `Churn = No` → customer stayed

## Workflow

The project follows an end-to-end ML workflow:

1. Data understanding and EDA
2. Data cleaning
3. Feature analysis and engineering
4. Train/test split
5. Numerical and categorical preprocessing
6. Logistic Regression baseline
7. Random Forest
8. Model evaluation
9. Cross-validation
10. Hyperparameter tuning
11. Model serialization
12. FastAPI deployment
13. Docker containerization

## Models

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.842 | 0.634 |
| Tuned Random Forest | 0.841 | 0.647 |

The tuned Random Forest achieved higher churn recall and slightly higher PR-AUC, while Logistic Regression provided higher precision at the default threshold.

## Key Findings

EDA revealed several strong associations with churn:

- Customers with shorter tenure were more likely to churn.
- Month-to-month contracts had substantially higher churn.
- Fiber-optic customers showed higher churn.
- Payment method was associated with different churn patterns.
- Monthly charges showed a positive association with churn.

These relationships are associations rather than causal conclusions.

## Deployment

The trained preprocessing and model are stored as a single scikit-learn Pipeline.

The model is exposed through a FastAPI REST API and containerized using Docker.

### API flow

```text
Customer data
     ↓
FastAPI
     ↓
Preprocessing Pipeline
     ↓
Random Forest
     ↓
Churn prediction + probability