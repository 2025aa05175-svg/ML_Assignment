# ML Assignment 2: Telco Customer Churn Prediction

## 1. Problem Statement
The goal of this assignment is to build and deploy a machine learning application to predict customer churn in a telecommunications company. Churn prediction helps businesses identify customers who are likely to leave, enabling proactive retention strategies. We implement multiple classification models, evaluate them, and deploy the best performing solution as an interactive Streamlit web application.

## 2. Dataset Description
**Dataset:** Telco Customer Churn (Kaggle/IBM)
**Source:** [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains information about:
- **Customers who left within the last month** – the column is called `Churn` (Target Variable).
- **Services that each customer has signed up for** – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- **Customer account information** – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
- **Demographic info about customers** – gender, age range, and if they have partners and dependents.

**Rows:** 7043
**Columns:** 21 (Data includes 1 target and 20 features)

## 3. Models Used and Evaluation Metrics

The following 6 classification models were implemented and evaluated:

| ML Model Name          | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
|------------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression    | 0.8176   | 0.8615   | 0.6824    | 0.5818   | 0.6281   | 0.5111   |
| Decision Tree          | 0.7246   | 0.6566   | 0.4809    | 0.5067   | 0.4935   | 0.3048   |
| kNN                    | 0.7757   | 0.7867   | 0.5877    | 0.5121   | 0.5473   | 0.4008   |
| Naive Bayes            | 0.7580   | 0.8429   | 0.5293    | 0.7748   | 0.6289   | 0.4770   |
| Random Forest          | 0.7970   | 0.8369   | 0.6629    | 0.4745   | 0.5531   | 0.4364   |
| XGBoost                | 0.7942   | 0.8365   | 0.6361    | 0.5201   | 0.5723   | 0.4424   |

## 4. Observations

- **Logistic Regression**: Achieved the highest **Accuracy (81.76%)** and **MCC (0.5111)**. It provides a balanced performance between Precision and Recall. It is a robust baseline and performs surprisingly well on this dataset.
- **Naive Bayes**: While having lower accuracy, it stands out with the highest **Recall (77.48%)**. This makes it useful if the business goal is to capture as many churners as possible, even at the cost of more false positives.
- **Decision Tree**: Performed the worst across almost all metrics, likely due to overfitting on the training data.
- **Ensemble Models (Random Forest & XGBoost)**: They performed well (around 79% accuracy) but did not outperform Logistic Regression in this specific split/configuration. They might require further hyperparameter tuning to surpass the linear model.
- **kNN**: Showed moderate performance but is computationally more expensive during inference compared to other models.

## 5. Deployment
The application is built using **Streamlit**.
- upload a dataset (CSV) to see predictions.
- Choose from any of the 6 trained models.
- View evaluation metrics and confusion matrix.
