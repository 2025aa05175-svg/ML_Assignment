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
| Logistic Regression    | 0.8197   | 0.8610   | 0.6901    | 0.5791   | 0.6297   | 0.5152   |
| Decision Tree          | 0.7246   | 0.6566   | 0.4809    | 0.5067   | 0.4935   | 0.3048   |
| kNN                    | 0.7793   | 0.7812   | 0.5975    | 0.5094   | 0.5499   | 0.4072   |
| Naive Bayes            | 0.7580   | 0.8429   | 0.5293    | 0.7748   | 0.6289   | 0.4770   |
| Random Forest          | 0.8006   | 0.8624   | 0.6901    | 0.4477   | 0.5431   | 0.4390   |
| XGBoost                | 0.7942   | 0.8365   | 0.6361    | 0.5201   | 0.5723   | 0.4424   |

## 4. Observations

- **Logistic Regression**: Achieved the best overall performer with the highest **Accuracy (81.97%)** and **MCC (0.5152)**.
- **Random Forest**: Achieved the highest **AUC Score (0.8624)** and **Precision (0.6901)** of all models. This indicates it is excellent at ranking customers by risk and minimizing false alarms (high precision), though its recall is lower (0.4477).
- **Naive Bayes**: While having lower overall accuracy, it retains the highest **Recall (77.48%)**. This model is the best choice if the primary business goal is to catch every potential churner, even if it means contacting some loyal customers by mistake (higher false positives).
- **kNN**: Performance has improved with recent tuning (Accuracy ~78%), making it a competitive mid-tier model, though still computationally heavier than linear models.
- **Decision Tree**: Showed moderate performance and performing model across almost all metrics, likely due to overfitting on the training data.

## 5. Deployment
The application is built using **Streamlit**.
- Upload a dataset (CSV) to see predictions.
- Choose from any of the 6 trained models.
- View evaluation metrics and confusion matrix.