import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import requests

st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("Telco Customer Churn Prediction")
st.markdown("""
This app predicts customer churn using various Machine Learning models.
Upload a sample CSV file to evaluate the model performance.
""")

# Function to download sample CSV
@st.cache_data
def get_sample_data():
    url = "https://raw.githubusercontent.com/2025aa05175-svg/ML_Assignment/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading sample file: {e}")
        return None

sample_data = get_sample_data()


# Load models and transformers
@st.cache_resource
def load_artifacts():
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f not in ['label_encoders.pkl', 'scaler.pkl']]
    
    for file in model_files:
        model_name = file.replace('.pkl', '').replace('_', ' ').title()
        
        if model_name == "Knn": model_name = "KNN"
        if "Xgboost" in model_name: model_name = "XGBoost"
        
        with open(os.path.join(model_dir, file), 'rb') as f:
            models[model_name] = pickle.load(f)
            
    with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
        
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    return models, label_encoders, scaler

try:
    models, label_encoders, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model/' directory exists and contains trained models.")
    st.stop()

# Sidebar
if sample_data:
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=sample_data,
        file_name="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        mime="text/csv",
        help="Download Sample CSV for testing"
    )

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
        
        # Preprocessing
        df_proc = df.copy()
        
        # 1. TotalCharges -> numeric
        df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'], errors='coerce')
        df_proc['TotalCharges'] = df_proc['TotalCharges'].fillna(0) # Simple fill for app
        
        # Drop customerID
        if 'customerID' in df_proc.columns:
            df_proc = df_proc.drop(columns=['customerID'])
            
        # 2. Encode
        for col, le in label_encoders.items():
            if col in df_proc.columns:
                # Handle unknown labels via applying map/lambda to avoid error
                # Or just use transform if we are sure data matches
                # For safety in app:
                df_proc[col] = df_proc[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                
        # 3. Scale
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        # Ensure cols exist
        if all(c in df_proc.columns for c in num_cols):
             df_proc[num_cols] = scaler.transform(df_proc[num_cols])
        
        # Model Selection
        clean_model_names = list(models.keys())
        selected_model_name = st.sidebar.selectbox("Select Model", clean_model_names)
        
        model = models[selected_model_name]
        
        if st.sidebar.button("Run Prediction"):
            # Separate features and target
            if 'Churn' in df_proc.columns:
                X = df_proc.drop('Churn', axis=1)
                y_true = df_proc['Churn']
                             
                # Predict
                y_pred = model.predict(X)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]
                else:
                    y_prob = None # KNN
                
                # Metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='binary', pos_label=1) 
                
                rec = recall_score(y_true, y_pred, pos_label=1)
                f1 = f1_score(y_true, y_pred, pos_label=1)
                mcc = matthews_corrcoef(y_true, y_pred)
                try:
                    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
                except:
                    auc = 0

                st.write(f"### Results for {selected_model_name}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("F1 Score", f"{f1:.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Precision", f"{prec:.4f}")
                col5.metric("Recall", f"{rec:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")
                
                # Confusion Matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
                
            else:
                st.warning("Uploaded CSV must contain 'Churn' column for evaluation.")
                
                # Just predict and show
                st.write("Predicting on data without ground truth...")
                y_pred = model.predict(df_proc)
                st.write(pd.DataFrame(y_pred, columns=['Predicted_Churn']))
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")