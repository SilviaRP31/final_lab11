import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
@st.cache_resource
def load_model_and_data():
    # Load the model
    with open('OF_PERSIS_ANYTIME_classification_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load the dataset
    data = pd.read_csv('Stage_2.csv', sep=';')
    return model, data

model, data = load_model_and_data()

# Sidebar navigation
st.sidebar.title("Navigation")
options = ["Data Exploration", "Predictive Model", "Global Explainability", "Local Explainability"]
choice = st.sidebar.radio("Go to", options)

if choice == "Data Exploration":
    st.title("Data Exploration")
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    st.write("### Correlation Matrix")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_columns].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    st.pyplot(plt)

    st.write("### Top 3 Correlated Features with 'OF_PERSIS_ANYTIME'")
    target_corr = correlation_matrix["OF_PERSIS_ANYTIME"].drop("OF_PERSIS_ANYTIME")
    top_3_corr = target_corr.abs().sort_values(ascending=False).head(3)
    st.write(top_3_corr)

elif choice == "Predictive Model":
    st.title("Predictive Model")
    st.write("### Input Patient Data")

    # Input form for user data
    user_data = {}
    for col in data.columns:
        if col not in ["ID_PATIENT", "OF_PERSIS_ANYTIME"]:
            user_data[col] = st.number_input(f"Enter {col}", value=0.0)

    input_data = pd.DataFrame([user_data])
    st.write("### Submitted Data")
    st.dataframe(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.write("### Prediction")
    st.write(f"Predicted Class: **{prediction}**")
    st.write(f"Probability of Class 1 (OF_PERSIS_ANYTIME): **{prediction_proba[1]:.2f}**")

elif choice == "Global Explainability":
    st.title("Global Explainability")
    st.write("### SHAP Summary Plot")

    explainer = shap.Explainer(model)
    X_test = data.drop(columns=["ID_PATIENT", "OF_PERSIS_ANYTIME"])
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values[:, :, 1], X_test)
    st.pyplot(plt)

    st.write("### SHAP Bar Plot")
    shap.plots.bar(shap_values[:, :, 1], max_display=15)
    st.pyplot(plt)

elif choice == "Local Explainability":
    st.title("Local Explainability")
    st.write("### SHAP Waterfall and Decision Plots")

    y_pred_proba = model.predict_proba(data.drop(columns=["ID_PATIENT", "OF_PERSIS_ANYTIME"]))

    # Patient with class 0 prediction
    patient_class_0_idx = np.where(y_pred_proba[:, 0] > 0.5)[0][0]
    shap_values_class_0 = shap_values[patient_class_0_idx]
    st.write("#### Patient with Prediction: Class 0")
    shap.plots.waterfall(shap_values_class_0[:, 0])
    st.pyplot(plt)

    # Patient with class 1 prediction
    patient_class_1_idx = np.where(y_pred_proba[:, 1] > 0.5)[0][0]
    shap_values_class_1 = shap_values[patient_class_1_idx]
    st.write("#### Patient with Prediction: Class 1")
    shap.plots.waterfall(shap_values_class_1[:, 1])
    st.pyplot(plt)
