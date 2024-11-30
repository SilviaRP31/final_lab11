import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page layout
st.set_page_config(
    page_title="Healthcare Analysis and Explainability",
    page_icon="🏥",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model_and_data():
    with open('OF_PERSIS_ANYTIME_classification_model.pkl', 'rb') as file:
        model = pickle.load(file)
    data = pd.read_csv('Stage_2.csv', sep=';')
    return model, data

model, data = load_model_and_data()

# Navigation Menu
st.title("🏥 Healthcare Analysis and Explainability")
menu = ["Data Exploration", "Predictive Model", "Explainability"]
choice = st.radio("Navigate to:", menu, horizontal=True)

# Define common utility functions
def plot_correlation_matrix(correlation_matrix):
    """Plots the correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    st.pyplot(plt)

def get_top_correlations(correlation_matrix, target_col, top_n=3):
    """Returns the top N correlated features with the target column."""
    target_corr = correlation_matrix[target_col].drop(target_col)
    return target_corr.abs().sort_values(ascending=False).head(top_n)

# Main content
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if choice == "Data Exploration":
    st.header("📊 Data Exploration")
    st.write("### Select Patients by ID Range")

    # Patient ID Range Selection
    min_id, max_id = st.select_slider(
        "Select range of Patient IDs:",
        options=sorted(data["ID_PATIENT"].unique()),
        value=(data["ID_PATIENT"].min(), data["ID_PATIENT"].max())
    )
    st.write(f"Selected Patient ID Range: **{min_id} to {max_id}**")
    
    # Filter data by selected range
    filtered_data = data[(data["ID_PATIENT"] >= min_id) & (data["ID_PATIENT"] <= max_id)]
    st.write("### Filtered Dataset")
    st.dataframe(filtered_data.set_index("ID_PATIENT"))

    # Plotting the confusion matrix for the selected patients
    if not filtered_data.empty:
        st.write("### Confusion Matrix for Selected Patients")
        X_filtered = filtered_data.drop(columns=["ID_PATIENT", "OF_PERSIS_ANYTIME"])
        y_filtered = filtered_data["OF_PERSIS_ANYTIME"]
        
        y_pred = model.predict(X_filtered)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_filtered, y_pred, labels=model.classes_)
        
        # Display confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        st.pyplot(fig)
    else:
        st.warning("No patients found in the selected range.")

    # Correlation Matrix
    st.write("### Global Correlation Matrix")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_columns].corr()
    plot_correlation_matrix(correlation_matrix)

    # Top Correlated Features
    st.write("### Top 3 Correlated Features with 'OF_PERSIS_ANYTIME'")
    top_3_corr = get_top_correlations(correlation_matrix, "OF_PERSIS_ANYTIME")
    st.write(top_3_corr)


elif choice == "Predictive Model":
    st.header("🧠 Predictive Model")
    st.write("### Input Patient Data")

    # Create columns for user inputs
    num_columns = 3  # You can adjust this number for more or fewer columns
    columns = st.columns(num_columns)
    
    # Create a dictionary to store the input data
    user_data = {}
    
    # Exclude the target and ID_PATIENT from the input features
    feature_columns = [col for col in data.columns if col not in ["ID_PATIENT", "OF_PERSIS_ANYTIME"]]
    
    # Iterate over the columns and input fields
    for i, col in enumerate(columns):
        # Distribute the features across the columns
        start_idx = i * len(feature_columns) // num_columns
        end_idx = (i + 1) * len(feature_columns) // num_columns
        feature_subset = feature_columns[start_idx:end_idx]
        
        for feature in feature_subset:
            user_data[feature] = col.number_input(f"Enter {feature}", value=0.0)
    
    # Convert the input data into a DataFrame, ensure correct column order
    input_data = pd.DataFrame([user_data])[feature_columns]  # Ensure columns are in the same order as during training

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the prediction
    st.write("### Prediction Results")
    st.write(f"Predicted Class: **{prediction}**")
    st.write(f"Probability of Class 1 (OF_PERSIS_ANYTIME): **{prediction_proba[1]:.2f}**")

elif choice == "Explainability":
    st.header("🌍 Global Explainability")
    st.write("""
        ### How the Predictive Model Works
        The predictive model is a machine learning classifier trained to predict the outcome of **OF_PERSIS_ANYTIME** based on several 
             input features. These features represent various patient characteristics used to predict whether the patient will experience the event 
             or condition described by the targe tvariable.

        The model works by learning patterns in the data that correlate with the target variable. We will use SHAP in order to visualize the data and break down each feature's 
             contribution to the model’s predictions, helping to understand which features are most relevant in the predictions.
    """)

    st.write("""
        ### SHAP Summary Plot
        SHAP summary plots provide an overview of how each feature affects the predictions made by the model. The plot shows the impact of each feature on the output across 
             all instances, helping to identify the most important features.
    """)

    # SHAP Explanation
    explainer = shap.Explainer(model)
    X_test = data.drop(columns=["ID_PATIENT", "OF_PERSIS_ANYTIME"])
    shap_values = explainer(X_test)

    # SHAP Summary Plot
    shap.summary_plot(shap_values[:, :, 1], X_test)
    st.pyplot(plt)

    st.write("""
        ### Key Inisghts from SHAP Summary Plot:
        - Features at the top of the plot (e.g., **SAT02%/FIO2%_ADMISS**, **HEARTRATED_ADMISS**, **AGE**, etc.) have the most significant impact on predictions, as their X range is wider, meaning that the 
             feature significantly influences the model’s prediction. 
        - The color of the points in the plot indicates whether the feature value is high (red) or low (blue), we can see that the lower **SAT02%/FIO2%_ADMISS** the most it 
        influences the prediction, also in the **HEARTRATED_ADMISS** the highest the most it influences or the **AGE** that when it's higher it will influence more 
        in the predictive model, showing in this cases more proximity to suffer from healthcare problems.
    """)

    st.header("🔍 Local Explainability")
    st.write("""
        ### SHAP Waterfall Plots
        Select a class (or both) to see detailed explanations for an individual prediction.
        - **Class 0**: Patients predicted with **No Persistence**.
        - **Class 1**: Patients predicted with **Persistence**.
    """)

    # SHAP values and probabilities
    explainer = shap.Explainer(model)
    X_test = data.drop(columns=["ID_PATIENT", "OF_PERSIS_ANYTIME"])
    shap_values = explainer(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Patient with class 0 prediction
    st.write("#### Patient with Prediction: Class 0")
    patient_class_0_idx = np.where(y_pred_proba[:, 0] > 0.5)[0][0]
    shap_values_class_0 = shap_values[patient_class_0_idx]
    shap.plots.waterfall(shap_values_class_0[:, 0])
    st.pyplot(plt)

    # Patient with class 1 prediction
    st.write("#### Patient with Prediction: Class 1")
    patient_class_1_idx = np.where(y_pred_proba[:, 1] > 0.5)[0][0]
    shap_values_class_1 = shap_values[patient_class_1_idx]
    shap.plots.waterfall(shap_values_class_1[:, 1])
    st.pyplot(plt)

    st.write("""
        ### Conclusion
        By using SHAP values, we can easily understand and explain how features influence the model’s predictions, and therefore the patient's health.
    """)

