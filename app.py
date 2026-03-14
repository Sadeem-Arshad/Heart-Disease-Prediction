import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Risk Predictor")
st.markdown("### CDC BRFSS 2015 Dataset Analysis")

@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    
    if os.path.exists('Heart_Disease_Prediction.pkl'):
        with open('Heart_Disease_Prediction.pkl', 'rb') as f:
            model = pickle.load(f)
    
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_model_and_scaler()

tab1, tab2, tab3 = st.tabs(["Predict", "Model Info", "About"])

with tab1:
    st.header("Enter Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age Group", [
            "18-24", "25-29", "30-34", "35-39", "40-44",
            "45-49", "50-54", "55-59", "60-64", "65-69",
            "70-74", "75-79", "80+"
        ])
        sex = st.radio("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        high_bp = st.checkbox("High Blood Pressure")
        high_chol = st.checkbox("High Cholesterol")
        stroke = st.checkbox("History of Stroke")
        diabetes = st.selectbox("Diabetes Status", ["No", "Pre-diabetes", "Yes"])

    with col3:
        physical_activity = st.checkbox("Physically Active")
        smoker = st.checkbox("Smoker")
        fruits = st.checkbox("Eats Fruits Daily")
        veggies = st.checkbox("Eats Vegetables Daily")
        heavy_alcohol = st.checkbox("Heavy Alcohol Consumption")
        
    st.markdown("---")
    col6, col7 = st.columns(2)
    
    with col6:
        any_healthcare = st.checkbox("Has Any Healthcare Coverage")
        chol_check = st.checkbox("Cholesterol Check in Last 5 Years")
        
    with col7:
        no_doc_cost = st.checkbox("Could Not See Doctor Due to Cost")

    col4, col5 = st.columns(2)

    with col4:
        general_health = st.select_slider(
            "General Health",
            options=["Excellent", "Very Good", "Good", "Fair", "Poor"]
        )
        mental_health = st.slider("Days of Poor Mental Health (last 30 days)", 0, 30, 0)

    with col5:
        physical_health = st.slider("Days of Poor Physical Health (last 30 days)", 0, 30, 0)
        diff_walk = st.checkbox("Difficulty Walking")

    if st.button("Predict Risk", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not found! Please train and save the model first.")
        elif scaler is None:
            st.error("Scaler not found! Please save the scaler as 'scaler.pkl'")
        else:
            try:
                age_map = {
                    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
                    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
                    "70-74": 11, "75-79": 12, "80+": 13
                }
                health_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
                diabetes_map = {"No": 0, "Pre-diabetes": 1, "Yes": 2}

                input_data = pd.DataFrame({
                    'HighBP': [1 if high_bp else 0],
                    'HighChol': [1 if high_chol else 0],
                    'CholCheck': [1 if chol_check else 0],
                    'BMI': [bmi],
                    'Smoker': [1 if smoker else 0],
                    'Stroke': [1 if stroke else 0],
                    'Diabetes': [diabetes_map[diabetes]],
                    'PhysActivity': [1 if physical_activity else 0],
                    'Fruits': [1 if fruits else 0],
                    'Veggies': [1 if veggies else 0],
                    'HvyAlcoholConsump': [1 if heavy_alcohol else 0],
                    'AnyHealthcare': [1 if any_healthcare else 0],
                    'NoDocbcCost': [1 if no_doc_cost else 0],
                    'GenHlth': [health_map[general_health]],
                    'MentHlth': [mental_health],
                    'PhysHlth': [physical_health],
                    'DiffWalk': [1 if diff_walk else 0],
                    'Sex': [1 if sex == "Male" else 0],
                    'Age': [age_map[age]]
                })

                input_scaled = scaler.transform(input_data)
                
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]

                st.markdown("---")
                st.subheader("Prediction Results")

                if prediction == 1:
                    st.error("HIGH RISK - Heart Disease Detected")
                    st.metric("Risk Probability", f"{probability[1]*100:.1f}%")
                else:
                    st.success("LOW RISK - No Heart Disease Detected")
                    st.metric("Risk Probability", f"{probability[1]*100:.1f}%")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("No Disease Probability", f"{probability[0]*100:.1f}%")
                with col_b:
                    st.metric("Disease Probability", f"{probability[1]*100:.1f}%")

                st.info("This is a prediction tool. Please consult with healthcare professionals for proper diagnosis.")
            
            except ValueError as e:
                st.error(f"Prediction Error: {str(e)}")
                if hasattr(model, 'n_features_in_'):
                    st.warning(f"Model expects {model.n_features_in_} features but received {input_data.shape[1]} features")
                st.info("Please ensure the model was trained with the same features in the same order")

with tab2:
    st.header("Model Performance")

    if model:
        st.success(f"Model Loaded: {type(model).__name__}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset Size", "253,680")
        with col2:
            st.metric("Target Recall", "80%+")
        with col3:
            st.metric("Model Type", "Random Forest" if isinstance(model, RandomForestClassifier) else "Logistic Regression")
        with col4:
            st.metric("Scaler", "Loaded" if scaler else "Missing")

        st.markdown("### Model Configuration")
        if isinstance(model, RandomForestClassifier):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Trees", "200")
            with col_b:
                st.metric("Max Depth", "15")
            with col_c:
                st.metric("Features", "19")

        st.markdown("### Key Features")
        st.markdown("""
        - High Blood Pressure
        - High Cholesterol
        - Cholesterol Check (Last 5 Years)
        - BMI (Body Mass Index)
        - Smoking Status
        - Stroke History
        - Diabetes Status
        - Physical Activity
        - Diet (Fruits & Vegetables)
        - Alcohol Consumption
        - Healthcare Coverage
        - Unable to See Doctor Due to Cost
        - General Health Status
        - Mental & Physical Health Days
        - Difficulty Walking
        - Sex & Age
        """)
    else:
        st.warning("No trained model found. Please train your model and save it as 'Heart_Disease_Prediction.pkl'")
        st.code("""
import pickle

with open('Heart_Disease_Prediction.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
        """)

with tab3:
    st.header("About This Application")
    st.markdown("""
    ### CDC BRFSS 2015 Heart Disease Prediction
    
    This application uses machine learning to predict heart disease risk based on the 
    CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset.
    
    **Dataset Information:**
    - Total Samples: 253,680
    - Source: CDC BRFSS 2015
    - Target: Heart Disease Prediction
    - Class Balance: Undersampled 50-50
    
    **Model Objective:**
    - Achieve 80%+ recall to minimize false negatives
    - Identify high-risk individuals for early intervention
    
    **Preprocessing:**
    - StandardScaler normalization
    - Train-Test Split: 80-20
    - Stratified sampling
    
    """)

    st.markdown("---")
    st.markdown("**Model Training Code:**")

    st.code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_balanced, y_train_balanced)

with open('Heart_Disease_Prediction.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    """, language="python")

st.sidebar.title("Navigation")
st.sidebar.info("Use the tabs above to navigate through the application")
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Dataset", "CDC BRFSS 2015")
st.sidebar.metric("Samples", "253,680")
st.sidebar.metric("Target Recall", "80%+")
st.sidebar.metric("Balance", "50-50")