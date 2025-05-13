import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_thyroid_model.cbm")


# Example input UI
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
tsh = st.number_input("TSH Level", value=1.0)
t3 = st.number_input("T3 Level", value=1.0)
t4 = st.number_input("T4 Level", value=1.0)

# Convert input to DataFrame
if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "TSH": [tsh],
        "T3": [t3],
        "T4": [t4]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Thyroid Condition: {prediction}")
