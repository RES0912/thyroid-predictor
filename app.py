import streamlit as st
from catboost import CatBoostClassifier
import pandas as pd

# Title of the app
st.title("Thyroid Disease Prediction")

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_thyroid_model.cbm")

# Upload a CSV file with user data
uploaded_file = st.file_uploader("Upload your data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    user_data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("User Data:", user_data)
    
    # Make predictions
    try:
        # Drop the target column if present
        user_data = user_data.drop('thyroid_condition', axis=1, errors='ignore')
        
        # Predict using the loaded model
        predictions = model.predict(user_data)
        
        # Show predictions
        st.write("Predictions:")
        st.write(predictions)
    except Exception as e:
        st.write("Error:", e)
