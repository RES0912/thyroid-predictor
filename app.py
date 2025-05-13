import streamlit as st
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample Data: Replace with your actual data or load from a CSV
data = {"D:\data.csv"}

df = pd.DataFrame(data)

# Split the data into features and target variable
X = df.drop('thyroid_condition', axis=1)
y = df['thyroid_condition']

# Train the CatBoost model
model = CatBoostClassifier(verbose=0)  # Silent mode to avoid unnecessary logs
model.fit(X, y)

# Save the trained model
model.save_model("catboost_thyroid_model.cbm")
st.write("Model has been trained and saved as 'catboost_thyroid_model.cbm'.")

# Allow the user to upload a file for prediction

uploaded_file = st.file_uploader("D:\data.csv", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    user_data = pd.read_csv(uploaded_file)
    st.write("User data", user_data)
    
    # Predict using the model
    try:
        # Load the saved model
        model.load_model("catboost_thyroid_model.cbm")
        
        # Prepare input features for prediction (assuming it matches training data format)
        user_data = user_data.drop('thyroid_condition', axis=1, errors='ignore')
        
        # Predict
        predictions = model.predict(user_data)
        
        # Display predictions
        st.write("Predictions:")
        st.write(predictions)

        # You can also add a classification report here, if needed
        # If you have labels, you could compare and generate the report
        if 'thyroid_condition' in user_data.columns:
            y_true = user_data['thyroid_condition']
            st.write(classification_report(y_true, predictions))

    except Exception as e:
        st.error(f"Error: {e}")

