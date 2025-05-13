import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier


# Load your dataset
# (Replace this with your actual dataset)
data = {
    'age': [34, 45, 29, 38, 50, 27],
    'sex': [1, 0, 1, 0, 1, 0],
    'TSH': [2.1, 4.5, 0.9, 3.2, 5.6, 0.6],
    'T3': [1.2, 1.0, 1.3, 0.8, 1.1, 1.5],
    'T4': [6.4, 6.9, 5.8, 4.2, 6.0, 5.2],
    'thyroid_condition': ['Congenital Hypothyroidism', 'Normal', 'Hyperthyroidism', 'Hypothyroidism', 'Hypothyroidism', 'Hyperthyroidism']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split into features and target
X = df.drop('thyroid_condition', axis=1)
y = df['thyroid_condition']

# Train the CatBoost model
model = CatBoostClassifier(verbose=0)
model.fit(X, y)

# Save the trained model to a file
model.save_model("catboost_thyroid_model.cbm")

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
