import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("medical_data.csv")
#data = pd.read_csv('C:\\Users\\saive\\Desktop\\myproject\\Quadrant_cognizant\\datasets\\medical_data.csv')


# Convert symptom column to lowercase
data['Symptoms'] = data['Symptoms'].str.lower()

# Encode categorical data
label_encoder_symptoms = LabelEncoder()
label_encoder_medicines = LabelEncoder()

data['symptoms_encoded'] = label_encoder_symptoms.fit_transform(data['Symptoms'])
data['medicines_encoded'] = label_encoder_medicines.fit_transform(data['Medicine'])

# Split the data
X = data['symptoms_encoded'].values.reshape(-1, 1)
y = data['medicines_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Define function to predict medicine
def predict_medicine(symptom):
    symptom = symptom.lower()  # Convert input to lowercase
    symptom_encoded = label_encoder_symptoms.transform([symptom])[0]
    medicine_encoded = model.predict([[symptom_encoded]])[0]
    medicine = label_encoder_medicines.inverse_transform([medicine_encoded])[0]
    return medicine

# Streamlit App
st.title('Medicine Recommendation System')

# User Input
symptom = st.text_input('Enter the symptom')

if st.button('Predict'):
    recommended_medicine = predict_medicine(symptom)
    st.write(f'Recommended Medicine: {recommended_medicine}')
    