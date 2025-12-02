import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Anil28053/Tourism-Prediction/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for comapany staff that predicts whether customers will buy the tourism package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to make purchase.")

# Collect user input
# Numeric features
Age = st.number_input("Age (years)", min_value=18, max_value=90, value=35)
DurationOfPitch = st.number_input("Duration of sales pitch (minutes)",min_value=0,max_value=200,value=15)
NumberOfPersonVisiting = st.number_input("Number of persons visiting with the customer",min_value=1,max_value=10,value=2)
NumberOfFollowups = st.number_input("Number of follow-ups done after the pitch",min_value=0,max_value=10,value=3)
PreferredPropertyStar = st.selectbox("Preferred hotel property star rating",[3, 4, 5],index=2)
NumberOfTrips = st.number_input("Average number of trips per year",min_value=0,max_value=30,value=1)
PitchSatisfactionScore = st.selectbox("Pitch satisfaction score (1–5)",[1, 2, 3, 4, 5],index=3)
NumberOfChildrenVisiting = st.number_input("Number of children visiting with the customer",min_value=0,max_value=10,value=0)
MonthlyIncome = st.number_input("Gross monthly income of the customer",min_value=0.0,value=100000.0,step=1000.0)

# Categorical features
TypeofContact = st.selectbox("Type of contact",["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City tier",[1, 2, 3],index=0)
Occupation = st.selectbox("Occupation",["Salaried", "Small Business", "Free Lancer", "Large Business"])
Gender = st.selectbox("Gender",["Male", "Female"])
ProductPitched = st.selectbox("Product pitched",["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital status",["Single", "Married", "Divorced", "Unmarried"])
Passport_display = st.selectbox("Does the customer have a passport?",["No", "Yes"])
Passport = 1 if Passport_display == "Yes" else 0
OwnCar_display = st.selectbox("Does the customer own a car?",["No", "Yes"])
OwnCar = 1 if OwnCar_display == "Yes" else 0
Designation = st.selectbox("Designation",["Executive", "Manager", "Senior Manager", "AVP", "VP"])


# Convert inputs to a single-row DataFrame matching model training features
input_data = pd.DataFrame([{
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Passport": Passport,        # 0/1 as in training data
    "OwnCar": OwnCar,            # 0/1 as in training data
    "Designation": Designation
}])


# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
