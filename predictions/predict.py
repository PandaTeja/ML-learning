import pickle
import numpy as np

# Load model
model = pickle.load(open("pickle/churnModel.pkl", "rb"))

# Example new customer data
customer = np.array([
    1,  # gender
    0,  # SeniorCitizen
    1,  # Partner
    0,  # Dependents
    5,  # tenure
    1,  # PhoneService
    0,  # MultipleLines
    1,  # InternetService
    0,  # OnlineSecurity
    0,  # OnlineBackup
    0,  # DeviceProtection
    0,  # TechSupport
    1,  # StreamingTV
    1,  # StreamingMovies
    0,  # Contract
    1,  # PaperlessBilling
    2,  # PaymentMethod
    70.5, # MonthlyCharges
    200.5 # TotalCharges
]).reshape(1,-1)


prediction = model.predict(customer)

if prediction[0] == 1:
    print("Customer will CHURN")
else:
    print("Customer will STAY")