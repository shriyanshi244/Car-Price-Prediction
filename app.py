import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the model
model = pk.load(open('model.pkl', 'rb'))

# Page title and description
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction App")
st.subheader("Predict the resale value of your car using machine learning!")

# Load and preprocess car data
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Sidebar for input collection
st.sidebar.header("Input Features")
st.sidebar.markdown("Please provide the following details about your car:")

name = st.sidebar.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.sidebar.slider('Car Manufactured Year', 1994, 2024, step=1)
km_driven = st.sidebar.slider('Number of Kilometers Driven', 11, 200000, step=1000)
fuel = st.sidebar.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.sidebar.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.sidebar.slider('Car Mileage (km/l)', 10, 40, step=1)
engine = st.sidebar.slider('Engine Capacity (CC)', 700, 5000, step=50)
max_power = st.sidebar.slider('Maximum Power (BHP)', 0, 200, step=1)
seats = st.sidebar.slider('Number of Seats', 5, 10, step=1)

# Display a car-themed GIF
st.image(
    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2gwbnFrMWR1ZDF0OHRhd3JtcWF3bm8xaTR6eGFmbzh5aXJ0M2s4MiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3ov9jWu7BuHufyLs7m/giphy.gif",
    caption="Drive into the future with AI!",
    # use_column_width=True,
)

# Predict Button
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Feature encoding
    input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5], inplace=True
    )
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        range(1, 32), inplace=True
    )

    # Predict the car price
    car_price = model.predict(input_data_model)

    # Display the prediction
    st.success(f"💰 Estimated Car Price: **₹{car_price[0]:,.2f} **")

# Footer
st.markdown("---")
st.markdown("© 2025 | Car Price Prediction App by Shriyanshi")
