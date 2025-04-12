import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

# API Details
API_KEY = '1afdd88fb14c4b25e2e9192d7eabbba9'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# --- Functions ---
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        st.error("City not found or API limit reached.")
        return None

    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind']['speed']
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    X = data[['MinTemp', 'MaxTemp', 'WindGustDir','WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_test_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

def get_compass_direction(deg):
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    return next((point for point, start, end in compass_points if start <= deg < end), "N")

# --- Streamlit UI ---
st.title("🌦️ Weather Forecast & Rain Prediction App")
city = st.text_input("Enter a city name", "Mumbai")

if city:
    current_weather = get_current_weather(city)
    if current_weather:
        df = read_historical_data("weather.csv")
        X, y, le = prepare_data(df)
        rain_model = train_test_model(X, y)

        compass_direction = get_compass_direction(current_weather['wind_gust_dir'])
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['WindGustSpeed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        # Temperature & Humidity Forecast
        X_temp, y_temp = prepare_regression_data(df, 'Temp')
        X_hum, y_hum = prepare_regression_data(df, 'Humidity')

        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Display
        st.subheader(f"📍 Weather in {current_weather['city']}, {current_weather['country']}")
        st.write(f"🌡️ Current Temp: {current_weather['current_temp']}°C (Feels like: {current_weather['feels_like']}°C)")
        st.write(f"🔽 Min Temp: {current_weather['temp_min']}°C | 🔼 Max Temp: {current_weather['temp_max']}°C")
        st.write(f"💧 Humidity: {current_weather['humidity']}% | 🌀 Wind: {current_weather['WindGustSpeed']} m/s ({compass_direction})")
        st.write(f"🔍 Condition: {current_weather['description'].capitalize()}")
        st.success(f"🌧️ Rain Prediction: {'Yes' if rain_prediction else 'No'}")

        st.subheader("📈 Forecast for Next 5 Hours")
        forecast_df = pd.DataFrame({
            "Time": future_times,
            "Temperature (°C)": [round(t, 1) for t in future_temp],
            "Humidity (%)": [round(h, 1) for h in future_humidity]
        })
        st.dataframe(forecast_df)

