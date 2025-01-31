## Weather Prediction System

This Python project allows users to get real-time weather updates for any city and predict future temperature and humidity values using historical weather data. The system fetches current weather data from the OpenWeather API and processes it for predictions using machine learning models. It also predicts whether it will rain the next day based on historical weather data.

## Features
Current Weather Data Fetching:

Retrieves the current weather conditions of a specified city.
Displays details like temperature, humidity, wind speed, and weather description.
Supports metric units for temperature (Celsius).
Rain Prediction:

Trains a Random Forest Classifier on historical weather data to predict if it will rain tomorrow.
Uses features like temperature, humidity, wind speed, and wind gust direction for prediction.
Future Temperature and Humidity Prediction:

Trains Random Forest Regressor models for temperature and humidity.
Predicts the future values of temperature and humidity for the next 5 hours based on the current values.
Data Preprocessing:

Reads historical weather data from a CSV file.
Cleans the data by removing missing values and duplicates.
Encodes categorical features such as wind direction using label encoding.
Timezone Support:

Uses the pytz library to handle timezone conversions and provide accurate future time predictions.
Requirements
requests for making API calls.
numpy for numerical operations.
pandas for data manipulation.
sklearn for machine learning models and metrics.
pytz for timezone handling.
API Key
To use the weather fetching functionality, you will need to sign up for an OpenWeather API key. Replace API_KEY with your own key.

## Code Overview
get_current_weather(city): Fetches current weather details for the specified city.

read_historical_data(filename): Reads historical weather data from a CSV file.

prepare_data(data): Prepares the dataset by encoding categorical features and splitting data into features and target.

train_test_model(X, y): Trains a Random Forest Classifier to predict if it will rain tomorrow.

prepare_regression_data(data, feature): Prepares data for regression model training, predicting the next value based on previous data.

train_regression_model(X, y): Trains a Random Forest Regressor for predicting continuous variables like temperature and humidity.

predict_future(model, current_value): Uses a trained regression model to predict future values.

weather_view(): The main function that prompts the user for a city name, retrieves the weather data, and displays predictions for rain, temperature, and humidity.


Usage

Ensure you have all the required libraries installed:

pip install requests numpy pandas scikit-learn pytz


To get weather updates and predictions, run the script:

weather_view()


The script will:

Ask for the city name.
Display current weather details.
Predict rain status for tomorrow.
Provide temperature and humidity predictions for the next few hours.
