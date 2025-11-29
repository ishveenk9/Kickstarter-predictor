### Kickstarter Success Predictor

This project is a Streamlit web application that predicts the likelihood of success for a Kickstarter project using a trained machine-learning model.
Users can enter project details, explore feature descriptions, and instantly receive a prediction based on their real and current Kickstarter data.

# 🚀 Overview
This application provides an interface for forecasting the success of a Kickstarter campaign.
A pre-trained model (stored using pickle) processes user input to return a prediction.

The app includes two core pages:
🏠 Home Page – Input fields + prediction output
📘 Feature Explanations Page – Definitions for all features

# 🧠 How the App Works

## 🔹 Model Loading

When the app launches, it loads a model_full.pkl file containing:
model — trained prediction model
scaler — numerical feature scaler
label_encoder — transforms predicted classes to readable labels
features — full model feature list
categorical_options — dropdown choices for categorical inputs

This ensures consistency with the preprocessing used during model training.

## 🏠 Home Page (Prediction Interface)
Passes the input to the ML model depending on wheather the fields are categorical or numerical. 
Display predicted class inside a custom-styled results box.

## 📘 Feature Explanations Page
This page, accessible from the sidebar, provides definitions for each input field.

## 🛠️ Technologies Used
Python 3.10+
Streamlit
pandas
numpy
pickle

## You can test it out for yourself [here.]([https://example.com](https://kickstarter-sucess-predictor.streamlit.app/?page=explain_features))
