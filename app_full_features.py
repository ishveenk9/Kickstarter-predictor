import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load model ---
with open("model_full.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]
features = data["features"]
categorical_options = data["categorical_options"]

# --- Custom CSS for clean modern look ---
st.markdown(
    """
    <style>
    /* Overall app styling */
    body, .stApp {
        background-color: #f0f2f6;  /* soft gray */
        color: #212529;              
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main title styling */
    .stTitle {
        color: #1a73e8;  /* pleasant blue */
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Section headers */
    h3 {
        color: #1a73e8;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: white;
        font-size: 16px;
        padding: 10px 25px;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1558b0;
        color: white;
    }

    /* Inputs styling */
    .stNumberInput > div > input, .stSelectbox > div > div > div > span {
        background-color: #ffffff !important;
        padding: 8px !important;
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
        margin-bottom: 12px !important;
    }

    /* Prediction output styling */
    .stMarkdown div {
        background-color: #e3f2fd;  /* soft blue highlight */
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid #1a73e8;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App content ---
st.title("Random Forest Predictor")

st.subheader("Select categorical values:")

# --- Collect categorical inputs ---
user_data = {}
for cat_col, options in categorical_options.items():
    choice = st.selectbox(f"{cat_col}", options)
    user_data[cat_col] = choice

st.subheader("Enter numeric features:")

# --- Identify numeric features ---
numeric_features = [f for f in features if all(not f.startswith(cat + "_") for cat in categorical_options.keys())]
numeric_input = {}
for num_feat in numeric_features:
    val = st.number_input(f"{num_feat}", value=0.0)
    numeric_input[num_feat] = val

# --- Build input DataFrame ---
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0

# Set categorical selections
for cat_col, choice in user_data.items():
    col_name = f"{cat_col}_{choice}"
    if col_name in input_df.columns:
        input_df.at[0, col_name] = 1

# Set numeric inputs
for num_feat, val in numeric_input.items():
    input_df.at[0, num_feat] = val

# --- Scale features ---
user_input_scaled = scaler.transform(input_df)

# --- Predict ---
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)
    st.markdown(f"### Predicted class: {pred_label[0]}")
