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

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Full page background */
    body, .stApp {
        background-color: #cce7ff;  /* light blue */
        color: #000000;             /* all text black */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Remove Streamlit extra headers and sidebar links */
    header, [data-testid="stSidebarNav"], .css-18ni7ap {display: none;}

    /* White input container */
    .input-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    /* Input fields */
    input, select {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid #ced4da;
        margin-bottom: 12px;
    }

    /* Button */
    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: #ffffff;
        font-size: 16px;
        padding: 12px 25px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1558b0;
    }

    /* Prediction output */
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ced4da;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        color: #000000;
        margin-top: 20px;
    }

    /* Custom title */
    .app-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #000000;
        margin-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Custom HTML title (removes 3-box problem) ---
st.markdown('<div class="app-title">Random Forest Predictor</div>', unsafe_allow_html=True)

# --- White container for inputs ---
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Collect categorical inputs
user_data = {}
for cat_col, options in categorical_options.items():
    user_data[cat_col] = st.selectbox(f"{cat_col}", options)

# Collect numeric inputs
numeric_features = [f for f in features if all(not f.startswith(cat + "_") for cat in categorical_options.keys())]
numeric_input = {}
for num_feat in numeric_features:
    numeric_input[num_feat] = st.number_input(f"{num_feat}", value=0.0)

# Build input DataFrame
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

# Scale features
user_input_scaled = scaler.transform(input_df)

st.markdown('</div>', unsafe_allow_html=True)  # close input container

# Predict button
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)
    st.markdown(f'<div class="prediction-box">Predicted class: {pred_label[0]}</div>', unsafe_allow_html=True)
