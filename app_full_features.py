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
    /* Remove section link anchors */
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Overall page styling */
    body, .stApp {
        background-color: #cce7ff;  /* light blue background */
        color: #000000;             /* all text black */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title styling: remove extra boxes */
    .stTitle {
        color: #000000 !important;
        font-size: 2.5em !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 40px !important;
        padding: 0 !important;
        background-color: transparent !important;
    }

    /* Section labels and input fields */
    label, .stTextInput, .stNumberInput, .stSelectbox {
        font-size: 16px;
        font-weight: 500;
        color: #000000 !important;
    }

    /* Input containers: white background boxes */
    .stNumberInput > div > input, 
    .stSelectbox > div > div > div > span {
        background-color: #ffffff !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
        margin-bottom: 12px !important;
        color: #000000 !important;
    }

    /* Button styling */
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
        color: #ffffff;
    }

    /* Prediction output box: white background */
    .stMarkdown div {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid #ced4da;
        color: #000000;
        text-align: center;
    }

    /* Remove top random box */
    header, .css-18ni7ap {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Content ---
st.title("Random Forest Predictor")

# White container for inputs
st.markdown('<div style="background-color:#ffffff; padding:20px; border-radius:10px;">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)  # Close white container

# Predict button
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)
    st.markdown(f"### Predicted class: {pred_label[0]}")
