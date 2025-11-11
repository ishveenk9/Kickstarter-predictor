# app_full_features.py
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
features = data["features"]  # all features including one-hot
categorical_options = data["categorical_options"]

st.title("Random Forest Predictor")

st.markdown("### Select categorical values:")

# --- Collect categorical inputs ---
user_data = {}
for cat_col, options in categorical_options.items():
    choice = st.selectbox(f"{cat_col}", options)
    user_data[cat_col] = choice

st.markdown("### Enter numeric features:")

# --- Identify numeric features ---
numeric_features = [f for f in features if all(not f.startswith(cat + "_") for cat in categorical_options.keys())]
numeric_input = {}
for num_feat in numeric_features:
    val = st.number_input(f"{num_feat}", value=0.0)
    numeric_input[num_feat] = val

# --- Build input DataFrame matching model features ---
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0  # initialize all zeros

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
    st.write(f"### Predicted class: {pred_label[0]}")
