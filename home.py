# This file deals with the main page of the application. It extracts in the features 
# from the trained model that is saved using pickle. It also uses what is saved in 
# the pickle file to predict the success of new samples.

# Necessary packages
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Loading in the model
with open("model_full.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]
features = data["features"]
categorical_options = data["categorical_options"]

# field name map
field_name_map = {
    "state": "State",
    "category": "Project Category",
    "subcategory": "Project Subcategory",
    "goal": "Goal (USD)",
    "levels": "Number of Donation Tiers",
    "updates": "Number of Updates",
    "comments": "Number of Comments",
    "duration_days": "Duration (Days)",
    "funded_month": "Funding Month (ex. February: 02)"
}

def rename_field(field):
    return field_name_map.get(field, field)

# Styling which will be changed later
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #FDFBD4;
        color: #000000 !important;  
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stTitle h1 {
        color: #000000 !important;
        text-decoration: none !important;
        pointer-events: none !important;
    }

    div[data-testid="stVerticalBlock"] > div:first-child:empty {
        display: none;
    }

    .stNumberInput, .stSelectbox, .stTextInput {
        color: #000000;  
    }

    div.stButton > button:first-child {
        background-color: #1a73e8;
        color: #ffffff;
        font-size: 16px;
        padding: 12px 25px;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #CCFFFF;
    }

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

    label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Kickstarter Success Predictor")

# categorical variables
user_data = {}

for cat_col, options in categorical_options.items():
    if cat_col == "funded_month":
        choice = st.selectbox(
            rename_field(cat_col),
            list(month_map.keys())
        )
        user_data[cat_col] = month_map[choice]

    else:
        user_data[cat_col] = st.selectbox(
            rename_field(cat_col),
            options
        )

# Numeric fields all which are now integers 
numeric_features = [
    f for f in features 
    if all(not f.startswith(cat + "_") for cat in categorical_options.keys())
]

numeric_input = {}

for num_feat in numeric_features:
    numeric_value = st.number_input(
        rename_field(num_feat),
        value=0,
        step=1,
        format="%d"
    )
    numeric_input[num_feat] = int(numeric_value)

input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0

for cat_col, choice in user_data.items():
    col_name = f"{cat_col}_{choice}"
    if col_name in input_df.columns:
        input_df.at[0, col_name] = 1

for num_feat, val in numeric_input.items():
    input_df.at[0, num_feat] = int(val)

user_input_scaled = scaler.transform(input_df)

# prediction button
if st.button("Predict"):
    pred = model.predict(user_input_scaled)
    pred_label = le.inverse_transform(pred)

    st.markdown(
        f'<div class="prediction-box">Predicted class: {pred_label[0]}</div>',
        unsafe_allow_html=True
    )
