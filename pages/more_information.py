# This file contains feature explanations so the users have a better idea of what each field means. Additionally, this page is a seprate tab from the 
# home page which can see seen in the side navbar 

import streamlit as st

# Custom CSS thats is the same as Home page
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: ##FDFBD4;
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stTitle h1 {
        color: #000000 !important;
        text-decoration: none !important;
        pointer-events: none !important;
    }

    label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# All the text on the page 
st.title("Feature Explanations")
st.write("Here are all the input-field descriptions to help you generate the most accurate results for your project!")

st.markdown("""
- **State**: The state in which the project was created.
- **Category**: The category your project falls under (e.g. Design).
- **Subcategory**: The subcategory your project falls under.
- **Goal**: The amount of money you are trying to raise for your project.
- **Levels**: The tiers of donations you offer to supporters (e.g. $1, $10, $100, $1000 = 4 levels).
- **Updates**: The number of updates made to the project at the start of the Kickstarter. This can increase throughout the campaign.
- **Comments**: The number of messages or posts about the project made by the team.
- **Duration**: The length of the Kickstarter in days.
- **Funded Month**: The month in which the project will attempt to receive funding.
""")
