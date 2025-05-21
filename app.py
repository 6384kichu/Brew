import streamlit as st
import pandas as pd
import joblib

st.title("Savvy Sips Brew Analytics – Ingredient Impact")

# Load model and data
model = joblib.load("brew_model.pkl")
df = pd.read_csv("SavvySips_FullDataset.csv")

# Input section
st.header("Predict Brew Performance Based on Ingredients")

ingredient_name = st.selectbox("Ingredient", df['ingredient_name'].unique())
ingredient_category = st.selectbox("Ingredient Category", df['ingredient_category'].unique())
beer_style = st.selectbox("Beer Style", df['beer_style'].unique())
customer_segment = st.selectbox("Customer Segment", df['customer_segment'].unique())
marketing_spend = st.number_input("Marketing Spend", min_value=0, value=10000)
social_media_engagement = st.number_input("Social Media Engagement", min_value=0, value=1000)

# Form input
input_df = pd.DataFrame([{
    "ingredient_name": ingredient_name,
    "ingredient_category": ingredient_category,
    "beer_style": beer_style,
    "customer_segment": customer_segment,
    "marketing_spend": marketing_spend,
    "social_media_engagement": social_media_engagement
}])

if st.button("Predict Metrics"):
    prediction = model.predict(input_df)
    st.subheader("Predicted Outcomes")
    st.write(f"🍺 **Ingredient Affinity Index (IAI):** {prediction[0][0]:.2f}")
    st.write(f"🌍 **Market Penetration Rate (MPR):** {prediction[0][1]:.2f}%")
    st.write(f"🔁 **Repeat Purchase Ratio (RPR):** {prediction[0][2]:.2f}%")
