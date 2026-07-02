import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_model.pkl")

model = load_model()
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction System")
st.write("Enter the patient's values below to predict whether the tumor is Benign or Malignant.")

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

if st.button("Predict"):
    prediction = model.predict([inputs])

    if prediction[0] == 0:
        st.error("⚠️ Prediction: Malignant")
    else:
        st.success("✅ Prediction: Benign")
