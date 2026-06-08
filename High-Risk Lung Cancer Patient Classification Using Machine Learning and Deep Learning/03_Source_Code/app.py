import streamlit as st
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")

st.title("Lung Cancer High Risk Prediction System")

st.write("Enter patient values to predict risk class.")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, value=30.0)
smoking_pack_years = st.number_input("Smoking Pack Years", min_value=0.0, value=10.0)
survival_months = st.number_input("Survival Months", min_value=0, value=24)

st.warning("This demo uses sample inputs. Full model input requires all selected dataset features.")

if st.button("Predict"):
    st.info("GUI interface created successfully. Full feature input connection will be completed after final selected feature list is fixed.")

# GUI Çalıştırmam için konsola python -m streamlit run app.py yazarak erişebilirim.