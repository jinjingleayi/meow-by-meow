import streamlit as st
import joblib

st.title('Meow-by-Meow')


# Replace with the path to your saved file
model_filename = '../../data/trained_models/knn.pkl'

# Load the model from disk
loaded_model = joblib.load(model_filename)

st.write(loaded_model)
