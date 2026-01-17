# pages/3_Similarity.py

import streamlit as st
import os
import importlib.util

# =============================
# Helper: Load Python module from file
# =============================
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# =============================
# Absolute path to clustering module
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
clustering_path = os.path.join(BASE_DIR, "clustering.py")

# Load clustering module dynamically
clustering = load_module_from_file("clustering", clustering_path)

# =============================
# Streamlit page config
# =============================
st.set_page_config(page_title="Defect Category Predictor", page_icon="🔍", layout="wide")
st.title("🔍 NCR Defect Category Prediction")
st.subheader("Enter a defect description to predict its defect category")

# =============================
# Single defect prediction
# =============================
user_input = st.text_area("Defect Description:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a defect description.")
    else:
        try:
            # Predict only defect category
            defect_cat = clustering.predict_defect_root_action(user_input)[0]

            # Display result
            st.markdown("### Predicted Defect Category")
            st.write(f"**{defect_cat}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
