# pages/3_Similarity.py

import streamlit as st
import pandas as pd
import os
import sys

# -----------------------
# Make sure Python sees src/
# -----------------------
# Get absolute path to the repo root (parent of pages/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add repo root to Python path
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Now import from src package
from src.clustering import predict_defect_root_action, predict_batch
from src.extraction import enrich_dataframe, load_prod_data


# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Similarity", page_icon="🔍", layout="wide")

st.title("🔍 NCR Defect Prediction")
st.subheader("Industrial NCR Copilot")

# =========================
# Load and enrich data
# =========================
df = load_prod_data()
enriched_df = enrich_dataframe(df)

st.success(f"Loaded {len(df)} NCRs")

# Show enriched dataframe
st.subheader("Enriched NCR Data")
st.dataframe(
    enriched_df,
    use_container_width=True
)

# =========================
# Predict defect, root cause, corrective action
# =========================
st.subheader("Predict Categories from Description")

query = st.text_area("Enter defect description to predict categories:")

if st.button("Predict"):
    if query.strip() == "":
        st.warning("Please enter a defect description.")
    else:
        defect_cat, root_cause_cat, action_cat = predict_defect_root_action(query)
        st.markdown("### Predicted Categories")
        st.write(f"- **Defect Category:** {defect_cat}")
        st.write(f"- **Root Cause Category:** {root_cause_cat}")
        st.write(f"- **Corrective Action:** {action_cat}")

# =========================
# Optional: batch predictions
# =========================
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with a column 'defect_description'", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    if 'defect_description' not in batch_df.columns:
        st.error("CSV must have a column named 'defect_description'.")
    else:
        batch_result = predict_batch(batch_df, description_col='defect_description')
        st.success("Batch prediction completed!")
        st.dataframe(batch_result, use_container_width=True)
