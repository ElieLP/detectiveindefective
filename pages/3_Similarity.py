# pages/3_Similarity.py

import streamlit as st
import pandas as pd
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
# Absolute paths to src modules
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
clustering_path = os.path.join(BASE_DIR, "clustering.py")
extraction_path = os.path.join(BASE_DIR, "extraction.py")

# Load modules dynamically
clustering = load_module_from_file("clustering", clustering_path)
extraction = load_module_from_file("extraction", extraction_path)

# =============================
# Streamlit page config
# =============================
st.set_page_config(page_title="NCR Copilot", page_icon="🔍", layout="wide")
st.title("🔍 NCR Defect Prediction")
st.subheader("Industrial NCR Copilot")

# =============================
# Load and enrich data
# =============================
try:
    df = extraction.load_prod_data()
    enriched_df = extraction.enrich_dataframe(df)
    st.success(f"Loaded {len(df)} NCRs")
except Exception as e:
    st.error(f"Failed to load NCR data: {e}")
    enriched_df = pd.DataFrame()

# Show enriched dataframe
if not enriched_df.empty:
    st.subheader("Enriched NCR Data")
    st.dataframe(enriched_df, use_container_width=True)

#

