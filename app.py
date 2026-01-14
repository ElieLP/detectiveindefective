import streamlit as st
import pandas as pd
from src.extraction import enrich_dataframe

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("NCR Analysis Copilot")

df = pd.read_csv("data/sample_ncrs.csv")
enriched_df = enrich_dataframe(df)

st.success(f"Loaded {len(df)} NCRs")

st.subheader("Extracted Fields")
st.dataframe(
    enriched_df[['ncr_id', 'date', 'severity', 'machines', 'suppliers', 'operators', 'defect_type']],
    use_container_width=True
)
