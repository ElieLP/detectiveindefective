import streamlit as st
from src.extraction import enrich_dataframe, load_prod_data

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("NCR Analysis Copilot")

df = load_prod_data()
enriched_df = enrich_dataframe(df)
enriched_df.to_csv('data/prod_data_enriched.csv', index=False, sep=';')

st.success(f"Loaded {len(enriched_df)} NCRs")

st.subheader("Cleaned Dataset")
st.dataframe(enriched_df, use_container_width=True)
