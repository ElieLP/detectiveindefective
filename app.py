import streamlit as st
import pandas as pd

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("NCR Analysis Copilot")

df = pd.read_csv("data/sample_ncrs.csv")

st.success(f"Loaded {len(df)} NCRs")
st.dataframe(df, use_container_width=True)
