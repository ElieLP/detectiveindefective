import streamlit as st
import pandas as pd

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Prediction")
st.subheader("Upload NCR Data for Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success(f"Loaded {len(df)} rows")
    st.dataframe(df, use_container_width=True)
