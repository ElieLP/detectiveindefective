import streamlit as st
import pandas as pd
from src.extraction import load_prod_data, enrich_dataframe

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")

df = load_prod_data()
enriched_df = enrich_dataframe(df)

st.success(f"Total NCRs: {len(enriched_df)}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏭 Top Defective Machines")
    machine_counts = enriched_df['MachineNum of detection'].dropna().value_counts().head(10).reset_index()
    machine_counts.columns = ['Machine', 'NCR Count']
    st.bar_chart(machine_counts.set_index('Machine'))

with col2:
    st.subheader("🔧 Top NC Codes")
    nc_counts = enriched_df['NC Code'].value_counts().head(10).reset_index()
    nc_counts.columns = ['NC Code', 'Count']
    st.bar_chart(nc_counts.set_index('NC Code'))

col3, col4 = st.columns(2)

with col3:
    st.subheader("📦 Defects by Part Type")
    part_counts = enriched_df['Part type'].value_counts().reset_index()
    part_counts.columns = ['Part Type', 'NCR Count']
    st.bar_chart(part_counts.set_index('Part Type'))

with col4:
    st.subheader("📅 NCR Trend Over Time")
    enriched_df['Date of detection'] = pd.to_datetime(enriched_df['Date of detection'], format='%m/%d/%y', errors='coerce')
    trend = enriched_df.groupby(enriched_df['Date of detection'].dt.date).size().reset_index(name='Count')
    trend.columns = ['Date', 'Count']
    st.line_chart(trend.set_index('Date'))

col5, col6 = st.columns(2)

with col5:
    st.subheader("🔍 Root Cause Categories")
    root_cause_cats = enriched_df['root_cause_category'].value_counts().reset_index()
    root_cause_cats.columns = ['Category', 'Count']
    st.bar_chart(root_cause_cats.set_index('Category'))

with col6:
    st.subheader("🛠️ Corrective Action Categories")
    corrective_cats = enriched_df['corrective_category'].value_counts().reset_index()
    corrective_cats.columns = ['Category', 'Count']
    st.bar_chart(corrective_cats.set_index('Category'))

st.subheader("📋 QA Comment Categories")
fqc_cats = enriched_df['fqc_category'].value_counts().reset_index()
fqc_cats.columns = ['Category', 'Count']
st.dataframe(fqc_cats, use_container_width=True, hide_index=True)
