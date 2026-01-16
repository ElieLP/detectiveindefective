import streamlit as st
import pandas as pd
from src.extraction import load_prod_data

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")

df = load_prod_data()

st.success(f"Total NCRs: {len(df)}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏭 Top Defective Machines")
    machine_counts = df[df['MachineNum of detection'] != '/']['MachineNum of detection'].value_counts().head(10).reset_index()
    machine_counts.columns = ['Machine', 'NCR Count']
    st.bar_chart(machine_counts.set_index('Machine'))

with col2:
    st.subheader("🔧 Top NC Codes")
    nc_counts = df['NC Code'].value_counts().head(10).reset_index()
    nc_counts.columns = ['NC Code', 'Count']
    st.bar_chart(nc_counts.set_index('NC Code'))

col3, col4 = st.columns(2)

with col3:
    st.subheader("📦 Defects by Part Type")
    part_counts = df['Part type'].value_counts().reset_index()
    part_counts.columns = ['Part Type', 'NCR Count']
    st.bar_chart(part_counts.set_index('Part Type'))

with col4:
    st.subheader("📅 NCR Trend Over Time")
    df['Date of detection'] = pd.to_datetime(df['Date of detection'], format='%m/%d/%y', errors='coerce')
    trend = df.groupby(df['Date of detection'].dt.date).size().reset_index(name='Count')
    trend.columns = ['Date', 'Count']
    st.line_chart(trend.set_index('Date'))

st.subheader("🔍 Root Cause Categories")
root_causes = df['Root cause of occurrence'].value_counts().head(10).reset_index()
root_causes.columns = ['Root Cause', 'Count']
st.dataframe(root_causes, use_container_width=True, hide_index=True)
