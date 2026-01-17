import streamlit as st
import pandas as pd
from src.extraction import load_prod_data, enrich_dataframe

st.set_page_config(page_title="📊 Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")
st.subheader("NCR Charts, may help you find patterns the AI cannot find")

df = load_prod_data()
enriched_df = enrich_dataframe(df)

st.success(f"Total NCRs: {len(enriched_df)}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏭 Top Defective Machines")
    st.caption("Which machines produce the most defects? Focus maintenance efforts here.")
    machine_counts = enriched_df['MachineNum of detection'].dropna().value_counts().head(10).reset_index()
    machine_counts.columns = ['Machine', 'NCR Count']
    st.bar_chart(machine_counts.set_index('Machine'))

with col2:
    st.subheader("🔧 Top NC Codes")
    st.caption("Most frequent non-conformance codes. Reveals recurring defect types.")
    nc_counts = enriched_df['NC Code'].value_counts().head(10).reset_index()
    nc_counts.columns = ['NC Code', 'Count']
    st.bar_chart(nc_counts.set_index('NC Code'))

col3, col4 = st.columns(2)

with col3:
    st.subheader("📦 Defects by Part Type")
    st.caption("Which parts fail most often? May indicate design or supplier issues.")
    part_counts = enriched_df['Part type'].value_counts().reset_index()
    part_counts.columns = ['Part Type', 'NCR Count']
    st.bar_chart(part_counts.set_index('Part Type'))

with col4:
    st.subheader("📅 NCR Trend Over Time")
    st.caption("Track defect volume over time. Spikes may indicate process changes.")
    dates = pd.to_datetime(df['Date of detection'], format='%m/%d/%y', errors='coerce')
    trend = dates.dropna().dt.date.value_counts().sort_index().reset_index()
    trend.columns = ['Date', 'Count']
    st.line_chart(trend.set_index('Date'))

col5, col6 = st.columns(2)

with col5:
    st.subheader("🔍 Root Cause Categories")
    st.caption("AI-extracted root cause themes. Shows systemic issues.")
    root_cause_cats = enriched_df['root_cause_category'].value_counts().reset_index()
    root_cause_cats.columns = ['Category', 'Count']
    st.bar_chart(root_cause_cats.set_index('Category'))

with col6:
    st.subheader("🛠️ Corrective Action Categories")
    st.caption("Types of fixes applied. Helps standardize responses.")
    corrective_cats = enriched_df['corrective_category'].value_counts().reset_index()
    corrective_cats.columns = ['Category', 'Count']
    st.bar_chart(corrective_cats.set_index('Category'))

st.subheader("📋 QA Comment Categories")
st.caption("Themes from QA comments. Reveals inspector concerns.")
fqc_cats = enriched_df['fqc_category'].value_counts().reset_index()
fqc_cats.columns = ['Category', 'Count']
st.dataframe(fqc_cats, use_container_width=True, hide_index=True)

st.subheader("🔗 Attribute Correlations")
st.caption("Explore relationships between attributes. Find hidden patterns like 'Machine X + Part Y = high defects'.")
correlation_fields = ['MachineNum of detection', 'NC Code', 'Part type', 'root_cause_category', 'corrective_category', 'defect_type']
available_fields = [f for f in correlation_fields if f in enriched_df.columns]

col_a, col_b = st.columns(2)
with col_a:
    field1 = st.selectbox("Select first attribute", available_fields, index=0)
with col_b:
    field2 = st.selectbox("Select second attribute", available_fields, index=1)

if field1 and field2 and field1 != field2:
    pair_counts = enriched_df.groupby([field1, field2]).size().reset_index(name='Count')
    pair_counts = pair_counts.sort_values('Count', ascending=False).head(15)
    pair_counts.columns = [field1, field2, 'Count']
    st.dataframe(pair_counts, use_container_width=True, hide_index=True)
else:
    st.info("Select two different attributes to see their correlation.")
