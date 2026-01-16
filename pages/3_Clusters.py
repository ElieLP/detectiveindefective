import streamlit as st
import pandas as pd
from src.extraction import enrich_dataframe, load_prod_data
from src.clustering import add_embeddings_and_clusters, compute_attribute_correlation

st.set_page_config(page_title="Clusters", page_icon="🔍", layout="wide")

st.title("🔍 Clusters")
st.subheader("NCR Analysis Copilot")

df = load_prod_data()
enriched_df = enrich_dataframe(df)
clustered_df = add_embeddings_and_clusters(enriched_df, description_col='NC description', n_clusters=4)

st.success(f"Loaded {len(df)} NCRs")

st.subheader("Extracted Fields")
st.dataframe(
    clustered_df.drop(columns=['embedding']),
    use_container_width=True
)

st.subheader("Clusters")
for cluster_id in sorted(clustered_df['cluster'].unique()):
    cluster_ncrs = clustered_df[clustered_df['cluster'] == cluster_id]
    with st.expander(f"Cluster {cluster_id} ({len(cluster_ncrs)} NCRs)"):
        st.dataframe(
            cluster_ncrs.drop(columns=['embedding']),
            use_container_width=True
        )

st.subheader("🔗 Attribute Correlations")
correlation_fields = ['MachineNum of detection', 'NC Code', 'Part type', 'root_cause_category', 'corrective_category', 'defect_type']
available_fields = [f for f in correlation_fields if f in enriched_df.columns]

col_a, col_b = st.columns(2)
with col_a:
    field1 = st.selectbox("Select first attribute", available_fields, index=0)
with col_b:
    field2 = st.selectbox("Select second attribute", available_fields, index=1)

if field1 and field2 and field1 != field2:
    crosstab = compute_attribute_correlation(enriched_df, field1, field2)
    st.dataframe(crosstab, use_container_width=True)
else:
    st.info("Select two different attributes to see their correlation.")