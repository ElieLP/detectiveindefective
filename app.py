import streamlit as st
import pandas as pd
from src.extraction import enrich_dataframe
from src.clustering import add_embeddings_and_clusters

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("NCR Analysis Copilot")

df = pd.read_csv("data/sample_ncrs.csv")
enriched_df = enrich_dataframe(df)
clustered_df = add_embeddings_and_clusters(enriched_df, n_clusters=4)

st.success(f"Loaded {len(df)} NCRs")

st.subheader("Extracted Fields")
st.dataframe(
    clustered_df[['ncr_id', 'date', 'severity', 'machines', 'suppliers', 'operators', 'defect_type', 'cluster']],
    use_container_width=True
)

st.subheader("Clusters")
for cluster_id in sorted(clustered_df['cluster'].unique()):
    cluster_ncrs = clustered_df[clustered_df['cluster'] == cluster_id]
    with st.expander(f"Cluster {cluster_id} ({len(cluster_ncrs)} NCRs)"):
        st.dataframe(
            cluster_ncrs[['ncr_id', 'severity', 'defect_type', 'description']],
            use_container_width=True
        )
