import streamlit as st
import pandas as pd
from src.extraction import enrich_dataframe, load_prod_data
from src.clustering import add_embeddings_and_clusters, compute_embeddings, find_similar
import numpy as np

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("NCR Analysis Copilot")

df = load_prod_data()
enriched_df = enrich_dataframe(df)
clustered_df = add_embeddings_and_clusters(enriched_df, description_col='NC description', n_clusters=4)

st.success(f"Loaded {len(df)} NCRs")

st.subheader("Extracted Fields")
st.dataframe(
    clustered_df[['Job order', 'NC Code', 'MachineNum of detection', 'Date of detection', 'defect_type', 'root_cause', 'cluster']],
    use_container_width=True
)

st.subheader("Clusters")
for cluster_id in sorted(clustered_df['cluster'].unique()):
    cluster_ncrs = clustered_df[clustered_df['cluster'] == cluster_id]
    with st.expander(f"Cluster {cluster_id} ({len(cluster_ncrs)} NCRs)"):
        st.dataframe(
            cluster_ncrs[['Job order', 'NC Code', 'defect_type', 'NC description', 'root_cause']],
            use_container_width=True
        )

st.subheader("🔎 Find Similar NCRs")
query = st.text_input("Describe a problem to find similar NCRs:")
if query:
    query_embedding = compute_embeddings([query])[0]
    embeddings = np.vstack(clustered_df['embedding'].values)
    similar = find_similar(query_embedding, embeddings, top_k=5)
    
    st.write("**Most similar NCRs:**")
    for idx, score in similar:
        row = clustered_df.iloc[idx]
        desc = str(row['NC description'])[:100]
        root_cause = str(row.get('root_cause', ''))[:50]
        st.markdown(f"- **{row['Job order']}** ({row['NC Code']}) - similarity: {score:.2f}")
        st.markdown(f"  - {desc}...")
        if root_cause:
            st.markdown(f"  - **Root cause:** {root_cause}...")
