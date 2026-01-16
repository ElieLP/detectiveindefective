import streamlit as st
import pandas as pd
import numpy as np
from src.extraction import enrich_dataframe, load_prod_data
from src.clustering import predict_defect_root_action, predict_batch


st.set_page_config(page_title="Similarity", page_icon="🔍", layout="wide")

st.title("🔍 Similarity")
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

st.subheader("🔎 Find Similar NCRs")
query = st.text_input("Describe a problem to find similar NCRs:")
if query:
    query_embedding = compute_embeddings([query])[0]
    embeddings = np.vstack(clustered_df['embedding'].values)
    similar = find_similar(query_embedding, embeddings, top_k=5)
    
    similar_indices = [idx for idx, _ in similar]
    similar_scores = [score for _, score in similar]
    
    similar_df = clustered_df.iloc[similar_indices].drop(columns=['embedding']).copy()
    similar_df.insert(0, 'Similarity', [f"{s:.2%}" for s in similar_scores])
    
    st.dataframe(similar_df, use_container_width=True)
