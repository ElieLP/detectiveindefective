from typing import List, Tuple
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_model = None

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR, device='cpu')
    return _model


def compute_embeddings(texts: List[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, show_progress_bar=False)


def find_similar(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def cluster_ncrs(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    n_samples = len(embeddings)
    n_clusters = min(n_clusters, n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def add_embeddings_and_clusters(df: pd.DataFrame, description_col: str = 'description', n_clusters: int = 5) -> pd.DataFrame:
    result = df.copy()
    texts = df[description_col].tolist()
    embeddings = compute_embeddings(texts)
    result['embedding'] = list(embeddings)
    result['cluster'] = cluster_ncrs(embeddings, n_clusters)
    return result


if __name__ == '__main__':
    df = pd.read_csv('data/sample_ncrs.csv')
    result = add_embeddings_and_clusters(df, n_clusters=4)
    print(result[['ncr_id', 'cluster']].to_string())
    print("\nCluster distribution:")
    print(result['cluster'].value_counts().sort_index())
