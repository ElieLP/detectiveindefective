from typing import List, Tuple
import os
import pandas as pd
import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity

# =========================
# ===== Old embedding model (commented out) =====
# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
# _model = None

# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

# def get_model() -> SentenceTransformer:
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR, device='cpu')
#     return _model

# def compute_embeddings(texts: List[str]) -> np.ndarray:
#     model = get_model()
#     return model.encode(texts, show_progress_bar=False)

# def find_similar(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_indices = np.argsort(similarities)[::-1][:top_k]
#     return [(int(idx), float(similarities[idx])) for idx in top_indices]

# def cluster_ncrs(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
#     n_samples = len(embeddings)
#     n_clusters = min(n_clusters, n_samples)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     return kmeans.fit_predict(embeddings)

# def add_embeddings_and_clusters(df: pd.DataFrame, description_col: str = 'root_cause', n_clusters: int = 5) -> pd.DataFrame:
#     result = df.copy()
#     texts = df[description_col].tolist()
#     embeddings = compute_embeddings(texts)
#     result['embedding'] = list(embeddings)
#     result['cluster'] = cluster_ncrs(embeddings, n_clusters)
#     return result

# =========================
# ===== Load your ML models =====
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths to models
stage1_path = os.path.join(BASE_DIR, "..", "stage1_defect_model.pkl")
stage2_path = os.path.join(BASE_DIR, "..", "stage2_root_cause_model.pkl")
stage3_path = os.path.join(BASE_DIR, "..", "stage3_corrective_action_model.pkl")
encoder_path = os.path.join(BASE_DIR, "..", "encoder_defect.pkl")
encoder2_path = os.path.join(BASE_DIR, "..", "encoder_defect_root.pkl")

# Load models
stage1_pipeline = joblib.load(stage1_path)
root_cause_model = joblib.load(stage2_path)
action_model = joblib.load(stage3_path)
encoder = joblib.load(encoder_path)
encoder2 = joblib.load(encoder2_path)

# =========================
# ===== Prediction function =====
# =========================
def predict_defect_root_action(defect_description: str):
    """
    Given a defect description, predicts:
    1. Defect Category
    2. Root Cause Category
    3. Corrective Action Category
    """
    # Stage 1: defect category
    predicted_defect = stage1_pipeline.predict([defect_description])[0]
    
    # Stage 2: root cause
    X_root = encoder.transform([[predicted_defect]])
    predicted_root = root_cause_model.predict(X_root)[0]
    
    # Stage 3: corrective action
    X_action = encoder2.transform([[predicted_defect, predicted_root]])
    predicted_action = action_model.predict(X_action)[0]
    
    return predicted_defect, predicted_root, predicted_action

# =========================
# ===== Main interactive interface =====
# =========================
if __name__ == "__main__":
    print("=== Industrial Defect Prediction System ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("Enter defect description:\n> ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        
        defect_cat, root_cause_cat, action_cat = predict_defect_root_action(user_input)
        
        print("\nPredicted categories:")
        print(f"- Defect Category       : {defect_cat}")
        print(f"- Root Cause Category   : {root_cause_cat}")
        print(f"- Corrective Action     : {action_cat}")
        print("="*50)

