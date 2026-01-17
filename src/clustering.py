from typing import List
import os
import joblib

# =========================
# ===== Load Stage 1 Model =====
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stage1_path = os.path.join(BASE_DIR, "stage1_defect_model.pkl")

# Load the model pipeline (TF-IDF + LogisticRegression)
stage1_pipeline = joblib.load(stage1_path)

# =========================
# ===== Prediction function =====
# =========================
def predict_defect_category(defect_description: str) -> str_
