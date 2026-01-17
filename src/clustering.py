import os
import re
import joblib

# =========================
# ===== Load your ML models =====
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

stage1_path = os.path.join(BASE_DIR, "stage1_defect_model.pkl")
stage2_path = os.path.join(BASE_DIR, "stage2_root_cause_model.pkl")
stage3_path = os.path.join(BASE_DIR, "stage3_corrective_action_model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder_defect.pkl")
encoder2_path = os.path.join(BASE_DIR, "encoder_defect_root.pkl")

stage1_pipeline = joblib.load(stage1_path)
root_cause_model = joblib.load(stage2_path)
action_model = joblib.load(stage3_path)
encoder = joblib.load(encoder_path)
encoder2 = joblib.load(encoder2_path)

# =========================
# ===== CONFIG =====
# =========================

UNKNOWN_LABEL = "unknown"
MIN_ALPHA_RATIO = 0.25
MIN_TEXT_LENGTH = 6

# =========================
# ===== GARBAGE DETECTION (ONLY) =====
# =========================

def is_garbage_text(text: str) -> bool:
    """
    Detects obvious nonsense:
    - random characters
    - symbols only
    - extremely short inputs
    """
    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH:
        return True

    alpha_count = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_count / max(len(text), 1)

    if
