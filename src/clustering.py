from typing import List, Tuple
import os
import pandas as pd
import joblib

# =========================
# ===== Load your ML models =====
# =========================

# Folder where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to your models (all in the same folder)
stage1_path = os.path.join(BASE_DIR, "stage1_defect_model.pkl")
stage2_path = os.path.join(BASE_DIR, "stage2_root_cause_model.pkl")
stage3_path = os.path.join(BASE_DIR, "stage3_corrective_action_model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder_defect.pkl")
encoder2_path = os.path.join(BASE_DIR, "encoder_defect_root.pkl")

# Load models
stage1_pipeline = joblib.load(stage1_path)
root_cause_model = joblib.load(stage2_path)
action_model = joblib.load(stage3_path)
encoder = joblib.load(encoder_path)
encoder2 = joblib.load(encoder2_path)

# =========================
# ===== UNKNOWN HANDLING CONFIG (NEW) =====
# =========================

UNKNOWN_LABEL = "unknown"
UNKNOWN_THRESHOLD = 0.5   # Tune between 0.4–0.7 depending on your data

# =========================
# ===== Prediction function =====
# =========================

def predict_defect_root_action(defect_description: str):
    """
    Given a defect description, predicts:
    1. Defect Category
    2. Root Cause Category
    3. Corrective Action Category

    If the input is low-confidence / out-of-distribution,
    returns 'unknown' for all outputs.
    """

    # ===== Stage 1: Defect category (with confidence check) =====
    probs = stage1_pipeline.predict_proba([defect_description])[0]
    max_confidence = probs.max()
    predicted_defect = stage1_pipeline.classes_[probs.argmax()]

    if max_confidence < UNKNOWN_THRESHOLD:
        return UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL

    # ===== Stage 2: Root cause =====
    X_root = encoder.transform([[predicted_defect]])
    predicted_root = root_cause_model.predict(X_root)[0]

    # ===== Stage 3: Corrective action =====
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
        user_input = input("Enter defect description:\n> ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        defect_cat, root_cause_cat, action_cat = predict_defect_root_action(user_input)

        print("\nPredicted categories:")
        print(f"- Defect Category       : {defect_cat}")
        print(f"- Root Cause Category   : {root_cause_cat}")
        print(f"- Corrective Action     : {action_cat}")
        print("=" * 50)

