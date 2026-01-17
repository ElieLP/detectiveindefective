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
UNKNOWN_THRESHOLD = 0.30   # LOWERED safely after normalization
MIN_ALPHA_RATIO = 0.3      # % of alphabetic characters required
MIN_TEXT_LENGTH = 5

# =========================
# ===== TEXT UTILITIES (NEW) =====
# =========================

def normalize_text(text: str) -> str:
    """
    Removes serial numbers / IDs and normalizes text
    """
    text = text.lower()
    text = re.sub(r"[a-z]{1,3}\d{4,}", " ", text)  # remove codes like DA2512100009
    text = re.sub(r"[^a-z\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful_text(text: str) -> bool:
    """
    Rejects pure noise like 'fsihfsh', '123123', '////'
    """
    if len(text) < MIN_TEXT_LENGTH:
        return False

    alpha_chars = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_chars / max(len(text), 1)

    return alpha_ratio >= MIN_ALPHA_RATIO

# =========================
# ===== Prediction function =====
# =========================

def predict_defect_root_action(defect_description: str):
    """
    Predicts defect, root cause, corrective action.
    Returns 'unknown' for OOD or meaningless input.
    """

    # ===== Normalize input =====
    cleaned_text = normalize_text(defect_description)

    # ===== Linguistic sanity check =====
    if not is_meaningful_text(cleaned_text):
        return UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL

    # ===== Stage 1: Defect category with confidence =====
    probs = stage1_pipeline.predict_proba([cleaned_text])[0]
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


