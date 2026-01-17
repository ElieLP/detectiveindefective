import streamlit as st
import pandas as pd
from src.prediction import load_context_data, build_context_prompt, predict_batch

st.set_page_config(page_title="🔮 Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Prediction")
st.subheader("Use the past data to predict root causes or corrective actions")

st.text("Upload NCR Data with empty Root Cause and/or Corrective actions fields")
st.image("pages/assets/prediction.png")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success(f"Loaded {len(df)} rows")
    
    st.markdown("### Input Data")
    st.dataframe(df, use_container_width=True)
    
    if st.button("Predict Root Causes", type="primary"):
        with st.spinner("Loading context data..."):
            context_df = load_context_data()
            context = build_context_prompt(context_df)
        
        with st.spinner(f"Predicting root causes for {len(df)} rows..."):
            predictions = predict_batch(df, context)
        
        root_causes = []
        corrective_actions = []
        for idx, row in df.iterrows():
            root_cause = row.get('Root cause of occurrence', '')
            corrective = row.get('Corrective actions', '')
            needs_root_cause = pd.isna(root_cause) or root_cause == ''
            needs_corrective = pd.isna(corrective) or corrective == ''
            
            pred_root, pred_action = predictions[idx]
            root_causes.append(pred_root if needs_root_cause else root_cause)
            corrective_actions.append(pred_action if needs_corrective else corrective)
        
        df['Root cause of occurrence'] = root_causes
        df['Corrective actions'] = corrective_actions
        
        st.markdown("### Prediction Results")
        
        def highlight_predictions(row):
            return ['background-color: #e6f3ff; color: #0066cc' if col in ['Root cause of occurrence', 'Corrective actions'] else '' for col in row.index]
        
        styled_df = df.style.apply(highlight_predictions, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        csv_output = df.to_csv(index=False, sep=';')
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
