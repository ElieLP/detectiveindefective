import streamlit as st
import pandas as pd
from src.prediction import predict_from_csv, load_context_data, build_context_prompt, predict_root_cause

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Root Cause Prediction")
st.subheader("Upload NCR Data with empty Root Cause field")

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
        
        results = []
        progress_bar = st.progress(0)
        
        for idx, row in df.iterrows():
            root_cause = row.get('Root cause of occurrence', '')
            if pd.isna(root_cause) or root_cause == '':
                with st.spinner(f"Predicting row {idx + 1}/{len(df)}..."):
                    predicted = predict_root_cause(row, context)
                    results.append(predicted)
            else:
                results.append(root_cause)
            progress_bar.progress((idx + 1) / len(df))
        
        df['Root cause of occurrence'] = results
        
        st.markdown("### Prediction Results")
        
        def highlight_root_cause(row):
            return ['background-color: #e6f3ff; color: #0066cc' if col == 'Root cause of occurrence' else '' for col in row.index]
        
        styled_df = df.style.apply(highlight_root_cause, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        csv_output = df.to_csv(index=False, sep=';')
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
