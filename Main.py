import streamlit as st

st.set_page_config(page_title="Industrial AI Detective", page_icon="🔍", layout="wide")

st.title("🔍 Industrial AI Detective")
st.subheader("Your NCR Analysis Copilot")

st.markdown("""
Welcome to the **Industrial AI Detective**! This tool helps you analyze Non-Conformance Reports (NCRs) 
using AI-powered insights. Choose a feature below to get started.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Dashboard")
    st.markdown("Visualize NCR trends, top defective machines, and identify patterns in your data.")
    if st.button("Go to Dashboard", key="dashboard", type="primary"):
        st.switch_page("pages/1_Dashboard.py")

with col2:
    st.markdown("### 🔮 Prediction")
    st.markdown("Upload NCR data to predict root causes and corrective actions using AI.")
    if st.button("Go to Prediction", key="prediction", type="primary"):
        st.switch_page("pages/2_Prediction.py")

with col3:
    st.markdown("### 🔍 Similarity")
    st.markdown("Find similar past NCRs and predict defect categories from descriptions.")
    if st.button("Go to Similarity", key="similarity", type="primary"):
        st.switch_page("pages/3_Similarity.py")

st.divider()

st.markdown("##### Quick Tips")
st.info("💡 Start with the **Dashboard** to explore your data, then use **Prediction** or **Similarity** for AI-powered analysis.")
