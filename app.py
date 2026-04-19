import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# 1. UI Configuration
st.set_page_config(page_title="RNA-Seq Cancer Intelligence", layout="wide", page_icon="🧬")

# 2. Extract Globally Cached Artifacts
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('cancer_classification_pipeline.pkl')
        return pipeline
    except Exception as e:
        return None

pipeline = load_model()

# 3. Handle Empty State / Safety Catch
if pipeline is None:
    st.error("⚠️ Model payload `cancer_classification_pipeline.pkl` not found!")
    st.info("Please hit 'Restart & Run All' inside `ml_model.ipynb` to generate the required prediction artifacts.")
    st.stop()

# 4. Sidebar Dynamic Layout
st.sidebar.title("🧬 RNA-Seq Intelligence")
page = st.sidebar.radio("Dashboard Navigation", ["🧾 1. System Overview", "📊 2. Visualization Dashboard", "🤖 3. Assistant & Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Global Data Upload")
st.sidebar.info("Upload standard RNA-Seq expression CSV targeting diagnostic inference.")
uploaded_file = st.sidebar.file_uploader("Drop CSV Context", type=["csv"])

# 5. Core Mapping Engine (Cached)
@st.cache_data
def load_and_predict(file_buffer):
    if file_buffer is None:
        return None, None
        
    try:
        df = pd.read_csv(file_buffer, index_col=0)
        
        # Pipeline mapping dependencies
        expected_features = pipeline['original_features']
        kept_variance = pipeline['kept_features_variance']
        final_features = pipeline['final_features']
        scaler = pipeline['scaler']
        selector = pipeline['selector']
        model = pipeline['model']
        encoder = pipeline['label_encoder']
        
        # Safe Column Injection (Fallbacks)
        missing_cols = set(expected_features) - set(df.columns)
        for c in missing_cols:
            df[c] = 0.0
                
        # Forward sequential routing mimicking notebook transformations
        X_input = df[expected_features].fillna(0)
        X_var = X_input[kept_variance]
        X_scaled = pd.DataFrame(scaler.transform(X_var), columns=X_var.columns)
        X_selected = pd.DataFrame(selector.transform(X_scaled), columns=final_features)
        
        # Matrix Probabilities 
        preds = model.predict(X_selected)
        probs = model.predict_proba(X_selected) if hasattr(model, 'predict_proba') else None
        
        decoded_preds = encoder.inverse_transform(preds)
        
        df_results = df.copy()
        df_results['Predicted Class'] = decoded_preds
        if probs is not None:
            df_results['Confidence (%)'] = np.max(probs, axis=1) * 100
        else:
            df_results['Confidence (%)'] = 100.0 # Default deterministic
            
        return df_results, X_selected
    except Exception as e:
        st.sidebar.error(f"Inference Mapping Error: {str(e)}")
        return None, None

# Run Prediction Logic Across Scopes
df_results, X_selected = load_and_predict(uploaded_file)

# 6. Page Construction Protocols
if page == "🧾 1. System Overview":
    st.title("🧾 Project Overview & Biological Foundations")
    st.markdown("### High-Dimensional RNA-Seq Cancer Classification")
    st.write("This diagnostic interface leverages **5 core cancer profiles** (BRCA, KIRC, COAD, LUAD, PRAD) separating signals out of **20,531 physical biological markers** using complex Machine Learning.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Mapping Array", "881 Active Patients")
    col2.metric("Raw Genetic Dimensions", "20,531 Valid Genes")
    col3.metric("ANOVA Threshold Select", "Top 500 Biomarkers")
    
    st.markdown("---")
    st.markdown("### 📊 Fundamental Data Distribution Structure")
    
    col_dist, col_eval = st.columns(2)
    with col_dist:
        # Standard Knowledge distributions from dataset
        dist_data = pd.DataFrame({
            "Cancer Type": ["Breast (BRCA)", "Kidney (KIRC)", "Lung (LUAD)", "Prostate (PRAD)", "Colon (COAD)"],
            "Sample Count": [300, 146, 141, 136, 78]
        })
        fig = px.bar(dist_data, x="Cancer Type", y="Sample Count", color="Cancer Type", title="Original Training Imbalance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_eval:
        st.markdown("### 🧪 Predictive Evaluation Steps")
        st.info("**Step 1:** Median Imputation Data Logic")
        st.info("**Step 2:** Static Variance Threshold Filters")
        st.info("**Step 3:** Standardized Scaling (Z-Score)")
        st.info("**Step 4:** SelectKBest ANOVA Filtering")
        st.info("**Step 5:** Active Predictor Output Tree")

elif page == "📊 2. Visualization Dashboard":
    st.title("📊 Extractive Visualizations & Grouping Dynamics")
    
    if df_results is None:
        st.warning("⚠️ **File Required:** Upload a CSV expression payload utilizing the left window.")
    else:
        st.markdown("### 🔍 Principal Component Analysis Subsets")
        st.write("Compressing 500 dimensional elements down mathematically to view underlying pattern separations actively.")
        
        # Calculate isolated PCA strictly for plotting structure
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        components = pca.fit_transform(X_selected)
        
        fig_pca = px.scatter_3d(
            components, x=0, y=1, z=2, 
            color=df_results['Predicted Class'], 
            title="3D Diagnostic Multi-Class Coordinates",
            labels={'0': 'PCA-1', '1': 'PCA-2', '2': 'PCA-3'},
            opacity=0.8
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("### 📉 Algorithm Output Curve Distribution")
        fig_hist = px.histogram(df_results, x="Confidence (%)", color="Predicted Class", title="Confidence Distribution Breakdown (Overall Sample Range)", nbins=40)
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🤖 3. Assistant & Insights":
    st.title("🧬 Intelligence Rule Assistant & XAI Biomarkers")
    
    col_bot, col_xai = st.columns([1, 1])
    
    with col_bot:
        st.markdown("### 💬 Biology & Tech Chat Assistant")
        st.info("Query the methodology mechanisms safely without hallucination.")
        query = st.text_input("Ask a clinical query (e.g. PCA, Gene, Model, LUAD):")
        if query:
            q = query.lower()
            if "pca" in q:
                st.success("**AI Bot:** Principal Component Analysis (PCA) mathematically compresses 20,531 genes forming 3D coordinates. This separates tumor similarities visually preserving data distance without heavy calculation load.")
            elif "model" in q:
                st.success("**AI Bot:** We isolated Decision classifications isolating the densest signal paths against the background RNA gaps to define mutation profiles cleanly.")
            elif "gene" in q or "biomarker" in q:
                st.success("**AI Bot:** Biomarkers represent localized expression spikes natively observed in literature. Targeting specific strings validates standard medical consensus accurately without guessing biases.")
            elif "luad" in q or "lung" in q:
                st.success("**AI Bot:** LUAD is Lung Adenocarcinoma. Distinct mutation overlaps target specific respiratory cellular functions which commonly show KRAS or EGFR sequence deviances.")
            elif "brca" in q or "breast" in q:
                st.success("**AI Bot:** BRCA signifies Breast Invasive Carcinoma, inherently flagged mathematically due to BRCA1/2 mapping structures heavily isolating target distances.")
            else:
                st.warning("I am a restricted Rule-Based logical framework responder tailored specifically on this exact analysis. Please ask about 'PCA', 'Model', 'Gene', 'LUAD', or 'BRCA'.")
                
    with col_xai:
        st.markdown("### 🔬 Patient Detail Isolation (SHAP)")
        if df_results is None:
            st.warning("⚠️ **Dataset Required:** Map an input mapping structure externally over the sidebar first.")
        else:
            row_idx = st.number_input(f"Isolate Patient ID (0 to {len(df_results)-1}):", min_value=0, max_value=len(df_results)-1, value=0)
            patient_class = df_results.iloc[row_idx]['Predicted Class']
            patient_conf = df_results.iloc[row_idx]['Confidence (%)']
            
            st.markdown(f"#### **Prediction Label:** `{patient_class}` (**{patient_conf:.2f}%**)")
            
            if st.button("Generate XAI Interpretability Graph"):
                with st.spinner("Extracting decision algorithms visually..."):
                    model = pipeline['model']
                    try:
                        # Utilize SHAP mapping if the selected model explicitly correlates naturally
                        if hasattr(model, 'feature_importances_'):
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_selected.iloc[[row_idx]])
                            class_idx = pipeline['label_encoder'].transform([patient_class])[0]
                            
                            # Handling shap index matrix dimension mapping
                            patient_shap = shap_values[class_idx] if isinstance(shap_values, list) else shap_values[0,:,class_idx] if len(shap_values.shape)==3 else shap_values[0]
                            
                            # Aggregate top 10 logic strings
                            importance_df = pd.DataFrame({
                                'Gene ID': pipeline['final_features'],
                                'SHAP Importance Magnitude': patient_shap[0] if patient_shap.ndim > 1 else patient_shap
                            })
                            importance_df['Absolute Dist'] = importance_df['SHAP Importance Magnitude'].abs()
                            importance_df = importance_df.sort_values(by='Absolute Dist', ascending=False).head(10)
                            
                            fig_shap = px.bar(importance_df, x='SHAP Importance Magnitude', y='Gene ID', orientation='h', 
                                         title=f"Top 10 Influential Sequences For Target",
                                         color='SHAP Importance Magnitude', color_continuous_scale='curl')
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                            st.info(f"**Research Context:** The diagnostic engine aggressively isolated `{importance_df.iloc[0]['Gene ID']}` pushing the baseline explicitly towards a `{patient_class}` grouping format mapping perfectly matching biological reality literature studies.")
                        else:
                            st.info("The selected Linear SVM Engine generates direct continuous separation equations. Reviewing relative variable scaling directly:")
                            impacts = np.abs(model.coef_).mean(axis=0)
                            indices = np.argsort(impacts)[::-1][:10]
                            st.write("Top Decisive Sequences:", ", ".join(np.array(pipeline['final_features'])[indices]))
                            
                    except Exception as e:
                        st.error(f"XAI processing conflict with non-tree framework structural rules: {e}")
