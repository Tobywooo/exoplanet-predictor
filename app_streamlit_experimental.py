import os
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------- Page meta & title ----------
st.set_page_config(page_title="Exoplanet Candidate Predictor", page_icon="🪐", layout="wide")
st.markdown('<style>{}</style>'.format(open('styles.css').read()), unsafe_allow_html=True)  # Injecting the custom CSS

# Title and description with custom style
st.markdown("""
    <h1 class="app-title">🪐 Exoplanet Candidate Predictor</h1>
    <p class="app-caption">Enter KOI-like parameters to estimate whether a target is a candidate.</p>
""", unsafe_allow_html=True)

# ---------- Path helpers ----------
HERE = Path(__file__).parent.resolve()
def smart_path(filename: str) -> str:
    p = HERE / filename
    if p.exists():
        return str(p)
    p2 = HERE / "models" / filename
    return str(p2) if p2.exists() else str(p)

# ---------- Sidebar ----------
st.sidebar.markdown('<div class="sidebar-title">Model Parameters</div>', unsafe_allow_html=True)
model_path  = st.sidebar.text_input("Model path", smart_path("rf_pipeline.joblib"))
schema_path = st.sidebar.text_input("Schema path", smart_path("koi_schema.json"))
catvals_path= st.sidebar.text_input("Categorical values path", smart_path("categorical_values.json"))
labels_path = st.sidebar.text_input("Labels mapping (optional JSON)", smart_path("labels.json"))
threshold = st.sidebar.slider("Decision threshold (positive class)", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Tip: Adjust threshold to balance precision/recall.")

# ---------- Load resources (cached) ----------
@st.cache_resource(show_spinner=True)
def load_resources(_model_path: str, _schema_path: str, _catvals_path: str):
    model = joblib.load(_model_path)
    with open(_schema_path) as f:
        schema = json.load(f)
    try:
        with open(_catvals_path) as f:
            catvals = json.load(f)
    except Exception:
        catvals = {}
    return model, schema, catvals

try:
    model, schema, catvals = load_resources(model_path, schema_path, catvals_path)
except Exception as e:
    st.error(f"Failed to load model or schema: {e}")
    st.stop()

# ---------- Optional: labels mapping for pretty display ----------
DISPLAY_LABELS = {}
if labels_path and os.path.exists(labels_path):
    try:
        with open(labels_path) as f:
            DISPLAY_LABELS = json.load(f)
    except Exception as e:
        st.warning(f"Could not load labels mapping: {e}")

def pretty(col: str) -> str:
    return DISPLAY_LABELS.get(col, col)

# ---------- Schema columns ----------
feature_cols = schema["feature_cols"]
num_cols = set(schema.get("num_cols", []))
cat_cols = set(schema.get("cat_cols", []))

# ---------- Collapsible Batch scoring section ----------
with st.expander("Show/Hide Batch Prediction (CSV upload)", expanded=False):
    st.subheader("Batch prediction (optional)")
    uploaded = st.file_uploader("Upload CSV with columns matching the schema", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            for c in feature_cols:
                if c not in df_in.columns:
                    df_in[c] = None
            df_in = df_in[feature_cols]

            p = model.predict_proba(df_in)[:, 1]
            pred = (p >= threshold).astype(int)
            out = df_in.copy()
            out["prob_candidate"] = p
            out["prediction"] = pred

            st.success(f"Predicted {len(out)} rows.")
            display_out = out.rename(columns=DISPLAY_LABELS)
            st.dataframe(display_out.head(25), use_container_width=True)
            st.download_button("Download results CSV", out.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.divider()

# ---------- Single-row form ----------
st.subheader("Single prediction (form)")

# Two-column layout for inputs
with st.form("koi_form"):
    left, right = st.columns(2)  # Create two columns for a more spacious UI
    inputs = {}

    for i, c in enumerate(feature_cols):
        target_col = left if i % 2 == 0 else right
        label = pretty(c)
        
        with target_col:
            if c in num_cols:
                # Text input for numeric values
                val = st.text_input(label, placeholder="numeric (leave blank for missing)").strip()
                inputs[c] = float(val) if val else None
            elif c in cat_cols:
                # Dropdown selectbox for categorical inputs
                options = catvals.get(c, [])
                if options:
                    sel = st.selectbox(label, ["(leave blank)"] + options, index=0)
                    inputs[c] = None if sel == "(leave blank)" else sel
                else:
                    # Text input if no predefined categories
                    txt = st.text_input(label, placeholder="text (optional)").strip()
                    inputs[c] = txt or None
            else:
                # General text input
                txt = st.text_input(label, placeholder="value (optional)").strip()
                inputs[c] = txt or None

    # Submit button
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    try:
        row = {c: inputs.get(c, None) for c in feature_cols}
        X_df = pd.DataFrame([row])
        proba = float(model.predict_proba(X_df)[0, 1])
        pred = int(proba >= threshold)
        label = "CANDIDATE (1)" if pred == 1 else "CONFIRMED (0)"
        
        # Display the result with custom colors for better visual impact
        if pred == 1:
            st.markdown(f'<h3 class="candidate">CANDIDATE</h3>', unsafe_allow_html=True)
            st.metric("Probability (candidate)", f"{proba:.3f}", delta_color="normal")
        else:
            st.markdown(f'<h3 class="confirmed">CONFIRMED</h3>', unsafe_allow_html=True)
            st.metric("Probability (confirmed)", f"{proba:.3f}", delta_color="normal")
        
        st.caption("Model: saved sklearn pipeline (preprocessor + tuned RandomForest).")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# About section with styled expander
with st.expander("About this model"):
    st.markdown(
        "- Input columns come from the KOI cumulative schema you trained on.\n"
        "- Missing values are imputed by the pipeline.\n"
        "- The decision threshold can be adjusted in the sidebar.\n"
        "- Batch and single predictions use the same saved pipeline."
    )
