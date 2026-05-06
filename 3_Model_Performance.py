"""
pages/3_Model_Performance.py
Side-by-side model comparison, ROC curve, confusion matrix.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0f172a,#1e293b);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
</style>
""", unsafe_allow_html=True)

DARK = "plotly_dark"
CLF_COLORS = px.colors.qualitative.Vivid
REG_COLORS = px.colors.qualitative.Safe

@st.cache_resource
def load_artefacts():
    results      = joblib.load(MODEL_DIR / "model_results.pkl")
    clf_model    = joblib.load(MODEL_DIR / "best_clf.pkl")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return results, clf_model, scaler, feature_cols

@st.cache_data
def load_test_data(feature_cols):
    df  = pd.read_csv(DATA_PATH)
    DROP = ["Future_Price_5Y","Good_Investment","ROI","Growth_Rate","Amenities","Security"]
    X   = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    X   = pd.get_dummies(X, drop_first=True)
    X   = X.reindex(columns=feature_cols, fill_value=0)
    y_clf = df["Good_Investment"]
    _, X_test, _, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    return X_test, y_test

st.title("🤖 Model Performance")
st.markdown("Compare all trained models and inspect evaluation metrics.")

try:
    results, clf_model, scaler, feature_cols = load_artefacts()
    X_test, y_test = load_test_data(feature_cols)
    X_test_sc = scaler.transform(X_test)
    ready = True
except FileNotFoundError:
    ready = False

if not ready:
    st.error("⚠️ Models not found. Please run `python src/train.py` first.")
    st.stop()

tab_clf, tab_reg = st.tabs(["🏷️ Classification", "📈 Regression"])

# ═══════════════════════════════════════════════════
# Classification Tab
# ═══════════════════════════════════════════════════
with tab_clf:
    clf_data = results["classification"]
    clf_df   = pd.DataFrame(clf_data).T.reset_index()
    clf_df.columns = ["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]
    clf_df = clf_df.sort_values("Accuracy", ascending=False).round(4)

    st.subheader("Classification Metrics")
    st.dataframe(clf_df, use_container_width=True, hide_index=True)

    # Grouped bar chart
    metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=m,
            x=clf_df["Model"],
            y=clf_df[m],
            marker_color=CLF_COLORS[i]
        ))
    fig.update_layout(
        barmode="group", template=DARK,
        title="Classification Metrics Comparison",
        yaxis=dict(range=[0, 1.05]),
        height=400, margin=dict(t=40,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve for best classifier
    st.subheader("ROC Curve — Best Classifier")
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_score    = roc_auc_score(y_test, proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#4f46e5", width=2.5),
            name=f"AUC = {auc_score:.3f}"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Random", showlegend=True
        ))
        fig_roc.update_layout(
            template=DARK, height=380,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=20,b=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix — Best Classifier")
    preds = clf_model.predict(X_test_sc)
    cm    = confusion_matrix(y_test, preds)
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Not Good","Good Investment"],
        y=["Not Good","Good Investment"],
        template=DARK, aspect="auto"
    )
    fig_cm.update_layout(height=350, margin=dict(t=20,b=10),
                         xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance
    if hasattr(clf_model, "feature_importances_"):
        st.subheader("Feature Importance — Classifier")
        fi = pd.Series(clf_model.feature_importances_, index=feature_cols)
        top = fi.nlargest(15).sort_values()
        fig_fi = px.bar(x=top.values, y=top.index, orientation="h",
                        color=top.values, color_continuous_scale="Purples",
                        template=DARK,
                        labels={"x":"Importance","y":"Feature"})
        fig_fi.update_layout(height=430, margin=dict(t=20,b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════
# Regression Tab
# ═══════════════════════════════════════════════════
with tab_reg:
    reg_data = results["regression"]
    reg_df   = pd.DataFrame(reg_data).T.reset_index()
    reg_df.columns = ["Model","RMSE","MAE","R²"]
    reg_df = reg_df.sort_values("R²", ascending=False).round(4)

    st.subheader("Regression Metrics")
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = px.bar(
            reg_df.sort_values("R²"), x="R²", y="Model",
            orientation="h", color="R²",
            color_continuous_scale="Greens", template=DARK,
            title="R² Score by Model"
        )
        fig_r2.update_layout(height=340, margin=dict(t=40,b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_rmse = px.bar(
            reg_df.sort_values("RMSE", ascending=False),
            x="RMSE", y="Model", orientation="h",
            color="RMSE", color_continuous_scale="Reds_r",
            template=DARK, title="RMSE by Model (lower = better)"
        )
        fig_rmse.update_layout(height=340, margin=dict(t=40,b=10),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
