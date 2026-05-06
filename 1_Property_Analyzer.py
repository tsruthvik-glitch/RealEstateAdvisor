"""
pages/1_Property_Analyzer.py
Investment classification + 5-year price forecast page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Property Analyzer", page_icon="🔍", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_models():
    clf          = joblib.load(MODEL_DIR / "best_clf.pkl")
    reg          = joblib.load(MODEL_DIR / "best_reg.pkl")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return clf, reg, scaler, feature_cols

try:
    clf_model, reg_model, scaler, feature_cols = load_models()
    models_ready = True
except FileNotFoundError:
    models_ready = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0f172a,#1e293b);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;
  border:none;border-radius:8px;padding:.5rem 2rem;font-weight:600;}
.result-card{border-radius:16px;padding:24px;margin:8px 0;border:1px solid;}
.good{background:linear-gradient(135deg,#052e16,#14532d);border-color:#16a34a;}
.bad{background:linear-gradient(135deg,#450a0a,#7f1d1d);border-color:#dc2626;}
.price-card{background:linear-gradient(135deg,#1e1b4b,#312e81);
  border:1px solid #4f46e5;border-radius:16px;padding:24px;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Property Analyzer")
st.markdown("Enter property details below to get an AI-powered investment recommendation.")

if not models_ready:
    st.error("⚠️ Models not found. Please run `python src/train.py` first.")
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("property_form"):
    st.subheader("📋 Property Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        city       = st.selectbox("City", ["Mumbai","Bangalore","Delhi","Gurgaon","Hyderabad",
                                            "Pune","Chennai","Kolkata","Noida","Ahmedabad",
                                            "Jaipur","Nagpur","Mysore","Lucknow","Chandigarh",
                                            "Kochi","Bhopal","Indore"])
        prop_type  = st.selectbox("Property Type", ["Apartment","Villa","House","Studio","Penthouse"])
        bhk        = st.selectbox("BHK", [1,2,3,4,5])
        furnished  = st.selectbox("Furnished Status", ["Fully Furnished","Semi Furnished","Unfurnished"])

    with c2:
        size       = st.number_input("Size (sq ft)", 300, 10000, 1200, step=50)
        price      = st.number_input("Current Price (Lakhs)", 5.0, 50000.0, 85.0, step=5.0)
        floor_no   = st.number_input("Floor Number", 0, 50, 3)
        total_fl   = st.number_input("Total Floors", 1, 60, 10)

    with c3:
        schools    = st.slider("Nearby Schools", 0, 10, 3)
        hospitals  = st.slider("Nearby Hospitals", 0, 8, 2)
        transport  = st.slider("Public Transport Score (1-9)", 1, 9, 5)
        parking    = st.slider("Parking Spaces", 0, 4, 1)

    c4, c5 = st.columns(2)
    with c4:
        facing     = st.selectbox("Facing", ["North","South","East","West","North-East","North-West"])
        owner_type = st.selectbox("Owner Type", ["Individual","Builder","Agent"])
    with c5:
        availability = st.selectbox("Availability Status", ["Available","Under Construction","Sold"])
        amenities_sel = st.multiselect("Amenities", ["Gym","Swimming Pool","Clubhouse","Garden","Playground","Library"])

    submitted = st.form_submit_button("🚀 Analyze Property", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    amenity_score  = len(amenities_sel)
    infra_score    = schools + hospitals + transport
    floor_ratio    = floor_no / max(total_fl, 1)
    age_prop       = 2024 - 2015   # assumed mid-vintage
    price_psf      = (price * 100_000) / max(size, 1)
    school_density = schools / 10
    hospital_density = hospitals / 8
    is_ready       = 1 if availability == "Available" else 0
    luxury_score   = amenity_score + parking
    bhk_cat        = {1:"Studio/1BHK", 2:"2BHK", 3:"3BHK"}.get(bhk, "4+BHK")

    HIGH = {"Bangalore","Mumbai","Delhi","Gurgaon"}
    MED  = {"Hyderabad","Pune","Chennai","Noida","Ahmedabad"}
    growth = 0.10 if city in HIGH else (0.08 if city in MED else 0.07)
    future_price_target = price * ((1 + growth) ** 5)

    # Build raw row matching training schema
    row = {
        "State": "Maharashtra",   # placeholder (encoded via dummies)
        "City": city,
        "Locality": "Central",
        "Property_Type": prop_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": price_psf,
        "Year_Built": 2015,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_fl,
        "Age_of_Property": age_prop,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Infra_Score": infra_score,
        "Floor_Ratio": floor_ratio,
        "Amenity_Score": amenity_score,
        "BHK_Category": bhk_cat,
        "Is_Ready": is_ready,
        "School_Density": school_density,
        "Hospital_Density": hospital_density,
        "Security_Score": 2,
        "Luxury_Score": luxury_score,
    }

    row_df = pd.DataFrame([row])
    row_df = pd.get_dummies(row_df, drop_first=True)
    row_df = row_df.reindex(columns=feature_cols, fill_value=0)
    row_sc = scaler.transform(row_df)

    clf_pred  = clf_model.predict(row_sc)[0]
    clf_proba = clf_model.predict_proba(row_sc)[0] if hasattr(clf_model, "predict_proba") else [0.5, 0.5]
    reg_pred  = reg_model.predict(row_sc)[0]

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if clf_pred == 1:
            st.markdown(f"""
            <div class='result-card good'>
              <h2 style='color:#4ade80;margin:0'>✅ Good Investment</h2>
              <p style='color:#86efac;margin:4px 0 0'>Confidence: {clf_proba[1]*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card bad'>
              <h2 style='color:#f87171;margin:0'>❌ Not Recommended</h2>
              <p style='color:#fca5a5;margin:4px 0 0'>Confidence: {clf_proba[0]*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        st.markdown("**Investment Probability**")
        st.progress(float(clf_proba[1]))
        st.caption(f"Good: {clf_proba[1]*100:.1f}%  |  Not Good: {clf_proba[0]*100:.1f}%")

    with res_col2:
        roi_est = ((reg_pred - price) / price) * 100
        st.markdown(f"""
        <div class='price-card'>
          <p style='color:#a5b4fc;margin:0;font-size:.9rem'>Estimated Price in 5 Years</p>
          <h1 style='color:#e0e7ff;margin:8px 0'>₹ {reg_pred:,.2f}L</h1>
          <p style='color:#6ee7b7;margin:0'>📈 Expected ROI: {roi_est:.1f}%</p>
          <p style='color:#94a3b8;font-size:.8rem;margin:4px 0 0'>
            Growth Rate: {growth*100:.0f}% p.a. ({city})</p>
        </div>""", unsafe_allow_html=True)

    # Feature importance (if tree-based)
    if hasattr(clf_model, "feature_importances_"):
        import plotly.graph_objects as go
        fi = pd.Series(clf_model.feature_importances_, index=feature_cols)
        top = fi.nlargest(12).sort_values()
        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color="#4f46e5"
        ))
        fig.update_layout(
            title="Top 12 Feature Importances (Classifier)",
            template="plotly_dark", height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
