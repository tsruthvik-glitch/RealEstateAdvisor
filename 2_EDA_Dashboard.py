"""
pages/2_EDA_Dashboard.py
Interactive EDA covering all 20 analytical questions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
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

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not DATA_PATH.exists():
    st.error("⚠️ cleaned_data.csv not found. Run `python src/preprocessing.py` first.")
    st.stop()

df = load_data()

st.title("📊 EDA Dashboard")
st.markdown(f"Dataset: **{len(df):,} properties** | **{df.shape[1]} features**")

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    if "City" in df.columns:
        cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)
    else:
        sel_city = "All"

    price_min, price_max = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    sel_price = st.slider("Price Range (Lakhs)", price_min, price_max,
                          (price_min, price_max), step=10.0)

fdf = df.copy()
if sel_city != "All":
    fdf = fdf[fdf["City"] == sel_city]
fdf = fdf[(fdf["Price_in_Lakhs"] >= sel_price[0]) & (fdf["Price_in_Lakhs"] <= sel_price[1])]

tabs = st.tabs([
    "💰 Price & Size", "📍 Location", "🔗 Correlations", "🏗️ Investment & Ownership"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Price & Size (Q1-Q5)
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Price & Size Analysis")

    c1, c2 = st.columns(2)

    # Q1 — Price distribution
    with c1:
        st.markdown("**Q1 · Price Distribution**")
        fig = px.histogram(fdf, x="Price_in_Lakhs", nbins=60,
                           color_discrete_sequence=["#4f46e5"],
                           labels={"Price_in_Lakhs":"Price (Lakhs)"},
                           template=DARK)
        fig.update_layout(margin=dict(t=20,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Q2 — Size distribution
    with c2:
        st.markdown("**Q2 · Size Distribution (sq ft)**")
        fig = px.histogram(fdf, x="Size_in_SqFt", nbins=60,
                           color_discrete_sequence=["#7c3aed"],
                           labels={"Size_in_SqFt":"Size (sq ft)"},
                           template=DARK)
        fig.update_layout(margin=dict(t=20,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Q3 — Price/sqft by property type
    with c3:
        st.markdown("**Q3 · Price/SqFt by Property Type**")
        if "Property_Type" in fdf.columns:
            fig = px.box(fdf, x="Property_Type", y="Price_per_SqFt",
                         color="Property_Type",
                         labels={"Price_per_SqFt":"₹/sq ft"},
                         template=DARK)
            fig.update_layout(margin=dict(t=20,b=10), height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Q4 — Size vs Price scatter
    with c4:
        st.markdown("**Q4 · Size vs Price**")
        fig = px.scatter(fdf.sample(min(2000,len(fdf))),
                         x="Size_in_SqFt", y="Price_in_Lakhs",
                         color="BHK" if "BHK" in fdf.columns else None,
                         opacity=0.5,
                         labels={"Size_in_SqFt":"Size (sq ft)","Price_in_Lakhs":"Price (L)"},
                         template=DARK)
        fig.update_layout(margin=dict(t=20,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Q5 — Outliers
    st.markdown("**Q5 · Outlier Detection — Price/SqFt & Size**")
    co1, co2 = st.columns(2)
    with co1:
        fig = px.box(fdf, y="Price_per_SqFt", color_discrete_sequence=["#f59e0b"],
                     template=DARK, labels={"Price_per_SqFt":"₹/sq ft"})
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with co2:
        fig = px.box(fdf, y="Size_in_SqFt", color_discrete_sequence=["#10b981"],
                     template=DARK, labels={"Size_in_SqFt":"sq ft"})
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Location (Q6-Q10)
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Location-based Analysis")

    # Q6 — Avg price/sqft by state
    if "State" in fdf.columns:
        st.markdown("**Q6 · Avg Price/SqFt by State**")
        state_avg = fdf.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=True)
        fig = px.bar(state_avg, orientation="h",
                     labels={"value":"Avg ₹/sq ft","index":"State"},
                     color=state_avg.values,
                     color_continuous_scale="Viridis",
                     template=DARK)
        fig.update_layout(height=420, margin=dict(t=20,b=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # Q7 — Avg price by city
    with c1:
        st.markdown("**Q7 · Avg Price by City**")
        if "City" in fdf.columns:
            city_avg = fdf.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15)
            fig = px.bar(city_avg, color=city_avg.values,
                         color_continuous_scale="Blues",
                         labels={"value":"Avg Price (L)","index":"City"},
                         template=DARK)
            fig.update_layout(height=360, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Q8 — Median age by locality (top 15)
    with c2:
        st.markdown("**Q8 · Median Property Age by Locality (Top 15)**")
        if "Locality" in fdf.columns and "Age_of_Property" in fdf.columns:
            loc_age = fdf.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False).head(15)
            fig = px.bar(loc_age, color=loc_age.values,
                         color_continuous_scale="Oranges",
                         labels={"value":"Median Age (yrs)","index":"Locality"},
                         template=DARK)
            fig.update_layout(height=360, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Q9 — BHK distribution by city
    with c3:
        st.markdown("**Q9 · BHK Distribution by City**")
        if "City" in fdf.columns and "BHK" in fdf.columns:
            bhk_city = fdf.groupby(["City","BHK"]).size().reset_index(name="Count")
            fig = px.bar(bhk_city, x="City", y="Count", color="BHK",
                         barmode="stack", template=DARK,
                         color_continuous_scale="Purples")
            fig.update_layout(height=360, margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Q10 — Top 5 expensive localities
    with c4:
        st.markdown("**Q10 · Top 5 Most Expensive Localities**")
        if "Locality" in fdf.columns:
            top5_loc = (fdf.groupby("Locality")["Price_in_Lakhs"]
                          .mean().sort_values(ascending=False).head(5))
            fig = px.bar(top5_loc, color=top5_loc.values,
                         color_continuous_scale="Reds",
                         labels={"value":"Avg Price (L)","index":"Locality"},
                         template=DARK)
            fig.update_layout(height=360, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Correlations (Q11-Q15)
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Feature Relationships & Correlations")

    # Q11 — Correlation heatmap
    st.markdown("**Q11 · Numeric Feature Correlation Heatmap**")
    num_df = fdf.select_dtypes(include=np.number).drop(
        columns=["Good_Investment","Future_Price_5Y","ROI","Growth_Rate"], errors="ignore"
    )
    corr = num_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    template=DARK, aspect="auto")
    fig.update_layout(height=500, margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # Q12 — Schools vs Price/SqFt
    with c1:
        st.markdown("**Q12 · Schools vs Price/SqFt**")
        if "Nearby_Schools" in fdf.columns:
            avg = fdf.groupby("Nearby_Schools")["Price_per_SqFt"].mean().reset_index()
            fig = px.line(avg, x="Nearby_Schools", y="Price_per_SqFt",
                          markers=True, template=DARK,
                          labels={"Nearby_Schools":"Schools Nearby","Price_per_SqFt":"Avg ₹/sq ft"})
            fig.update_traces(line_color="#4f46e5")
            fig.update_layout(height=300, margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Q13 — Hospitals vs Price/SqFt
    with c2:
        st.markdown("**Q13 · Hospitals vs Price/SqFt**")
        if "Nearby_Hospitals" in fdf.columns:
            avg = fdf.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().reset_index()
            fig = px.line(avg, x="Nearby_Hospitals", y="Price_per_SqFt",
                          markers=True, template=DARK,
                          labels={"Nearby_Hospitals":"Hospitals Nearby","Price_per_SqFt":"Avg ₹/sq ft"})
            fig.update_traces(line_color="#10b981")
            fig.update_layout(height=300, margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Q14 — Price by furnished status
    with c3:
        st.markdown("**Q14 · Price by Furnished Status**")
        if "Furnished_Status" in fdf.columns:
            fig = px.violin(fdf, x="Furnished_Status", y="Price_in_Lakhs",
                            color="Furnished_Status", box=True, template=DARK,
                            labels={"Price_in_Lakhs":"Price (Lakhs)"})
            fig.update_layout(height=330, margin=dict(t=20,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Q15 — Price/SqFt by facing
    with c4:
        st.markdown("**Q15 · Price/SqFt by Facing Direction**")
        if "Facing" in fdf.columns:
            avg_f = fdf.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False)
            fig = px.bar(avg_f, color=avg_f.values,
                         color_continuous_scale="Turbo",
                         labels={"value":"Avg ₹/sq ft","index":"Facing"},
                         template=DARK)
            fig.update_layout(height=330, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Investment & Ownership (Q16-Q20)
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Investment, Amenities & Ownership")

    c1, c2 = st.columns(2)

    # Q16 — Owner type distribution
    with c1:
        st.markdown("**Q16 · Owner Type Distribution**")
        if "Owner_Type" in fdf.columns:
            ot = fdf["Owner_Type"].value_counts()
            fig = px.pie(values=ot.values, names=ot.index,
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template=DARK)
            fig.update_layout(height=320, margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Q17 — Availability status
    with c2:
        st.markdown("**Q17 · Availability Status**")
        if "Availability_Status" in fdf.columns:
            avail = fdf["Availability_Status"].value_counts()
            fig = px.pie(values=avail.values, names=avail.index,
                         color_discrete_sequence=["#4f46e5","#7c3aed","#a855f7"],
                         hole=0.4, template=DARK)
            fig.update_layout(height=320, margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Q18 — Parking vs price
    with c3:
        st.markdown("**Q18 · Parking Spaces vs Price**")
        if "Parking_Space" in fdf.columns:
            avg_p = fdf.groupby("Parking_Space")["Price_in_Lakhs"].mean().reset_index()
            fig = px.bar(avg_p, x="Parking_Space", y="Price_in_Lakhs",
                         color="Price_in_Lakhs",
                         color_continuous_scale="Teal",
                         labels={"Price_in_Lakhs":"Avg Price (L)","Parking_Space":"Parking Spots"},
                         template=DARK)
            fig.update_layout(height=320, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Q19 — Amenity Score vs Price/SqFt
    with c4:
        st.markdown("**Q19 · Amenity Score vs Price/SqFt**")
        if "Amenity_Score" in fdf.columns:
            avg_a = fdf.groupby("Amenity_Score")["Price_per_SqFt"].mean().reset_index()
            fig = px.bar(avg_a, x="Amenity_Score", y="Price_per_SqFt",
                         color="Price_per_SqFt",
                         color_continuous_scale="Magenta",
                         labels={"Amenity_Score":"Amenity Count","Price_per_SqFt":"Avg ₹/sq ft"},
                         template=DARK)
            fig.update_layout(height=320, margin=dict(t=20,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Q20 — Public transport vs Price/SqFt
    st.markdown("**Q20 · Public Transport Accessibility vs Price/SqFt**")
    if "Public_Transport_Accessibility" in fdf.columns:
        avg_t = fdf.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().reset_index()
        fig = px.line(avg_t, x="Public_Transport_Accessibility", y="Price_per_SqFt",
                      markers=True, template=DARK,
                      labels={"Public_Transport_Accessibility":"Transport Score (1-9)",
                              "Price_per_SqFt":"Avg ₹/sq ft"})
        fig.update_traces(line_color="#f59e0b", line_width=2)
        fig.update_layout(height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Good Investment summary
    if "Good_Investment" in fdf.columns:
        st.markdown("---")
        st.markdown("**🏆 Good Investment Distribution**")
        gi = fdf["Good_Investment"].value_counts().rename({0:"Not Recommended", 1:"Good Investment"})
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("✅ Good Investments", f"{gi.get('Good Investment',0):,}")
        mc2.metric("❌ Not Recommended", f"{gi.get('Not Recommended',0):,}")
        mc3.metric("📈 % Good",
                   f"{fdf['Good_Investment'].mean()*100:.1f}%")
