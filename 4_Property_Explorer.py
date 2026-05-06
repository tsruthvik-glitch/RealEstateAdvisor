"""
pages/4_Property_Explorer.py
Filter and browse the dataset interactively.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Property Explorer", page_icon="🔎", layout="wide")

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0f172a,#1e293b);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;
  border:none;border-radius:8px;padding:.4rem 1.6rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not DATA_PATH.exists():
    st.error("⚠️ cleaned_data.csv not found. Run `python src/preprocessing.py` first.")
    st.stop()

df = load_data()

st.title("🔎 Property Explorer")
st.markdown("Filter properties and explore the dataset interactively.")

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")

    cities = ["All"] + sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else ["All"]
    sel_city = st.selectbox("City", cities)

    prop_types = ["All"] + sorted(df["Property_Type"].dropna().unique().tolist()) if "Property_Type" in df.columns else ["All"]
    sel_type = st.selectbox("Property Type", prop_types)

    bhk_opts = sorted(df["BHK"].dropna().unique().tolist()) if "BHK" in df.columns else [1,2,3,4,5]
    sel_bhk = st.multiselect("BHK", bhk_opts, default=bhk_opts)

    price_min = float(df["Price_in_Lakhs"].min())
    price_max = float(df["Price_in_Lakhs"].max())
    sel_price = st.slider("Price Range (Lakhs)", price_min, price_max,
                          (price_min, price_max), step=5.0)

    size_min = float(df["Size_in_SqFt"].min())
    size_max = float(df["Size_in_SqFt"].max())
    sel_size = st.slider("Size Range (sq ft)", size_min, size_max,
                         (size_min, size_max), step=50.0)

    if "Good_Investment" in df.columns:
        inv_filter = st.radio("Investment Status", ["All","Good Only","Not Recommended"])
    else:
        inv_filter = "All"

    if "Furnished_Status" in df.columns:
        furnished_opts = ["All"] + sorted(df["Furnished_Status"].dropna().unique().tolist())
        sel_furnished = st.selectbox("Furnished Status", furnished_opts)
    else:
        sel_furnished = "All"

# ── Apply Filters ─────────────────────────────────────────────────────────────
fdf = df.copy()

if sel_city != "All" and "City" in fdf.columns:
    fdf = fdf[fdf["City"] == sel_city]
if sel_type != "All" and "Property_Type" in fdf.columns:
    fdf = fdf[fdf["Property_Type"] == sel_type]
if sel_bhk and "BHK" in fdf.columns:
    fdf = fdf[fdf["BHK"].isin(sel_bhk)]
if "Price_in_Lakhs" in fdf.columns:
    fdf = fdf[(fdf["Price_in_Lakhs"] >= sel_price[0]) & (fdf["Price_in_Lakhs"] <= sel_price[1])]
if "Size_in_SqFt" in fdf.columns:
    fdf = fdf[(fdf["Size_in_SqFt"] >= sel_size[0]) & (fdf["Size_in_SqFt"] <= sel_size[1])]
if inv_filter == "Good Only" and "Good_Investment" in fdf.columns:
    fdf = fdf[fdf["Good_Investment"] == 1]
elif inv_filter == "Not Recommended" and "Good_Investment" in fdf.columns:
    fdf = fdf[fdf["Good_Investment"] == 0]
if sel_furnished != "All" and "Furnished_Status" in fdf.columns:
    fdf = fdf[fdf["Furnished_Status"] == sel_furnished]

# ── Summary Metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🏠 Properties", f"{len(fdf):,}")
m2.metric("💰 Avg Price (L)", f"₹{fdf['Price_in_Lakhs'].mean():.1f}" if len(fdf) else "—")
m3.metric("📐 Avg Size (sqft)", f"{fdf['Size_in_SqFt'].mean():.0f}" if len(fdf) else "—")
m4.metric("📊 Avg ₹/sqft", f"₹{fdf['Price_per_SqFt'].mean():.0f}" if len(fdf) else "—")
if "Good_Investment" in fdf.columns and len(fdf):
    m5.metric("✅ Good Inv %", f"{fdf['Good_Investment'].mean()*100:.1f}%")

st.markdown("---")

# ── Sort Controls ─────────────────────────────────────────────────────────────
sort_cols = ["Price_in_Lakhs","Size_in_SqFt","Price_per_SqFt","Future_Price_5Y","ROI","Age_of_Property"]
sort_cols = [c for c in sort_cols if c in fdf.columns]

sc1, sc2 = st.columns([3,1])
with sc1:
    sort_by = st.selectbox("Sort By", sort_cols, index=0)
with sc2:
    sort_asc = st.radio("Order", ["↑ Ascending","↓ Descending"], index=1) == "↑ Ascending"

fdf_sorted = fdf.sort_values(sort_by, ascending=sort_asc) if sort_by in fdf.columns else fdf

# ── Display Columns ───────────────────────────────────────────────────────────
display_cols = [c for c in [
    "City","Locality","Property_Type","BHK","Size_in_SqFt",
    "Price_in_Lakhs","Price_per_SqFt","Furnished_Status",
    "Future_Price_5Y","ROI","Good_Investment"
] if c in fdf_sorted.columns]

st.dataframe(
    fdf_sorted[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=460
)

st.caption(f"Showing {len(fdf_sorted):,} of {len(df):,} properties")

# ── Download ──────────────────────────────────────────────────────────────────
csv = fdf_sorted[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Data",
    data=csv,
    file_name="filtered_properties.csv",
    mime="text/csv"
)
