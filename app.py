import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

st.set_page_config(page_title="Company Intelligence Explorer", layout="wide")
st.title("Company Segmentation & Intelligence Explorer")

# ---------- Load (upload to avoid path issues) ----------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload the dataset to begin.")
    st.stop()

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    # standardize to snake_case-ish for your rule code
    df.columns = [c.lower().strip().replace(" ", "_").replace(".", "") for c in df.columns]
    return df

df = load_data(uploaded)

# ---------- Helpers ----------
def pick_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand2 = cand.lower().strip().replace(" ", "_").replace(".", "")
        if cand2 in lower:
            return lower[cand2]
    return None

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def add_rule_segments(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()

    def _digits_only(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = re.sub(r"\D+", "", s)
        return s if s != "" else np.nan

    def sic_prefix(x, n=2):
        s = _digits_only(x)
        if pd.isna(s):
            return np.nan
        return s[:n] if len(s) >= n else s.zfill(n)

    def safe_qcut(series, q=4, labels=None):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() < q * 5:
            q = 3
            labels = labels[:3] if labels is not None else None
        try:
            return pd.qcut(s, q=q, labels=labels, duplicates="drop")
        except Exception:
            return pd.Series(pd.NA, index=series.index, dtype="string")

    # ---------- 1) Industry bucket ----------
    sic_col = "8_digit_sic_code" if "8_digit_sic_code" in df.columns else ("sic_code" if "sic_code" in df.columns else None)
    df["sic_2digit"] = df[sic_col].map(lambda x: sic_prefix(x, n=2)) if sic_col else "Unknown"

    # ---------- 2) Size tiers ----------
    for c in ["employees_total", "revenue_usd", "it_spend", "it_budget"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "employees_total" in df.columns:
        df["log_employees"] = np.log1p(df["employees_total"])
        df["size_emp_tier"] = safe_qcut(df["log_employees"], q=4, labels=["emp_s", "emp_m", "emp_l", "emp_xl"]).astype("string")
    else:
        df["size_emp_tier"] = "Unknown"

    if "revenue_usd" in df.columns:
        df["log_revenue"] = np.log1p(df["revenue_usd"])
        df["size_rev_tier"] = safe_qcut(df["log_revenue"], q=4, labels=["rev_s", "rev_m", "rev_l", "rev_xl"]).astype("string")
    else:
        df["size_rev_tier"] = "Unknown"

    # ---------- 3) Structure tier ----------
    for b in ["is_headquarters", "is_domestic_ultimate"]:
        if b in df.columns:
            if df[b].dtype.name in ["string", "object"]:
                df[b] = df[b].astype("string").str.lower().map({"true": True, "false": False})
            df[b] = df[b].fillna(False).astype(bool)
        else:
            df[b] = False

    df["has_parent_company"] = df["parent_company"].notna() if "parent_company" in df.columns else False
    df["has_global_ultimate"] = df["global_ultimate_company"].notna() if "global_ultimate_company" in df.columns else False
    df["has_domestic_ultimate_company"] = df["domestic_ultimate_company"].notna() if "domestic_ultimate_company" in df.columns else False

    def structure_tier(row):
        if row.get("is_headquarters", False):
            return "hq"
        if row.get("is_domestic_ultimate", False):
            return "domestic_ultimate"
        et = str(row.get("entity_type", "")).lower()
        if "subsidi" in et:
            return "subsidiary"
        if "branch" in et:
            return "branch"
        if row.get("has_parent_company", False):
            return "subsidiary_like"
        if row.get("has_global_ultimate", False) or row.get("has_domestic_ultimate_company", False):
            return "member_of_group"
        return "standalone_like"

    df["structure_tier"] = df.apply(structure_tier, axis=1).astype("string")

    # ---------- 4) IT tiers + device tiers ----------
    if "it_spend" in df.columns:
        df["log_it_spend"] = np.log1p(df["it_spend"])
        df["it_spend_tier"] = safe_qcut(df["log_it_spend"], q=4, labels=["it_low", "it_mid", "it_high", "it_top"]).astype("string")
    else:
        df["it_spend_tier"] = "Unknown"

    device_cols = [c for c in ["no_of_pc", "no_of_desktops", "no_of_laptops", "no_of_routers", "no_of_servers", "no_of_storage_devices"] if c in df.columns]
    df["device_total"] = df[device_cols].sum(axis=1, min_count=1) if device_cols else np.nan
    df["log_device_total"] = np.log1p(df["device_total"])
    df["device_tier"] = safe_qcut(df["log_device_total"], q=4, labels=["dev_low", "dev_mid", "dev_high", "dev_top"]).astype("string")

    # ---------- 5) Geo tiers ----------
    if "region" in df.columns:
        df["geo_tier"] = df["region"].astype("string")
    elif "country" in df.columns:
        df["geo_tier"] = df["country"].astype("string")
    else:
        df["geo_tier"] = "Unknown"

    # ---------- 6) Final label/id ----------
    seg_parts = ["sic_2digit", "size_emp_tier", "size_rev_tier", "structure_tier", "it_spend_tier", "device_tier", "geo_tier"]
    for c in seg_parts:
        df[c] = df[c].fillna("Unknown").astype("string")

    df["segment_label"] = df[seg_parts].agg("|".join, axis=1)
    seg_order = df["segment_label"].value_counts().index.tolist()
    seg_map = {lab: i for i, lab in enumerate(seg_order)}
    df["segment_id"] = df["segment_label"].map(seg_map).astype(int)

    return df

df = add_rule_segments(df)

# ---------- Sidebar filters ----------
st.sidebar.subheader("Filters")
col_country = pick_col(df, ["country"])
col_entity  = pick_col(df, ["entity_type"])
col_name    = pick_col(df, ["company_name", "name", "company"])

def multiselect_filter(label, col):
    if col is None:
        st.sidebar.caption(f"⚠️ {label}: column not found")
        return []
    vals = sorted([v for v in df[col].dropna().astype(str).unique()])
    return st.sidebar.multiselect(label, vals)

sel_country = multiselect_filter("Country", col_country)
sel_entity  = multiselect_filter("Entity Type", col_entity)

seg_vals = sorted(df["segment_label"].dropna().astype(str).unique())
sel_segs = st.sidebar.multiselect("Segment", seg_vals)

filtered = df.copy()
if col_country and sel_country:
    filtered = filtered[filtered[col_country].astype(str).isin(sel_country)]
if col_entity and sel_entity:
    filtered = filtered[filtered[col_entity].astype(str).isin(sel_entity)]
if sel_segs:
    filtered = filtered[filtered["segment_label"].astype(str).isin(sel_segs)]

st.sidebar.caption(f"Filtered rows: {len(filtered):,}")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Explore Companies", "Explore Segments"])

with tab1:
    st.subheader("Company Explorer")

    if col_name:
        company = st.selectbox("Select a company", sorted(filtered[col_name].fillna("UNKNOWN").astype(str).unique()))
        row = filtered[filtered[col_name].astype(str) == str(company)].head(1)
        st.write("Company record")
        st.dataframe(row, use_container_width=True)

    st.subheader("Filtered preview (top 200 rows)")
    st.dataframe(filtered.head(200), use_container_width=True)

with tab2:
    st.subheader("Segment Summary")
    seg_counts = filtered["segment_label"].value_counts().reset_index()
    seg_counts.columns = ["segment_label", "count"]
    st.dataframe(seg_counts.head(30), use_container_width=True)

    st.subheader("Top segments chart")
    top = seg_counts.head(15)
    fig = plt.figure()
    plt.bar(top["segment_label"].astype(str), top["count"])
    plt.xticks(rotation=90)
    st.pyplot(fig, clear_figure=True)
