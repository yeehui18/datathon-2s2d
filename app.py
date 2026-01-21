"""
Streamlit app (clean + segment + explore) for Champions Group dataset.

Fixes in this version:
- NO pandas "string" dtype anywhere (uses plain object/str), so it won't crash with:
  TypeError: data type 'string' not understood
- Robust bucket parsing for device/server fields (eg '1 to 10', '1,001 to 5,000', '100000+')
- Content normalisation (country casing, missing tokens, phone formatting)
- Keeps only columns needed for the 5 attribute groups + a few UI display fields
- Adds rule-based segments (interpretable)

Run:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from huggingface_hub import InferenceClient

# Initialize Client (Best practice: use st.secrets, but for now we paste directly)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()
repo_id = "Qwen/Qwen2.5-72B-Instruct"
llm_client = InferenceClient(model=repo_id, token=HF_TOKEN)

def get_dataframe_context(df, max_rows=5):
    """
    Creates a text summary of the current dataframe to send to the LLM.
    Uses to_string() to avoid dependency on 'tabulate'.
    """
    if df.empty:
        return "The dataset is currently empty."

    # metrics summary
    row_count = len(df)
    col_names = ", ".join(df.columns.tolist())

    # CHANGE: Use to_string instead of to_markdown
    preview = df.head(max_rows).to_string(index=False)

    context = f"""
    Dataset Summary:
    - Total Rows in current view: {row_count}
    - Columns: {col_names}

    Data Preview (First {max_rows} rows):
    {preview}
    """
    return context

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="Company Intelligence Explorer", layout="wide")
st.title("Company Segmentation and Intelligence Explorer")

# -----------------------------
# Upload (avoid file path issues)
# -----------------------------
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload the dataset to begin.")
    st.stop()

# -----------------------------
# Helpers: column name + content cleaning
# -----------------------------
def to_snake(s: str) -> str:
    """Convert column names to snake_case-like strings."""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)  # replace non-alphanumeric with _
    return s.strip("_")

def normalise_missing_text(series: pd.Series) -> pd.Series:
    """
    Standardise common missing tokens into actual NA.
    Uses plain object dtype (not pandas StringDtype).
    """
    s = series.astype(object)
    miss = {"", "na", "n/a", "none", "null", "unknown", "nan"}
    out = []
    for v in s.values:
        if pd.isna(v):
            out.append(pd.NA)
            continue
        t = str(v).strip()
        if t.lower() in miss:
            out.append(pd.NA)
        else:
            out.append(t)
    return pd.Series(out, index=series.index, dtype=object)

def clean_phone(x):
    """
    Phone numbers sometimes appear as floats/scientific notation in Excel.
    Convert to a clean digit string where possible.
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    # Try convert scientific notation -> int string
    try:
        f = float(s)
        if np.isfinite(f):
            return str(int(f))
    except Exception:
        pass

    # Remove trailing .0
    if re.match(r"^\d+\.0$", s):
        return s[:-2]

    return s

def bucket_to_midpoint(x):
    """
    Parse range buckets commonly found in device/server columns.
    Examples:
      '1 to 10' -> 5.5
      '1,001 to 5,000' -> 3000.5
      '100000+' -> 100000
      '12' -> 12
    Returns np.nan if not parseable.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(",", "")
    if s in {"", "na", "n/a", "none", "null", "unknown"}:
        return np.nan

    # '100000+' style
    m = re.match(r"^(\d+)\s*\+$", s)
    if m:
        return float(m.group(1))

    # 'a to b' or 'a - b' style
    m = re.match(r"^(\d+)\s*(to|-)\s*(\d+)$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(3))
        return (a + b) / 2.0

    # plain number
    m = re.match(r"^\d+(\.\d+)?$", s)
    if m:
        return float(s)

    return np.nan

def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to numeric safely.
    If most values are not numeric, try bucket_to_midpoint (useful for device bucket fields).
    """
    x = pd.to_numeric(series, errors="coerce")
    # If too many NaNs after numeric conversion, attempt bucket parsing
    if x.notna().mean() < 0.30:
        x2 = series.map(bucket_to_midpoint)
        if pd.Series(x2).notna().mean() > x.notna().mean():
            return pd.to_numeric(x2, errors="coerce")
    return x

def zero_to_nan(series: pd.Series) -> pd.Series:
    """
    Many columns use 0 as placeholder for missing.
    Convert 0 -> NaN for columns where true 0 is unlikely (revenue, employees, IT spend, etc.).
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.mask(x == 0, np.nan)

def pick_col(df: pd.DataFrame, candidates):
    """Find first existing column name from a list of candidates."""
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None

# -----------------------------
# Load + Clean + Keep relevant columns
# -----------------------------
@st.cache_data
def load_and_clean(file) -> pd.DataFrame:
    """
    Load Excel and apply:
    - snake_case columns
    - content normalisation (country casing, phone, missing tokens)
    - numeric conversion for key fields
    - device bucket parsing into numeric midpoints (keeping raw bucket text too)
    - keep only columns relevant to the 5 attribute groups + UI identifiers
    """
    df = pd.read_excel(file)
    df.columns = [to_snake(c) for c in df.columns]

    # ---------- General content normalisation ----------
    for c in [
        "country", "city", "state", "state_or_province_abbreviation",
        "entity_type", "ownership_type", "legal_status",
        "company_status_active_inactive", "sic_description", "8_digit_sic_description"
    ]:
        if c in df.columns:
            df[c] = normalise_missing_text(df[c])

    # Country: case normalisation (CHINA vs China)
    if "country" in df.columns:
        df["country"] = df["country"].astype(object).apply(lambda x: str(x).upper().strip() if not pd.isna(x) else pd.NA)

    # Entity type: consistent casing for UI
    if "entity_type" in df.columns:
        df["entity_type"] = df["entity_type"].astype(object).apply(lambda x: str(x).strip().title() if not pd.isna(x) else pd.NA)

    # Phone number: fix float/scientific notation
    if "phone_number" in df.columns:
        df["phone_number"] = df["phone_number"].apply(clean_phone)

    # ---------- Numeric conversions ----------
    placeholder_zero_cols = [
        "employees_total", "employees_single_site",
        "revenue_usd", "market_value_usd",
        "it_budget", "it_spend",
        "corporate_family_members"
    ]
    for c in placeholder_zero_cols:
        if c in df.columns:
            df[c] = zero_to_nan(df[c])

    # Year found: sanity bounds
    if "year_found" in df.columns:
        y = pd.to_numeric(df["year_found"], errors="coerce")
        df["year_found"] = y.mask((y <= 1700) | (y > 2026), np.nan)

    # Device/server columns: parse bucket ranges -> numeric midpoints
    device_cols = [
        "no_of_pc", "no_of_desktops", "no_of_laptops",
        "no_of_routers", "no_of_servers", "no_of_storage_devices"
    ]
    for c in device_cols:
        if c in df.columns:
            # keep raw bucket text for display/debug
            df[c + "_bucket"] = normalise_missing_text(df[c])
            # numeric for modelling
            df[c] = safe_numeric(df[c])

    # device_total derived
    dev_present = [c for c in device_cols if c in df.columns]
    df["device_total"] = df[dev_present].sum(axis=1, min_count=1) if dev_present else np.nan

    # Derived metrics (safe division)
    if "employees_total" in df.columns and "revenue_usd" in df.columns:
        df["revenue_per_employee"] = df["revenue_usd"] / df["employees_total"]
    if "it_spend" in df.columns and "revenue_usd" in df.columns:
        df["it_spend_to_revenue"] = df["it_spend"] / df["revenue_usd"]

    # ---------- Keep only relevant columns ----------
    industry_cols = ["sic_code", "sic_description", "8_digit_sic_code", "8_digit_sic_description"]
    size_cols = ["employees_single_site", "employees_total", "revenue_usd", "market_value_usd", "year_found", "revenue_per_employee"]
    structure_cols = [
        "entity_type", "ownership_type", "corporate_family_members",
        "is_headquarters", "is_domestic_ultimate",
        "parent_company", "global_ultimate_company", "domestic_ultimate_company"
    ]
    it_cols = ["it_budget", "it_spend", "it_spend_to_revenue", "device_total"] + device_cols + [c + "_bucket" for c in device_cols if c in df.columns]
    geo_cols = ["country", "state", "state_or_province_abbreviation", "city", "postal_code", "lattitude", "longitude"]
    ui_cols = [
        "duns_number", "company_sites", "website", "address_line_1",
        "phone_number", "registration_number",
        "company_description", "company_status_active_inactive", "legal_status"
    ]

    keep = []
    for grp in [industry_cols, size_cols, structure_cols, it_cols, geo_cols, ui_cols]:
        keep += [c for c in grp if c in df.columns]
    keep = list(dict.fromkeys(keep))  # de-dupe preserving order

    df = df[keep].copy()
    return df

df = load_and_clean(uploaded)

# -----------------------------
# Segmentation (rule-based, interpretable) - NO "string" dtype used
# -----------------------------
@st.cache_data
def add_rule_segments(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable segments based on:
    - Industry: SIC prefix (2 digits)
    - Size: employee/revenue tiers
    - Corporate structure: HQ / domestic ultimate / subsidiary / branch / etc.
    - IT footprint: IT spend tiers + device tiers
    - Geo: country

    All text columns are handled as plain object/str to avoid pandas StringDtype issues.
    """
    df = df_in.copy()

    # ---------- Industry bucket ----------
    sic_col = "8_digit_sic_code" if "8_digit_sic_code" in df.columns else ("sic_code" if "sic_code" in df.columns else None)

    def digits_only(x):
        if pd.isna(x):
            return np.nan
        s = re.sub(r"\D+", "", str(x))
        return s if s else np.nan

    def sic_prefix(x, n=2):
        s = digits_only(x)
        if pd.isna(s):
            return "Unknown"
        return s[:n].zfill(n)

    df["sic_2digit"] = df[sic_col].map(lambda x: sic_prefix(x, 2)) if sic_col else "Unknown"

    # ---------- Size tiers ----------
    def safe_qcut(series, q=4, labels=None):
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() < q * 10:
            q = 3
            if labels is not None:
                labels = labels[:3]
        try:
            return pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(object)
        except Exception:
            return pd.Series(["Unknown"] * len(series), index=series.index, dtype=object)

    if "employees_total" in df.columns:
        df["log_employees"] = np.log1p(df["employees_total"])
        df["size_emp_tier"] = safe_qcut(df["log_employees"], q=4, labels=["emp_s", "emp_m", "emp_l", "emp_xl"])
    else:
        df["size_emp_tier"] = "Unknown"

    if "revenue_usd" in df.columns:
        df["log_revenue"] = np.log1p(df["revenue_usd"])
        df["size_rev_tier"] = safe_qcut(df["log_revenue"], q=4, labels=["rev_s", "rev_m", "rev_l", "rev_xl"])
    else:
        df["size_rev_tier"] = "Unknown"

    # ---------- Corporate structure ----------
    def to_bool_col(colname):
        if colname not in df.columns:
            return pd.Series([False] * len(df), index=df.index, dtype=bool)
        s = df[colname].astype(object)
        out = []
        for v in s.values:
            if pd.isna(v):
                out.append(False)
            else:
                t = str(v).strip().lower()
                out.append(True if t == "true" else False)
        return pd.Series(out, index=df.index, dtype=bool)

    is_hq = to_bool_col("is_headquarters")
    is_du = to_bool_col("is_domestic_ultimate")

    df["has_parent_company"] = df["parent_company"].notna() if "parent_company" in df.columns else False
    df["has_global_ultimate"] = df["global_ultimate_company"].notna() if "global_ultimate_company" in df.columns else False
    df["has_domestic_ultimate_company"] = df["domestic_ultimate_company"].notna() if "domestic_ultimate_company" in df.columns else False

    if "entity_type" in df.columns:
        et = df["entity_type"].astype(object).apply(lambda x: str(x).lower() if not pd.isna(x) else "")
    else:
        et = pd.Series([""] * len(df), index=df.index, dtype=object)

    df["structure_tier"] = np.select(
        [
            is_hq,
            is_du,
            et.str.contains("subsidi", na=False),
            et.str.contains("branch", na=False),
            df["has_parent_company"] == True,
            (df["has_global_ultimate"] == True) | (df["has_domestic_ultimate_company"] == True),
        ],
        [
            "hq",
            "domestic_ultimate",
            "subsidiary",
            "branch",
            "subsidiary_like",
            "member_of_group",
        ],
        default="standalone_like"
    ).astype(object)

    # ---------- IT tiers ----------
    if "it_spend" in df.columns:
        df["log_it_spend"] = np.log1p(df["it_spend"])
        df["it_spend_tier"] = safe_qcut(df["log_it_spend"], q=4, labels=["it_low", "it_mid", "it_high", "it_top"])
    else:
        df["it_spend_tier"] = "Unknown"

    if "device_total" in df.columns:
        df["log_device_total"] = np.log1p(df["device_total"])
        df["device_tier"] = safe_qcut(df["log_device_total"], q=4, labels=["dev_low", "dev_mid", "dev_high", "dev_top"])
    else:
        df["device_tier"] = "Unknown"

    # ---------- Geo tier ----------
    if "country" in df.columns:
        df["geo_tier"] = df["country"].astype(object)
        df["geo_tier"] = df["geo_tier"].fillna("Unknown")
    else:
        df["geo_tier"] = "Unknown"

    # ---------- Final label/id ----------
    seg_parts = ["sic_2digit", "size_emp_tier", "size_rev_tier", "structure_tier", "it_spend_tier", "device_tier", "geo_tier"]
    for c in seg_parts:
        df[c] = df[c].astype(object)
        df[c] = df[c].where(~pd.isna(df[c]), "Unknown")  # fill NA with "Unknown"

    df["segment_label"] = df[seg_parts].astype(str).agg("|".join, axis=1)
    df["segment_id"] = df["segment_label"].astype("category").cat.codes

    return df

df = add_rule_segments(df)

# -----------------------------
# Quick data health summary
# -----------------------------
with st.expander("Data health (quick checks)", expanded=False):
    st.write(f"Rows: {len(df):,}  Columns: {df.shape[1]}")

    if "country" in df.columns:
        st.write("Country breakdown")
        ctab = df["country"].fillna("Unknown").astype(str).value_counts(dropna=False).reset_index()
        ctab.columns = ["country", "count"]
        st.dataframe(ctab, use_container_width=True)

    key_cols = [c for c in [
        "sic_code", "8_digit_sic_code",
        "employees_total", "revenue_usd",
        "it_spend", "device_total",
        "corporate_family_members", "entity_type",
        "country", "state", "city"
    ] if c in df.columns]

    if key_cols:
        miss = (df[key_cols].isna().mean() * 100).round(1).sort_values(ascending=False)
        miss_df = miss.reset_index()
        miss_df.columns = ["column", "missing_percent"]
        st.write("Missing % for key columns")
        st.dataframe(miss_df, use_container_width=True)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.subheader("Filters")

col_country = pick_col(df, ["country"])
col_entity = pick_col(df, ["entity_type"])

def multiselect_filter(label, col):
    if col is None:
        st.sidebar.caption(f"⚠️ {label}: column not found")
        return []
    vals = sorted([v for v in df[col].dropna().astype(str).unique()])
    return st.sidebar.multiselect(label, vals)

sel_country = multiselect_filter("Country", col_country)
sel_entity = multiselect_filter("Entity Type", col_entity)

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

# -----------------------------
# Tabs (basic exploration)
# -----------------------------
tab1, tab2 = st.tabs(["Explore Companies", "Explore Segments"])

with tab1:
    st.subheader("Company Explorer")

    col_name = pick_col(filtered, ["company_name", "name", "company"])
    if col_name:
        company = st.selectbox("Select a company", sorted(filtered[col_name].fillna("UNKNOWN").astype(str).unique()))
        row = filtered[filtered[col_name].astype(str) == str(company)].head(1)
        st.write("Company record")
        st.dataframe(row, use_container_width=True)
    else:
        st.info("No company name column detected (expected something like company_name). Showing table only.")

    st.subheader("Filtered preview (top 200 rows)")
    st.dataframe(filtered.head(200), use_container_width=True)

with tab2:
    st.subheader("Segment Summary")
    seg_counts = filtered["segment_label"].value_counts().reset_index()
    seg_counts.columns = ["segment_label", "count"]
    st.dataframe(seg_counts.head(50), use_container_width=True)

    st.subheader("Top segments chart")
    top = seg_counts.head(15)
    fig = plt.figure()
    plt.bar(top["segment_label"].astype(str), top["count"])
    plt.xticks(rotation=90)
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# AI Assistant (Sidebar)
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 AI Data Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about the data..."):
        # 1. Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Build Context (The filtered data from your app)
        # Use the 'filtered' dataframe from your main script
        data_context = get_dataframe_context(filtered, max_rows=5)

        # 3. Construct the prompt for the LLM
        full_prompt = f"""
        You are a helpful Data Analyst assistant. 
        Analyze the following dataset snippet and answer the user's question.

        CONTEXT DATA:
        {data_context}

        USER QUESTION: 
        {prompt}

        Answer concisely and based ONLY on the provided data context.
        """

        # 4. Stream the response
        with st.chat_message("assistant"):
            try:
                # Helper: Yields text chunks from the API response
                def stream_generator():
                    stream = llm_client.chat_completion(
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=500,
                        stream=True
                    )
                    for chunk in stream:
                        # Extract text from the "delta" in the chunk
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                # Streamlit writes the stream to the UI
                response = st.write_stream(stream_generator())

                # Save the final response to history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error communicating with API: {e}")
