

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Optional AI
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# =========================
# Page config
# =========================
st.set_page_config(page_title="Company Intelligence Explorer", layout="wide")
st.title("Company Segmentation and Intelligence Explorer")


# =========================
# Upload
# =========================
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload the dataset to begin.")
    st.stop()


# =========================
# Utility helpers
# =========================
def to_snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")


def normalise_missing_text(series: pd.Series) -> pd.Series:
    """
    Normalise common missing tokens into NA.
    Keeps dtype as plain object.
    """
    s = series.astype(object)
    miss = {"", "na", "n/a", "none", "null", "unknown", "nan"}
    out = []
    for v in s.values:
        if pd.isna(v):
            out.append(pd.NA)
            continue
        t = str(v).strip()
        out.append(pd.NA if t.lower() in miss else t)
    return pd.Series(out, index=series.index, dtype=object)


def clean_phone(x):
    """
    Excel sometimes stores phone numbers as floats/scientific notation.
    Convert to a clean digits string where possible.
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    try:
        f = float(s)
        if np.isfinite(f):
            return str(int(f))
    except Exception:
        pass

    if re.match(r"^\d+\.0$", s):
        return s[:-2]

    return s


def bucket_to_midpoint(x):
    """
    Parse bucket ranges commonly found in device/server columns.
    Examples:
      '1 to 10' -> 5.5
      '1,001 to 5,000' -> 3000.5
      '100000+' -> 100000
      '12' -> 12
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(",", "")
    if s in {"", "na", "n/a", "none", "null", "unknown"}:
        return np.nan

    m = re.match(r"^(\d+)\s*\+$", s)
    if m:
        return float(m.group(1))

    m = re.match(r"^(\d+)\s*(to|-)\s*(\d+)$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(3))
        return (a + b) / 2.0

    m = re.match(r"^\d+(\.\d+)?$", s)
    if m:
        return float(s)

    return np.nan


def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to numeric safely.
    If most values fail numeric conversion, attempt bucket_to_midpoint parsing.
    """
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().mean() < 0.30:
        x2 = series.map(bucket_to_midpoint)
        if pd.Series(x2).notna().mean() > x.notna().mean():
            return pd.to_numeric(x2, errors="coerce")
    return x


def zero_to_nan(series: pd.Series) -> pd.Series:
    """
    Convert 0 -> NaN for columns where 0 is likely a missing placeholder.
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.mask(x == 0, np.nan)


def pick_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def robust_z(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return (v - med) / (1.4826 * mad)


def percentile_within(group: pd.Series, value: float) -> float:
    g = pd.to_numeric(group, errors="coerce").dropna()
    if len(g) == 0 or not np.isfinite(value):
        return np.nan
    return float((g <= value).mean() * 100.0)


def safe_log1p(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.log1p(x)


def digits_only(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\D+", "", str(x))


def sic_prefix(x, n=2):
    s = digits_only(x)
    if not s:
        return "Unknown"
    if len(s) < n:
        s = s.zfill(n)
    return s[:n]


def build_display_name(df: pd.DataFrame) -> pd.Series:
    """
    Build a human-friendly label for UI dropdowns.
    IMPORTANT: prioritises company_sites (your dataset's "company name" column).
    """
    candidates = [
        "company_sites",          # <-- key fix: treat as primary company name
        "company_name",
        "name",
        "company",
        "website",
        "duns_number",
        "address_line_1",
    ]
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(object)
            out = s.apply(lambda v: str(v).strip() if not pd.isna(v) else "")
            # choose this field if it is reasonably populated
            if (out != "").mean() > 0.30:
                out = out.replace({"": "UNKNOWN"})
                return out
    return pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype=object)


# =========================
# Load and clean
# =========================
@st.cache_data
def load_and_clean(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [to_snake(c) for c in df.columns]

    # --- Dedup: exact duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    after_exact = len(df)

    # --- Dedup: by DUNS, keep most complete row ---
    if "duns_number" in df.columns:
        completeness = df.notna().sum(axis=1)
        df = df.assign(_completeness=completeness).sort_values("_completeness", ascending=False)
        df = df.drop_duplicates(subset=["duns_number"], keep="first").drop(columns=["_completeness"])

    df = df.reset_index(drop=True)

    # --- Normalise content text fields ---
    for c in [
        "company_sites",  # <-- name column (normalise missing tokens only)
        "website",
        "address_line_1",
        "country", "city", "state", "state_or_province_abbreviation",
        "entity_type", "ownership_type", "legal_status",
        "company_status_active_inactive",
        "sic_description", "8_digit_sic_description",
        "parent_company", "global_ultimate_company", "domestic_ultimate_company",
    ]:
        if c in df.columns:
            df[c] = normalise_missing_text(df[c])

    # Country: case normalisation
    if "country" in df.columns:
        df["country"] = df["country"].astype(object).apply(lambda x: str(x).upper().strip() if not pd.isna(x) else pd.NA)

    # Entity type: consistent casing for UI
    if "entity_type" in df.columns:
        df["entity_type"] = df["entity_type"].astype(object).apply(lambda x: str(x).strip().title() if not pd.isna(x) else pd.NA)

    # Phone number
    if "phone_number" in df.columns:
        df["phone_number"] = df["phone_number"].apply(clean_phone)

    # --- Numeric conversions and placeholder zeros ---
    placeholder_zero_cols = [
        "employees_total", "employees_single_site",
        "revenue_usd", "market_value_usd",
        "it_budget", "it_spend",
        "corporate_family_members"
    ]
    for c in placeholder_zero_cols:
        if c in df.columns:
            df[c] = zero_to_nan(df[c])

    # Year found sanity
    if "year_found" in df.columns:
        y = pd.to_numeric(df["year_found"], errors="coerce")
        df["year_found"] = y.mask((y <= 1700) | (y > 2026), np.nan)

    # --- Device/server columns: parse buckets -> numeric midpoints, keep raw bucket too ---
    device_cols = [
        "no_of_pc", "no_of_desktops", "no_of_laptops",
        "no_of_routers", "no_of_servers", "no_of_storage_devices"
    ]
    for c in device_cols:
        if c in df.columns:
            df[c + "_bucket"] = normalise_missing_text(df[c])  # raw bucket string for UI
            df[c] = safe_numeric(df[c])                        # numeric midpoints for modelling

    dev_present = [c for c in device_cols if c in df.columns]
    df["device_total"] = df[dev_present].sum(axis=1, min_count=1) if dev_present else np.nan

    # --- Derived metrics for insights ---
    if "revenue_usd" in df.columns and "employees_total" in df.columns:
        df["revenue_per_employee"] = df["revenue_usd"] / df["employees_total"]

    if "it_spend" in df.columns and "revenue_usd" in df.columns:
        df["it_spend_to_revenue"] = df["it_spend"] / df["revenue_usd"]

    if "it_spend" in df.columns and "employees_total" in df.columns:
        df["it_spend_per_employee"] = df["it_spend"] / df["employees_total"]

    if "no_of_servers" in df.columns and "device_total" in df.columns:
        df["server_to_device_ratio"] = df["no_of_servers"] / df["device_total"]

    if "no_of_laptops" in df.columns and "device_total" in df.columns:
        df["laptop_to_device_ratio"] = df["no_of_laptops"] / df["device_total"]

    if "no_of_desktops" in df.columns and "device_total" in df.columns:
        df["desktop_to_device_ratio"] = df["no_of_desktops"] / df["device_total"]

    # --- Keep only columns relevant to objectives (5 attribute groups + UI) ---
    industry_cols = ["sic_code", "sic_description", "8_digit_sic_code", "8_digit_sic_description", "naics_code", "naics_description"]
    size_cols = ["employees_single_site", "employees_total", "revenue_usd", "market_value_usd", "year_found", "revenue_per_employee"]
    structure_cols = [
        "entity_type", "ownership_type", "corporate_family_members",
        "is_headquarters", "is_domestic_ultimate",
        "parent_company", "global_ultimate_company", "domestic_ultimate_company"
    ]
    it_cols = [
        "it_budget", "it_spend", "it_spend_to_revenue", "it_spend_per_employee",
        "device_total", "server_to_device_ratio", "laptop_to_device_ratio", "desktop_to_device_ratio"
    ] + device_cols + [c + "_bucket" for c in device_cols if c in df.columns]
    geo_cols = ["country", "state", "state_or_province_abbreviation", "city", "postal_code", "lattitude", "longitude"]
    ui_cols = [
        "duns_number",
        "company_sites",  # <-- keep it explicitly
        "website",
        "address_line_1",
        "phone_number",
        "registration_number",
        "company_description",
        "company_status_active_inactive",
        "legal_status",
    ]

    keep = []
    for grp in [industry_cols, size_cols, structure_cols, it_cols, geo_cols, ui_cols]:
        keep += [c for c in grp if c in df.columns]
    keep = list(dict.fromkeys(keep))
    df = df[keep].copy()

    # --- Add display_name for dropdowns (company_sites first) ---
    df["display_name"] = build_display_name(df)

    # Store dedup stats
    df.attrs["dedup_before"] = before
    df.attrs["dedup_after_exact"] = after_exact
    df.attrs["dedup_after_key"] = len(df)

    return df


raw_df = load_and_clean(uploaded)


# =========================
# Segmentation
# =========================
@st.cache_data
def add_rule_segments(df_in: pd.DataFrame, min_industry_count: int, simple_segments: bool) -> pd.DataFrame:
    """
    Interpretable segments based on:
    - Industry bucket (SIC 2-digit, rare -> Other)
    - Size tiers (employees, revenue)
    - Structure tier (HQ / domestic ultimate / subsidiary / branch / etc.)
    - IT footprint tiers (it_spend, device_total)
    - Geography (country) if full mode
    """
    df = df_in.copy()

    # Industry bucket
    sic_col = "8_digit_sic_code" if "8_digit_sic_code" in df.columns else ("sic_code" if "sic_code" in df.columns else None)
    if sic_col:
        df["sic_2digit"] = df[sic_col].map(lambda x: sic_prefix(x, 2))
    else:
        df["sic_2digit"] = "Unknown"

    vc = df["sic_2digit"].value_counts(dropna=False)
    rare = vc[vc < int(min_industry_count)].index
    df["sic_bucket"] = df["sic_2digit"].where(~df["sic_2digit"].isin(rare), "Other")
    df["sic_bucket"] = df["sic_bucket"].fillna("Unknown")

    # Safer qcut using ranks
    def qcut_rank(series: pd.Series, labels):
        x = pd.to_numeric(series, errors="coerce")
        if x.notna().sum() < 40:
            return pd.Series(["Unknown"] * len(series), index=series.index, dtype=object)
        r = x.rank(method="average")
        try:
            return pd.qcut(r, q=len(labels), labels=labels, duplicates="drop").astype(object)
        except Exception:
            return pd.Series(["Unknown"] * len(series), index=series.index, dtype=object)

    # Size tiers
    df["size_emp_tier"] = qcut_rank(safe_log1p(df["employees_total"]), ["emp_s", "emp_m", "emp_l", "emp_xl"]) if "employees_total" in df.columns else "Unknown"
    df["size_rev_tier"] = qcut_rank(safe_log1p(df["revenue_usd"]), ["rev_s", "rev_m", "rev_l", "rev_xl"]) if "revenue_usd" in df.columns else "Unknown"

    # Structure tier
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
        ["hq", "domestic_ultimate", "subsidiary", "branch", "subsidiary_like", "member_of_group"],
        default="standalone_like"
    ).astype(object)

    # IT tiers
    df["it_spend_tier"] = qcut_rank(safe_log1p(df["it_spend"]), ["it_low", "it_mid", "it_high", "it_top"]) if "it_spend" in df.columns else "Unknown"
    df["device_tier"] = qcut_rank(safe_log1p(df["device_total"]), ["dev_low", "dev_mid", "dev_high", "dev_top"]) if "device_total" in df.columns else "Unknown"

    # Geo tier
    df["geo_tier"] = df["country"].fillna("Unknown") if "country" in df.columns else "Unknown"

    # Segment label
    if simple_segments:
        seg_parts = ["sic_bucket", "size_emp_tier", "size_rev_tier", "it_spend_tier", "device_tier"]
    else:
        seg_parts = ["sic_bucket", "size_emp_tier", "size_rev_tier", "structure_tier", "it_spend_tier", "device_tier", "geo_tier"]

    for c in seg_parts:
        df[c] = df[c].astype(object)
        df[c] = df[c].where(~pd.isna(df[c]), "Unknown")

    df["segment_label"] = df[seg_parts].astype(str).agg("|".join, axis=1)
    df["segment_id"] = df["segment_label"].astype("category").cat.codes

    return df


# Sidebar segmentation controls
st.sidebar.subheader("Segmentation settings")
simple_segments = st.sidebar.toggle("Simple segments (recommended)", value=True)
min_industry_count = st.sidebar.slider("Min companies per industry bucket", 10, 300, 80, 10)

df = add_rule_segments(raw_df, min_industry_count=min_industry_count, simple_segments=simple_segments)


# =========================
# Filters
# =========================
st.sidebar.subheader("Filters")
col_country = pick_col(df, ["country"])
col_entity = pick_col(df, ["entity_type"])
col_state = pick_col(df, ["state"])
col_city = pick_col(df, ["city"])

def multiselect_filter(label, col):
    if col is None:
        st.sidebar.caption(f"{label}: column not found")
        return []
    vals = sorted([v for v in df[col].dropna().astype(str).unique()])
    return st.sidebar.multiselect(label, vals)

sel_country = multiselect_filter("Country", col_country)
sel_entity = multiselect_filter("Entity type", col_entity)
sel_state = multiselect_filter("State", col_state)
sel_city = multiselect_filter("City", col_city)

seg_vals = sorted(df["segment_label"].dropna().astype(str).unique())
sel_segs = st.sidebar.multiselect("Segment", seg_vals)

filtered = df.copy()
if col_country and sel_country:
    filtered = filtered[filtered[col_country].astype(str).isin(sel_country)]
if col_entity and sel_entity:
    filtered = filtered[filtered[col_entity].astype(str).isin(sel_entity)]
if col_state and sel_state:
    filtered = filtered[filtered[col_state].astype(str).isin(sel_state)]
if col_city and sel_city:
    filtered = filtered[filtered[col_city].astype(str).isin(sel_city)]
if sel_segs:
    filtered = filtered[filtered["segment_label"].astype(str).isin(sel_segs)]

st.sidebar.caption(f"Filtered rows: {len(filtered):,}")


# =========================
# Segment profiling (no merge collisions)
# =========================
@st.cache_data
def build_segment_profiles(d: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in [
        "employees_total", "revenue_usd", "market_value_usd",
        "it_budget", "it_spend", "device_total",
        "it_spend_to_revenue", "it_spend_per_employee",
        "corporate_family_members",
        "server_to_device_ratio", "laptop_to_device_ratio", "desktop_to_device_ratio"
    ] if c in d.columns]

    agg = {m: "median" for m in metrics}
    prof = d.groupby(["segment_id", "segment_label"], dropna=False).agg(agg)
    prof["count"] = d.groupby(["segment_id", "segment_label"], dropna=False).size()
    prof = prof.reset_index().sort_values("count", ascending=False).reset_index(drop=True)

    def top_cat_and_share(df_seg: pd.DataFrame, col: str):
        s = df_seg[col].fillna("Unknown").astype(str)
        vc = s.value_counts()
        if len(vc) == 0:
            return ("Unknown", np.nan)
        top = vc.index[0]
        share = float(vc.iloc[0] / vc.sum() * 100.0)
        return (top, share)

    comp_cols = []
    for col in [
        "country", "entity_type", "sic_description", "8_digit_sic_description",
        "sic_bucket", "structure_tier", "state", "city"
    ]:
        if col in d.columns:
            comp_cols.append(col)

    comp_rows = []
    for seg_id, df_seg in d.groupby("segment_id", dropna=False):
        row = {"segment_id": int(seg_id)}
        for col in comp_cols:
            top, share = top_cat_and_share(df_seg, col)
            row[f"top_{col}"] = top
            row[f"top_{col}_share"] = round(share, 1) if np.isfinite(share) else np.nan
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows) if comp_rows else pd.DataFrame({"segment_id": prof["segment_id"]})
    out = prof.merge(comp_df, on="segment_id", how="left")
    return out


# =========================
# Anomaly detection
# =========================
@st.cache_data
def compute_anomalies(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()

    # 1) IT spend relative to size (log regression residual)
    y_col = "it_spend" if "it_spend" in out.columns else None
    x_cols = [c for c in ["employees_total", "revenue_usd"] if c in out.columns]
    if y_col and x_cols:
        y = safe_log1p(out[y_col])
        X = [np.ones(len(out))]
        for c in x_cols:
            X.append(safe_log1p(out[c]))
        X = np.vstack(X).T

        mask = np.isfinite(y.values)
        for j in range(X.shape[1]):
            mask = mask & np.isfinite(X[:, j])

        if mask.sum() >= 50:
            Xf = X[mask]
            yf = y.values[mask]
            beta, _, _, _ = np.linalg.lstsq(Xf, yf, rcond=None)
            yhat = X @ beta
            resid = y.values - yhat
            out["it_residual"] = resid
            out["it_residual_z"] = robust_z(pd.Series(resid, index=out.index))
        else:
            out["it_residual"] = np.nan
            out["it_residual_z"] = np.nan
    else:
        out["it_residual"] = np.nan
        out["it_residual_z"] = np.nan

    # 2) Large subsidiaries
    out["is_subsidiary_type"] = out["structure_tier"].astype(str).isin(["subsidiary", "subsidiary_like"]) if "structure_tier" in out.columns else False
    out["subsidiary_emp_pct"] = np.nan
    if "employees_total" in out.columns:
        subs = out[out["is_subsidiary_type"] == True]
        if len(subs) >= 20:
            subs_emp = pd.to_numeric(subs["employees_total"], errors="coerce")
            subs_emp_rank = subs_emp.rank(pct=True) * 100.0
            out.loc[subs.index, "subsidiary_emp_pct"] = subs_emp_rank

    # 3) Atypical device/server distributions (robust z within segment)
    ratio_cols = [c for c in ["server_to_device_ratio", "laptop_to_device_ratio", "desktop_to_device_ratio"] if c in out.columns]
    for c in ratio_cols:
        out[c + "_seg_z"] = out.groupby("segment_id")[c].transform(lambda s: robust_z(s))

    # 4) Large corporate families
    if "corporate_family_members" in out.columns:
        fam = pd.to_numeric(out["corporate_family_members"], errors="coerce")
        out["family_pct"] = fam.rank(pct=True) * 100.0
    else:
        out["family_pct"] = np.nan

    # Flags + severity
    severity = np.zeros(len(out), dtype=float)

    out["flag_it_high_relative"] = (pd.to_numeric(out["it_residual_z"], errors="coerce") >= 2.5).fillna(False)
    out["flag_it_low_relative"] = (pd.to_numeric(out["it_residual_z"], errors="coerce") <= -2.5).fillna(False)
    severity += np.where(out["flag_it_high_relative"], 2.0, 0.0)
    severity += np.where(out["flag_it_low_relative"], 2.0, 0.0)

    out["flag_large_subsidiary"] = ((pd.to_numeric(out["subsidiary_emp_pct"], errors="coerce") >= 95) & (out["is_subsidiary_type"] == True)).fillna(False)
    severity += np.where(out["flag_large_subsidiary"], 1.5, 0.0)

    for c in ratio_cols:
        flagname = "flag_" + c.replace("_ratio", "") + "_atypical"
        out[flagname] = (pd.to_numeric(out[c + "_seg_z"], errors="coerce").abs() >= 3.0).fillna(False)
        severity += np.where(out[flagname], 1.0, 0.0)

    out["flag_large_family"] = (pd.to_numeric(out["family_pct"], errors="coerce") >= 95).fillna(False)
    severity += np.where(out["flag_large_family"], 1.0, 0.0)

    out["anomaly_severity"] = severity

    def explain_row(r):
        bullets = []
        if r.get("flag_it_high_relative", False):
            z = r.get("it_residual_z", np.nan)
            bullets.append(f"IT spend high relative to size (robust z={z:.2f})")
        if r.get("flag_it_low_relative", False):
            z = r.get("it_residual_z", np.nan)
            bullets.append(f"IT spend low relative to size (robust z={z:.2f})")
        if r.get("flag_large_subsidiary", False):
            p = r.get("subsidiary_emp_pct", np.nan)
            bullets.append(f"Subsidiary type but very large among subsidiaries (pct={p:.0f})")
        if r.get("flag_server_to_device_atypical", False):
            z = r.get("server_to_device_ratio_seg_z", np.nan)
            bullets.append(f"Server share atypical within segment (robust z={z:.2f})")
        if r.get("flag_laptop_to_device_atypical", False):
            z = r.get("laptop_to_device_ratio_seg_z", np.nan)
            bullets.append(f"Laptop share atypical within segment (robust z={z:.2f})")
        if r.get("flag_desktop_to_device_atypical", False):
            z = r.get("desktop_to_device_ratio_seg_z", np.nan)
            bullets.append(f"Desktop share atypical within segment (robust z={z:.2f})")
        if r.get("flag_large_family", False):
            p = r.get("family_pct", np.nan)
            bullets.append(f"Corporate family unusually large (pct={p:.0f})")
        return "; ".join(bullets) if bullets else ""

    out["anomaly_explanation"] = out.apply(explain_row, axis=1)
    return out


# =========================
# Compute profiles/anomalies safely (prevents NameError)
# =========================
try:
    profiles = build_segment_profiles(filtered)
except Exception as e:
    profiles = pd.DataFrame()
    st.error(f"Segment profiling failed: {e}")

try:
    an_df = compute_anomalies(filtered)
except Exception as e:
    an_df = filtered.copy()
    an_df["anomaly_severity"] = 0.0
    an_df["anomaly_explanation"] = ""
    st.error(f"Anomaly detection failed: {e}")


# =========================
# Benchmark helpers
# =========================
def compute_company_percentiles(d: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    row = d.loc[row_idx]
    seg_id = int(row["segment_id"])
    peers = d[d["segment_id"] == seg_id].copy()

    metrics = [c for c in [
        "employees_total", "revenue_usd", "it_spend", "it_budget",
        "device_total", "it_spend_to_revenue", "it_spend_per_employee",
        "corporate_family_members",
        "server_to_device_ratio", "laptop_to_device_ratio", "desktop_to_device_ratio"
    ] if c in d.columns]

    records = []
    for m in metrics:
        v = pd.to_numeric(pd.Series([row[m]]), errors="coerce").iloc[0]
        if not np.isfinite(v):
            continue
        pct = percentile_within(peers[m], v)
        med = float(np.nanmedian(pd.to_numeric(peers[m], errors="coerce"))) if peers[m].notna().any() else np.nan
        records.append({
            "metric": m,
            "company_value": float(v),
            "segment_median": med,
            "percentile_in_segment": pct
        })
    if not records:
        return pd.DataFrame(columns=["metric", "company_value", "segment_median", "percentile_in_segment"])
    return pd.DataFrame(records).sort_values("percentile_in_segment", ascending=False)


def nearest_peers(d: pd.DataFrame, row_idx: int, k: int = 10) -> pd.DataFrame:
    row = d.loc[row_idx]
    seg_id = int(row["segment_id"])
    peers = d[d["segment_id"] == seg_id].copy()

    if len(peers) <= 1:
        return peers.head(0)

    feat_cols = [c for c in ["employees_total", "revenue_usd", "it_spend", "device_total", "corporate_family_members"] if c in d.columns]
    if not feat_cols:
        return peers.head(0)

    X = peers[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = np.log1p(X)
    mu = np.nanmean(X.values, axis=0)
    sd = np.nanstd(X.values, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X.values - mu) / sd

    x0 = np.log1p(pd.to_numeric(row[feat_cols], errors="coerce")).values.astype(float)
    z0 = (x0 - mu) / sd

    dist = np.nanmean((Z - z0) ** 2, axis=1) ** 0.5
    peers = peers.assign(peer_distance=dist).sort_values("peer_distance", ascending=True)
    peers = peers[peers.index != row_idx]

    cols_show = ["display_name", "segment_label", "peer_distance"]
    for c in ["company_sites", "country", "entity_type", "employees_total", "revenue_usd", "it_spend", "device_total"]:
        if c in peers.columns and c not in cols_show:
            cols_show.append(c)
    return peers[cols_show].head(k)


# =========================
# Tabs
# =========================
tab_overview, tab_segments, tab_company, tab_risk, tab_usecases = st.tabs(
    ["Overview", "Segments", "Company benchmarking", "Risks and anomalies", "Buyer use cases"]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies (filtered)", f"{len(filtered):,}")
    c2.metric("Segments", f"{filtered['segment_id'].nunique():,}")
    flagged = int((an_df.get("anomaly_severity", 0) > 0).sum()) if "anomaly_severity" in an_df.columns else 0
    c3.metric("Flagged companies", f"{flagged:,}")
    c4.metric("Countries", f"{filtered['country'].nunique(dropna=True) if 'country' in filtered.columns else 0:,}")

    with st.expander("Data health", expanded=False):
        st.write(
            f"Dedup stats: {raw_df.attrs.get('dedup_before', 'NA')} -> "
            f"{raw_df.attrs.get('dedup_after_exact', 'NA')} -> "
            f"{raw_df.attrs.get('dedup_after_key', 'NA')}"
        )
        key_cols = [c for c in ["company_sites", "sic_code", "8_digit_sic_code", "employees_total", "revenue_usd", "it_spend", "device_total", "corporate_family_members", "country", "entity_type"] if c in filtered.columns]
        if key_cols:
            miss = (filtered[key_cols].isna().mean() * 100).round(1).sort_values(ascending=False)
            miss_df = miss.reset_index()
            miss_df.columns = ["column", "missing_percent"]
            st.dataframe(miss_df, use_container_width=True)

    st.markdown("Top segments")
    seg_counts = filtered.groupby(["segment_id", "segment_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    st.dataframe(seg_counts.head(30), use_container_width=True)

    if "country" in filtered.columns:
        st.markdown("Top countries")
        ctab = filtered["country"].fillna("Unknown").astype(str).value_counts().head(20).reset_index()
        ctab.columns = ["country", "count"]
        st.dataframe(ctab, use_container_width=True)

# ---------- Segments ----------
with tab_segments:
    st.subheader("Segment profiles")
    st.caption("Profiles summarise typical values (median) and dominant categories per segment.")

    if profiles is None or profiles.empty:
        st.warning("No segment profiles available (profiling failed or no data after filters).")
    else:
        st.dataframe(profiles.head(100), use_container_width=True, height=420)

        seg_ids = profiles["segment_id"].astype(int).tolist()
        selected_seg = st.selectbox("Select a segment to deep dive", options=seg_ids, format_func=lambda x: f"S{int(x)}")

        seg_row = profiles[profiles["segment_id"] == selected_seg].head(1)
        seg_label = seg_row["segment_label"].iloc[0] if len(seg_row) else "Unknown"
        seg_df = filtered[filtered["segment_id"] == selected_seg].copy()

        st.write(f"Selected segment: S{int(selected_seg)}")
        with st.expander("Segment label (definition)", expanded=False):
            st.code(str(seg_label))

        st.markdown("Typical values (median) and top composition fields")
        st.dataframe(seg_row, use_container_width=True)

        cA, cB, cC = st.columns(3)
        with cA:
            if "country" in seg_df.columns:
                st.write("Country composition (top 10)")
                t = seg_df["country"].fillna("Unknown").astype(str).value_counts().head(10).reset_index()
                t.columns = ["country", "count"]
                st.dataframe(t, use_container_width=True, height=260)

        with cB:
            if "entity_type" in seg_df.columns:
                st.write("Entity type composition (top 10)")
                t = seg_df["entity_type"].fillna("Unknown").astype(str).value_counts().head(10).reset_index()
                t.columns = ["entity_type", "count"]
                st.dataframe(t, use_container_width=True, height=260)

        with cC:
            ind_col = "8_digit_sic_description" if "8_digit_sic_description" in seg_df.columns else ("sic_description" if "sic_description" in seg_df.columns else None)
            if ind_col:
                st.write("Industry composition (top 10)")
                t = seg_df[ind_col].fillna("Unknown").astype(str).value_counts().head(10).reset_index()
                t.columns = ["industry", "count"]
                st.dataframe(t, use_container_width=True, height=260)

# ---------- Company benchmarking ----------
with tab_company:
    st.subheader("Company benchmarking within segment peers")

    companies = sorted(filtered["display_name"].fillna("UNKNOWN").astype(str).unique())
    if not companies:
        st.info("No companies found in current filtered view.")
    else:
        company = st.selectbox("Select a company", companies)
        row_df = filtered[filtered["display_name"].astype(str) == str(company)]
        if len(row_df) == 0:
            st.warning("Company not found after filters.")
        else:
            row_idx = row_df.index[0]
            row = filtered.loc[row_idx]
            seg_id = int(row["segment_id"])

            st.write(f"Segment: S{seg_id}")
            with st.expander("Segment label", expanded=False):
                st.code(str(row.get("segment_label", "Unknown")))

            show_cols = [
                "company_sites", "display_name", "country", "entity_type",
                "sic_description", "8_digit_sic_description",
                "employees_total", "revenue_usd",
                "it_spend", "it_budget",
                "device_total", "corporate_family_members",
                "structure_tier",
                "website", "phone_number", "address_line_1"
            ]
            show_cols = [c for c in show_cols if c in filtered.columns]
            st.markdown("Company record (selected columns)")
            st.dataframe(filtered.loc[[row_idx], show_cols], use_container_width=True)

            st.markdown("Benchmark vs segment peers (percentiles and medians)")
            st.dataframe(compute_company_percentiles(filtered, row_idx), use_container_width=True)

            st.markdown("Nearest peers in the same segment")
            st.dataframe(nearest_peers(filtered, row_idx, k=10), use_container_width=True)

            st.markdown("Auto insights (data grounded)")
            insights = []
            bench = compute_company_percentiles(filtered, row_idx)

            if not bench.empty:
                def get_pct(metric):
                    s = bench[bench["metric"] == metric]
                    return float(s["percentile_in_segment"].iloc[0]) if len(s) else np.nan

                p_it = get_pct("it_spend")
                if np.isfinite(p_it):
                    if p_it >= 85:
                        insights.append(f"IT spend is higher than {p_it:.0f}% of peers in this segment.")
                    elif p_it <= 15:
                        insights.append(f"IT spend is lower than {100 - p_it:.0f}% of peers in this segment.")

                p_emp = get_pct("employees_total")
                if np.isfinite(p_emp) and str(row.get("structure_tier", "")) in {"subsidiary", "subsidiary_like"} and p_emp >= 90:
                    insights.append(f"This company is subsidiary type but large for its peer group (employees around {p_emp:.0f}th percentile).")

                p_fam = get_pct("corporate_family_members")
                if np.isfinite(p_fam) and p_fam >= 95:
                    insights.append(f"Corporate family size is unusually large (around {p_fam:.0f}th percentile in segment).")

            if row_idx in an_df.index and "anomaly_explanation" in an_df.columns:
                expl = str(an_df.loc[row_idx].get("anomaly_explanation", "")).strip()
                if expl:
                    insights.append(f"Anomaly flags: {expl}")

            if insights:
                for s in insights:
                    st.write("• " + s)
            else:
                st.caption("No strong signals detected from the current metrics and thresholds.")

# ---------- Risks and anomalies ----------
with tab_risk:
    st.subheader("Risks and anomalies")
    st.caption("Flagged companies are surfaced using robust statistics with evidence-based explanations.")

    flagged_only = st.toggle("Show flagged companies only", value=True)
    view = an_df.copy()
    if "anomaly_severity" not in view.columns:
        view["anomaly_severity"] = 0.0
    if "anomaly_explanation" not in view.columns:
        view["anomaly_explanation"] = ""

    if flagged_only:
        view = view[view["anomaly_severity"] > 0]

    view = view.sort_values(["anomaly_severity"], ascending=False)

    cols = ["company_sites", "display_name", "country", "entity_type", "segment_label", "anomaly_severity", "anomaly_explanation"]
    cols = [c for c in cols if c in view.columns]
    for c in ["employees_total", "revenue_usd", "it_spend", "device_total", "corporate_family_members", "it_spend_to_revenue", "server_to_device_ratio"]:
        if c in view.columns:
            cols.append(c)

    st.dataframe(view[cols].head(300), use_container_width=True, height=520)
    st.download_button(
        "Download risk list as CSV",
        data=view[cols].to_csv(index=False).encode("utf-8"),
        file_name="risk_flags.csv",
        mime="text/csv"
    )

# ---------- Buyer use cases ----------
with tab_usecases:
    st.subheader("Buyer workflows (commercial value)")

    st.markdown("### 1) Market segmentation and lead targeting")
    lead_cols = [
        "company_sites", "display_name", "country", "state", "city", "segment_label",
        "sic_description", "8_digit_sic_description",
        "employees_total", "revenue_usd", "it_spend", "device_total",
        "website", "phone_number"
    ]
    lead_cols = [c for c in lead_cols if c in filtered.columns]
    st.dataframe(filtered[lead_cols].head(200), use_container_width=True)
    st.download_button(
        "Download leads as CSV",
        data=filtered[lead_cols].to_csv(index=False).encode("utf-8"),
        file_name="leads.csv",
        mime="text/csv"
    )

    st.markdown("### 2) Competitive benchmarking")
    companies2 = sorted(filtered["display_name"].fillna("UNKNOWN").astype(str).unique())
    if companies2:
        comp2 = st.selectbox("Select a company for benchmarking", companies2, key="usecase_company")
        row_df2 = filtered[filtered["display_name"].astype(str) == str(comp2)]
        if len(row_df2):
            idx2 = row_df2.index[0]
            st.dataframe(compute_company_percentiles(filtered, idx2), use_container_width=True)
            st.dataframe(nearest_peers(filtered, idx2, k=15), use_container_width=True)

    st.markdown("### 3) Risk assessment and compliance screening")
    risk_view = an_df.copy()
    if "anomaly_severity" in risk_view.columns:
        risk_view = risk_view[risk_view["anomaly_severity"] > 0].sort_values("anomaly_severity", ascending=False)
    risk_cols = [c for c in ["company_sites", "display_name", "country", "segment_label", "anomaly_severity", "anomaly_explanation"] if c in risk_view.columns]
    st.dataframe(risk_view[risk_cols].head(200), use_container_width=True)
    st.download_button(
        "Download screening list as CSV",
        data=risk_view[risk_cols].to_csv(index=False).encode("utf-8"),
        file_name="screening.csv",
        mime="text/csv"
    )

    st.markdown("### 4) Technology investment analysis")
    if profiles is None or profiles.empty:
        st.info("Segment profiles not available, cannot rank segments by IT intensity.")
    else:
        rank_metric_opts = [c for c in ["it_spend_to_revenue", "it_spend_per_employee", "it_spend", "device_total"] if c in profiles.columns]
        if rank_metric_opts:
            rank_metric = st.selectbox("Rank segments by", options=rank_metric_opts, index=0)
            seg_rank = profiles.sort_values(rank_metric, ascending=False)
            seg_it_cols = ["segment_id", "segment_label", "count"] + [c for c in rank_metric_opts if c in seg_rank.columns]
            st.dataframe(seg_rank[seg_it_cols].head(100), use_container_width=True)
        else:
            st.info("No IT intensity metrics found in profiles.")


# =========================
# Optional AI Assistant (Sidebar)
# =========================
def get_llm_client():
    if not HF_AVAILABLE:
        return None
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None
    if not token:
        return None
    try:
        return InferenceClient(model="Qwen/Qwen2.5-72B-Instruct", token=token)
    except Exception:
        return None


def get_dataframe_context(df_in: pd.DataFrame, max_rows=8) -> str:
    if df_in.empty:
        return "The dataset view is empty."
    row_count = len(df_in)
    col_names = ", ".join(df_in.columns.tolist())
    preview = df_in.head(max_rows).to_string(index=False)
    return (
        f"Dataset Summary:\n"
        f"Total Rows in current view: {row_count}\n"
        f"Columns: {col_names}\n\n"
        f"Data Preview (first {max_rows} rows):\n{preview}\n"
    )


with st.sidebar:
    st.markdown("---")
    st.subheader("AI Data Assistant")

    enable_assistant = st.toggle("Enable assistant", value=False, disabled=(not HF_AVAILABLE))
    llm_client = get_llm_client() if enable_assistant else None

    if enable_assistant and llm_client is None:
        st.caption("Assistant unavailable. Add HF_TOKEN in .streamlit/secrets.toml or disable.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if enable_assistant and llm_client is not None:
        if prompt := st.chat_input("Ask about the filtered data..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            context = get_dataframe_context(filtered, max_rows=8)
            seg_summary = filtered.groupby("segment_label").size().sort_values(ascending=False).head(10).to_string()

            full_prompt = (
                "You are a careful data analyst assistant.\n"
                "Answer ONLY from the provided context.\n"
                "If context is insufficient, say what is missing.\n\n"
                f"CONTEXT:\n{context}\n"
                f"Top 10 segments by count:\n{seg_summary}\n\n"
                f"USER QUESTION:\n{prompt}\n"
            )

            with st.chat_message("assistant"):
                try:
                    resp = llm_client.chat_completion(
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=220,
                        stream=False
                    )
                    txt = ""
                    if hasattr(resp, "choices") and resp.choices:
                        txt = resp.choices[0].message.content
                    txt = (txt or "").strip()
                    st.markdown(txt if txt else "No response returned.")
                    st.session_state.messages.append({"role": "assistant", "content": txt if txt else "No response returned."})
                except Exception as e:
                    st.error(f"Error communicating with API: {e}")
