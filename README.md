# datathon-2s2d
# NUS Datathon Category A: Company Intelligence Explorer

**Team:** 2son2daughters
**Folder name (submission):** `NUS_DATATHON_CAT_A_2son2daughters`

## Overview
This project transforms raw company records into **data-grounded company intelligence** by:
- Grouping companies into **interpretable segments** using industry, size, corporate structure, IT footprint, and geography.
- Producing **segment profiles** that summarise typical values and composition.
- Allowing **benchmarking** of an individual company against its segment peers.
- Detecting **patterns, strengths, risks, and anomalies** with evidence-based explanations.
- Demonstrating **commercial value** through buyer workflows and exports.
- **Bonus:** Optional **LLM assistant** that answers questions based only on the currently filtered dataset view.

## What is included

### Required submission files:
- `CAT_A_2son2daughters.ipynb`  
  End-to-end notebook (loading, cleaning, segmentation, profiling, insights, outputs).
- `requirements.txt`  
  All Python packages with versions.
- **Model file** (only if used)  
  Example: `model.pkl` or `model.joblib`.
- `README.md`  
  This file.
- `Report.pdf` or `Report.docx`  
  Sections: Introduction, Dataset Overview, Methodology, Results, Insights, Conclusion.
- **NDA forms** One per team member in the same folder.

### Demo app:
- `app.py`  
  Streamlit dashboard for segmentation, benchmarking, risk screening, and exports.

---

## Environment setup
**Recommended:** Python 3.10 or 3.11

### 1. Create and activate virtual environment

**For macOS or Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

**For Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Run the notebook

1. Launch Jupyter:
```bash
jupyter notebook

```


2. Open `CAT_A_2son2daughters.ipynb`.
3. Run all cells from top to bottom.

**Notebook outputs may include:**

* Cleaned dataset exports
* Segment tables and segment profiles
* Benchmarking tables
* Anomaly and risk tables
* Figures used in the report

### 3. Run the Streamlit dashboard

Start the app:

```bash
streamlit run app.py

```

**Upload the dataset:**

* In the left sidebar, use **Upload Excel (.xlsx)** to upload the dataset.
* The app reads from the uploaded file to avoid path issues.

---

## How to navigate the Streamlit app

The dashboard is organised into tabs. The recommended flow is left to right.

### 1) Overview

*Purpose: Fast summary of the dataset and segmentation quality.*

* **Key counts:** Companies, segments, flagged anomalies, countries.
* **Data health:** Missingness checks and dedup stats.
* Top segments table.
* Optional country breakdown.

### 2) Segments

*Purpose: Understand groups and what makes them different.*

* **Segment profile table includes:**
* Segment size (company count).
* Median typical values (employees, revenue, IT spend, devices, etc.).
* Dominant categories (top country, top entity type, top industry).


* **Segment deep dive shows:**
* Raw segment definition (segment label).
* Composition tables for country, entity type, and industry.



### 3) Company benchmarking

*Purpose: Compare one company to its segment peers.*
**Steps:**

1. Select a company from the dropdown (uses `Company Sites` as the primary company name).
2. View the company record (selected key columns).
3. **View benchmarking:**
* Company value vs segment median.
* Percentile rank within segment peers.


4. View nearest peers within the same segment.
5. Read auto-insights generated from evidence (percentiles and anomaly flags).

### 4) Risks and anomalies

*Purpose: Surface unusual companies and explain why.*

* Toggle **“Show flagged companies only”** to focus on anomalies.
* **Each flagged row includes:**
* Anomaly severity.
* Evidence-based explanation.


* Download flagged list as CSV for screening workflows.

### 5) Buyer use cases

*Purpose: Demonstrate commercial value for data buyers.*

* Market segmentation and lead targeting + CSV export.
* Competitive benchmarking (percentiles and nearest peers).
* Risk screening + CSV export.
* Technology investment analysis (rank segments by IT intensity metrics).

---

## Segmentation settings and filters (Sidebar)

* **Segmentation controls:**
* **Simple segments (recommended):** Fewer splits, larger segments, easier to interpret.
* **Min companies per industry bucket:** Rare industry buckets are grouped into “Other” to reduce fragmentation.


* **Filters:**
* Country, Entity Type, State, City, Segment.
* *Note: All tables, insights, risk flags, and exports update based on the filtered view.*



---

## Optional AI Data Assistant (HF_TOKEN)

The AI assistant is optional. If enabled, it answers questions based only on the current filtered view.

**1. Create Streamlit secrets file**
In the project root (same level as `app.py`), create a folder and file:

```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml

```

**2. Add your token**
Add this line inside `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"

```

**3. Restart the app**

```bash
streamlit run app.py

```

**4. Usage**

* In the sidebar, toggle **“Enable assistant” ON**.
* Use the chat input to ask questions about the filtered data.

> **Security Note:** Do not commit `.streamlit/secrets.toml` to a public repository. If you do not have a token, keep the assistant disabled; the rest of the dashboard works normally.

---

## Reproducibility notes

* Install exactly the versions in `requirements.txt`.
* Run the notebook end-to-end without skipping cells.
* Use the same dataset file for the notebook and Streamlit to ensure consistent segmentation and metrics.

## Troubleshooting

**Company dropdown shows IDs instead of names:**

* The app expects `Company Sites` to exist (it becomes `company_sites` after cleaning).
* Re-upload the correct dataset file and confirm it contains a "Company Sites" column.

**Tables appear empty:**

* Clear filters in the sidebar.
* Check missingness in **Overview -> Data health**.

## Contact

* **Primary:** <NAME> <EMAIL>
* **Backup:** <NAME> <EMAIL>

```

```
