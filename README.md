# datathon-2s2d
# NUS Datathon Category A: Company Intelligence Explorer
Team: <TEAM_NAME>  
Folder name (submission): `NUS_DATATHON_CAT_A_<TEAM_NAME>`

## Overview
This project transforms raw company records into **data grounded company intelligence** by:
- Grouping companies into **interpretable segments** using industry, size, corporate structure, IT footprint, and geography
- Producing **segment profiles** that summarise typical values and composition
- Allowing **benchmarking** of an individual company against its segment peers
- Detecting **patterns, strengths, risks, and anomalies** with evidence based explanations
- Demonstrating **commercial value** through buyer workflows and exports
- Bonus: optional **LLM assistant** that answers questions based only on the currently filtered dataset view

## What is included
Required submission files:
- `CAT_A_<TEAM_NAME>.ipynb`  
  End to end notebook (loading, cleaning, segmentation, profiling, insights, outputs)
- `requirements.txt`  
  All Python packages with versions
- Model file (only if used)  
  Example: `model.pkl` or `model.joblib`
- `README.md`  
  This file
- `Report.pdf` or `Report.docx`  
  Sections: Introduction, Dataset Overview, Methodology, Results, Insights, Conclusion
- NDA forms  
  One per team member in the same folder

Demo app:
- `app.py`  
  Streamlit dashboard for segmentation, benchmarking, risk screening, and exports

## Environment setup
Recommended:
- Python 3.10 or 3.11

### Create and activate virtual environment
macOS or Linux
```bash
python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Run the notebook

Launch Jupyter

jupyter notebook


Open CAT_A_<TEAM_NAME>.ipynb

Run all cells from top to bottom

Notebook outputs may include:

cleaned dataset exports

segment tables and segment profiles

benchmarking tables

anomaly and risk tables

figures used in the report

Run the Streamlit dashboard

Start the app:

streamlit run app.py


Upload the dataset:

In the left sidebar, use Upload Excel (.xlsx) to upload the dataset

The app reads from the uploaded file to avoid path issues

How to navigate the Streamlit app

The dashboard is organised into tabs. Recommended flow is left to right.

1) Overview

Purpose: fast summary of the dataset and segmentation quality.

Key counts: companies, segments, flagged anomalies, countries

Data health: missingness checks and dedup stats

Top segments table

Optional country breakdown

2) Segments

Purpose: understand groups and what makes them different.

Segment profile table includes:

segment size (company count)

median typical values (employees, revenue, IT spend, devices, etc.)

dominant categories (top country, top entity type, top industry)

Segment deep dive shows:

raw segment definition (segment label)

composition tables for country, entity type, and industry

3) Company benchmarking

Purpose: compare one company to its segment peers.
Steps:

Select a company from the dropdown (uses Company Sites as the primary company name)

View the company record (selected key columns)

View benchmarking:

company value vs segment median

percentile rank within segment peers

View nearest peers within the same segment

Read auto insights generated from evidence (percentiles and anomaly flags)

4) Risks and anomalies

Purpose: surface unusual companies and explain why.

Toggle “Show flagged companies only” to focus on anomalies

Each flagged row includes:

anomaly severity

evidence based explanation

Download flagged list as CSV for screening workflows

5) Buyer use cases

Purpose: demonstrate commercial value for data buyers.
Includes:

Market segmentation and lead targeting + CSV export

Competitive benchmarking (percentiles and nearest peers)

Risk screening + CSV export

Technology investment analysis (rank segments by IT intensity metrics)

Segmentation settings and filters (sidebar)

Segmentation controls:

Simple segments (recommended): fewer splits, larger segments, easier to interpret

Min companies per industry bucket: rare industry buckets are grouped into “Other” to reduce fragmentation

Filters:

Country, Entity Type, State, City, Segment

All tables, insights, risk flags, and exports update based on the filtered view

Optional AI Data Assistant (HF_TOKEN)

The AI assistant is optional. If enabled, it answers questions based only on the current filtered view.

Create Streamlit secrets file

In the project root (same level as app.py), create a folder:

mkdir -p .streamlit


Create the secrets file:

touch .streamlit/secrets.toml


Add this line inside .streamlit/secrets.toml:

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"


Restart the app:

streamlit run app.py


In the sidebar:

Toggle “Enable assistant” ON

Use the chat input to ask questions about the filtered data

Security notes:

Do not commit .streamlit/secrets.toml to a public repository

If you do not have a token, keep the assistant disabled. The rest of the dashboard works normally.

Reproducibility notes

Install exactly the versions in requirements.txt

Run the notebook end to end without skipping cells

Use the same dataset file for notebook and Streamlit to ensure consistent segmentation and metrics

Troubleshooting

Company dropdown shows IDs instead of names:

The app expects Company Sites to exist (it becomes company_sites after cleaning)

Re upload the correct dataset file and confirm it contains a Company Sites column

Tables appear empty:

Clear filters in the sidebar

Check missingness in Overview -> Data health

Contact

Primary: <NAME> <EMAIL>
Backup: <NAME> <EMAIL>


python -m pip install --upgrade pip
pip install -r requirements.txt
