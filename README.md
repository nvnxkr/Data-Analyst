# 📊 AI Data Analyst

A single-file Streamlit app: upload a CSV/Excel file, get an instant data
summary, ask questions about your data in plain English, see auto-generated
insights, run a simple forecast, and export a shareable report — all backed
by the Mistral API.

## Features

- **File upload** — CSV/XLSX up to 10MB, validated and parsed with pandas
- **Automatic data summary** — row/column counts, dtypes, null counts, numeric
  stats, detected date/categorical columns
- **Natural language Q&A** — ask things like *"top 10 customers by revenue"*
  or *"monthly sales trend"*; the LLM returns structured JSON (pandas code +
  chart spec), which is validated by an AST-based sandbox and executed safely
  (no raw `eval`/`exec` on untrusted output)
- **Auto-generated insights** — 3–5 bullet points on trends, outliers, and
  top/bottom performers, generated on demand
- **Charts** — bar/line/pie/scatter via Plotly, auto-picked by the LLM, with
  a manual dropdown to switch chart type
- **Simple forecasting** — linear-regression trend forecast (+ moving-average
  sanity check) for "predict next month" style questions
- **Export report** — download a self-contained HTML report or a PDF report
  containing the summary, insights, and charts

## Project structure

```
ai-data-analyst/
├── app.py               # Streamlit UI and orchestration
├── data_utils.py         # File loading, validation, summary stats
├── safe_exec.py           # AST-based sandbox for LLM-generated pandas code
├── llm_client.py          # Mistral API client (Q&A intent + insights)
├── charts.py               # Plotly chart builders
├── forecasting.py          # Linear regression / moving average forecast
├── report_export.py        # HTML + PDF report builders
├── sample_data/
│   └── sample_sales_data.csv
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone/copy this project**, then create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your Mistral API key**:

   ```bash
   cp .env.example .env
   # then edit .env and set MISTRAL_API_KEY=your_key_here
   ```

   Get a key at https://console.mistral.ai/. (Q&A and auto-insights need
   this; file upload, summary stats, and manual charting work without it.)

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   Streamlit will open the app at `http://localhost:8501`.

## Using the sample dataset

Click **"Use sample dataset"** on the upload screen to load
`sample_data/sample_sales_data.csv` — a small dummy sales dataset (order
date, region, customer, product, category, units, unit price, revenue) with
a few intentionally missing values, useful for testing null-handling.

## Safety notes on the Q&A feature

The LLM never gets to run arbitrary code directly. Its JSON response
includes a `pandas_code` field, which:

1. Is parsed into a Python AST.
2. Is walked against a strict allow-list — imports, function/class
   definitions, `eval`/`exec`/`open`/`__import__`, dunder attribute access,
   and file-writing methods (`to_csv`, `to_sql`, etc.) are all rejected.
3. Only if it passes validation is it executed, in a restricted namespace
   exposing only `df`, `pd`, `np`, and a small set of safe builtins.

If validation fails or the code errors out, the app shows a friendly error
instead of crashing.

## 60-second demo script

1. **Open the app** — point out the clean upload card. Click **"Use sample
   dataset"** instead of uploading your own file (saves time).
2. **Point at the preview table and summary metrics** — "It automatically
   detected 3 numeric columns, 1 date column, and flagged 1 missing value in
   revenue."
3. **Click "Generate insights"** — while it loads: "This calls Mistral with
   the schema and summary stats, and asks for 3–5 concrete bullet-point
   insights." Read one or two aloud once they appear.
4. **Ask a question**: type *"Show total revenue by region"* and hit Ask.
   Point out: the plain-English explanation, the result table, the
   auto-picked bar chart, and — expand "Show generated pandas code" — "this
   is the actual code the AI wrote; it's validated by an AST sandbox before
   we ever run it."
5. **Switch the chart type** to *pie* via the dropdown to show manual
   override.
6. **Ask a forecasting question or use the Forecasting section**: pick
   `order_date` / `revenue`, click **Forecast** — "simple linear regression
   trend over the monthly aggregates, done without any heavy ML
   dependencies."
7. **Click "Generate HTML report"** and download it — "one click gets you a
   shareable report with the summary, insights, and charts baked in."
8. **Wrap up**: "Everything — the file parsing, the sandboxing, the charts,
   the forecast, the report — runs from this single Streamlit app."

## Known limitations

- The AST sandbox intentionally restricts the LLM to simple, safe pandas
  expressions — very complex multi-step analyses may need to be broken into
  a few separate questions.
- Forecasting uses a simple linear trend; it's meant for quick
  directional estimates, not production-grade forecasting.
- PDF export renders charts as static images (via kaleido); the HTML export
  keeps them interactive.
