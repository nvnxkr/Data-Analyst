"""
AI Data Analyst — Streamlit app
Upload a CSV/XLSX, get an automatic summary, ask questions in plain English,
see auto-insights, simple forecasts, and export a report.
"""

import os
import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_utils import load_dataframe, build_summary, schema_summary_text, DataLoadError
from safe_exec import run_generated_code, UnsafeCodeError
from llm_client import get_query_intent, get_auto_insights, LLMError
from charts import make_chart, SUPPORTED_TYPES
from forecasting import forecast_next_periods, ForecastError
from report_export import build_html_report, build_pdf_report

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    div[data-testid="stMetric"] {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] * {
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #4b5563 !important;
    }
    .card {
        background: white; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "df": None,
    "dataset_name": None,
    "summary": None,
    "schema_text": None,
    "insights": None,
    "qa_history": [],  # list of dicts: question, explanation, result(df), fig, chart_type
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_dataset():
    for key in defaults:
        st.session_state[key] = defaults[key]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 AI Data Analyst")
st.caption("Upload a spreadsheet, get instant insights, and ask questions in plain English.")

if not os.environ.get("MISTRAL_API_KEY"):
    st.warning(
        "⚠️ `MISTRAL_API_KEY` is not set. Copy `.env.example` to `.env` and add your key "
        "to enable Q&A and auto-insights. File upload, summary stats, and manual charts "
        "still work without it.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# 1. File upload
# ---------------------------------------------------------------------------
with st.container():
    st.subheader("1. Upload your data")
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "CSV or Excel file (max 10MB)", type=["csv", "xlsx", "xls"]
        )
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("Use sample dataset", use_container_width=True)

if use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_sales_data.csv")
    with open(sample_path, "rb") as f:
        class _FakeUpload:
            def __init__(self, name, data):
                self.name = name
                self._data = data
                self.size = len(data)

            def read(self):
                return self._data

        data = f.read()
    uploaded_file = _FakeUpload("sample_sales_data.csv", data)

if uploaded_file is not None:
    try:
        if st.session_state.dataset_name != getattr(uploaded_file, "name", None):
            df = load_dataframe(uploaded_file)
            st.session_state.df = df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.summary = build_summary(df)
            st.session_state.schema_text = schema_summary_text(df)
            st.session_state.insights = None
            st.session_state.qa_history = []
    except DataLoadError as e:
        st.error(f"Could not load file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading file: {e}")
        st.stop()

df = st.session_state.df

if df is None:
    st.info("👆 Upload a CSV/XLSX file or click **Use sample dataset** to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# 2. Preview + summary
# ---------------------------------------------------------------------------
st.subheader("2. Data preview & summary")

with st.container():
    st.markdown(f"**File:** `{st.session_state.dataset_name}`")
    st.dataframe(df.head(20), use_container_width=True)

summary = st.session_state.summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", summary["n_rows"])
m2.metric("Columns", summary["n_cols"])
m3.metric("Numeric columns", len(summary["column_types"]["numeric"]))
m4.metric("Date columns", len(summary["column_types"]["datetime"]))

with st.expander("Column details & basic statistics"):
    dtype_df = pd.DataFrame(
        {
            "Column": summary["columns"],
            "Type": [summary["dtypes"][c] for c in summary["columns"]],
            "Nulls": [summary["null_counts"][c] for c in summary["columns"]],
        }
    )
    st.dataframe(dtype_df, use_container_width=True)

    if summary["numeric_stats"]:
        stats_df = pd.DataFrame(summary["numeric_stats"]).T
        st.markdown("**Numeric column statistics**")
        st.dataframe(stats_df, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Auto-generated insights
# ---------------------------------------------------------------------------
st.subheader("3. Auto-generated insights")

insight_col1, insight_col2 = st.columns([1, 5])
with insight_col1:
    generate_clicked = st.button("✨ Generate insights")

if generate_clicked:
    if not os.environ.get("MISTRAL_API_KEY"):
        st.error("Set MISTRAL_API_KEY in your .env file to use this feature.")
    else:
        with st.spinner("Analyzing your data..."):
            try:
                st.session_state.insights = get_auto_insights(
                    st.session_state.schema_text, summary
                )
            except LLMError as e:
                st.error(f"Couldn't generate insights: {e}")
            except Exception:
                st.error("An unexpected error occurred while generating insights.")

if st.session_state.insights:
    with st.container():
        for insight in st.session_state.insights:
            st.markdown(f"- {insight}")
else:
    st.caption("Click **Generate insights** to have the AI summarize trends, outliers, and top performers.")

# ---------------------------------------------------------------------------
# 4. Natural language Q&A
# ---------------------------------------------------------------------------
st.subheader("4. Ask a question about your data")

with st.form("qa_form", clear_on_submit=False):
    question = st.text_input(
        "e.g. 'Show top 5 customers by revenue' or 'What are the monthly sales trends?'",
        key="question_input",
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not os.environ.get("MISTRAL_API_KEY"):
        st.error("Set MISTRAL_API_KEY in your .env file to use Q&A.")
    else:
        with st.spinner("Thinking..."):
            entry = {"question": question, "error": None}
            try:
                intent = get_query_intent(question, st.session_state.schema_text)
                result = run_generated_code(intent["pandas_code"], df)
                entry["explanation"] = intent.get("explanation", "")
                entry["chart_type"] = intent.get("chart_type", "none")
                entry["chart_x"] = intent.get("chart_x")
                entry["chart_y"] = intent.get("chart_y")
                entry["result"] = result
                entry["code"] = intent["pandas_code"]
            except LLMError as e:
                entry["error"] = f"AI service issue: {e}"
            except UnsafeCodeError as e:
                entry["error"] = f"The AI's generated code was rejected for safety reasons: {e}"
            except Exception as e:
                entry["error"] = f"Couldn't answer that question: {e}"
            st.session_state.qa_history.insert(0, entry)

for i, entry in enumerate(st.session_state.qa_history):
    with st.container():
        st.markdown(f"**Q: {entry['question']}**")
        if entry.get("error"):
            st.error(entry["error"])
            continue

        if entry.get("explanation"):
            st.write(entry["explanation"])

        result = entry.get("result")
        if isinstance(result, (pd.DataFrame, pd.Series)):
            display_df = result.to_frame() if isinstance(result, pd.Series) else result
            st.dataframe(display_df, use_container_width=True)

            chart_type_options = ["none"] + SUPPORTED_TYPES
            default_type = entry.get("chart_type", "none")
            if default_type not in chart_type_options:
                default_type = "none"
            chosen_type = st.selectbox(
                "Chart type",
                chart_type_options,
                index=chart_type_options.index(default_type),
                key=f"chart_type_{i}",
            )
            if chosen_type != "none":
                fig = make_chart(
                    result, chosen_type, entry.get("chart_x"), entry.get("chart_y"), title=entry["question"]
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}")
                else:
                    st.caption("Couldn't build a chart from this result.")
        else:
            st.write(result)

        with st.expander("Show generated pandas code"):
            st.code(entry.get("code", ""), language="python")
        st.divider()

# ---------------------------------------------------------------------------
# 5. Forecasting
# ---------------------------------------------------------------------------
st.subheader("5. Simple forecasting")

date_cols = summary["column_types"]["datetime"]
numeric_cols = summary["column_types"]["numeric"]

if not date_cols or not numeric_cols:
    st.caption("Forecasting requires at least one date column and one numeric column.")
else:
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fc_date_col = st.selectbox("Date column", date_cols, key="fc_date_col")
    with fc2:
        fc_value_col = st.selectbox("Value column", numeric_cols, key="fc_value_col")
    with fc3:
        fc_periods = st.number_input("Periods ahead", min_value=1, max_value=12, value=1)

    if st.button("📈 Forecast"):
        try:
            history, forecast = forecast_next_periods(
                df, fc_date_col, fc_value_col, periods=int(fc_periods)
            )
            combined = pd.concat(
                [
                    history.assign(kind="Actual"),
                    forecast.rename(columns={"value": "value"})[["period", "value"]].assign(kind="Forecast"),
                ],
                ignore_index=True,
            )
            import plotly.express as px

            fig = px.line(
                combined, x="period", y="value", color="kind", markers=True,
                title=f"{fc_value_col} — history & {fc_periods}-period forecast",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast, use_container_width=True)
        except ForecastError as e:
            st.error(str(e))
        except Exception:
            st.error("Couldn't compute a forecast for the selected columns.")

# ---------------------------------------------------------------------------
# 6. Export report
# ---------------------------------------------------------------------------
st.subheader("6. Export report")

chart_figs = []
for i, entry in enumerate(st.session_state.qa_history):
    result = entry.get("result")
    if isinstance(result, (pd.DataFrame, pd.Series)) and entry.get("chart_type") not in (None, "none"):
        fig = make_chart(result, entry["chart_type"], entry.get("chart_x"), entry.get("chart_y"), title=entry["question"])
        if fig is not None:
            chart_figs.append((entry["question"], fig))

exp1, exp2 = st.columns(2)
with exp1:
    if st.button("Generate HTML report"):
        html = build_html_report(
            st.session_state.dataset_name, summary, st.session_state.insights or [], chart_figs
        )
        st.download_button(
            "⬇️ Download HTML report",
            data=html,
            file_name="data_analysis_report.html",
            mime="text/html",
        )
with exp2:
    if st.button("Generate PDF report"):
        try:
            pdf_bytes = build_pdf_report(
                st.session_state.dataset_name, summary, st.session_state.insights or [], chart_figs
            )
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name="data_analysis_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Couldn't generate PDF report: {e}")

st.caption("Tip: generate insights and ask a couple of questions first so the report has content to include.")
