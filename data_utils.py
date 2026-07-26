"""
data_utils.py
Handles file loading, validation, and automatic data summarization.
"""

import io
import pandas as pd
import numpy as np

MAX_FILE_SIZE_MB = 10


class DataLoadError(Exception):
    pass


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load a CSV or XLSX Streamlit UploadedFile into a pandas DataFrame with validation."""
    if uploaded_file is None:
        raise DataLoadError("No file provided.")

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DataLoadError(
            f"File is {size_mb:.1f}MB, which exceeds the {MAX_FILE_SIZE_MB}MB limit."
        )

    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", engine="python")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            raise DataLoadError("Unsupported file type. Please upload a .csv or .xlsx file.")
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Could not parse file: {e}")

    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The uploaded file appears to be empty or has no columns.")

    # Clean column names: strip whitespace, keep original for display
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully-empty rows/columns which often come from bad exports
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Attempt to auto-detect date columns (only if not already datetime/numeric)
    for col in df.columns:
        dtype = df[col].dtype
        is_textlike = (
            pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        )
        if is_textlike:
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                except Exception:
                    pass

    return df.reset_index(drop=True)


def classify_columns(df: pd.DataFrame) -> dict:
    """Bucket columns into numeric, datetime, categorical, and boolean."""
    numeric_cols, date_cols, cat_cols, bool_cols = [], [], [], []

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            bool_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    return {
        "numeric": numeric_cols,
        "datetime": date_cols,
        "categorical": cat_cols,
        "boolean": bool_cols,
    }


def build_summary(df: pd.DataFrame) -> dict:
    """Build the automatic data summary: shape, dtypes, stats, nulls."""
    cols = classify_columns(df)

    stats = {}
    for col in cols["numeric"]:
        series = df[col]
        stats[col] = {
            "mean": round(float(series.mean()), 2) if series.notna().any() else None,
            "median": round(float(series.median()), 2) if series.notna().any() else None,
            "min": round(float(series.min()), 2) if series.notna().any() else None,
            "max": round(float(series.max()), 2) if series.notna().any() else None,
            "nulls": int(series.isna().sum()),
        }

    dtypes = {col: str(df[col].dtype) for col in df.columns}
    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "null_counts": null_counts,
        "numeric_stats": stats,
        "column_types": cols,
    }


def schema_summary_text(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """A compact textual schema + sample used as LLM context. Keeps prompts small."""
    cols = classify_columns(df)
    lines = [
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        f"Numeric columns: {cols['numeric']}",
        f"Datetime columns: {cols['datetime']}",
        f"Categorical columns: {cols['categorical']}",
        "",
        "Sample rows (as CSV):",
        df.head(max_sample_rows).to_csv(index=False),
    ]
    return "\n".join(lines)
