"""
charts.py
Builds Plotly figures from a result DataFrame/Series given a chart type.
Used both for LLM-driven charts and manual chart-type switching.
"""

import pandas as pd
import plotly.express as px


SUPPORTED_TYPES = ["bar", "line", "pie", "scatter"]


def _coerce_to_frame(result):
    """Normalize a pandas Series or DataFrame (or LLM result) into a plot-ready DataFrame."""
    if isinstance(result, pd.Series):
        df = result.reset_index()
        df.columns = ["category", "value"]
        return df
    if isinstance(result, pd.DataFrame):
        return result
    raise ValueError("Result is not chartable (must be a DataFrame or Series).")


def guess_axes(df: pd.DataFrame):
    """Pick reasonable default x/y columns if the LLM didn't specify them."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    x = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
    y = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    return x, y


def make_chart(result, chart_type: str, x: str = None, y: str = None, title: str = ""):
    """Build a Plotly figure. Returns None if chart_type is 'none' or data isn't chartable."""
    if chart_type not in SUPPORTED_TYPES:
        return None

    try:
        df = _coerce_to_frame(result)
    except ValueError:
        return None

    if df.empty:
        return None

    if not x or x not in df.columns:
        x, guessed_y = guess_axes(df)
    else:
        _, guessed_y = guess_axes(df)

    if not y or y not in df.columns:
        y = guessed_y

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title, markers=True)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title)
        else:
            return None
    except Exception:
        return None

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=13),
    )
    return fig
