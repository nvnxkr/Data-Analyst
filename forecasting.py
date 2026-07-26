"""
forecasting.py
Lightweight forecasting: linear regression trend line + simple moving
average, applied to a date column and a numeric value column aggregated
by month. No heavy dependencies (just numpy/pandas + sklearn).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class ForecastError(Exception):
    pass


def forecast_next_periods(
    df: pd.DataFrame, date_col: str, value_col: str, periods: int = 1, freq: str = "ME"
):
    """
    Aggregate value_col by calendar period (month by default), fit a simple
    linear regression on the period index, and forecast `periods` ahead.
    Returns (history_df, forecast_df) both with columns [period, value].
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise ForecastError("Selected date or value column not found in data.")

    work = df[[date_col, value_col]].dropna()
    if work.empty:
        raise ForecastError("No data available after dropping missing values.")

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    grouped = (
        work.set_index(date_col)[value_col]
        .resample(freq)
        .sum()
        .reset_index()
        .rename(columns={date_col: "period", value_col: "value"})
    )

    if len(grouped) < 2:
        raise ForecastError("Need at least 2 time periods of data to forecast a trend.")

    grouped["t"] = np.arange(len(grouped))
    model = LinearRegression()
    model.fit(grouped[["t"]], grouped["value"])

    future_t = pd.DataFrame({"t": np.arange(len(grouped), len(grouped) + periods)})
    future_periods = pd.date_range(
        start=grouped["period"].iloc[-1], periods=periods + 1, freq=freq
    )[1:]

    predictions = model.predict(future_t)
    forecast_df = pd.DataFrame({"period": future_periods, "value": predictions})

    # Simple 3-period moving average as a secondary sanity-check estimate
    ma_window = min(3, len(grouped))
    moving_avg = grouped["value"].rolling(ma_window).mean().iloc[-1]
    forecast_df["moving_avg_estimate"] = moving_avg

    return grouped[["period", "value"]], forecast_df
