"""Data loading and preprocessing utilities for time series data."""

import io
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)


async def load_upload_file(file: UploadFile) -> pd.DataFrame:
    """Load an uploaded file into a pandas DataFrame."""
    content = await file.read()
    filename = file.filename or "data.csv"

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(content))
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    elif filename.endswith(".json"):
        df = pd.read_json(io.BytesIO(content))
    else:
        # Try CSV as default
        df = pd.read_csv(io.BytesIO(content))

    return df


def detect_frequency(df: pd.DataFrame, ds_column: str = "ds") -> str:
    """Auto-detect time series frequency from the data."""
    try:
        dates = pd.to_datetime(df[ds_column].sort_values().unique())
        if len(dates) < 3:
            return "D"  # Default to daily

        freq = pd.infer_freq(dates)
        if freq is None:
            # Estimate from median difference
            diffs = pd.Series(dates).diff().dropna()
            median_diff = diffs.median()

            if median_diff <= pd.Timedelta(seconds=1):
                return "S"
            elif median_diff <= pd.Timedelta(minutes=1):
                return "T"
            elif median_diff <= pd.Timedelta(hours=1):
                return "h"
            elif median_diff <= pd.Timedelta(days=1):
                return "D"
            elif median_diff <= pd.Timedelta(days=7):
                return "W"
            elif median_diff <= pd.Timedelta(days=31):
                return "M"
            elif median_diff <= pd.Timedelta(days=92):
                return "Q"
            else:
                return "Y"

        # Normalize frequency aliases
        freq_map = {
            "MS": "M", "QS": "Q", "YS": "Y", "AS": "Y",
            "BMS": "M", "BQS": "Q", "BYS": "Y",
            "ME": "M", "QE": "Q", "YE": "Y",
            "BME": "M", "BQE": "Q", "BYE": "Y",
        }
        return freq_map.get(freq, freq)
    except Exception as e:
        logger.warning(f"Could not detect frequency: {e}. Defaulting to D.")
        return "D"


def prepare_nixtla_df(
    df: pd.DataFrame,
    ds_column: str = "ds",
    y_column: str = "y",
    unique_id_column: str = "unique_id",
) -> pd.DataFrame:
    """Prepare a DataFrame for Nixtla library consumption.

    Nixtla expects columns: unique_id, ds, y
    """
    result = df.copy()

    # Rename columns to standard Nixtla format
    rename_map = {}
    if ds_column != "ds":
        rename_map[ds_column] = "ds"
    if y_column != "y":
        rename_map[y_column] = "y"
    if unique_id_column != "unique_id" and unique_id_column in result.columns:
        rename_map[unique_id_column] = "unique_id"

    if rename_map:
        result = result.rename(columns=rename_map)

    # Ensure ds is datetime
    result["ds"] = pd.to_datetime(result["ds"])

    # Ensure y is numeric
    result["y"] = pd.to_numeric(result["y"], errors="coerce")

    # Add unique_id if not present (single series)
    if "unique_id" not in result.columns:
        result["unique_id"] = "series_1"

    # Sort by unique_id and ds
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Drop rows with NaN in y
    result = result.dropna(subset=["y"])

    return result[["unique_id", "ds", "y"]]


def get_series_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for the time series data."""
    summaries = {}

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]
        y = series["y"]

        summaries[str(uid)] = {
            "n_observations": len(series),
            "start_date": str(series["ds"].min()),
            "end_date": str(series["ds"].max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "median": float(y.median()),
            "skewness": float(y.skew()),
            "kurtosis": float(y.kurtosis()),
            "missing_values": int(y.isna().sum()),
            "q25": float(y.quantile(0.25)),
            "q75": float(y.quantile(0.75)),
            "iqr": float(y.quantile(0.75) - y.quantile(0.25)),
            "coefficient_of_variation": float(y.std() / y.mean()) if y.mean() != 0 else None,
        }

    return {
        "n_series": len(df["unique_id"].unique()),
        "series_ids": list(df["unique_id"].unique().astype(str)),
        "total_observations": len(df),
        "series_details": summaries,
    }


def detect_seasonality(df: pd.DataFrame, frequency: str) -> Dict[str, Any]:
    """Detect seasonality patterns in the time series."""
    seasonality_info = {}

    season_periods = {
        "h": [24, 168],     # daily, weekly
        "D": [7, 30, 365],  # weekly, monthly, yearly
        "W": [4, 52],       # monthly, yearly
        "M": [12],          # yearly
        "Q": [4],           # yearly
        "T": [60, 1440],    # hourly, daily
    }

    periods = season_periods.get(frequency, [])

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]["y"].values
        uid_str = str(uid)
        seasonality_info[uid_str] = {"periods_tested": periods, "detected_periods": []}

        for period in periods:
            if len(series) >= 2 * period:
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomp = seasonal_decompose(series, period=period, extrapolate_trend="freq")
                    seasonal_strength = 1 - (np.var(decomp.resid[~np.isnan(decomp.resid)]) /
                                              np.var(decomp.observed - decomp.trend[~np.isnan(decomp.trend)]))
                    seasonal_strength = max(0, seasonal_strength)

                    if seasonal_strength > 0.1:
                        seasonality_info[uid_str]["detected_periods"].append({
                            "period": period,
                            "strength": round(float(seasonal_strength), 4),
                        })
                except Exception:
                    pass

    return seasonality_info


def compute_stationarity_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run stationarity tests (ADF, KPSS) on each series."""
    results = {}

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]["y"].values
        uid_str = str(uid)
        results[uid_str] = {}

        # ADF Test
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series, autolag="AIC")
            results[uid_str]["adf_test"] = {
                "test_statistic": round(float(adf_result[0]), 4),
                "p_value": round(float(adf_result[1]), 4),
                "lags_used": int(adf_result[2]),
                "n_observations": int(adf_result[3]),
                "critical_values": {k: round(float(v), 4) for k, v in adf_result[4].items()},
                "is_stationary": bool(adf_result[1] < 0.05),
            }
        except Exception as e:
            results[uid_str]["adf_test"] = {"error": str(e)}

        # KPSS Test
        try:
            from statsmodels.tsa.stattools import kpss
            kpss_result = kpss(series, regression="c", nlags="auto")
            results[uid_str]["kpss_test"] = {
                "test_statistic": round(float(kpss_result[0]), 4),
                "p_value": round(float(kpss_result[1]), 4),
                "lags_used": int(kpss_result[2]),
                "critical_values": {k: round(float(v), 4) for k, v in kpss_result[3].items()},
                "is_stationary": bool(kpss_result[1] > 0.05),
            }
        except Exception as e:
            results[uid_str]["kpss_test"] = {"error": str(e)}

    return results


def compute_acf_pacf(df: pd.DataFrame, nlags: int = 40) -> Dict[str, Any]:
    """Compute ACF and PACF for each series."""
    results = {}

    for uid in df["unique_id"].unique():
        series = df[df["unique_id"] == uid]["y"].values
        uid_str = str(uid)
        max_lags = min(nlags, len(series) // 2 - 1)
        if max_lags < 1:
            results[uid_str] = {"error": "Not enough data points"}
            continue

        try:
            from statsmodels.tsa.stattools import acf, pacf

            acf_values = acf(series, nlags=max_lags, fft=True)
            pacf_values = pacf(series, nlags=max_lags)

            results[uid_str] = {
                "acf": [round(float(v), 4) for v in acf_values],
                "pacf": [round(float(v), 4) for v in pacf_values],
                "nlags": max_lags,
                "confidence_bound": round(float(1.96 / np.sqrt(len(series))), 4),
            }
        except Exception as e:
            results[uid_str] = {"error": str(e)}

    return results
