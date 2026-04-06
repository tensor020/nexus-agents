"""Analytics router - full end-to-end time series analytics endpoints."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import pandas as pd
import json
import logging

from timeseries_api.utils.data_loader import (
    load_upload_file,
    prepare_nixtla_df,
    detect_frequency,
    get_series_summary,
    detect_seasonality,
    compute_stationarity_tests,
    compute_acf_pacf,
)
from timeseries_api.services.analytics_engine import run_full_analytics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.post("/full")
async def full_analytics(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    forecast_horizon: int = Form(default=12),
    include_forecasting: bool = Form(default=True),
    include_decomposition: bool = Form(default=True),
    include_stationarity_tests: bool = Form(default=True),
    include_cross_validation: bool = Form(default=True),
):
    """Run full end-to-end time series analytics on uploaded data.

    This endpoint performs complete automated analysis including:
    - Data summary and descriptive statistics
    - Frequency detection
    - Seasonality detection and visualization
    - Stationarity tests (ADF, KPSS)
    - ACF/PACF analysis
    - Time series decomposition
    - Statistical forecasting with confidence intervals
    - Cross-validation and model evaluation
    - All Plotly visualizations (time series, forecast, decomposition, etc.)
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        results = run_full_analytics(
            df=df,
            frequency=frequency,
            forecast_horizon=forecast_horizon,
            include_forecasting=include_forecasting,
            include_decomposition=include_decomposition,
            include_stationarity_tests=include_stationarity_tests,
            include_cross_validation=include_cross_validation,
        )

        return results
    except Exception as e:
        logger.error(f"Full analytics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def data_summary(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
):
    """Get summary statistics for time series data."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)
        frequency = detect_frequency(df)

        summary = get_series_summary(df)
        summary["detected_frequency"] = frequency

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stationarity")
async def stationarity_tests(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
):
    """Run stationarity tests (ADF and KPSS) on the data."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)
        return compute_stationarity_tests(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seasonality")
async def seasonality_detection(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
):
    """Detect seasonality patterns in the data."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)
        if frequency is None:
            frequency = detect_frequency(df)
        return detect_seasonality(df, frequency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acf-pacf")
async def acf_pacf_analysis(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    nlags: int = Form(default=40),
):
    """Compute ACF and PACF for each series."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)
        return compute_acf_pacf(df, nlags=nlags)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
