"""Visualization router - Plotly-based time series visualization endpoints."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import pandas as pd
import logging

from timeseries_api.utils.data_loader import (
    load_upload_file,
    prepare_nixtla_df,
    detect_frequency,
    compute_acf_pacf,
)
from timeseries_api.services.visualization_service import (
    plot_time_series,
    plot_decomposition,
    plot_acf_pacf,
    plot_seasonality,
    plot_distribution,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/plots", tags=["Visualization"])


@router.post("/time-series")
async def generate_time_series_plot(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    title: str = Form(default="Time Series Data"),
    width: int = Form(default=1200),
    height: int = Form(default=600),
    theme: str = Form(default="plotly_white"),
    series_ids: Optional[str] = Form(default=None),
):
    """Generate an interactive Plotly time series plot."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        sid_list = [s.strip() for s in series_ids.split(",")] if series_ids else None

        plot_json = plot_time_series(
            df=df,
            series_ids=sid_list,
            title=title,
            width=width,
            height=height,
            theme=theme,
        )
        return {"plot": plot_json, "type": "time_series"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decomposition")
async def generate_decomposition_plot(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    series_id: Optional[str] = Form(default=None),
    frequency: Optional[str] = Form(default=None),
    width: int = Form(default=1200),
    height: int = Form(default=800),
    theme: str = Form(default="plotly_white"),
):
    """Generate time series decomposition plot (trend, seasonal, residual)."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        if frequency is None:
            frequency = detect_frequency(df)

        plot_json = plot_decomposition(
            df=df,
            series_id=series_id,
            frequency=frequency,
            width=width,
            height=height,
            theme=theme,
        )
        return {"plot": plot_json, "type": "decomposition"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acf-pacf")
async def generate_acf_pacf_plot(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    nlags: int = Form(default=40),
    width: int = Form(default=1200),
    height: int = Form(default=500),
    theme: str = Form(default="plotly_white"),
):
    """Generate ACF and PACF plots."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        acf_pacf_data = compute_acf_pacf(df, nlags=nlags)
        plots = {}

        for uid, data in acf_pacf_data.items():
            if "error" not in data:
                plots[uid] = plot_acf_pacf(
                    acf_values=data["acf"],
                    pacf_values=data["pacf"],
                    confidence_bound=data["confidence_bound"],
                    series_id=uid,
                    width=width,
                    height=height,
                    theme=theme,
                )

        return {"plots": plots, "type": "acf_pacf"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seasonality")
async def generate_seasonality_plot(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    series_id: Optional[str] = Form(default=None),
    width: int = Form(default=1200),
    height: int = Form(default=500),
    theme: str = Form(default="plotly_white"),
):
    """Generate seasonal pattern plots (box plots by period)."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        plot_json = plot_seasonality(
            df=df,
            series_id=series_id,
            width=width,
            height=height,
            theme=theme,
        )
        return {"plot": plot_json, "type": "seasonality"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/distribution")
async def generate_distribution_plot(
    file: UploadFile = File(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    series_ids: Optional[str] = Form(default=None),
    width: int = Form(default=1200),
    height: int = Form(default=500),
    theme: str = Form(default="plotly_white"),
):
    """Generate value distribution plots (histogram + box plot)."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        sid_list = [s.strip() for s in series_ids.split(",")] if series_ids else None

        plot_json = plot_distribution(
            df=df,
            series_ids=sid_list,
            width=width,
            height=height,
            theme=theme,
        )
        return {"plot": plot_json, "type": "distribution"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
