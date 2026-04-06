"""Forecasting router - single and batch forecasting endpoints."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import pandas as pd
import json
import logging

from timeseries_api.utils.data_loader import (
    load_upload_file,
    prepare_nixtla_df,
    detect_frequency,
)
from timeseries_api.services.stats_forecast_service import (
    run_stats_forecast,
    run_stats_cross_validation,
)
from timeseries_api.services.ml_forecast_service import (
    run_ml_forecast,
    run_ml_cross_validation,
)
from timeseries_api.services.neural_forecast_service import (
    run_neural_forecast,
    run_neural_cross_validation,
)
from timeseries_api.services.hierarchical_forecast_service import (
    run_hierarchical_forecast,
)
from timeseries_api.services.visualization_service import (
    plot_forecast,
    plot_cross_validation,
    plot_model_comparison,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/forecast", tags=["Forecasting"])


@router.post("/stats")
async def stats_forecast(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    methods: str = Form(default="AutoARIMA,AutoETS,SeasonalNaive"),
    confidence_levels: str = Form(default="80,90,95"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run StatsForecast models (AutoARIMA, AutoETS, AutoTheta, MSTL, etc.).

    Supports single series and batch forecasting with confidence intervals.
    Available methods: AutoARIMA, AutoETS, AutoTheta, AutoCES, MSTL,
    SeasonalNaive, Naive, HistoricAverage, WindowAverage,
    SeasonalWindowAverage, ADIDA, CrostonClassic, CrostonOptimized,
    CrostonSBA, IMAPA, TSB, HoltWinters
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]
        ci_levels = [int(l.strip()) for l in confidence_levels.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_stats_forecast(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            confidence_levels=ci_levels,
        )

        if include_plot and "error" not in result:
            forecasts_df = pd.DataFrame(result["forecasts"])
            if "ds" in forecasts_df.columns:
                forecasts_df["ds"] = pd.to_datetime(forecasts_df["ds"])

            result["plot"] = plot_forecast(
                historical=df,
                forecasts=forecasts_df,
                methods=result.get("methods_used", method_list),
                confidence_levels=ci_levels,
            )

        return result
    except Exception as e:
        logger.error(f"Stats forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/cross-validation")
async def stats_cross_validation(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    n_windows: int = Form(default=3),
    methods: str = Form(default="AutoARIMA,AutoETS,SeasonalNaive"),
    confidence_levels: str = Form(default="90"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run cross-validation with StatsForecast models."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]
        ci_levels = [int(l.strip()) for l in confidence_levels.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_stats_cross_validation(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            n_windows=n_windows,
            confidence_levels=ci_levels,
        )

        if include_plot and "error" not in result and "metrics" in result:
            cv_df = pd.DataFrame(result["cv_results"])
            if "ds" in cv_df.columns:
                cv_df["ds"] = pd.to_datetime(cv_df["ds"])

            result["plot_cv"] = plot_cross_validation(
                cv_results=cv_df,
                methods=result.get("methods_used", method_list),
                metrics=result["metrics"],
            )
            result["plot_comparison"] = plot_model_comparison(result["metrics"])

        return result
    except Exception as e:
        logger.error(f"Stats CV failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml")
async def ml_forecast(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    methods: str = Form(default="LightGBM,LinearRegression"),
    lags: str = Form(default="1,2,3,4,5,6,7,12,24"),
    confidence_levels: str = Form(default="80,90,95"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run MLForecast models (LightGBM, XGBoost, RandomForest, etc.).

    Supports single series and batch forecasting with lag features.
    Available methods: LightGBM, XGBoost, LinearRegression, Ridge, Lasso,
    RandomForest, KNN
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]
        lag_list = [int(l.strip()) for l in lags.split(",")]
        ci_levels = [int(l.strip()) for l in confidence_levels.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_ml_forecast(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            lags=lag_list,
            confidence_levels=ci_levels,
        )

        if include_plot and "error" not in result:
            forecasts_df = pd.DataFrame(result["forecasts"])
            if "ds" in forecasts_df.columns:
                forecasts_df["ds"] = pd.to_datetime(forecasts_df["ds"])

            result["plot"] = plot_forecast(
                historical=df,
                forecasts=forecasts_df,
                methods=result.get("methods_used", method_list),
                confidence_levels=ci_levels,
            )

        return result
    except Exception as e:
        logger.error(f"ML forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/cross-validation")
async def ml_cross_validation(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    n_windows: int = Form(default=3),
    methods: str = Form(default="LightGBM,LinearRegression"),
    lags: str = Form(default="1,2,3,4,5,6,7,12,24"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run cross-validation with MLForecast models."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]
        lag_list = [int(l.strip()) for l in lags.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_ml_cross_validation(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            n_windows=n_windows,
            lags=lag_list,
        )

        if include_plot and "error" not in result and "metrics" in result:
            result["plot_comparison"] = plot_model_comparison(result["metrics"])

        return result
    except Exception as e:
        logger.error(f"ML CV failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neural")
async def neural_forecast(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    methods: str = Form(default="NBEATS,NHITS"),
    max_steps: int = Form(default=100),
    confidence_levels: str = Form(default="80,90,95"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run NeuralForecast models (NBEATS, NHITS, LSTM, TFT, PatchTST, etc.).

    Supports deep learning-based time series forecasting.
    Available methods: NBEATS, NHITS, LSTM, GRU, TCN, TFT, PatchTST,
    TimesNet, DeepAR, FedFormer, Informer, Autoformer
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]
        ci_levels = [int(l.strip()) for l in confidence_levels.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_neural_forecast(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            max_steps=max_steps,
            confidence_levels=ci_levels,
        )

        if include_plot and "error" not in result:
            forecasts_df = pd.DataFrame(result["forecasts"])
            if "ds" in forecasts_df.columns:
                forecasts_df["ds"] = pd.to_datetime(forecasts_df["ds"])

            result["plot"] = plot_forecast(
                historical=df,
                forecasts=forecasts_df,
                methods=result.get("methods_used", method_list),
                confidence_levels=ci_levels,
            )

        return result
    except Exception as e:
        logger.error(f"Neural forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neural/cross-validation")
async def neural_cross_validation(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    n_windows: int = Form(default=2),
    methods: str = Form(default="NBEATS,NHITS"),
    max_steps: int = Form(default=100),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Run cross-validation with NeuralForecast models."""
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        method_list = [m.strip() for m in methods.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_neural_cross_validation(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            n_windows=n_windows,
            max_steps=max_steps,
        )

        if include_plot and "error" not in result and "metrics" in result:
            result["plot_comparison"] = plot_model_comparison(result["metrics"])

        return result
    except Exception as e:
        logger.error(f"Neural CV failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hierarchical")
async def hierarchical_forecast(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    methods: str = Form(default="AutoARIMA"),
    reconciliation_methods: str = Form(default="BottomUp,MinTrace"),
    hierarchy_levels: str = Form(...),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
):
    """Run hierarchical forecast with reconciliation.

    Available reconciliation methods: BottomUp, TopDown, MinTrace, ERM
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        # Re-add hierarchy columns from raw data
        hierarchy_list = [h.strip() for h in hierarchy_levels.split(",")]
        for col in hierarchy_list:
            if col in raw_df.columns and col not in df.columns:
                df[col] = raw_df[col].values[:len(df)]

        method_list = [m.strip() for m in methods.split(",")]
        recon_list = [r.strip() for r in reconciliation_methods.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        result = run_hierarchical_forecast(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            hierarchy_levels=hierarchy_list,
            reconciliation_methods=recon_list,
            unique_id_column="unique_id",
        )

        return result
    except Exception as e:
        logger.error(f"Hierarchical forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_forecast(
    file: UploadFile = File(...),
    horizon: int = Form(default=12),
    methods: str = Form(default="AutoARIMA,AutoETS,SeasonalNaive"),
    confidence_levels: str = Form(default="80,90,95"),
    ds_column: str = Form(default="ds"),
    y_column: str = Form(default="y"),
    unique_id_column: str = Form(default="unique_id"),
    frequency: Optional[str] = Form(default=None),
    series_ids: Optional[str] = Form(default=None),
    include_plot: bool = Form(default=True),
):
    """Batch forecast for multiple time series simultaneously.

    Upload a dataset with multiple series (identified by unique_id column).
    Optionally filter specific series with the series_ids parameter.
    """
    try:
        raw_df = await load_upload_file(file)
        df = prepare_nixtla_df(raw_df, ds_column, y_column, unique_id_column)

        # Filter specific series if requested
        if series_ids:
            id_list = [s.strip() for s in series_ids.split(",")]
            df = df[df["unique_id"].isin(id_list)]
            if df.empty:
                raise HTTPException(status_code=400, detail="No matching series found")

        method_list = [m.strip() for m in methods.split(",")]
        ci_levels = [int(l.strip()) for l in confidence_levels.split(",")]

        if frequency is None:
            frequency = detect_frequency(df)

        n_series = df["unique_id"].nunique()

        result = run_stats_forecast(
            df=df,
            horizon=horizon,
            methods=method_list,
            frequency=frequency,
            confidence_levels=ci_levels,
        )

        result["batch_info"] = {
            "n_series": n_series,
            "series_ids": df["unique_id"].unique().tolist(),
        }

        if include_plot and "error" not in result:
            forecasts_df = pd.DataFrame(result["forecasts"])
            if "ds" in forecasts_df.columns:
                forecasts_df["ds"] = pd.to_datetime(forecasts_df["ds"])

            result["plot"] = plot_forecast(
                historical=df,
                forecasts=forecasts_df,
                methods=result.get("methods_used", method_list),
                confidence_levels=ci_levels,
                title=f"Batch Forecast - {n_series} Series",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch forecast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
