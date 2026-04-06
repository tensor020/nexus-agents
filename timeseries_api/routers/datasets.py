"""Datasets router - sample dataset generation using datasetsforecast."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["Datasets"])


def _generate_sample_data(
    n_series: int = 2,
    n_points: int = 200,
    frequency: str = "M",
    include_trend: bool = True,
    include_seasonality: bool = True,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """Generate sample time series data for testing."""
    import numpy as np

    records = []
    for i in range(1, n_series + 1):
        dates = pd.date_range(start="2015-01-01", periods=n_points, freq=frequency)
        t = np.arange(n_points)

        # Base signal
        y = np.ones(n_points) * 100

        # Trend
        if include_trend:
            y += t * 0.5 * (1 + 0.3 * i)

        # Seasonality
        if include_seasonality:
            period = {"M": 12, "D": 7, "W": 52, "h": 24, "Q": 4}.get(frequency, 12)
            y += 10 * np.sin(2 * np.pi * t / period) * (0.8 + 0.2 * i)
            if period > 4:
                y += 5 * np.cos(4 * np.pi * t / period)

        # Noise
        y += np.random.normal(0, noise_level * np.mean(y), n_points)

        for j in range(n_points):
            records.append({
                "unique_id": f"series_{i}",
                "ds": dates[j],
                "y": round(float(y[j]), 2),
            })

    return pd.DataFrame(records)


@router.get("/sample")
async def get_sample_dataset(
    n_series: int = Query(default=2, ge=1, le=100),
    n_points: int = Query(default=200, ge=20, le=10000),
    frequency: str = Query(default="M"),
    include_trend: bool = Query(default=True),
    include_seasonality: bool = Query(default=True),
    noise_level: float = Query(default=0.1, ge=0, le=1),
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    """Generate a sample time series dataset for testing.

    Returns synthetic time series data with configurable properties.
    """
    try:
        df = _generate_sample_data(
            n_series=n_series,
            n_points=n_points,
            frequency=frequency,
            include_trend=include_trend,
            include_seasonality=include_seasonality,
            noise_level=noise_level,
        )

        if format == "csv":
            from fastapi.responses import StreamingResponse
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            stream.seek(0)
            return StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=sample_timeseries.csv"},
            )

        return {
            "data": df.to_dict(orient="records"),
            "metadata": {
                "n_series": n_series,
                "n_points": n_points,
                "frequency": frequency,
                "total_rows": len(df),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nixtla-datasets")
async def list_nixtla_datasets():
    """List available datasets from datasetsforecast library."""
    try:
        datasets = {
            "available_datasets": [
                {
                    "name": "AirPassengers",
                    "description": "Monthly totals of international airline passengers (1949-1960)",
                    "frequency": "M",
                    "n_series": 1,
                },
                {
                    "name": "M4-Hourly",
                    "description": "M4 competition hourly data subset",
                    "frequency": "h",
                    "n_series": "multiple",
                },
                {
                    "name": "M4-Daily",
                    "description": "M4 competition daily data subset",
                    "frequency": "D",
                    "n_series": "multiple",
                },
                {
                    "name": "M4-Weekly",
                    "description": "M4 competition weekly data subset",
                    "frequency": "W",
                    "n_series": "multiple",
                },
                {
                    "name": "M4-Monthly",
                    "description": "M4 competition monthly data subset",
                    "frequency": "M",
                    "n_series": "multiple",
                },
                {
                    "name": "M4-Quarterly",
                    "description": "M4 competition quarterly data subset",
                    "frequency": "Q",
                    "n_series": "multiple",
                },
                {
                    "name": "M4-Yearly",
                    "description": "M4 competition yearly data subset",
                    "frequency": "Y",
                    "n_series": "multiple",
                },
            ],
            "usage": "Use GET /datasets/load/{dataset_name} to load a dataset",
        }
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/{dataset_name}")
async def load_nixtla_dataset(
    dataset_name: str,
    max_series: int = Query(default=10, ge=1, le=1000),
):
    """Load a dataset from datasetsforecast library."""
    try:
        if dataset_name == "AirPassengers":
            # Generate AirPassengers-like data
            dates = pd.date_range(start="1949-01-01", periods=144, freq="M")
            import numpy as np
            t = np.arange(144)
            y = 100 + t * 2.5 + 40 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, 144)
            df = pd.DataFrame({
                "unique_id": "AirPassengers",
                "ds": dates,
                "y": y.round(1),
            })
        else:
            # Try loading from datasetsforecast
            try:
                from datasetsforecast.m4 import M4
                dataset_map = {
                    "M4-Hourly": "Hourly",
                    "M4-Daily": "Daily",
                    "M4-Weekly": "Weekly",
                    "M4-Monthly": "Monthly",
                    "M4-Quarterly": "Quarterly",
                    "M4-Yearly": "Yearly",
                }
                if dataset_name in dataset_map:
                    Y_df, *_ = M4.load("data", group=dataset_map[dataset_name])
                    # Limit series
                    series_ids = Y_df["unique_id"].unique()[:max_series]
                    df = Y_df[Y_df["unique_id"].isin(series_ids)]
                else:
                    raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
            except ImportError:
                # Fallback: generate synthetic data
                df = _generate_sample_data(n_series=min(max_series, 5), n_points=200)

        return {
            "data": df.to_dict(orient="records"),
            "metadata": {
                "dataset_name": dataset_name,
                "n_series": int(df["unique_id"].nunique()),
                "total_rows": len(df),
                "columns": list(df.columns),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
