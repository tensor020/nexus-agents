"""Tests for the Nixtla Time Series Analytics API.

Tests cover:
- Sample data generation
- Data loading and preprocessing
- StatsForecast integration
- Full analytics pipeline
- API endpoints
- Plotly visualization generation
"""

import pytest
import pandas as pd
import numpy as np
import io
import json
from fastapi.testclient import TestClient

from timeseries_api.main import app
from timeseries_api.utils.data_loader import (
    prepare_nixtla_df,
    detect_frequency,
    get_series_summary,
    detect_seasonality,
    compute_stationarity_tests,
    compute_acf_pacf,
)

client = TestClient(app)


# --- Fixtures ---

def generate_sample_csv(n_series=2, n_points=120, frequency="M"):
    """Generate sample time series CSV data."""
    records = []
    for i in range(1, n_series + 1):
        dates = pd.date_range(start="2015-01-01", periods=n_points, freq=frequency)
        t = np.arange(n_points)
        y = 100 + t * 0.5 + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, n_points)
        for j in range(n_points):
            records.append({
                "unique_id": f"series_{i}",
                "ds": str(dates[j].date()),
                "y": round(float(y[j]), 2),
            })
    df = pd.DataFrame(records)
    return df


def get_csv_bytes(df):
    """Convert DataFrame to CSV bytes for upload."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# --- Unit Tests ---

class TestDataLoader:
    """Test data loading and preprocessing utilities."""

    def test_prepare_nixtla_df(self):
        df = generate_sample_csv(n_series=1, n_points=50)
        result = prepare_nixtla_df(df)
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns
        assert len(result) == 50

    def test_prepare_nixtla_df_custom_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "value": np.random.randn(30),
            "id": "test_series",
        })
        result = prepare_nixtla_df(df, ds_column="date", y_column="value", unique_id_column="id")
        assert "ds" in result.columns
        assert "y" in result.columns
        assert "unique_id" in result.columns

    def test_detect_frequency_monthly(self):
        df = generate_sample_csv(n_series=1, n_points=50, frequency="M")
        df = prepare_nixtla_df(df)
        freq = detect_frequency(df)
        assert freq in ("M", "MS", "ME")

    def test_detect_frequency_daily(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "unique_id": "s1",
            "ds": dates,
            "y": np.random.randn(100),
        })
        freq = detect_frequency(df)
        assert freq == "D"

    def test_get_series_summary(self):
        df = generate_sample_csv(n_series=2, n_points=50)
        df = prepare_nixtla_df(df)
        summary = get_series_summary(df)
        assert summary["n_series"] == 2
        assert "series_details" in summary
        assert "series_1" in summary["series_details"]

    def test_compute_stationarity_tests(self):
        df = generate_sample_csv(n_series=1, n_points=100)
        df = prepare_nixtla_df(df)
        results = compute_stationarity_tests(df)
        assert "series_1" in results
        assert "adf_test" in results["series_1"]

    def test_compute_acf_pacf(self):
        df = generate_sample_csv(n_series=1, n_points=100)
        df = prepare_nixtla_df(df)
        results = compute_acf_pacf(df, nlags=20)
        assert "series_1" in results
        assert "acf" in results["series_1"]
        assert "pacf" in results["series_1"]


class TestStatsForecast:
    """Test StatsForecast integration."""

    def test_stats_forecast_basic(self):
        from timeseries_api.services.stats_forecast_service import run_stats_forecast

        df = generate_sample_csv(n_series=1, n_points=120)
        df = prepare_nixtla_df(df)

        result = run_stats_forecast(
            df=df,
            horizon=12,
            methods=["SeasonalNaive", "Naive", "HistoricAverage"],
            frequency="M",
            confidence_levels=[90],
        )
        assert "error" not in result
        assert "forecasts" in result
        assert len(result["forecasts"]) > 0

    def test_stats_forecast_multiple_methods(self):
        from timeseries_api.services.stats_forecast_service import run_stats_forecast

        df = generate_sample_csv(n_series=1, n_points=120)
        df = prepare_nixtla_df(df)

        result = run_stats_forecast(
            df=df,
            horizon=6,
            methods=["AutoARIMA", "AutoETS", "SeasonalNaive"],
            frequency="M",
            confidence_levels=[80, 90, 95],
        )
        assert "error" not in result
        assert len(result["methods_used"]) >= 1

    def test_stats_cross_validation(self):
        from timeseries_api.services.stats_forecast_service import run_stats_cross_validation

        df = generate_sample_csv(n_series=1, n_points=120)
        df = prepare_nixtla_df(df)

        result = run_stats_cross_validation(
            df=df,
            horizon=6,
            methods=["SeasonalNaive", "Naive"],
            frequency="M",
            n_windows=2,
        )
        assert "error" not in result
        assert "metrics" in result

    def test_batch_forecast(self):
        from timeseries_api.services.stats_forecast_service import run_stats_forecast

        df = generate_sample_csv(n_series=3, n_points=120)
        df = prepare_nixtla_df(df)

        result = run_stats_forecast(
            df=df,
            horizon=12,
            methods=["SeasonalNaive"],
            frequency="M",
            confidence_levels=[90],
        )
        assert "error" not in result
        forecasts_df = pd.DataFrame(result["forecasts"])
        assert forecasts_df["unique_id"].nunique() == 3


class TestVisualization:
    """Test Plotly visualization generation."""

    def test_plot_time_series(self):
        from timeseries_api.services.visualization_service import plot_time_series

        df = generate_sample_csv(n_series=2, n_points=50)
        df = prepare_nixtla_df(df)
        plot_json = plot_time_series(df)
        data = json.loads(plot_json)
        assert "data" in data
        assert len(data["data"]) == 2

    def test_plot_distribution(self):
        from timeseries_api.services.visualization_service import plot_distribution

        df = generate_sample_csv(n_series=1, n_points=100)
        df = prepare_nixtla_df(df)
        plot_json = plot_distribution(df)
        data = json.loads(plot_json)
        assert "data" in data

    def test_plot_decomposition(self):
        from timeseries_api.services.visualization_service import plot_decomposition

        df = generate_sample_csv(n_series=1, n_points=120)
        df = prepare_nixtla_df(df)
        plot_json = plot_decomposition(df, frequency="M")
        data = json.loads(plot_json)
        assert "data" in data
        assert len(data["data"]) == 4  # observed, trend, seasonal, residual

    def test_plot_acf_pacf(self):
        from timeseries_api.services.visualization_service import plot_acf_pacf

        acf_vals = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        pacf_vals = [1.0, 0.7, 0.1, -0.1, 0.05, 0.02]
        plot_json = plot_acf_pacf(acf_vals, pacf_vals, confidence_bound=0.2)
        data = json.loads(plot_json)
        assert "data" in data


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Nixtla Time Series Analytics API"

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "libraries" in data

    def test_sample_dataset(self):
        response = client.get("/datasets/sample?n_series=2&n_points=50&frequency=M")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 100

    def test_summary_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=50)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/analytics/summary",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "n_series" in data

    def test_stationarity_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=100)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/analytics/stationarity",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200

    def test_stats_forecast_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=120)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/forecast/stats",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={
                "horizon": 6,
                "methods": "SeasonalNaive,Naive",
                "confidence_levels": "90",
                "include_plot": "true",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "forecasts" in data

    def test_batch_forecast_endpoint(self):
        df = generate_sample_csv(n_series=3, n_points=120)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/forecast/batch",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={
                "horizon": 6,
                "methods": "SeasonalNaive",
                "confidence_levels": "90",
                "include_plot": "true",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "batch_info" in data
        assert data["batch_info"]["n_series"] == 3

    def test_time_series_plot_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=50)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/plots/time-series",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "plot" in data

    def test_decomposition_plot_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=120)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/plots/decomposition",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "plot" in data

    def test_distribution_plot_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=100)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/plots/distribution",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "plot" in data

    def test_full_analytics_endpoint(self):
        df = generate_sample_csv(n_series=1, n_points=120)
        csv_bytes = get_csv_bytes(df)
        response = client.post(
            "/analytics/full",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={
                "forecast_horizon": 6,
                "include_forecasting": "true",
                "include_decomposition": "true",
                "include_stationarity_tests": "true",
                "include_cross_validation": "true",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "plots" in data
        assert "metadata" in data


class TestAnalyticsEngine:
    """Test the full analytics engine."""

    def test_full_analytics_pipeline(self):
        from timeseries_api.services.analytics_engine import run_full_analytics

        df = generate_sample_csv(n_series=1, n_points=120)
        df = prepare_nixtla_df(df)

        results = run_full_analytics(
            df=df,
            frequency="M",
            forecast_horizon=6,
            include_forecasting=True,
            include_decomposition=True,
            include_stationarity_tests=True,
            include_cross_validation=True,
        )

        assert results["metadata"]["status"] == "completed"
        assert "summary" in results
        assert "plots" in results
        assert "time_series" in results["plots"]

    def test_full_analytics_batch(self):
        from timeseries_api.services.analytics_engine import run_full_analytics

        df = generate_sample_csv(n_series=3, n_points=120)
        df = prepare_nixtla_df(df)

        results = run_full_analytics(
            df=df,
            frequency="M",
            forecast_horizon=6,
        )

        assert results["metadata"]["n_series"] == 3
        assert results["metadata"]["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
