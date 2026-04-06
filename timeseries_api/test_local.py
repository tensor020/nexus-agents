"""
Local test script - Run the full analytics pipeline and view results.

Usage:
    1. Install all dependencies:
       pip install statsforecast mlforecast utilsforecast datasetsforecast coreforecast
       pip install statsmodels lightgbm xgboost scikit-learn
       pip install fastapi uvicorn python-multipart plotly pandas numpy httpx

    2. Run this script:
       python3 timeseries_api/test_local.py

    3. Or start the server and use the Swagger UI:
       uvicorn timeseries_api.main:app --reload --port 8000
       Then open http://localhost:8000/docs in your browser.

       In the Swagger UI:
       - Go to POST /analytics/full
       - Click "Try it out"
       - Upload timeseries_api/sample_data.csv as the file
       - Set forecast_horizon to 12
       - Click "Execute"
       - The response contains all analytics + Plotly plot JSON

    4. To view plots, copy any plot JSON from the response and paste it at:
       https://plotly.com/chart-studio/ or use the HTML viewer below.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi.testclient import TestClient
from timeseries_api.main import app

client = TestClient(app)


def save_plot_html(plot_json_str: str, filename: str):
    """Save a Plotly JSON plot as a standalone HTML file."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body {{ margin: 0; padding: 20px; font-family: sans-serif; }}</style>
</head>
<body>
    <div id="plot" style="width:100%;height:90vh;"></div>
    <script>
        var plotData = {plot_json_str};
        Plotly.newPlot('plot', plotData.data, plotData.layout);
    </script>
</body>
</html>"""
    with open(filename, "w") as f:
        f.write(html)
    print(f"  -> Saved plot: {filename}")


def test_full_analytics():
    """Test the full analytics pipeline with sample data."""
    print("=" * 70)
    print("NIXTLA TIME SERIES ANALYTICS - FULL TEST")
    print("=" * 70)

    # Load sample data
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    with open(sample_path, "rb") as f:
        csv_bytes = f.read()

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Test 1: Health check ---
    print("\n1. Health Check")
    resp = client.get("/health")
    health = resp.json()
    print(f"   Status: {health['status']}")
    for lib, status in health["libraries"].items():
        print(f"   {lib}: {status}")

    # --- Test 2: Data Summary ---
    print("\n2. Data Summary")
    resp = client.post(
        "/analytics/summary",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
    )
    summary = resp.json()
    print(f"   Series: {summary['n_series']}")
    print(f"   Total observations: {summary['total_observations']}")
    for sid, details in summary["series_details"].items():
        print(f"   {sid}: mean={details['mean']:.1f}, std={details['std']:.1f}, "
              f"range=[{details['min']:.1f}, {details['max']:.1f}]")

    # --- Test 3: Stationarity Tests ---
    print("\n3. Stationarity Tests")
    resp = client.post(
        "/analytics/stationarity",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
    )
    if resp.status_code == 200:
        stationarity = resp.json()
        for sid, tests in stationarity.items():
            if "adf_test" in tests and "error" not in tests["adf_test"]:
                adf = tests["adf_test"]
                print(f"   {sid} ADF: stat={adf['test_statistic']}, p={adf['p_value']}, "
                      f"stationary={adf['is_stationary']}")
            if "kpss_test" in tests and "error" not in tests["kpss_test"]:
                kpss = tests["kpss_test"]
                print(f"   {sid} KPSS: stat={kpss['test_statistic']}, p={kpss['p_value']}, "
                      f"stationary={kpss['is_stationary']}")
    else:
        print(f"   Skipped (statsmodels not available)")

    # --- Test 4: Time Series Plot ---
    print("\n4. Time Series Plot")
    resp = client.post(
        "/plots/time-series",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={"title": "Sample Time Series Data"},
    )
    if resp.status_code == 200:
        save_plot_html(resp.json()["plot"], os.path.join(output_dir, "time_series.html"))

    # --- Test 5: Decomposition Plot ---
    print("\n5. Decomposition Plot")
    resp = client.post(
        "/plots/decomposition",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={"series_id": "series_1"},
    )
    if resp.status_code == 200:
        save_plot_html(resp.json()["plot"], os.path.join(output_dir, "decomposition.html"))
    else:
        print(f"   Skipped (statsmodels not available)")

    # --- Test 6: Distribution Plot ---
    print("\n6. Distribution Plot")
    resp = client.post(
        "/plots/distribution",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
    )
    if resp.status_code == 200:
        save_plot_html(resp.json()["plot"], os.path.join(output_dir, "distribution.html"))

    # --- Test 7: ACF/PACF Plot ---
    print("\n7. ACF/PACF Plot")
    resp = client.post(
        "/plots/acf-pacf",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
    )
    if resp.status_code == 200:
        data = resp.json()
        for sid, plot_json in data.get("plots", {}).items():
            save_plot_html(plot_json, os.path.join(output_dir, f"acf_pacf_{sid}.html"))
    else:
        print(f"   Skipped (statsmodels not available)")

    # --- Test 8: Seasonality Plot ---
    print("\n8. Seasonality Plot")
    resp = client.post(
        "/plots/seasonality",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={"series_id": "series_1"},
    )
    if resp.status_code == 200:
        save_plot_html(resp.json()["plot"], os.path.join(output_dir, "seasonality.html"))

    # --- Test 9: StatsForecast (single series) ---
    print("\n9. StatsForecast - Single Series with Confidence Intervals")
    resp = client.post(
        "/forecast/stats",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "horizon": "12",
            "methods": "AutoARIMA,AutoETS,SeasonalNaive",
            "confidence_levels": "80,90,95",
            "include_plot": "true",
        },
    )
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Methods used: {result.get('methods_used', [])}")
        print(f"   Forecast points: {len(result.get('forecasts', []))}")
        if "plot" in result:
            save_plot_html(result["plot"], os.path.join(output_dir, "forecast_stats.html"))
    else:
        print(f"   Skipped (statsforecast not available): {resp.json().get('detail', '')[:100]}")

    # --- Test 10: Batch Forecast ---
    print("\n10. Batch Forecast - Multiple Series")
    resp = client.post(
        "/forecast/batch",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "horizon": "12",
            "methods": "AutoARIMA,SeasonalNaive",
            "confidence_levels": "90,95",
            "include_plot": "true",
        },
    )
    if resp.status_code == 200:
        result = resp.json()
        batch_info = result.get("batch_info", {})
        print(f"   Series forecasted: {batch_info.get('n_series', 'N/A')}")
        print(f"   Series IDs: {batch_info.get('series_ids', [])}")
        if "plot" in result:
            save_plot_html(result["plot"], os.path.join(output_dir, "forecast_batch.html"))
    else:
        print(f"   Skipped: {resp.json().get('detail', '')[:100]}")

    # --- Test 11: Cross-Validation ---
    print("\n11. Cross-Validation with Model Comparison")
    resp = client.post(
        "/forecast/stats/cross-validation",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "horizon": "6",
            "n_windows": "3",
            "methods": "AutoARIMA,AutoETS,SeasonalNaive",
            "confidence_levels": "90",
            "include_plot": "true",
        },
    )
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Methods: {result.get('methods_used', [])}")
        metrics = result.get("metrics", {})
        for model, m in metrics.items():
            if "error" not in m:
                print(f"   {model}: MAE={m.get('mae', 'N/A')}, "
                      f"RMSE={m.get('rmse', 'N/A')}, MAPE={m.get('mape', 'N/A')}%")
        if "plot_comparison" in result:
            save_plot_html(result["plot_comparison"], os.path.join(output_dir, "model_comparison.html"))
        if "plot_cv" in result:
            save_plot_html(result["plot_cv"], os.path.join(output_dir, "cross_validation.html"))
    else:
        print(f"   Skipped: {resp.json().get('detail', '')[:100]}")

    # --- Test 12: ML Forecast ---
    print("\n12. ML Forecast (LightGBM, LinearRegression)")
    resp = client.post(
        "/forecast/ml",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "horizon": "12",
            "methods": "LightGBM,LinearRegression",
            "lags": "1,2,3,6,12",
            "confidence_levels": "90",
            "include_plot": "true",
        },
    )
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Methods: {result.get('methods_used', [])}")
        print(f"   Lags used: {result.get('lags_used', [])}")
        if "plot" in result:
            save_plot_html(result["plot"], os.path.join(output_dir, "forecast_ml.html"))
    else:
        print(f"   Skipped: {resp.json().get('detail', '')[:100]}")

    # --- Test 13: Full Analytics ---
    print("\n13. FULL END-TO-END ANALYTICS")
    resp = client.post(
        "/analytics/full",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "forecast_horizon": "12",
            "include_forecasting": "true",
            "include_decomposition": "true",
            "include_stationarity_tests": "true",
            "include_cross_validation": "true",
        },
    )
    if resp.status_code == 200:
        result = resp.json()
        meta = result.get("metadata", {})
        print(f"   Status: {meta.get('status', 'unknown')}")
        print(f"   Frequency: {meta.get('frequency', 'unknown')}")
        print(f"   Series: {meta.get('n_series', 'unknown')}")
        print(f"   Best model: {meta.get('best_model', 'N/A')}")
        print(f"   Plots generated: {list(result.get('plots', {}).keys())}")

        # Save all plots from full analytics
        for plot_name, plot_json in result.get("plots", {}).items():
            save_plot_html(plot_json, os.path.join(output_dir, f"full_{plot_name}.html"))
    else:
        print(f"   Error: {resp.json().get('detail', '')[:200]}")

    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}/")
    print("Open any .html file in your browser to view the interactive Plotly charts.")
    print("=" * 70)


if __name__ == "__main__":
    test_full_analytics()
