"""Full analytics engine - orchestrates all Nixtla services for end-to-end analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from timeseries_api.utils.data_loader import (
    detect_frequency,
    get_series_summary,
    detect_seasonality,
    compute_stationarity_tests,
    compute_acf_pacf,
)
from timeseries_api.services.stats_forecast_service import run_stats_forecast, run_stats_cross_validation
from timeseries_api.services.visualization_service import (
    plot_time_series,
    plot_forecast,
    plot_decomposition,
    plot_acf_pacf,
    plot_seasonality,
    plot_distribution,
    plot_cross_validation,
    plot_residuals,
    plot_model_comparison,
)

logger = logging.getLogger(__name__)


def run_full_analytics(
    df: pd.DataFrame,
    frequency: Optional[str] = None,
    forecast_horizon: int = 12,
    include_forecasting: bool = True,
    include_decomposition: bool = True,
    include_stationarity_tests: bool = True,
    include_cross_validation: bool = True,
) -> Dict[str, Any]:
    """Run complete end-to-end time series analytics pipeline.

    This function orchestrates all available analytics:
    1. Data summary and descriptive statistics
    2. Frequency detection
    3. Seasonality detection
    4. Stationarity tests (ADF, KPSS)
    5. ACF/PACF analysis
    6. Time series decomposition
    7. Statistical forecasting with confidence intervals
    8. Cross-validation and model evaluation
    9. All Plotly visualizations
    """
    results = {
        "metadata": {},
        "summary": {},
        "plots": {},
    }

    # Step 1: Detect frequency
    if frequency is None:
        frequency = detect_frequency(df)
    results["metadata"]["frequency"] = frequency
    results["metadata"]["n_series"] = int(df["unique_id"].nunique())
    results["metadata"]["total_observations"] = len(df)

    # Step 2: Data summary
    logger.info("Computing data summary...")
    results["summary"] = get_series_summary(df)

    # Step 3: Generate time series plot
    logger.info("Generating time series plot...")
    results["plots"]["time_series"] = plot_time_series(
        df, title="Time Series Overview"
    )

    # Step 4: Distribution plot
    logger.info("Generating distribution plot...")
    results["plots"]["distribution"] = plot_distribution(df)

    # Step 5: Seasonality detection
    logger.info("Detecting seasonality...")
    results["seasonality"] = detect_seasonality(df, frequency)

    # Step 6: Seasonal patterns plot
    logger.info("Generating seasonality plot...")
    for uid in list(df["unique_id"].unique())[:3]:
        results["plots"][f"seasonality_{uid}"] = plot_seasonality(
            df, series_id=uid
        )

    # Step 7: Stationarity tests
    if include_stationarity_tests:
        logger.info("Running stationarity tests...")
        results["stationarity_tests"] = compute_stationarity_tests(df)

    # Step 8: ACF/PACF
    logger.info("Computing ACF/PACF...")
    acf_pacf_results = compute_acf_pacf(df)
    results["acf_pacf"] = acf_pacf_results

    for uid, acf_data in acf_pacf_results.items():
        if "error" not in acf_data:
            results["plots"][f"acf_pacf_{uid}"] = plot_acf_pacf(
                acf_values=acf_data["acf"],
                pacf_values=acf_data["pacf"],
                confidence_bound=acf_data["confidence_bound"],
                series_id=uid,
            )

    # Step 9: Decomposition
    if include_decomposition:
        logger.info("Running decomposition...")
        results["decomposition"] = {}
        for uid in list(df["unique_id"].unique())[:3]:
            try:
                results["plots"][f"decomposition_{uid}"] = plot_decomposition(
                    df, series_id=uid, frequency=frequency
                )
                results["decomposition"][str(uid)] = {"status": "completed"}
            except Exception as e:
                results["decomposition"][str(uid)] = {"error": str(e)}

    # Step 10: Forecasting
    if include_forecasting:
        logger.info("Running statistical forecasts...")
        forecast_methods = ["AutoARIMA", "AutoETS", "SeasonalNaive", "Naive", "HistoricAverage"]
        confidence_levels = [80, 90, 95]

        try:
            forecast_result = run_stats_forecast(
                df=df,
                horizon=forecast_horizon,
                methods=forecast_methods,
                frequency=frequency,
                confidence_levels=confidence_levels,
            )
            results["forecasts"] = forecast_result

            # Generate forecast plot
            if "error" not in forecast_result:
                forecasts_df = pd.DataFrame(forecast_result["forecasts"])
                if "ds" in forecasts_df.columns:
                    forecasts_df["ds"] = pd.to_datetime(forecasts_df["ds"])

                results["plots"]["forecast"] = plot_forecast(
                    historical=df,
                    forecasts=forecasts_df,
                    methods=forecast_result.get("methods_used", forecast_methods),
                    confidence_levels=confidence_levels,
                    title="Statistical Forecast Results",
                )
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            results["forecasts"] = {"error": str(e)}

    # Step 11: Cross-validation
    if include_cross_validation:
        logger.info("Running cross-validation...")
        cv_methods = ["AutoARIMA", "AutoETS", "SeasonalNaive"]

        try:
            cv_result = run_stats_cross_validation(
                df=df,
                horizon=forecast_horizon,
                methods=cv_methods,
                frequency=frequency,
                n_windows=3,
                confidence_levels=[90],
            )
            results["cross_validation"] = cv_result

            if "error" not in cv_result and "metrics" in cv_result:
                # Model comparison plot
                results["plots"]["model_comparison"] = plot_model_comparison(
                    cv_result["metrics"]
                )

                # Cross-validation plot
                cv_df = pd.DataFrame(cv_result["cv_results"])
                if "ds" in cv_df.columns:
                    cv_df["ds"] = pd.to_datetime(cv_df["ds"])

                results["plots"]["cross_validation"] = plot_cross_validation(
                    cv_results=cv_df,
                    methods=cv_result.get("methods_used", cv_methods),
                    metrics=cv_result["metrics"],
                )

                # Residual analysis for best model
                if cv_result["metrics"]:
                    best_model = min(
                        cv_result["metrics"].items(),
                        key=lambda x: x[1].get("mae", float("inf"))
                    )[0]
                    results["plots"]["residuals"] = plot_residuals(
                        cv_results=cv_df,
                        method=best_model,
                    )
                    results["metadata"]["best_model"] = best_model
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            results["cross_validation"] = {"error": str(e)}

    results["metadata"]["status"] = "completed"
    logger.info("Full analytics pipeline completed.")

    return results
