"""MLForecast service - Machine Learning forecasting models from Nixtla."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_ml_models(methods: List[str]):
    """Get ML model instances based on method names."""
    models = {}

    for method in methods:
        try:
            if method == "LightGBM":
                import lightgbm as lgb
                models[method] = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    verbose=-1,
                )
            elif method == "XGBoost":
                import xgboost as xgb
                models[method] = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    verbosity=0,
                )
            elif method == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                models[method] = LinearRegression()
            elif method == "Ridge":
                from sklearn.linear_model import Ridge
                models[method] = Ridge(alpha=1.0)
            elif method == "Lasso":
                from sklearn.linear_model import Lasso
                models[method] = Lasso(alpha=0.1)
            elif method == "RandomForest":
                from sklearn.ensemble import RandomForestRegressor
                models[method] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=1,
                )
            elif method == "KNN":
                from sklearn.neighbors import KNeighborsRegressor
                models[method] = KNeighborsRegressor(n_neighbors=5)
            else:
                logger.warning(f"Unknown ML method: {method}")
        except ImportError as e:
            logger.warning(f"Could not import {method}: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize {method}: {e}")

    return models


def run_ml_forecast(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    lags: List[int] = None,
    confidence_levels: List[int] = None,
) -> Dict[str, Any]:
    """Run MLForecast models on the data."""
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    import mlforecast.lag_transforms as lag_transforms

    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7, 12, 24]
    if confidence_levels is None:
        confidence_levels = [80, 90, 95]

    models = get_ml_models(methods)

    if not models:
        return {"error": "No valid ML models could be initialized"}

    # Filter lags to be valid for the data
    min_series_len = df.groupby("unique_id").size().min()
    valid_lags = [lag for lag in lags if lag < min_series_len - horizon]
    if not valid_lags:
        valid_lags = [1]

    # Define lag transforms
    lag_tfms = {}
    if len(valid_lags) >= 2:
        window = min(4, len(valid_lags))
        lag_tfms = {
            1: [
                lag_transforms.RollingMean(window_size=window),
                lag_transforms.RollingStd(window_size=window),
            ],
        }

    mlf = MLForecast(
        models=models,
        freq=frequency,
        lags=valid_lags,
        lag_transforms=lag_tfms,
        date_features=["dayofweek", "month", "year"],
    )

    # Fit
    mlf.fit(df)

    # Predict with confidence intervals
    level = confidence_levels
    try:
        forecasts = mlf.predict(h=horizon, level=level)
    except Exception:
        # Fallback without confidence intervals
        forecasts = mlf.predict(h=horizon)

    forecasts = forecasts.reset_index() if "unique_id" not in forecasts.columns else forecasts

    result = {
        "forecasts": forecasts.to_dict(orient="records"),
        "columns": list(forecasts.columns),
        "methods_used": list(models.keys()),
        "horizon": horizon,
        "frequency": frequency,
        "lags_used": valid_lags,
        "confidence_levels": confidence_levels,
    }

    return result


def run_ml_cross_validation(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    n_windows: int = 3,
    lags: List[int] = None,
) -> Dict[str, Any]:
    """Run cross-validation with MLForecast models."""
    from mlforecast import MLForecast

    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7, 12, 24]

    models = get_ml_models(methods)

    if not models:
        return {"error": "No valid ML models could be initialized"}

    min_series_len = df.groupby("unique_id").size().min()
    valid_lags = [lag for lag in lags if lag < min_series_len - horizon]
    if not valid_lags:
        valid_lags = [1]

    mlf = MLForecast(
        models=models,
        freq=frequency,
        lags=valid_lags,
        date_features=["dayofweek", "month", "year"],
    )

    cv_results = mlf.cross_validation(
        df=df,
        h=horizon,
        n_windows=n_windows,
    )
    cv_results = cv_results.reset_index() if "unique_id" not in cv_results.columns else cv_results

    # Compute metrics
    from timeseries_api.services.stats_forecast_service import compute_cv_metrics
    metrics = compute_cv_metrics(cv_results, list(models.keys()))

    return {
        "cv_results": cv_results.to_dict(orient="records"),
        "metrics": metrics,
        "n_windows": n_windows,
        "methods_used": list(models.keys()),
    }
