"""StatsForecast service - Statistical forecasting models from Nixtla."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_stats_models(methods: List[str], season_length: int = 1):
    """Get StatsForecast model instances based on method names."""
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        AutoCES,
        MSTL,
        SeasonalNaive,
        Naive,
        HistoricAverage,
        WindowAverage,
        SeasonalWindowAverage,
        ADIDA,
        CrostonClassic,
        CrostonOptimized,
        CrostonSBA,
        IMAPA,
        TSB,
        HoltWinters,
    )

    model_map = {
        "AutoARIMA": lambda: AutoARIMA(season_length=season_length),
        "AutoETS": lambda: AutoETS(season_length=season_length),
        "AutoTheta": lambda: AutoTheta(season_length=season_length),
        "AutoCES": lambda: AutoCES(season_length=season_length),
        "MSTL": lambda: MSTL(season_length=season_length),
        "SeasonalNaive": lambda: SeasonalNaive(season_length=season_length),
        "Naive": lambda: Naive(),
        "HistoricAverage": lambda: HistoricAverage(),
        "WindowAverage": lambda: WindowAverage(window_size=min(12, season_length * 2) if season_length > 1 else 12),
        "SeasonalWindowAverage": lambda: SeasonalWindowAverage(
            season_length=season_length,
            window_size=2,
        ),
        "ADIDA": lambda: ADIDA(),
        "CrostonClassic": lambda: CrostonClassic(),
        "CrostonOptimized": lambda: CrostonOptimized(),
        "CrostonSBA": lambda: CrostonSBA(),
        "IMAPA": lambda: IMAPA(),
        "TSB": lambda: TSB(alpha_d=0.2, alpha_p=0.2),
        "HoltWinters": lambda: HoltWinters(
            season_length=season_length,
            error_type="A",
        ),
    }

    models = []
    for method in methods:
        if method in model_map:
            try:
                models.append(model_map[method]())
            except Exception as e:
                logger.warning(f"Could not initialize {method}: {e}")
        else:
            logger.warning(f"Unknown stats method: {method}")

    return models


def get_season_length(frequency: str) -> int:
    """Get default season length for a given frequency."""
    season_map = {
        "h": 24,
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "Y": 1,
        "T": 60,
        "S": 60,
        "B": 5,
    }
    return season_map.get(frequency, 1)


def run_stats_forecast(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    confidence_levels: List[int] = None,
) -> Dict[str, Any]:
    """Run StatsForecast models on the data."""
    from statsforecast import StatsForecast

    if confidence_levels is None:
        confidence_levels = [80, 90, 95]

    season_length = get_season_length(frequency)
    models = get_stats_models(methods, season_length=season_length)

    if not models:
        return {"error": "No valid models could be initialized"}

    sf = StatsForecast(
        models=models,
        freq=frequency,
        n_jobs=1,
    )

    # Fit and predict
    level = confidence_levels
    forecasts = sf.forecast(df=df, h=horizon, level=level)
    forecasts = forecasts.reset_index()

    # Convert to serializable format
    result = {
        "forecasts": forecasts.to_dict(orient="records"),
        "columns": list(forecasts.columns),
        "methods_used": [type(m).__name__ for m in models],
        "horizon": horizon,
        "frequency": frequency,
        "confidence_levels": confidence_levels,
    }

    return result


def run_stats_cross_validation(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    n_windows: int = 3,
    step_size: Optional[int] = None,
    confidence_levels: List[int] = None,
) -> Dict[str, Any]:
    """Run cross-validation with StatsForecast models."""
    from statsforecast import StatsForecast

    if confidence_levels is None:
        confidence_levels = [90]

    season_length = get_season_length(frequency)
    models = get_stats_models(methods, season_length=season_length)

    if not models:
        return {"error": "No valid models could be initialized"}

    sf = StatsForecast(
        models=models,
        freq=frequency,
        n_jobs=1,
    )

    if step_size is None:
        step_size = horizon

    cv_results = sf.cross_validation(
        df=df,
        h=horizon,
        n_windows=n_windows,
        step_size=step_size,
        level=confidence_levels,
    )
    cv_results = cv_results.reset_index()

    # Compute metrics
    metrics = compute_cv_metrics(cv_results, [type(m).__name__ for m in models])

    return {
        "cv_results": cv_results.to_dict(orient="records"),
        "metrics": metrics,
        "n_windows": n_windows,
        "methods_used": [type(m).__name__ for m in models],
    }


def compute_cv_metrics(cv_df: pd.DataFrame, model_names: List[str]) -> Dict[str, Any]:
    """Compute cross-validation metrics for each model."""
    from utilsforecast.losses import mae, mse, rmse, mape, smape

    metrics = {}
    for model_name in model_names:
        if model_name not in cv_df.columns:
            continue

        try:
            y_true = cv_df["y"].values
            y_pred = cv_df[model_name].values

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            if len(y_true_clean) == 0:
                continue

            errors = y_true_clean - y_pred_clean
            abs_errors = np.abs(errors)

            metrics[model_name] = {
                "mae": round(float(np.mean(abs_errors)), 4),
                "mse": round(float(np.mean(errors ** 2)), 4),
                "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
                "mape": round(float(np.mean(np.abs(errors / np.where(y_true_clean == 0, 1, y_true_clean))) * 100), 4),
                "smape": round(float(np.mean(2 * abs_errors / (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8)) * 100), 4),
            }
        except Exception as e:
            metrics[model_name] = {"error": str(e)}

    return metrics
