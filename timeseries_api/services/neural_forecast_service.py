"""NeuralForecast service - Deep learning forecasting models from Nixtla."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_neural_models(methods: List[str], horizon: int, max_steps: int = 100):
    """Get NeuralForecast model instances based on method names."""
    models = []

    for method in methods:
        try:
            if method == "NBEATS":
                from neuralforecast.models import NBEATS
                models.append(NBEATS(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "NHITS":
                from neuralforecast.models import NHITS
                models.append(NHITS(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "LSTM":
                from neuralforecast.models import LSTM
                models.append(LSTM(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "GRU":
                from neuralforecast.models import GRU
                models.append(GRU(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "TCN":
                from neuralforecast.models import TCN
                models.append(TCN(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "TFT":
                from neuralforecast.models import TFT
                models.append(TFT(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "PatchTST":
                from neuralforecast.models import PatchTST
                models.append(PatchTST(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "TimesNet":
                from neuralforecast.models import TimesNet
                models.append(TimesNet(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "DeepAR":
                from neuralforecast.models import DeepAR
                models.append(DeepAR(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "FedFormer":
                from neuralforecast.models import FEDformer
                models.append(FEDformer(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "Informer":
                from neuralforecast.models import Informer
                models.append(Informer(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            elif method == "Autoformer":
                from neuralforecast.models import Autoformer
                models.append(Autoformer(
                    h=horizon,
                    input_size=2 * horizon,
                    max_steps=max_steps,
                    scaler_type="standard",
                ))
            else:
                logger.warning(f"Unknown neural method: {method}")
        except ImportError as e:
            logger.warning(f"Could not import {method}: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize {method}: {e}")

    return models


def run_neural_forecast(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    max_steps: int = 100,
    confidence_levels: List[int] = None,
) -> Dict[str, Any]:
    """Run NeuralForecast models on the data."""
    from neuralforecast import NeuralForecast

    if confidence_levels is None:
        confidence_levels = [80, 90, 95]

    models = get_neural_models(methods, horizon=horizon, max_steps=max_steps)

    if not models:
        return {"error": "No valid neural models could be initialized"}

    nf = NeuralForecast(
        models=models,
        freq=frequency,
    )

    # Fit and predict
    nf.fit(df=df)

    try:
        forecasts = nf.predict()
        # Try prediction intervals
        try:
            insample = nf.predict_insample()
        except Exception:
            insample = None
    except Exception as e:
        return {"error": f"Neural forecast prediction failed: {str(e)}"}

    forecasts = forecasts.reset_index()

    result = {
        "forecasts": forecasts.to_dict(orient="records"),
        "columns": list(forecasts.columns),
        "methods_used": [type(m).__name__ for m in models],
        "horizon": horizon,
        "frequency": frequency,
        "max_steps": max_steps,
        "confidence_levels": confidence_levels,
    }

    return result


def run_neural_cross_validation(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    n_windows: int = 2,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Run cross-validation with NeuralForecast models."""
    from neuralforecast import NeuralForecast

    models = get_neural_models(methods, horizon=horizon, max_steps=max_steps)

    if not models:
        return {"error": "No valid neural models could be initialized"}

    nf = NeuralForecast(
        models=models,
        freq=frequency,
    )

    cv_results = nf.cross_validation(
        df=df,
        n_windows=n_windows,
    )
    cv_results = cv_results.reset_index()

    # Compute metrics
    model_names = [type(m).__name__ for m in models]
    from timeseries_api.services.stats_forecast_service import compute_cv_metrics
    metrics = compute_cv_metrics(cv_results, model_names)

    return {
        "cv_results": cv_results.to_dict(orient="records"),
        "metrics": metrics,
        "n_windows": n_windows,
        "methods_used": model_names,
    }
