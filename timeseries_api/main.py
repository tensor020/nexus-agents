"""Main FastAPI application for Nixtla Time Series Analytics API.

This API provides end-to-end automated time series analytics and forecasting
using the complete Nixtla ecosystem:

- **StatsForecast**: Statistical models (AutoARIMA, AutoETS, AutoTheta, MSTL, etc.)
- **MLForecast**: Machine Learning models (LightGBM, XGBoost, RandomForest, etc.)
- **NeuralForecast**: Deep Learning models (NBEATS, NHITS, LSTM, TFT, PatchTST, etc.)
- **HierarchicalForecast**: Hierarchical reconciliation (BottomUp, MinTrace, etc.)
- **UtilsForecast**: Evaluation metrics and utilities
- **DatasetsForecast**: Sample datasets for testing
- **CoreForecast**: Core forecasting engine

Features:
- Automated frequency detection
- Single and batch forecasting
- Confidence intervals at multiple levels
- Full analytics pipeline (decomposition, stationarity, seasonality)
- Cross-validation and model comparison
- Interactive Plotly visualizations
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from timeseries_api.routers import analytics, forecasting, visualization, datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Nixtla Time Series Analytics API",
    description="""
## End-to-End Automated Time Series Analytics & Forecasting

Powered by the complete **Nixtla** open-source ecosystem.

### Libraries Integrated

| Library | Purpose |
|---------|---------|
| **StatsForecast** | Statistical models (AutoARIMA, AutoETS, AutoTheta, MSTL, HoltWinters, etc.) |
| **MLForecast** | ML models (LightGBM, XGBoost, RandomForest, Ridge, Lasso, KNN) |
| **NeuralForecast** | Deep Learning (NBEATS, NHITS, LSTM, GRU, TFT, PatchTST, TimesNet, DeepAR) |
| **HierarchicalForecast** | Reconciliation (BottomUp, TopDown, MinTrace, ERM) |
| **UtilsForecast** | Evaluation metrics (MAE, MSE, RMSE, MAPE, sMAPE) |
| **DatasetsForecast** | Sample datasets (M4 competition data) |
| **CoreForecast** | Core forecasting engine |

### Key Features

- **Auto-detection**: Automatic frequency and seasonality detection
- **Single & Batch**: Forecast single series or hundreds simultaneously
- **Confidence Intervals**: Multiple levels (80%, 90%, 95%, 99%)
- **Full Analytics**: Decomposition, stationarity tests, ACF/PACF, cross-validation
- **Plotly Visualizations**: Interactive charts for all analytics
- **Model Comparison**: Cross-validation with metric comparison across models
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analytics.router)
app.include_router(forecasting.router)
app.include_router(visualization.router)
app.include_router(datasets.router)


@app.get("/")
async def root():
    """API root - health check and info."""
    return {
        "name": "Nixtla Time Series Analytics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "libraries": {
            "statsforecast": "Statistical forecasting (AutoARIMA, AutoETS, AutoTheta, AutoCES, MSTL, SeasonalNaive, Naive, HistoricAverage, WindowAverage, SeasonalWindowAverage, ADIDA, CrostonClassic, CrostonOptimized, CrostonSBA, IMAPA, TSB, HoltWinters)",
            "mlforecast": "ML forecasting (LightGBM, XGBoost, LinearRegression, Ridge, Lasso, RandomForest, KNN)",
            "neuralforecast": "Neural forecasting (NBEATS, NHITS, LSTM, GRU, TCN, TFT, PatchTST, TimesNet, DeepAR, FedFormer, Informer, Autoformer)",
            "hierarchicalforecast": "Hierarchical reconciliation (BottomUp, TopDown, MinTrace, ERM)",
            "utilsforecast": "Evaluation metrics and utilities",
            "datasetsforecast": "Sample datasets (M4 competition)",
            "coreforecast": "Core forecasting engine",
        },
        "endpoints": {
            "analytics": {
                "/analytics/full": "Complete end-to-end time series analytics",
                "/analytics/summary": "Data summary statistics",
                "/analytics/stationarity": "Stationarity tests (ADF, KPSS)",
                "/analytics/seasonality": "Seasonality detection",
                "/analytics/acf-pacf": "ACF and PACF analysis",
            },
            "forecasting": {
                "/forecast/stats": "StatsForecast models",
                "/forecast/stats/cross-validation": "StatsForecast cross-validation",
                "/forecast/ml": "MLForecast models",
                "/forecast/ml/cross-validation": "MLForecast cross-validation",
                "/forecast/neural": "NeuralForecast models",
                "/forecast/neural/cross-validation": "NeuralForecast cross-validation",
                "/forecast/hierarchical": "Hierarchical forecast with reconciliation",
                "/forecast/batch": "Batch forecasting for multiple series",
            },
            "visualization": {
                "/plots/time-series": "Time series plot",
                "/plots/decomposition": "Decomposition plot",
                "/plots/acf-pacf": "ACF/PACF plots",
                "/plots/seasonality": "Seasonal pattern plots",
                "/plots/distribution": "Distribution plots",
            },
            "datasets": {
                "/datasets/sample": "Generate sample data",
                "/datasets/nixtla-datasets": "List available datasets",
                "/datasets/load/{name}": "Load a specific dataset",
            },
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    lib_status = {}

    try:
        import statsforecast
        lib_status["statsforecast"] = "available"
    except ImportError:
        lib_status["statsforecast"] = "not installed"

    try:
        import mlforecast
        lib_status["mlforecast"] = "available"
    except ImportError:
        lib_status["mlforecast"] = "not installed"

    try:
        import neuralforecast
        lib_status["neuralforecast"] = "available"
    except ImportError:
        lib_status["neuralforecast"] = "not installed"

    try:
        import hierarchicalforecast
        lib_status["hierarchicalforecast"] = "available"
    except ImportError:
        lib_status["hierarchicalforecast"] = "not installed"

    try:
        import utilsforecast
        lib_status["utilsforecast"] = "available"
    except ImportError:
        lib_status["utilsforecast"] = "not installed"

    try:
        import datasetsforecast
        lib_status["datasetsforecast"] = "available"
    except ImportError:
        lib_status["datasetsforecast"] = "not installed"

    try:
        import coreforecast
        lib_status["coreforecast"] = "available"
    except ImportError:
        lib_status["coreforecast"] = "not installed"

    try:
        import plotly
        lib_status["plotly"] = "available"
    except ImportError:
        lib_status["plotly"] = "not installed"

    return {
        "status": "healthy",
        "libraries": lib_status,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
