"""Pydantic models for the Time Series Analytics API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class FrequencyEnum(str, Enum):
    """Supported time series frequencies."""
    HOURLY = "h"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    MINUTELY = "T"
    SECONDLY = "S"
    BUSINESS_DAILY = "B"


class ForecastMethod(str, Enum):
    """Available forecasting methods."""
    # StatsForecast models
    AUTO_ARIMA = "AutoARIMA"
    AUTO_ETS = "AutoETS"
    AUTO_THETA = "AutoTheta"
    AUTO_CES = "AutoCES"
    MSTL = "MSTL"
    SEASONAL_NAIVE = "SeasonalNaive"
    NAIVE = "Naive"
    HISTORIC_AVERAGE = "HistoricAverage"
    WINDOW_AVERAGE = "WindowAverage"
    SEASONAL_WINDOW_AVERAGE = "SeasonalWindowAverage"
    ADIDA = "ADIDA"
    CROSTON_CLASSIC = "CrostonClassic"
    CROSTON_OPTIMIZED = "CrostonOptimized"
    CROSTON_SBA = "CrostonSBA"
    IMAPA = "IMAPA"
    TSB = "TSB"
    HOLT_WINTERS = "HoltWinters"
    # MLForecast models
    LIGHTGBM = "LightGBM"
    XGBOOST = "XGBoost"
    LINEAR_REGRESSION = "LinearRegression"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    RANDOM_FOREST = "RandomForest"
    KNN = "KNN"
    # NeuralForecast models
    NBEATS = "NBEATS"
    NHITS = "NHITS"
    LSTM = "LSTM"
    GRU = "GRU"
    TCN = "TCN"
    TFT = "TFT"
    PATCH_TST = "PatchTST"
    TIMESNET = "TimesNet"
    DEEPAR = "DeepAR"
    FEDFORMER = "FedFormer"
    INFORMER = "Informer"
    AUTOFORMER = "Autoformer"


class ConfidenceLevel(str, Enum):
    """Confidence interval levels."""
    LOW = "80"
    MEDIUM = "90"
    HIGH = "95"
    VERY_HIGH = "99"


class TimeSeriesUpload(BaseModel):
    """Model for time series data upload."""
    ds_column: str = Field(default="ds", description="Name of the datetime column")
    y_column: str = Field(default="y", description="Name of the target value column")
    unique_id_column: Optional[str] = Field(default="unique_id", description="Column for series identifier (batch)")
    frequency: Optional[str] = Field(default=None, description="Time series frequency (auto-detected if not provided)")


class ForecastRequest(BaseModel):
    """Model for forecast requests."""
    horizon: int = Field(default=12, ge=1, le=365, description="Forecast horizon (number of periods)")
    methods: List[str] = Field(
        default=["AutoARIMA", "AutoETS", "SeasonalNaive"],
        description="Forecasting methods to use"
    )
    confidence_levels: List[int] = Field(
        default=[80, 90, 95],
        description="Confidence interval levels"
    )
    frequency: Optional[str] = Field(default=None, description="Override frequency")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")


class BatchForecastRequest(BaseModel):
    """Model for batch forecast requests with multiple series."""
    horizon: int = Field(default=12, ge=1, le=365, description="Forecast horizon")
    methods: List[str] = Field(
        default=["AutoARIMA", "AutoETS", "SeasonalNaive"],
        description="Forecasting methods to use"
    )
    confidence_levels: List[int] = Field(
        default=[80, 90, 95],
        description="Confidence interval levels"
    )
    frequency: Optional[str] = Field(default=None, description="Override frequency")
    series_ids: Optional[List[str]] = Field(default=None, description="Specific series IDs to forecast")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")


class AnalyticsRequest(BaseModel):
    """Model for full analytics requests."""
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")
    frequency: Optional[str] = Field(default=None, description="Override frequency")
    forecast_horizon: int = Field(default=12, ge=1, description="Forecast horizon for analytics")
    include_forecasting: bool = Field(default=True, description="Include forecasting in analytics")
    include_decomposition: bool = Field(default=True, description="Include decomposition")
    include_stationarity_tests: bool = Field(default=True, description="Include stationarity tests")
    include_cross_validation: bool = Field(default=True, description="Include cross-validation")


class MLForecastRequest(BaseModel):
    """Model for ML-based forecast requests."""
    horizon: int = Field(default=12, ge=1, le=365, description="Forecast horizon")
    methods: List[str] = Field(
        default=["LightGBM", "LinearRegression"],
        description="ML methods to use"
    )
    lags: List[int] = Field(default=[1, 2, 3, 4, 5, 6, 7, 12, 24], description="Lag features")
    confidence_levels: List[int] = Field(default=[80, 90, 95], description="Confidence levels")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")
    frequency: Optional[str] = Field(default=None, description="Override frequency")


class NeuralForecastRequest(BaseModel):
    """Model for neural forecast requests."""
    horizon: int = Field(default=12, ge=1, le=365, description="Forecast horizon")
    methods: List[str] = Field(
        default=["NBEATS", "NHITS"],
        description="Neural methods to use"
    )
    max_steps: int = Field(default=100, ge=10, le=10000, description="Max training steps")
    confidence_levels: List[int] = Field(default=[80, 90, 95], description="Confidence levels")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")
    frequency: Optional[str] = Field(default=None, description="Override frequency")


class HierarchicalRequest(BaseModel):
    """Model for hierarchical forecast reconciliation."""
    horizon: int = Field(default=12, ge=1, description="Forecast horizon")
    methods: List[str] = Field(default=["AutoARIMA"], description="Base forecasting methods")
    reconciliation_methods: List[str] = Field(
        default=["BottomUp", "MinTrace"],
        description="Reconciliation methods"
    )
    hierarchy_levels: List[str] = Field(description="Column names defining hierarchy levels")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")
    frequency: Optional[str] = Field(default=None, description="Override frequency")


class CrossValidationRequest(BaseModel):
    """Model for cross-validation requests."""
    horizon: int = Field(default=12, ge=1, description="Forecast horizon")
    n_windows: int = Field(default=3, ge=1, le=20, description="Number of CV windows")
    step_size: Optional[int] = Field(default=None, description="Step size between windows")
    methods: List[str] = Field(default=["AutoARIMA", "AutoETS"], description="Methods to evaluate")
    ds_column: str = Field(default="ds", description="Datetime column name")
    y_column: str = Field(default="y", description="Target column name")
    unique_id_column: str = Field(default="unique_id", description="Series ID column name")
    frequency: Optional[str] = Field(default=None, description="Override frequency")


class PlotRequest(BaseModel):
    """Model for plot generation requests."""
    plot_type: str = Field(default="forecast", description="Type of plot")
    series_ids: Optional[List[str]] = Field(default=None, description="Specific series to plot")
    width: int = Field(default=1200, description="Plot width")
    height: int = Field(default=600, description="Plot height")
    theme: str = Field(default="plotly_white", description="Plotly theme")


class ForecastResponse(BaseModel):
    """Response model for forecasts."""
    forecasts: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any]


class AnalyticsResponse(BaseModel):
    """Response model for full analytics."""
    summary: Dict[str, Any]
    decomposition: Optional[Dict[str, Any]] = None
    stationarity_tests: Optional[Dict[str, Any]] = None
    forecasts: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    plots: Dict[str, str]
    metadata: Dict[str, Any]
