"""Plotly visualization service for time series analytics."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


def plot_time_series(
    df: pd.DataFrame,
    series_ids: Optional[List[str]] = None,
    title: str = "Time Series Data",
    width: int = 1200,
    height: int = 600,
    theme: str = "plotly_white",
) -> str:
    """Plot raw time series data."""
    if series_ids is None:
        series_ids = df["unique_id"].unique().tolist()[:20]  # Limit to 20

    fig = go.Figure()

    for uid in series_ids:
        series = df[df["unique_id"] == uid].sort_values("ds")
        fig.add_trace(go.Scatter(
            x=series["ds"],
            y=series["y"],
            mode="lines",
            name=str(uid),
            hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template=theme,
        width=width,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )

    return fig.to_json()


def plot_forecast(
    historical: pd.DataFrame,
    forecasts: pd.DataFrame,
    methods: List[str],
    confidence_levels: List[int] = None,
    series_ids: Optional[List[str]] = None,
    title: str = "Forecast Results",
    width: int = 1200,
    height: int = 600,
    theme: str = "plotly_white",
) -> str:
    """Plot forecast results with confidence intervals."""
    if confidence_levels is None:
        confidence_levels = [90]

    if series_ids is None:
        series_ids = historical["unique_id"].unique().tolist()[:5]

    # Create subplots for each series
    n_series = len(series_ids)
    fig = make_subplots(
        rows=n_series,
        cols=1,
        subplot_titles=[f"Series: {uid}" for uid in series_ids],
        vertical_spacing=0.08,
    )

    colors = px.colors.qualitative.Set2

    for i, uid in enumerate(series_ids, 1):
        # Historical data
        hist = historical[historical["unique_id"] == uid].sort_values("ds")
        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=hist["y"],
                mode="lines",
                name="Historical" if i == 1 else None,
                line=dict(color="black", width=2),
                showlegend=(i == 1),
                legendgroup="historical",
                hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra>Historical</extra>",
            ),
            row=i,
            col=1,
        )

        # Forecasts for each method
        if "unique_id" in forecasts.columns:
            fc = forecasts[forecasts["unique_id"] == uid]
        else:
            fc = forecasts

        for j, method in enumerate(methods):
            if method not in fc.columns:
                continue

            color = colors[j % len(colors)]

            # Main forecast line
            fig.add_trace(
                go.Scatter(
                    x=fc["ds"],
                    y=fc[method],
                    mode="lines",
                    name=method if i == 1 else None,
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=(i == 1),
                    legendgroup=method,
                    hovertemplate=f"Date: %{{x}}<br>{method}: %{{y:.2f}}<extra></extra>",
                ),
                row=i,
                col=1,
            )

            # Confidence intervals
            for level in sorted(confidence_levels):
                lo_col = f"{method}-lo-{level}"
                hi_col = f"{method}-hi-{level}"

                if lo_col in fc.columns and hi_col in fc.columns:
                    opacity = 0.1 + (0.15 * (1 - confidence_levels.index(level) / max(len(confidence_levels), 1)))

                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                            y=pd.concat([fc[hi_col], fc[lo_col][::-1]]),
                            fill="toself",
                            fillcolor=f"rgba({_hex_to_rgb(color)},{opacity})",
                            line=dict(color="rgba(0,0,0,0)"),
                            name=f"{method} CI {level}%" if i == 1 else None,
                            showlegend=(i == 1),
                            legendgroup=f"{method}_ci_{level}",
                            hoverinfo="skip",
                        ),
                        row=i,
                        col=1,
                    )

    fig.update_layout(
        title=title,
        template=theme,
        width=width,
        height=max(height, n_series * 350),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )

    return fig.to_json()


def plot_decomposition(
    df: pd.DataFrame,
    series_id: str = None,
    frequency: str = "M",
    width: int = 1200,
    height: int = 800,
    theme: str = "plotly_white",
) -> str:
    """Plot time series decomposition (trend, seasonal, residual)."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from timeseries_api.services.stats_forecast_service import get_season_length

    if series_id is None:
        series_id = df["unique_id"].unique()[0]

    series = df[df["unique_id"] == series_id].sort_values("ds").set_index("ds")["y"]
    period = get_season_length(frequency)

    if len(series) < 2 * period:
        period = max(2, len(series) // 4)

    try:
        decomp = seasonal_decompose(series, period=period, extrapolate_trend="freq")
    except Exception:
        decomp = seasonal_decompose(series, period=2, extrapolate_trend="freq")

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(x=series.index, y=decomp.observed, mode="lines", name="Observed", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.trend, mode="lines", name="Trend", line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.seasonal, mode="lines", name="Seasonal", line=dict(color="#2ca02c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.resid, mode="lines+markers", name="Residual", line=dict(color="#d62728")), row=4, col=1)

    fig.update_layout(
        title=f"Time Series Decomposition - {series_id}",
        template=theme,
        width=width,
        height=height,
        showlegend=True,
    )

    return fig.to_json()


def plot_acf_pacf(
    acf_values: List[float],
    pacf_values: List[float],
    confidence_bound: float,
    series_id: str = "series",
    width: int = 1200,
    height: int = 500,
    theme: str = "plotly_white",
) -> str:
    """Plot ACF and PACF."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])

    nlags = len(acf_values)
    lags = list(range(nlags))

    # ACF
    for lag, val in zip(lags, acf_values):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines", line=dict(color="#1f77b4", width=2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=lags, y=acf_values, mode="markers", marker=dict(color="#1f77b4", size=6), name="ACF"), row=1, col=1)
    fig.add_hline(y=confidence_bound, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-confidence_bound, line_dash="dash", line_color="red", row=1, col=1)

    # PACF
    nlags_pacf = len(pacf_values)
    lags_pacf = list(range(nlags_pacf))
    for lag, val in zip(lags_pacf, pacf_values):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines", line=dict(color="#ff7f0e", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=lags_pacf, y=pacf_values, mode="markers", marker=dict(color="#ff7f0e", size=6), name="PACF"), row=1, col=2)
    fig.add_hline(y=confidence_bound, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-confidence_bound, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(
        title=f"ACF & PACF - {series_id}",
        template=theme,
        width=width,
        height=height,
    )

    return fig.to_json()


def plot_seasonality(
    df: pd.DataFrame,
    series_id: str = None,
    width: int = 1200,
    height: int = 500,
    theme: str = "plotly_white",
) -> str:
    """Plot seasonal patterns (box plots by period)."""
    if series_id is None:
        series_id = df["unique_id"].unique()[0]

    series = df[df["unique_id"] == series_id].copy()
    series["ds"] = pd.to_datetime(series["ds"])
    series["month"] = series["ds"].dt.month
    series["dayofweek"] = series["ds"].dt.dayofweek
    series["hour"] = series["ds"].dt.hour

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Monthly Pattern", "Day of Week Pattern"])

    # Monthly
    fig.add_trace(
        go.Box(x=series["month"], y=series["y"], name="Monthly", marker_color="#1f77b4"),
        row=1, col=1,
    )

    # Day of week
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    series["day_name"] = series["dayofweek"].map(lambda x: day_names[x] if x < 7 else str(x))
    fig.add_trace(
        go.Box(x=series["day_name"], y=series["y"], name="Day of Week", marker_color="#ff7f0e"),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Seasonal Patterns - {series_id}",
        template=theme,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig.to_json()


def plot_distribution(
    df: pd.DataFrame,
    series_ids: Optional[List[str]] = None,
    width: int = 1200,
    height: int = 500,
    theme: str = "plotly_white",
) -> str:
    """Plot value distribution (histogram + box plot)."""
    if series_ids is None:
        series_ids = df["unique_id"].unique().tolist()[:5]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])

    for uid in series_ids:
        series = df[df["unique_id"] == uid]
        fig.add_trace(
            go.Histogram(x=series["y"], name=str(uid), opacity=0.7, nbinsx=50),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=series["y"], name=str(uid)),
            row=1, col=2,
        )

    fig.update_layout(
        title="Value Distribution",
        template=theme,
        width=width,
        height=height,
        barmode="overlay",
    )

    return fig.to_json()


def plot_cross_validation(
    cv_results: pd.DataFrame,
    methods: List[str],
    metrics: Dict[str, Any],
    width: int = 1200,
    height: int = 600,
    theme: str = "plotly_white",
) -> str:
    """Plot cross-validation results."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Actual vs Predicted", "Error Metrics by Model"],
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
    )

    # Actual vs Predicted scatter
    for method in methods:
        if method not in cv_results.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=cv_results["y"],
                y=cv_results[method],
                mode="markers",
                name=method,
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # Add perfect prediction line
    y_min, y_max = cv_results["y"].min(), cv_results["y"].max()
    fig.add_trace(
        go.Scatter(
            x=[y_min, y_max],
            y=[y_min, y_max],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash", color="red"),
        ),
        row=1, col=1,
    )

    # Metrics bar chart
    metric_names = ["mae", "rmse", "mape"]
    for metric_name in metric_names:
        values = []
        model_names = []
        for method in methods:
            if method in metrics and metric_name in metrics[method]:
                model_names.append(method)
                values.append(metrics[method][metric_name])

        if values:
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric_name.upper()),
                row=1, col=2,
            )

    fig.update_layout(
        title="Cross-Validation Results",
        template=theme,
        width=width,
        height=height,
    )
    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)

    return fig.to_json()


def plot_residuals(
    cv_results: pd.DataFrame,
    method: str,
    width: int = 1200,
    height: int = 600,
    theme: str = "plotly_white",
) -> str:
    """Plot residual analysis."""
    if method not in cv_results.columns:
        return json.dumps({"error": f"Method {method} not in results"})

    residuals = cv_results["y"] - cv_results[method]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residuals Over Time",
            "Residual Distribution",
            "Q-Q Plot",
            "Residuals vs Fitted",
        ],
    )

    # Residuals over time
    fig.add_trace(
        go.Scatter(x=cv_results["ds"], y=residuals, mode="markers+lines", name="Residuals", line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Distribution
    fig.add_trace(
        go.Histogram(x=residuals, name="Distribution", nbinsx=30, marker_color="#2ca02c"),
        row=1, col=2,
    )

    # Q-Q Plot
    sorted_residuals = np.sort(residuals.dropna())
    n = len(sorted_residuals)
    theoretical = np.sort(np.random.normal(0, residuals.std(), n))
    fig.add_trace(
        go.Scatter(x=theoretical, y=sorted_residuals, mode="markers", name="Q-Q", marker=dict(color="#ff7f0e")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[theoretical.min(), theoretical.max()], y=[theoretical.min(), theoretical.max()],
                   mode="lines", name="Reference", line=dict(dash="dash", color="red")),
        row=2, col=1,
    )

    # Residuals vs Fitted
    fig.add_trace(
        go.Scatter(x=cv_results[method], y=residuals, mode="markers", name="Resid vs Fitted", marker=dict(color="#d62728")),
        row=2, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(
        title=f"Residual Analysis - {method}",
        template=theme,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig.to_json()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    width: int = 1000,
    height: int = 500,
    theme: str = "plotly_white",
) -> str:
    """Plot model comparison based on metrics."""
    metric_names = ["mae", "rmse", "mape", "smape"]

    fig = make_subplots(
        rows=1, cols=len(metric_names),
        subplot_titles=[m.upper() for m in metric_names],
    )

    for i, metric_name in enumerate(metric_names, 1):
        model_names = []
        values = []
        for model, model_metrics in metrics.items():
            if metric_name in model_metrics:
                model_names.append(model)
                values.append(model_metrics[metric_name])

        if values:
            colors = ["#2ecc71" if v == min(values) else "#3498db" for v in values]
            fig.add_trace(
                go.Bar(x=model_names, y=values, marker_color=colors, showlegend=False),
                row=1, col=i,
            )

    fig.update_layout(
        title="Model Comparison",
        template=theme,
        width=width,
        height=height,
    )

    return fig.to_json()


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"{r},{g},{b}"
    return "100,100,100"
