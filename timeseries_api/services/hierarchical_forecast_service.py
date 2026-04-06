"""HierarchicalForecast service - Hierarchical reconciliation from Nixtla."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def build_hierarchy_spec(
    df: pd.DataFrame,
    hierarchy_levels: List[str],
    unique_id_column: str = "unique_id",
) -> Dict[str, Any]:
    """Build hierarchy specification from data columns.

    Creates an S (summing) matrix for hierarchical reconciliation.
    """
    # Get unique combinations for each level
    all_ids = df[unique_id_column].unique().tolist()

    # Build hierarchy tags from the data
    tags = {}
    for level in hierarchy_levels:
        if level in df.columns:
            tags[level] = df.groupby(level)[unique_id_column].apply(list).to_dict()

    return {
        "all_ids": all_ids,
        "hierarchy_levels": hierarchy_levels,
        "tags": tags,
    }


def run_hierarchical_forecast(
    df: pd.DataFrame,
    horizon: int,
    methods: List[str],
    frequency: str,
    hierarchy_levels: List[str],
    reconciliation_methods: List[str] = None,
    unique_id_column: str = "unique_id",
) -> Dict[str, Any]:
    """Run hierarchical forecast with reconciliation."""
    try:
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.methods import (
            BottomUp,
            TopDown,
            MiddleOut,
            MinTrace,
            ERM,
        )
    except ImportError as e:
        return {"error": f"HierarchicalForecast not available: {str(e)}"}

    if reconciliation_methods is None:
        reconciliation_methods = ["BottomUp", "MinTrace"]

    # First generate base forecasts using StatsForecast
    from timeseries_api.services.stats_forecast_service import run_stats_forecast

    base_result = run_stats_forecast(
        df=df,
        horizon=horizon,
        methods=methods,
        frequency=frequency,
        confidence_levels=[90],
    )

    if "error" in base_result:
        return base_result

    base_forecasts = pd.DataFrame(base_result["forecasts"])

    # Build reconciliation models
    recon_models = {}
    recon_map = {
        "BottomUp": BottomUp,
        "TopDown": lambda: TopDown(method="average_proportions"),
        "MinTrace": lambda: MinTrace(method="mint_shrink"),
        "ERM": ERM,
    }

    for method_name in reconciliation_methods:
        if method_name in recon_map:
            try:
                model = recon_map[method_name]
                recon_models[method_name] = model() if callable(model) else model
            except Exception as e:
                logger.warning(f"Could not init reconciliation method {method_name}: {e}")

    if not recon_models:
        return {
            "base_forecasts": base_result,
            "reconciliation": {"error": "No valid reconciliation methods"},
        }

    # Build S matrix from hierarchy
    try:
        spec = build_hierarchy_spec(df, hierarchy_levels, unique_id_column)

        # Use tags-based approach
        tags_dict = {}
        for level, groups in spec["tags"].items():
            level_tags = []
            for uid in spec["all_ids"]:
                for group_name, group_ids in groups.items():
                    if uid in group_ids:
                        level_tags.append(group_name)
                        break
                else:
                    level_tags.append("unknown")
            tags_dict[level] = np.array(level_tags)

        # Build S matrix
        from hierarchicalforecast.utils import aggregate
        Y_df, S_df, tags = aggregate(df=df, spec=list(tags_dict.values()))

        # Run hierarchical reconciliation
        hrec = HierarchicalReconciliation(reconcilers=list(recon_models.values()))

        reconciled = hrec.reconcile(
            Y_hat_df=base_forecasts,
            Y_df=df,
            S=S_df,
            tags=tags,
        )
        reconciled = reconciled.reset_index()

        return {
            "base_forecasts": base_result,
            "reconciled_forecasts": reconciled.to_dict(orient="records"),
            "reconciliation_methods": list(recon_models.keys()),
            "hierarchy_spec": {
                "levels": hierarchy_levels,
                "n_bottom_series": len(spec["all_ids"]),
            },
        }
    except Exception as e:
        logger.error(f"Hierarchical reconciliation failed: {e}")
        return {
            "base_forecasts": base_result,
            "reconciliation": {"error": str(e)},
            "hierarchy_spec": {"levels": hierarchy_levels},
        }
