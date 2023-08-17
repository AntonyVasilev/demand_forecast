from typing import Dict, Tuple, Optional
import pandas as pd


def add_features(
    df: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> None:
    """
    Add rolling features to the DataFrame based on the specified aggregations.
    For each sku_id, the features are computed as the aggregations of the last N-days.
    Current date is always included into rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the feature to. Changes are applied inplace.
    features : Dict[str, Tuple[str, int, str, Optional[int]]]
        Dictionary with the following structure:
        {
            "feature_name": ("agg_col", "days", "aggregation_function", "quantile"),
            ...
        }
        where:
            - feature_name: name of the feature to add
            - agg_col: name of the column to aggregate
            - int: number of days to include into rolling window
            - aggregation_function: one of the following: "quantile", "avg"
            - int: quantile to compute (only for "quantile" aggregation_function)

    Raises
    ------
    ValueError
        If aggregation_function is not one of the following: "quantile", "avg"
    """
    for feature_name, (agg_col, days, agg_func, quantile) in features.items():
        if agg_func == "quantile":
            df[feature_name] = (
                df.groupby("sku_id")[agg_col]
                .rolling(window=days)
                .quantile(quantile / 100)
                .reset_index(level=0, drop=True)
            )
        elif agg_func == "avg":
            df[feature_name] = (
                df.groupby("sku_id")[agg_col]
                .rolling(window=days)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")


def add_targets(df: pd.DataFrame, targets: Dict[str, Tuple[str, int]]) -> None:
    """
    Add targets to the DataFrame based on the specified aggregations.
    For each sku_id, the targets is computed as the aggregations of the next N-days.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to add the target to. Changes are applied inplace.
    targets : Dict[str, Tuple[str, int]]
        Dictionary with the following structure:
        {
            "target_name": ("agg_col", "days"),
            ...
        }
        where:
            - target_name: name of the target to add
            - agg_col: name of the column to aggregate
            - days: number of next days to include into rolling window
            (current date is always excluded from the rolling window)
    """
    for feature_name, (agg_col, days) in targets.items():
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=days)
        df[feature_name] = (
            df.groupby("sku_id")[agg_col]
            .shift(-1)
            .rolling(window=indexer)
            .sum()
            .reset_index(level=0, drop=True)
        )


if __name__ == '__main__':
    sales_df = pd.read_csv('data/sales.csv')

    FEATURES = {
        "qty_7d_avg": ("qty", 7, "avg", None),
        "qty_7d_q10": ("qty", 7, "quantile", 10),
        "qty_7d_q50": ("qty", 7, "quantile", 50),
        "qty_7d_q90": ("qty", 7, "quantile", 90),
        "qty_14d_avg": ("qty", 14, "avg", None),
        "qty_14d_q10": ("qty", 14, "quantile", 10),
        "qty_14d_q50": ("qty", 14, "quantile", 50),
        "qty_14d_q90": ("qty", 14, "quantile", 90),
        "qty_21d_avg": ("qty", 21, "avg", None),
        "qty_21d_q10": ("qty", 21, "quantile", 10),
        "qty_21d_q50": ("qty", 21, "quantile", 50),
        "qty_21d_q90": ("qty", 21, "quantile", 90),
    }

    TARGETS = {
        "next_7d": ("qty", 7),
        "next_14d": ("qty", 14),
        "next_21d": ("qty", 21),
    }

    add_features(df=sales_df, features=FEATURES)
    add_targets(df=sales_df, targets=TARGETS)

    sales_df.to_csv('features.csv', index=False)
