from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils import resample

from model import MultiTargetModel, split_train_test


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    loss = (quantile * np.maximum(y_true - y_pred, 0) +
            (1 - quantile) * np.maximum(y_pred - y_true, 0)).mean()
    return loss


def evaluate_model(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    horizons: List[int] = [7, 14, 21],
) -> pd.DataFrame:
    """Evaluate model on data.

    Parameters
    ----------
    df_true : pd.DataFrame
        True values.
    df_pred : pd.DataFrame
        Predicted values.
    quantiles : List[float], optional
        Quantiles to evaluate on, by default [0.1, 0.5, 0.9].
    horizons : List[int], optional
        Horizons to evaluate on, by default [7, 14, 21].

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    losses = {}

    for quantile in quantiles:
        for horizon in horizons:
            true = df_true[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile*100)}"].values
            loss = quantile_loss(true, pred, quantile)

            losses[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]  # type: ignore

    return losses


def week_missed_profits(
    df: pd.DataFrame,
    sales_col: str,
    forecast_col: str,
    date_col: str = "day",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """
    data_ = df.copy()
    data_[date_col] = pd.to_datetime(data_[date_col])

    # Calculate revenue
    data_['revenue'] = data_[sales_col] * data_[price_col]

    # Calculate missed profits
    data_['missed_sales'] = np.where(data_[sales_col] < data_[forecast_col],
                                     data_[forecast_col] - data_[sales_col],
                                     0)
    data_['missed_profits'] = data_['missed_sales'] * data_[price_col]
    result_df = data_.groupby(pd.Grouper(key=date_col, freq='W'))[['revenue', 'missed_profits']].sum().reset_index()
    result_df['missed_profits'] = result_df['missed_profits'].astype(int)
    return result_df


def missed_profits_ci(
    df: pd.DataFrame,
    missed_profits_col: str,
    confidence_level: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates
    the 95% confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.

    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval,
        by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average
        missed profits with its CI, and the second is the relative average missed
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """
    alpha = 1 - confidence_level

    df_ = df.copy()
    avg_profits = df_['revenue'].mean()
    avg_missed_profits = df_[missed_profits_col].mean()

    # Make a bootstrap & calculate a confidence interval
    bootstrap_missed_profits = []
    for _ in range(n_bootstraps):
        mp = resample(df_[missed_profits_col])
        bootstrap_missed_profits.append(mp.mean())
    lcb, ucb = np.quantile(bootstrap_missed_profits, [alpha / 2, 1 - alpha / 2])

    return (
        (avg_missed_profits, (lcb, ucb)),
        (avg_missed_profits / avg_profits,
         (lcb / avg_profits, ucb / avg_profits))
    )


if __name__ == '__main__':
    sales_data = pd.read_csv('data/features.csv')
    train_, test_ = split_train_test(sales_data, test_days=550)
    model = MultiTargetModel(
        features=[
            "price",
            "qty",
            "qty_7d_avg",
            "qty_7d_q10",
            "qty_7d_q50",
            "qty_7d_q90",
            "qty_14d_avg",
            "qty_14d_q10",
            "qty_14d_q50",
            "qty_14d_q90",
            "qty_21d_avg",
            "qty_21d_q10",
            "qty_21d_q50",
            "qty_21d_q90",
        ],
        horizons=[7],
        quantiles=[0.5]
    )
    model.fit(train_, verbose=True)
    predictions_ = model.predict(test_)
    losses_ = evaluate_model(test_, predictions_)
    union_data = test_.merge(predictions_, on=['sku_id', 'day'])
    missed_profits_df = week_missed_profits(df=union_data, sales_col='qty', forecast_col='pred_7d_q50')
    missed_profits_ci_ = missed_profits_ci(df=missed_profits_df, missed_profits_col='missed_profits')
