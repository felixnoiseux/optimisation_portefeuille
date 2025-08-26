from __future__ import annotations

import numpy as np
import pandas as pd


def describe_returns(returns: pd.DataFrame, trading_days: int = 252, rf_annual: float = 0.02) -> pd.DataFrame:
    mean_daily = returns.mean()
    vol_daily = returns.std()
    mean_ann = (1 + mean_daily) ** trading_days - 1
    vol_ann = vol_daily * np.sqrt(trading_days)
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    sharpe = (mean_daily - rf_daily) / vol_daily.replace(0.0, np.nan)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan) * np.sqrt(trading_days)

    stats = pd.DataFrame({
        'mean_ann': mean_ann,
        'vol_ann': vol_ann,
        'sharpe': sharpe,
    })
    return stats


def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()


def drawdown_stats(returns: pd.Series) -> dict:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    mdd = dd.min()
    # duration
    below = dd < 0
    if below.any():
        durations = (below.astype(int).groupby((below != below.shift()).cumsum()).cumsum() * below).max()
    else:
        durations = 0
    return {"max_drawdown": float(mdd), "max_duration": int(durations)}


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()
