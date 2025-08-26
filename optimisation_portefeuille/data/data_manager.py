from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


logger = logging.getLogger(__name__)


@dataclass
class DataFetchResult:
    prices: pd.DataFrame  # Adjusted Close
    returns: pd.DataFrame  # daily simple returns


class DataManager:
    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    def fetch(self, tickers: List[str], years: int = 5) -> DataFetchResult:
        if yf is None:
            raise ImportError("yfinance n'est pas installé. Ajoutez-le à requirements.txt et installez les dépendances.")
        end = datetime.today()
        start = end - timedelta(days=int(years * 365.25))
        logger.info("Téléchargement des données ajustées: %s, de %s à %s", ",".join(tickers), start.date(), end.date())
        data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by='column')

        # Handle multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Supporte les deux organisations:
            # - group_by='ticker' -> (ticker, field)
            # - group_by='column' -> (field, ticker)
            lvl0 = data.columns.get_level_values(0)
            lvl1 = data.columns.get_level_values(1)

            if 'Adj Close' in lvl1:
                # (ticker, 'Adj Close')
                adj = data.xs('Adj Close', axis=1, level=1)
                adj.columns = adj.columns.astype(str)
            elif 'Adj Close' in lvl0:
                # ('Adj Close', ticker)
                adj = data.xs('Adj Close', axis=1, level=0)
                adj.columns = adj.columns.astype(str)
            elif 'Close' in lvl1:
                adj = data.xs('Close', axis=1, level=1)
                adj.columns = adj.columns.astype(str)
            elif 'Close' in lvl0:
                adj = data.xs('Close', axis=1, level=0)
                adj.columns = adj.columns.astype(str)
            else:
                raise ValueError("Colonnes 'Adj Close' ou 'Close' introuvables dans les données téléchargées (MultiIndex).")
        else:
            # Single ticker may return a single-index frame
            if 'Adj Close' in data.columns:
                adj = data['Adj Close'].to_frame() if isinstance(data['Adj Close'], pd.Series) else data[['Adj Close']]
                adj.columns = [tickers[0]]
            elif 'Close' in data.columns:
                adj = data['Close'].to_frame() if isinstance(data['Close'], pd.Series) else data[['Close']]
                adj.columns = [tickers[0]]
            else:
                raise ValueError("Colonnes 'Adj Close' ou 'Close' introuvables dans les données téléchargées.")

        adj = adj.sort_index().astype(float)

        # Cleaning: remove columns with excessive NaNs
        nan_ratio = adj.isna().mean()
        keep = nan_ratio[nan_ratio <= 0.05].index.tolist()
        dropped = [c for c in adj.columns if c not in keep]
        if dropped:
            logger.warning("Colonnes supprimées pour données manquantes (>5%% NaN): %s", ",".join(dropped))
        adj = adj[keep] if keep else adj

        # Smart missing handling: forward fill then backfill small gaps
        adj = adj.ffill().bfill()

        # Drop rows still NaN
        adj = adj.dropna(how='any')
        if adj.empty:
            raise ValueError("Toutes les séries sont vides après nettoyage.")

        # Compute daily simple returns
        rets = adj.pct_change().dropna(how='any')

        # Remove constant series
        stds = rets.std()
        const_cols = stds[stds <= 1e-12].index.tolist()
        if const_cols:
            logger.warning("Colonnes supprimées (retours constants): %s", ",".join(const_cols))
            rets = rets.drop(columns=const_cols)
            adj = adj.drop(columns=const_cols)

        return DataFetchResult(prices=adj, returns=rets)

    @staticmethod
    def annualize_return(mean_daily: pd.Series, trading_days: int = 252) -> pd.Series:
        return (1 + mean_daily) ** trading_days - 1

    @staticmethod
    def annualize_vol(std_daily: pd.Series, trading_days: int = 252) -> pd.Series:
        return std_daily * np.sqrt(trading_days)
