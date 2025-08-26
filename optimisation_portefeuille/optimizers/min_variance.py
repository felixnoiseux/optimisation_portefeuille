from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class MinVarianceResult:
    weights: pd.Series
    cov: pd.DataFrame
    success: bool
    message: str
    near_equal: bool = False
    diagnostics: Optional[Dict[str, float]] = None


class MinVarianceOptimizer:
    def __init__(self, w_min: float = 0.00, w_max: float = 1.00, ridge: float = 1e-6):
        self.default_w_min = w_min
        self.default_w_max = w_max
        self.ridge = ridge

    def _make_cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.cov()
        # Regularize covariance for numerical stability using labeled DataFrame to preserve alignment
        if self.ridge and cov.shape[0] > 0:
            ridge_mat = pd.DataFrame(np.eye(cov.shape[0]) * self.ridge, index=cov.index, columns=cov.columns)
            cov = cov + ridge_mat
        return cov

    def _feasible_bounds(self, n: int, w_min: float, w_max: float) -> tuple[float, float]:
        # Ensure feasibility of sum(w)=1 with simple per-asset bounds
        if w_min * n > 1.0:
            logger.warning("Somme des bornes min dépasse 100%% (n=%d). Abaissement de la borne min à 0.", n)
            w_min = 0.0
        if w_max * n < 1.0:
            logger.warning("Somme des bornes max inférieure à 100%% (n=%d). Relèvement de la borne max à 1.0.", n)
            w_max = 1.0
        if w_min > w_max:
            w_min = 0.0
        if w_max < 1.0 / n and n > 3:  # Si même pas 1/n possible
            logger.warning("w_max=%.3f trop restrictif pour n=%d actifs. Relèvement à %.3f",
                           w_max, n, min(1.0, 2.0 / n))
            w_max = min(1.0, 2.0 / n)
        return w_min, w_max

    def optimize(self, returns: pd.DataFrame, tickers: Optional[list[str]] = None,
                 w_min: Optional[float] = None, w_max: Optional[float] = None) -> MinVarianceResult:
        # Validate input DataFrame
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("'returns' doit être un pandas.DataFrame")
        # Keep only numeric columns
        numeric_cols = returns.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in returns.columns if c not in numeric_cols]
        if non_numeric:
            logger.warning("Colonnes non numériques ignorées: %s", ",".join(map(str, non_numeric)))
        clean_returns = returns[numeric_cols]
        # Resolve tickers and ensure intersection with available columns
        if tickers is None:
            tickers = list(clean_returns.columns)
        else:
            # Deduplicate while preserving order
            seen = set()
            tickers = [t for t in tickers if not (t in seen or seen.add(t))]
        available = [t for t in tickers if t in clean_returns.columns]
        missing = [t for t in tickers if t not in clean_returns.columns]
        if missing:
            logger.warning("Tickers absents des données de retours et ignorés: %s", ",".join(missing))
        rets = clean_returns[available]
        # Drop any rows with NaNs to avoid NaNs in covariance
        if rets.isna().any().any():
            before = len(rets)
            rets = rets.dropna(how='any')
            logger.warning("Lignes contenant des NaN supprimées avant covariance: %d -> %d", before, len(rets))
        if rets.shape[1] < 2:
            raise ValueError("Au moins 2 actifs sont requis pour l'optimisation de variance minimale.")
        if rets.empty:
            raise ValueError("Aucune ligne de données de retours après nettoyage.")
        n = rets.shape[1]
        cov = self._make_cov(rets)
        if not np.isfinite(cov.values).all():
            raise ValueError("La matrice de covariance contient des valeurs non finies.")

        # Diagnostics on covariance structure
        var_spread = float('nan')
        cond_num = float('nan')
        offdiag_norm_ratio = float('nan')
        try:
            diag = np.diag(cov.values)
            diag_min = float(np.min(diag))
            diag_max = float(np.max(diag))
            diag_mean = float(np.mean(diag)) if diag.size else float('nan')
            var_spread = (diag_max - diag_min) / (diag_mean if diag_mean != 0 else 1.0)
            eigvals = np.linalg.eigvalsh(cov.values)
            cond_num = float((np.max(eigvals) / np.min(eigvals))) if np.min(eigvals) > 0 else float('inf')
            offdiag = cov.values - np.diag(diag)
            offdiag_norm_ratio = float(np.linalg.norm(offdiag) / (np.linalg.norm(np.diag(diag)) or 1.0))
            logger.debug("MV: cov diag min=%.3e max=%.3e mean=%.3e spread=%.3e, eig cond=%.3e, offdiag_norm_ratio=%.3e",
                         diag_min, diag_max, diag_mean, var_spread, cond_num, offdiag_norm_ratio)
            if var_spread < 1e-3 and offdiag_norm_ratio < 1e-2:
                logger.warning("Covariance proche sphérique (variances homogènes et corrélations faibles) -> poids quasi-égaux attendus.")
        except Exception as e:
            logger.debug("MV: échec du diagnostic de covariance: %s", e)

        w_min = self.default_w_min if w_min is None else w_min
        w_max = self.default_w_max if w_max is None else w_max
        w_min, w_max = self._feasible_bounds(n, w_min, w_max)
        logger.info("MV: n=%d, bornes effectives w in [%.2f, %.2f]", n, w_min, w_max)

        bounds = [(w_min, w_max) for _ in range(n)]

        def obj(w: np.ndarray) -> float:
            return float(w.T @ cov.values @ w)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        # Point de départ faisable: commence à w_min puis distribue le reliquat jusqu'à w_max
        def _feasible_x0(n_: int, wmin_: float, wmax_: float) -> np.ndarray:
            w0 = np.full(n_, wmin_, dtype=float)
            rem = 1.0 - w0.sum()
            if rem < 0:
                # Fallback de sécurité (ne devrait pas arriver si _feasible_bounds a fait son job)
                w0 = np.full(n_, 1.0 / n_, dtype=float)
                w0 = np.clip(w0, wmin_, wmax_)
                s0 = w0.sum()
                return (w0 / s0) if s0 > 0 else np.full(n_, 1.0 / n_, dtype=float)
            cap = np.full(n_, wmax_ - wmin_, dtype=float)
            cap_sum = cap.sum()
            if cap_sum > 0 and rem > 0:
                # Répartition proportionnelle à la capacité restante
                incr = np.minimum(cap, rem * (cap / cap_sum))
                w0 += incr
                # Ajuste un résidu numérique éventuel
                resid = 1.0 - w0.sum()
                if abs(resid) > 1e-12:
                    mask_up = (w0 < wmax_)
                    if mask_up.any():
                        w0[mask_up] += resid / mask_up.sum()
                        w0 = np.clip(w0, wmin_, wmax_)
            return w0





        x0 = _feasible_x0(n, w_min, w_max)

        print(f"🔍 x0 = {x0}")
        print(f"🔍 obj(x0) = {obj(x0)}")

        # # Test manuel : poids différents
        # w_test = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])  # Poids inégaux
        # w_test = w_test / w_test.sum()  # Normalise
        # print(f"🔍 obj(w_test) = {obj(w_test)} (devrait être différent)")
        #
        # print(f"🔍 cov matrix shape: {cov.shape}")
        # print(f"🔍 cov diagonal: {np.diag(cov.values)}")

        res = minimize(obj, x0=x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'eps': 1e-10})

        if not res.success:
            logger.warning("Optimisation MV a échoué avec bornes par défaut: %s. Nouvelle tentative avec [0,1].", res.message)
            bounds = [(0.0, 1.0) for _ in range(n)]
            x0 = _feasible_x0(n, 0.0, 1.0)
            res = minimize(obj, x0=x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'ftol': 1e-12, 'eps': 1e-10})

        # Construire les poids avec l'index des colonnes effectivement utilisées (available)
        w_arr = res.x.astype(float, copy=True)
        # Ne corriger que de minuscules négatifs dus au bruit numérique
        tiny_neg_tol = 1e-12
        w_arr = np.where(np.logical_and(w_arr < 0.0, w_arr > -tiny_neg_tol), 0.0, w_arr)
        w = pd.Series(w_arr, index=available)
        s = float(w.sum())
        if not np.isfinite(s) or s == 0.0:
            # Fallback sûr
            w[:] = 1.0 / n
        elif abs(s - 1.0) > 1e-8:
            logger.warning("Somme des poids différente de 1 (%.12f). Renormalisation proportionnelle.", s)
            w /= s

        # Diagnostic: near-equal detection
        near_equal = (w.max() - w.min()) < 1e-4
        if near_equal:
            logger.warning("Solution MV quasi-égale entre actifs (écart max < 1e-4). Vérifiez les bornes et les données.")

        diagnostics: Dict[str, float] = {
            "var_spread": float(var_spread) if var_spread == var_spread else float('nan'),
            "eig_cond": float(cond_num) if cond_num == cond_num else float('nan'),
            "offdiag_norm_ratio": float(offdiag_norm_ratio) if offdiag_norm_ratio == offdiag_norm_ratio else float(
                'nan'),
            "n": float(n),
            "n_obs": float(len(rets)),
            "w_min_eff": float(w_min),
            "w_max_eff": float(w_max),
        }

        # Ligne après "return MinVarianceResult(...)"
        print(f"🔍 DEBUG POIDS RÉELS:")
        print(f"w.min()={w.min():.6f}, w.max()={w.max():.6f}")
        print(f"Poids détaillés:\n{w}")
        print(f"res.success={res.success}")
        print(f"res.x={res.x}")

        return MinVarianceResult(weights=w, cov=cov, success=bool(res.success), message=str(res.message),
                                 near_equal=near_equal, diagnostics=diagnostics)



      