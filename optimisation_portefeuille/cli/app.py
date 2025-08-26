from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd

from optimisation_portefeuille.config.settings import get_settings
from optimisation_portefeuille.utils.logging import setup_logging
from optimisation_portefeuille.utils.validation import validate_tickers
from optimisation_portefeuille.data.data_manager import DataManager
from optimisation_portefeuille.analytics.stats import describe_returns, correlation_matrix, drawdown_stats
from optimisation_portefeuille.optimizers.min_variance import MinVarianceOptimizer


logger = logging.getLogger(__name__)


def run_cli(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimisation de portefeuille - Variance Minimale (MVP)")
    parser.add_argument("tickers", type=str, help="Tickers séparés par des virgules, ex: AAPL,MSFT,GOOGL,SPY")
    parser.add_argument("--years", type=int, default=None, help="Nombre d'années d'historique (défaut: settings)")
    parser.add_argument("--rf", type=float, default=None, help="Taux sans risque annualisé (défaut: settings)")
    parser.add_argument("--export", type=str, default=None, help="Chemin CSV pour exporter les poids")
    parser.add_argument("--wmin", type=float, default=None, help="Borne inférieure par actif (défaut: settings)")
    parser.add_argument("--wmax", type=float, default=None, help="Borne supérieure par actif (défaut: settings)")
    parser.add_argument("--log-level", type=str, default=None, help="Niveau de logs (DEBUG, INFO, WARNING, ERROR). Défaut: settings")

    args = parser.parse_args(argv)

    settings = get_settings()
    log_level = args.log_level if args.log_level is not None else settings.log_level
    setup_logging(log_level)

    years = args.years if args.years is not None else settings.years_history
    rf = args.rf if args.rf is not None else settings.risk_free_rate

    tickers = validate_tickers(args.tickers.split(","))
    logger.info("Tickers: %s", ",".join(tickers))

    # Data
    dm = DataManager(trading_days=settings.trading_days)
    data = dm.fetch(tickers, years=years)

    # Basic analytics
    asset_stats = describe_returns(data.returns, trading_days=settings.trading_days, rf_annual=rf)
    corr = correlation_matrix(data.returns)

    # Optimization - Min Variance
    mv = MinVarianceOptimizer(w_min=settings.weight_min, w_max=settings.weight_max)
    mv_res = mv.optimize(data.returns, w_min=(args.wmin if args.wmin is not None else None), w_max=(args.wmax if args.wmax is not None else None))

    # Portfolio summary
    weights = mv_res.weights
    port_ret_daily = (data.returns @ weights)
    port_vol_daily = port_ret_daily.std()
    port_ret_ann = (1 + port_ret_daily.mean()) ** settings.trading_days - 1
    port_vol_ann = port_vol_daily * (settings.trading_days ** 0.5)
    rf_daily = (1 + rf) ** (1 / settings.trading_days) - 1
    sharpe = ((port_ret_daily.mean() - rf_daily) / (port_vol_daily if port_vol_daily != 0 else float('nan'))) * (settings.trading_days ** 0.5)
    dd_port = drawdown_stats(port_ret_daily)

    # Output
    if getattr(mv_res, 'near_equal', False):
        diag = mv_res.diagnostics or {}
        print("\n[Note] Les poids optimisés sont quasi-égaux.")
        print("- Cela peut survenir si la covariance est proche d'une sphère (variances homogènes, corrélations faibles) ou si les bornes sont trop contraignantes.")
        print(f"- Bornes effectives: w in [{diag.get('w_min_eff', float('nan')):.2f}, {diag.get('w_max_eff', float('nan')):.2f}] pour n={int(diag.get('n', len(weights)))} actifs")
        print(f"- Indicateurs: var_spread={diag.get('var_spread', float('nan')):.3e}, eig_cond={diag.get('eig_cond', float('nan')):.3e}, offdiag_norm_ratio={diag.get('offdiag_norm_ratio', float('nan')):.3e}")
        print("- Conseils: essayez d'ajuster --wmin/--wmax, de changer la fenêtre --years, ou d'inspecter les données/corrélations.")
    poids_pct = (weights.values * 100)
    summary = pd.DataFrame({
        'Symbole': weights.index,
        'Poids Optimal': [f"{v:.2f}%" for v in poids_pct],
    })

    print("\n=== Tableau de Synthèse (Min Variance) ===")
    print(summary.to_string(index=False))

    print("\n=== Métriques du Portefeuille ===")
    print(f"Rendement attendu (ann.): {port_ret_ann:.2%}")
    print(f"Volatilité (ann.): {port_vol_ann:.2%}")
    print(f"Ratio de Sharpe: {sharpe:.3f}")
    print(f"Max Drawdown (portefeuille): {dd_port['max_drawdown']:.2%} | Durée max: {dd_port['max_duration']} jours")

    if args.export:
        export_path = args.export
        summary.to_csv(export_path, index=False)
        print(f"\nRésultats exportés vers: {export_path}")

    return 0
