# Optimisation de Portefeuille — Variance Minimale (MVP)

Un outil en ligne de commande (CLI) pour construire un portefeuille à variance minimale à partir de séries de prix historiques téléchargées via Yahoo Finance (yfinance). Il calcule les poids optimaux, affiche des métriques de portefeuille et exporte les résultats si souhaité.


## À quoi sert ce programme ?

- À estimer un portefeuille de type « Minimum Variance »: les poids sont choisis pour minimiser la volatilité (variance) du portefeuille sous la contrainte que la somme des poids = 1 et que chaque poids respecte des bornes simples (w_min ≤ w_i ≤ w_max).
- À analyser rapidement un panier d’actifs: téléchargements de données ajustées (Adj Close), calculs des rendements journaliers, statistiques de base (rendement/volatilité annualisés, Sharpe), corrélations et drawdown sur le portefeuille optimisé.
- À fournir un export CSV des poids pour intégration dans d’autres workflows.


## Pour qui ?

- Investisseurs individuels souhaitant tester une allocation défensive (faible variance) sur une sélection d’actions/ETF.
- Étudiants et data scientists voulant un exemple clair d’optimisation de portefeuille avec Python, pandas, NumPy et SciPy.
- Analystes qui ont besoin d’un outil rapide en CLI pour comparer des paniers d’actifs sur différentes fenêtres historiques et contraintes.


## Principales fonctionnalités

- Téléchargement automatique des données via yfinance.
- Nettoyage des données (remplissage des petites lacunes, suppression de colonnes problématiques, filtrage des NaN).
- Optimisation de variance minimale avec SciPy (SLSQP), bornes par actif et contrainte de somme des poids.
- Régularisation de la matrice de covariance (ridge) pour la stabilité numérique.
- Diagnostics de covariance et de solution (détection de solutions quasi-égales et explications).
- Calculs de métriques de portefeuille: rendement annualisé, volatilité annualisée, Sharpe (ajusté du taux sans risque), max drawdown et durée maximale du drawdown.
- Export CSV des poids.
- Configuration par défaut centralisée (settings.py) + options CLI pour overrides.


## Inspirations et fondements

- Modèle de Markowitz / Modern Portfolio Theory (MPT) — cas particulier de la frontière efficiente avec objectif de variance minimale.
- Outils Python classiques pour la finance de marché: pandas, NumPy, SciPy, yfinance.


## Aperçu de l’architecture

- `main.py` — point d’entrée qui appelle le CLI.
- `optimisation_portefeuille/cli/app.py` — parsing des arguments, téléchargement des données, optimisation et affichage/exports.
- `optimisation_portefeuille/data/data_manager.py` — gestion des données (yfinance, nettoyage, rendements journaliers).
- `optimisation_portefeuille/optimizers/min_variance.py` — optimiseur MVP (SLSQP), construction de la covariance, bornes, diagnostics.
- `optimisation_portefeuille/analytics/stats.py` — statistiques (annualisation, Sharpe, corrélation, drawdown).
- `optimisation_portefeuille/config/settings.py` — paramètres par défaut: historique, bornes, jours de bourse, taux sans risque, niveau de logs.
- `optimisation_portefeuille/utils/logging.py` — configuration du logging.


## Installation

Prérequis: Python 3.10+ recommandé.

1) Cloner ou copier ce dépôt.
2) Créer un environnement virtuel et installer les dépendances:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Remarque: yfinance est utilisé pour le téléchargement; il est listé dans requirements.txt.


## Utilisation (CLI)

Commande générale:

```bash
python main.py TICKER1,TICKER2,... [options]
```

Exemple:

```bash
python main.py AAPL,JNJ,PG,XOM,VZ,WMT,KO,JPM,ABBV,NEE --years 7 --rf 0.03 --wmin 0.00 --wmax 0.20 --export .\poids.csv --log-level DEBUG
```

Options disponibles:

- `--years` Nombre d’années d’historique (défaut: settings.years_history = 5)
- `--rf` Taux sans risque annualisé (défaut: settings.risk_free_rate = 0.02)
- `--export` Chemin CSV pour exporter le tableau des poids
- `--wmin` Borne inférieure des poids par actif (override; défaut: settings.weight_min = 0.00)
- `--wmax` Borne supérieure des poids par actif (override; défaut: settings.weight_max = 0.20)
- `--log-level` Niveau de logs: DEBUG, INFO, WARNING, ERROR (défaut: settings.log_level = INFO)

Sortie typique:

- Tableau de synthèse des poids optimaux (en %).
- Métriques du portefeuille optimisé: rendement annualisé, vol annualisée, Sharpe, Max Drawdown et sa durée.
- Note explicative si la solution est « quasi-égale » entre actifs, avec diagnostics (bornes effectives, spread de variance, conditionnement eigen, ratio de norme hors-diagonale).

Export CSV (si `--export` est fourni):

```csv
Symbole,Poids Optimal
AAPL,11.11%
JNJ,11.11%
...
```


## Détails méthodologiques

- Rendements journaliers: simples (pct_change), nettoyage ffill/bfill, suppression des lignes NaN restantes et des séries constantes.
- Covariance: covariance des rendements journaliers + petite régularisation « ridge » (valeur très faible) pour la stabilité numérique.
- Optimisation: SLSQP minimise w^T Σ w avec contraintes:
  - Somme des poids = 1
  - w_min ≤ w_i ≤ w_max
- Bornes faisables: l’outil ajuste prudemment les bornes si nécessaire (ex. si n*w_max < 1 ou n*w_min > 1) pour garantir une solution faisable et loggue les bornes « effectives ».
- Annualisation: vol multipliée par √252; rendement annualisé par composition (1 + mean_daily)^252 - 1; Sharpe ajusté d’un taux sans risque converti en quotidien.
- Drawdown: calculé sur les rendements quotidiens du portefeuille optimisé.


## Diagnostics et solutions quasi-égales

Si tous les poids sortent quasi-égaux, le programme:
- Loggue un avertissement et affiche une Note avec des diagnostics (var_spread, eig_cond, offdiag_norm_ratio, bornes effectives, nombre d’actifs, nombre d’observations). 
- Causes fréquentes:
  - Covariance « proche sphérique » (variances très homogènes et corrélations faibles), 
  - Bornes trop contraignantes ou trop lâches aboutissant à une solution indifférenciée.
- Pistes:
  - Ajuster `--wmin`/`--wmax` (ex. réduire `w_min`, augmenter `w_max`),
  - Modifier `--years` pour changer la fenêtre historique,
  - Inspecter les corrélations et variances individuelles, 
  - Activer `--log-level DEBUG` pour plus de détails.


## Paramètres par défaut (settings.py)

- `years_history = 5`
- `trading_days = 252`
- `weight_min = 0.00`
- `weight_max = 0.20`
- `turnover_max = 0.20` (réservé pour extensions)
- `risk_free_rate = 0.02`
- `log_level = "INFO"`


## Dépendances principales

- pandas, numpy, scipy, yfinance

Voir `requirements.txt` pour la liste complète.


## Limitations et bonnes pratiques

- Données Yahoo Finance: peuvent contenir des lacunes ou ajustements; vérifiez l’intégrité des données si les résultats surprennent.
- Les poids peuvent être contraints par `w_min` et `w_max`. Des réglages inadaptés peuvent forcer des allocations uniformes.
- L’optimisation MVP ne tient pas compte du rendement attendu (seule la variance est minimisée). Pour des stratégies axées sur Sharpe ou rendement cible, il faudrait un autre objectif.
- Les résultats passés ne préjugent pas des performances futures.


## Dépannage (FAQ)

- « Pourquoi tous mes poids sont identiques ? »
  - Vérifiez la Note dans la sortie et les logs DEBUG. Si la covariance est quasi-sphérique ou les bornes trop restrictives, des poids égaux sont logiques. Essayez d’ajuster `--wmin`/`--wmax` et la fenêtre `--years`.
- « Erreur: yfinance n’est pas installé »
  - Installez les dépendances avec `pip install -r requirements.txt`.
- « Max drawdown semble incohérent »
  - Le programme calcule le drawdown sur les rendements du portefeuille optimisé (pas l’égal-pondéré). Assurez-vous d’utiliser la version récente.


## Exemple complet

```bash
python main.py AAPL,MSFT,GOOGL,SPY --years 3 --rf 0.025 --wmin 0.00 --wmax 0.40 --export .\poids.csv --log-level DEBUG
```

Vous obtiendrez:
- Un tableau des poids (en %),
- Des métriques de portefeuille,
- Un fichier `poids.csv` dans le chemin spécifié.


## Avertissement

Cet outil est fourni à des fins éducatives et exploratoires. Il ne constitue pas un conseil en investissement.
