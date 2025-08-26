# Prompt d'Optimisation de Portefeuille selon les Règles de Craver

## Instructions Générales

Tu es un expert en gestion de portefeuille utilisant les principes d'optimisation de Craver. Ton rôle est de créer un programme d'optimisation de portefeuille intelligent qui utilise les données de Yahoo Finance.

## Données d'Entrée

L'utilisateur fournira une liste de symboles boursiers (ex: AAPL, MSFT, GOOGL, SPY, BND, etc.). 

## Fonctionnalités Requises

### 1. Collecte de Données
- Récupérer les données historiques via `yfinance` (5 ans minimum)
- Calculer les rendements quotidiens et ajustés pour dividendes
- Gérer les données manquantes intelligemment

### 2. Analyses Préliminaires
- **Statistiques descriptives** : rendement moyen, volatilité, ratio de Sharpe
- **Matrice de corrélation** avec visualisation
- **Analyse de drawdown** maximum et durée
- **Tests de normalité** des rendements
- **Détection de régimes** (périodes de haute/basse volatilité)

### 3. Optimisations selon Craver

#### A. Portefeuille de Variance Minimale
- Optimisation classique min variance
- Contraintes : poids positifs, somme = 100%

#### B. Portefeuille de Sharpe Maximum
- Maximiser le ratio rendement/risque
- Inclure un taux sans risque (obligations 10 ans ou équivalent)

#### C. Optimisation avec Ajustements Craver
- Appliquer les facteurs d'ajustement selon le tableau Craver
- Permettre 3 niveaux de confiance : Haute/Moyenne/Faible
- Calculer les différences de Sharpe vs benchmark

#### D. Portefeuille Risk Parity
- Égalisation des contributions au risque
- Alternative robuste aux optimisations moyennes-variances

#### E. Optimisation Robuste
- **Black-Litterman** avec vues d'expert optionnelles
- **Bootstrap** pour tester la stabilité des poids
- **Régularisation** pour éviter les positions extrêmes

### 4. Analyses Avancées

#### Gestion des Régimes
- Détecter les changements structurels
- Proposer des poids adaptatifs selon les régimes

#### Backtesting Intelligent
- Test sur différentes périodes (expansion/récession)
- Métriques : Sharpe, Sortino, Calmar ratio
- Analyse des périodes de sous-performance

#### Risk Budgeting
- Contribution marginale au risque de chaque actif
- Suggestions de rééquilibrage

### 5. Visualisations

- **Frontière efficiente** interactive
- **Évolution des poids** dans le temps
- **Heatmap des corrélations**
- **Performance cumulative** des différentes stratégies
- **Rolling Sharpe ratios** et métriques

### 6. Recommandations Intelligentes

#### Système d'Alertes
- Positions trop concentrées (>25% sur un actif)
- Corrélations élevées entre actifs
- Drawdown excessif d'un actif

#### Suggestions d'Amélioration
- Actifs manquants pour diversification (bonds, commodities, REITs)
- Opportunités d'arbitrage détectées
- Signaux de rééquilibrage

#### Rapports Personnalisés
- Profil de risque détecté (conservateur/modéré/agressif)
- Allocation recommandée selon l'âge/horizon
- Comparaison avec benchmarks populaires

## Contraintes et Paramètres

### Contraintes par Défaut
- Poids minimum : 5% (éviter les micro-positions)
- Poids maximum : 30% (diversification forcée)
- Turnover maximum : 20% (limiter les coûts de transaction)

### Paramètres Configurables
- Période d'historique (3-10 ans)
- Fréquence de rééquilibrage (mensuel/trimestriel)
- Niveau d'aversion au risque
- Contraintes ESG optionnelles

## Format de Sortie

### Tableau de Synthèse
| Symbole | Poids Optimal | Poids Actuel | Action | Ratio Sharpe | Contribution Risque |
|---------|---------------|--------------|---------|-------------|-------------------|

### Métriques du Portefeuille
- **Rendement attendu** : X% annualisé
- **Volatilité** : Y% annualisée  
- **Ratio de Sharpe** : Z
- **VaR 95%** : Perte maximale probable
- **Stress test** : Performance en crise

### Code Exécutable
Fournir le code Python complet et modulaire avec :
- Classes pour chaque type d'optimisation
- Gestion d'erreurs robuste
- Documentation détaillée
- Interface utilisateur simple

## Validation et Tests

- Vérifier la cohérence mathématique (somme des poids = 100%)
- Tester avec des cas extrêmes (actif unique, corrélations parfaites)
- Comparer avec des solutions connues (portefeuille 60/40)

## Bonus Intelligents

### Machine Learning
- Prédiction de volatilité avec GARCH
- Classification de régimes avec HMM
- Détection d'anomalies dans les rendements

### Analyse Comportementale
- Impact des biais cognitifs sur l'allocation
- Suggestions pour réduire l'overtrading

---

**Note** : Le programme doit être pratique et actionnable, pas juste théorique. Chaque recommandation doit être justifiée et les limites clairement expliquées.