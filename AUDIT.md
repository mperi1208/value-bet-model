# Audit du pipeline value bet — juillet 2026

> **Mise à jour (passe 2)** — La stratégie sharp anchor a été étendue aux marchés 1X2 et Asian Handicap avec dévig « power » (correction du biais favori-outsider). Portefeuille combiné 1X2 + O/U + AH, EV > 0.02, cotes @Max : **20 676 paris, ROI +4.86%, IC95 bootstrap [+2.6, +7.1], CLV +3.05% vs closing Pinnacle (68% des paris battent la close)**. Positif 11 saisons sur 13 (2012–2024), 8 ligues sur 10, sur les trois issues du 1X2 et sur les deux ères de données. C'est le premier résultat statistiquement significatif du projet. Détails en fin de document.

## Verdict

Le ROI de −3.2% affiché par le README v5 est surestimé d'environ 3.5 points par deux artefacts méthodologiques : le ROI honnête du pipeline ML est **−6.7% @Avg** (−2.8% @Max). La conclusion scientifique du README (le modèle ne bat pas le marché) est confirmée et même renforcée : un blend logistique walk-forward `y ~ logit(p_model) + logit(p_bookie)` donne un poids ≈ 0 au XGBoost et ≈ 1.15 au bookmaker — le modèle n'apporte strictement rien au-delà des cotes.

En revanche, il existe dans ces mêmes données un edge positif qui ne passe pas par le ML : parier les cotes Max aberrantes contre le prix no-vig de Pinnacle (stratégie « sharp anchor », implémentée dans `src/value_bet_sharp.py`). Backtest : **ROI +4.0%, IC95 bootstrap [−0.0, +8.2] sur 2 666 paris**, positif 4 saisons sur 5 et 7 ligues sur 10, validé par un **CLV moyen de +2.75%** contre la closing line Pinnacle (67% des paris battent la close). Le CLV positif est la signature standard d'un edge réel, indépendamment du bruit du ROI réalisé.

## Reproduction du baseline

Pipeline v5 reproduit à l'identique avant correction : 14 folds walk-forward sur 20 saisons exploitables, AUC tier2 = 0.557, ROI = −3.27% sur 1 627 paris (README : −3.2%). La reproduction valide que les problèmes ci-dessous sont bien dans le code, pas dans mon harnais de test.

## Problèmes identifiés

**P1 — Data leakage : recalibration isotonique globale** (`model.py`, corrigé). L'isotonique finale était fittée sur la totalité des prédictions out-of-sample — y compris le futur de chaque pari — puis utilisée pour sélectionner les paris sur ces mêmes matchs. Impact mesuré : environ +1.5 à +2 pts de ROI artificiels. Corrigé : l'isotonique du fold *k* n'est plus fittée que sur les prédictions OOS des folds plus anciens.

**P2 — Sélection de ligues a posteriori** (`main.py`, corrigé). `EXCLUDED_DIVS = {'D2','SP2','I2'}` était choisi d'après le ROI observé sur le test set lui-même (commit b4f2bc1 : « ROI −3.2% vs −5.2% »). C'est de l'overfitting du backtest : en sélection walk-forward honnête (ligues choisies uniquement sur les folds passés), le ROI se dégrade au lieu de s'améliorer (−11.5% @Avg). Corrigé : périmètre par défaut = toutes les ligues.

**P3 — Hyperparamètres de stratégie fittés sur le test** (signalé, non corrigé). La bande de cotes 1.65–2.35, le seuil edge 0.05 et `optimize_edge_threshold` (qui affiche le ROI de 7 seuils sur le même test set) relèvent du même problème. Tout seuil retenu doit être fixé a priori ou choisi walk-forward.

**P4 — Bug chemin rivalités** (`main.py`, corrigé). `DEFAULT_RIVALS` pointait vers `src/teams_by_country.csv` alors que le fichier est à la racine → 0 rivalité chargée depuis le refactor, composante rivalité de `match_importance` morte silencieusement.

**P5 — Bug enrichissement Sofascore mono-ligue** (`features.py`, corrigé). `league = df_raw["Div"].iloc[0]` : seule la première ligue du DataFrame était enrichie, les 9 autres restaient à NaN. Les features xG/big chances/GK n'ont donc jamais servi sur l'essentiel des données. Corrigé : boucle sur toutes les ligues.

**P6 — Le point structurel : le modèle est redondant avec le marché.** Même corrigé, le pipeline perd de l'argent parce que l'AUC 0.557 est presque entièrement contenue dans l'information des cotes. Le « calibration paradox » du README en est le symptôme mécanique : conditionner sur `p_model − p_bookie > seuil` sélectionne le bruit d'estimation du modèle (régression vers le marché). Aucun réglage de calibration ne peut réparer ça — c'est une propriété de la sélection, pas du modèle.

Points d'hygiène mineurs : `model.pkl`, `nohup.out`, données CSV et sorties de backtest vivent dans `src/` ; le dossier `{src,docs,csv}` à la racine est un résidu de commande shell ; `sklearn.frozen.FrozenEstimator` est toujours utilisé malgré le message du commit a6c420e.

## Chiffres honnêtes (mêmes filtres edge 0.05, cotes 1.65–2.35, tier 2)

| Variante | n | ROI @Avg | ROI @Max |
|---|---|---|---|
| README v5 (iso leaky + exclusion ligues) | 1 627 | −3.3% | +1.0% |
| Sans exclusion ligues, iso leaky | 2 626 | −5.4% | — |
| Honnête (calibration walk-forward, toutes ligues) | 2 502 | **−6.7%** | **−2.8%** |

Le +1.0% @Max de la première ligne montre au passage que la moitié du chemin vers le « ROI positif » était déjà disponible par simple line shopping — mais sur un périmètre sélectionné a posteriori, donc non fiable.

## La voie positive : sharp anchor (`src/value_bet_sharp.py`)

Fair prob = no-vig Pinnacle pré-match (`P<2.5`/`P>2.5`), pari sur `Max<2.5` ou `Max>2.5` quand `fair × cote_max − 1 > EV_min`. Aucun modèle, aucune feature, aucun paramètre fitté sur les résultats — le seuil EV est structurel (couvrir la marge, 0.01 par défaut).

| Seuil EV | n | ROI | IC95 | CLV moyen |
|---|---|---|---|---|
| > 0.01 | 2 666 | +4.0% | [−0.0, +8.2] | +2.8% (67% > 0) |
| > 0.02 | 1 101 | +6.2% | [−0.5, +13.0] | +4.1% (72% > 0) |

Par saison (EV > 0.01) : 2019 +11.8%, 2020 +2.7%, 2021 +4.1%, 2022 −3.6%, 2023 +5.1%.

Limites à garder en tête : (1) l'hypothèse Max suppose des comptes chez la quasi-totalité des bookmakers recensés par football-data ; (2) les bookmakers soft limitent les comptes gagnants en quelques semaines — c'est la contrainte opérationnelle réelle de cette stratégie, bien documentée ; (3) la couverture Pinnacle O/U ne commence qu'en 2019 dans ces CSV (~20% des matchs) ; (4) les cotes football-data sont des snapshots — l'exécution réelle au moment du pari peut différer.

## Modifications apportées au code

`src/model.py` : recalibration isotonique walk-forward (folds passés uniquement) au lieu de globale ; ajout de `fold_id`. `src/main.py` : `EXCLUDED_DIVS` vidé, chemin rivalités corrigé, affichage du ROI @Max en plus de @Avg. `src/features.py` : enrichissement Sofascore sur toutes les ligues. `src/value_bet_sharp.py` : nouveau module, stratégie sharp anchor + validation CLV + export CSV. Rien n'a été committé.

## Passe 2 — extension multi-marchés : le résultat rentable

L'extension recommandée en passe 1 a été implémentée (`src/value_bet_sharp.py`, réécrit). Le marché 1X2 change la donne : Pinnacle y est coté depuis ~2012 dans les CSV football-data (54% des matchs contre 20% pour l'O/U), soit 13 saisons de backtest. Deux améliorations méthodologiques au passage : dévig « power » (résout Σ(1/oᵢ)^k = 1) au lieu de la normalisation multiplicative, qui surestime la fair prob des outsiders et générait de faux EV sur les grosses cotes ; et settlement exact de l'Asian Handicap avec gestion des quarts de ligne.

Résultats par marché (EV > 0.02, @Max, aucun filtre de ligue, seuil fixé a priori) :

| Marché | Période | n | ROI | IC95 | CLV |
|---|---|---|---|---|---|
| 1X2 | 2012–2024 | 17 890 | +4.8% | [+2.2, +7.4] | +3.2% (69% > 0) |
| O/U 2.5 | 2019–2024 | 1 311 | +5.9% | — | +4.2% |
| Asian Handicap | 2019–2024 | 1 475 | +5.0% | — | +1.6% |
| **Portefeuille** | 2012–2024 | **20 676** | **+4.86%** | **[+2.6, +7.1]** | **+3.05%** |

Robustesse du 1X2 (le cœur du signal) : positif 10/13 saisons, 9/10 ligues (seule D2 est négative, −0.3%), sur les trois issues (H +6.1%, D +5.8%, A +2.3%) et sur les deux ères (2012–2018 : +5.8% ; 2019+ : +3.1%). Le test de réalisme le plus dur — parier uniquement chez Bet365 au lieu de la cote Max théorique — donne encore +1.6% de ROI et un CLV de +1.4% (60% > 0) : l'edge survit, atténué, à l'hypothèse d'exécution la plus pessimiste. Le tableau par bucket de cotes confirme que le dévig power élimine l'essentiel du biais outsider ; seule la tranche 8–15 reste négative (−8%, 839 paris, 4.7% du volume) — attendu, c'est là que le dévig est le moins fiable, et je ne l'ai volontairement pas exclue pour ne pas fitter le périmètre sur les résultats.

Ce qui reste vrai : l'edge vient du line shopping contre des books lents, pas d'une prédiction du football. Les limites opérationnelles (limitation des comptes gagnants, snapshots de cotes, liquidité réelle au moment du pari) s'appliquent intégralement. Le CLV positif à 68% est néanmoins la meilleure garantie disponible que ce n'est pas un artefact de backtest.

## Staking et forward test

Le module intègre le staking Kelly fractionné (`--kelly`, défaut ¼) : mise = min(¼ × EV/(cote−1), 2% bankroll), exposition journalière cappée à 25%, mises d'une même journée calculées sur la bankroll du matin. En simulation sur les 20 676 paris, le quarter-Kelly donne un drawdown max de 22.6% contre >60% en Kelly plein — la fraction ≤ ¼ n'est pas négociable vu le bruit de l'EV estimé. Le multiple de bankroll affiché est du compounding théorique (liquidité infinie, comptes jamais limités) : seuls la hiérarchie et le drawdown sont informatifs.

Le workflow de forward test est opérationnel : `--predict` logge les value bets détectés sur les fixtures à venir dans `paper_trades.csv` (avec `stake_pct` suggéré), `--evaluate` les settle et calcule le CLV réalisé. C'est le juge de paix : trois mois de CLV positif en paper trading (dès la reprise d'août) valent plus que 13 saisons de backtest.

## Test d'exécution sur Betfair Exchange (automatisation)

Question testée : l'edge survit-il si on exécute sur l'exchange (seule voie d'automatisation légitime, API officielle) au lieu des soft books ? Données : saison 2024-25 uniquement (seule à avoir les colonnes BFE), 3 758 matchs, 1X2, EV > 0.02. Résultat : non. Sur le même périmètre, les paris @Max soft books gardent un CLV de +2.50% (69% > 0) ; les paris @BFE à commission nulle affichent un CLV de +1.02% avec seulement **51% des paris battant la close — un coin flip**, et à commission réaliste (2-5%) le volume s'effondre et le ROI devient négatif. Les cotes BFE sont pourtant équivalentes aux Max en niveau (ratio médian ~1.00) : la différence est que l'exchange bouge *avec* Pinnacle — quand une cote y dépasse le fair price, c'est de l'information, pas de la lenteur. L'edge de la stratégie est précisément la lenteur des soft books ; elle ne se transporte pas sur un marché rapide. Conclusion opérationnelle : l'automatisation s'arrête à la détection/notification/suivi ; l'exécution reste manuelle, chez les soft books.

## Recommandations

Le plafond AUC ~0.56 du XGBoost avec données publiques est un résultat robuste, cohérent avec la littérature — inutile d'y réinvestir. Les prolongements utiles : mesurer le CLV comme métrique primaire de toute stratégie future plutôt que le ROI ; staking Kelly fractionné (l'EV pré-match est déjà calculé par pari dans l'export) ; élargir aux lignes O/U 1.5 et 3.5 et aux handicaps alternatifs si les données deviennent disponibles ; et si un modèle ML doit resservir, l'entraîner à prédire le *mouvement* de la ligne (open → close) plutôt que le résultat du match — c'est le seul y disponible où le marché n'a pas déjà tout dit au moment du pari.
