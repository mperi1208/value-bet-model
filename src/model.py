"""
model.py
--------
Walk-forward XGBoost calibré pour la prédiction under 2.5 :
  - TARGET fixé à under_25 (binaire : 1 si total buts < 3, 0 sinon)
  - prob_bookie absent des features (indépendance du modèle)
  - scale_pos_weight pour rééquilibrer under/over (ratio ~1.4 en Div2)
  - Calibration sigmoid sur le fold de validation (moins sujette à l'overfitting
    que l'isotonique sur de petits jeux de calibration)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

FEATURE_COLS = [
    # Stats domicile
    'h_avg_goals_scored', 'h_avg_goals_conceded',
    'h_avg_shots', 'h_avg_sot', 'h_avg_corners',
    'h_avg_yellow', 'h_null_pct', 'h_form_5', 'h_rank',
    'h_under_rate_3', 'h_under_rate_5', 'h_under_rate_10',
    'h_avg_low_score_rate', 'h_avg_total_goals', 'h_goals_var_5',
    # Stats extérieur
    'a_avg_goals_scored', 'a_avg_goals_conceded',
    'a_avg_shots', 'a_avg_sot', 'a_avg_corners',
    'a_avg_yellow', 'a_null_pct', 'a_form_5', 'a_rank',
    'a_under_rate_3', 'a_under_rate_5', 'a_under_rate_10',
    'a_avg_low_score_rate', 'a_avg_total_goals', 'a_goals_var_5',
    # Features combinées
    'delta_form', 'delta_rank',
    'delta_goals_scored', 'delta_goals_against', 'delta_sot',
    'combined_under_rate_5', 'combined_under_rate_10',
    'combined_goals_var', 'combined_avg_goals', 'combined_low_score',
    'match_importance',
    # Qualité de tir (proxy xG)
    'h_shot_acc', 'a_shot_acc', 'combined_shot_acc',
    'h_xg_proxy', 'a_xg_proxy', 'combined_xg_proxy',
    # Arbitre
    'ref_under_rate', 'ref_avg_goals',
    # Signal marché (mouvement de cote)
    'odds_spread_under',
    # Head-to-head historique entre les deux équipes
    'h2h_under_rate',
    # Fixture congestion (jours de repos)
    'h_days_rest', 'a_days_rest',
]

TARGET = 'under_25'


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

def walk_forward_splits(df: pd.DataFrame, min_train_seasons: int = 5):
    """
    Saison 1 = plus récente, N = plus ancienne.
    Splits : train (≥ train_s) | val (= val_s) | test (= test_s)
    """
    n = df['season'].max()
    splits = []

    for test_s in range(1, n - min_train_seasons):
        val_s   = test_s + 1
        train_s = test_s + 2

        train = df[df['season'] >= train_s].dropna(subset=FEATURE_COLS + [TARGET]).copy()
        val   = df[df['season'] == val_s].dropna(subset=FEATURE_COLS + [TARGET]).copy()
        test  = df[df['season'] == test_s].dropna(subset=FEATURE_COLS + [TARGET]).copy()

        if len(train) < 500 or len(val) < 100 or len(test) < 100:
            continue
        splits.append((train, val, test))

    print(f'Walk-forward : {len(splits)} folds sur {n} saisons')
    for i, (tr, vl, te) in enumerate(splits):
        print(f'   Fold {i+1} : train={len(tr):>5} | val={len(vl):>4} | '
              f'test={len(te):>4} (saison {te["season"].iloc[0]})')
    return splits


# ═══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def _make_xgb(scale_pos_weight: float = 1.0):
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.7,
        min_child_weight=8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def train_model(train: pd.DataFrame, val: pd.DataFrame):
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]

    n_under = y_train.sum()
    n_over  = len(y_train) - n_under
    spw = n_over / n_under if n_under > 0 else 1.0

    xgb_model = _make_xgb(scale_pos_weight=spw)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibration sigmoid (Platt scaling) — plus régularisée qu'isotonique
    # sur de petits sets de calibration, réduit la surconfiance sélective.
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_model), method='sigmoid')
    calibrated.fit(X_val, y_val)
    return calibrated


def run_walk_forward(df: pd.DataFrame,
                     min_train_seasons: int = 5,
                     predict_tier: int = None):
    """
    Entraîne sur toutes les données disponibles dans df.
    Si predict_tier est fourni, ne retourne les prédictions que pour ce tier.
    (ex: predict_tier=2 pour ne parier que sur Div2 même si entraîné sur Div1+Div2)
    """
    splits    = walk_forward_splits(df, min_train_seasons)
    all_preds = []

    for i, (train, val, test) in enumerate(splits):
        print(f'  Fold {i+1}/{len(splits)}...', end=' ', flush=True)
        model = train_model(train, val)
        test  = test.copy()
        test['prob_model'] = model.predict_proba(test[FEATURE_COLS])[:, 1]

        if predict_tier is not None:
            test = test[test['tier'] == predict_tier]

        all_preds.append(test)
        print('OK')

    df_wf = pd.concat(all_preds, ignore_index=True)
    auc   = roc_auc_score(df_wf[TARGET].values, df_wf['prob_model'].values)
    acc   = ((df_wf['prob_model'] > 0.5) == df_wf[TARGET]).mean()
    tier_info = f' | tier={predict_tier}' if predict_tier else ''
    print(f'\n  Walk-forward : {len(df_wf)} matchs{tier_info} | AUC={auc:.4f} | Acc={acc:.1%}')
    return df_wf, model


# ═══════════════════════════════════════════════════════════════
# ÉVALUATION / UTILS
# ═══════════════════════════════════════════════════════════════

def feature_importance(model, top_n: int = 15):
    base = None
    try:
        cc = model.calibrated_classifiers_[0]
        if hasattr(cc, 'estimator'):
            inner = cc.estimator
            base = inner.estimator if hasattr(inner, 'estimator') else inner
        else:
            base = cc
    except Exception:
        base = model

    if base is None or not hasattr(base, 'feature_importances_'):
        print('  feature_importance : impossible d\'extraire')
        return None

    imp = pd.Series(base.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    top5 = imp.head(5)
    print('  Top features : ' + ' | '.join(f'{f}={s:.3f}' for f, s in top5.items()))
    return imp


def save_model(model, path: str = 'model.pkl'):
    joblib.dump(model, path)
    print(f'Modèle sauvegardé -> {path}')


def load_model(path: str = 'model.pkl'):
    return joblib.load(path)
