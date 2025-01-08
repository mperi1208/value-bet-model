"""
model.py
--------
Walk-forward XGBoost calibré pour la prédiction under 2.5 :
  - TARGET fixé à under_25
  - prob_bookie absent des features (indépendance du modèle)
  - scale_pos_weight pour rééquilibrer under/over
  - Calibration isotonique sur le fold de validation
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
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
]

TARGET = 'under_25'


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

def walk_forward_splits(df: pd.DataFrame, min_train_seasons: int = 5):
    """
    Saison 1 = plus récente, N = plus ancienne.
    min_train_seasons augmenté à 5 pour éviter les folds trop petits.
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

    # Rééquilibrage selon ratio under/over dans le train
    n_under = y_train.sum()
    n_over  = len(y_train) - n_under
    spw = n_over / n_under if n_under > 0 else 1.0

    xgb_model = _make_xgb(scale_pos_weight=spw)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    try:
        from sklearn.frozen import FrozenEstimator
        frozen = FrozenEstimator(xgb_model)
        calibrated = CalibratedClassifierCV(frozen, method='isotonic', cv='prefit')
    except ImportError:
        calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')

    calibrated.fit(X_val, y_val)
    return calibrated


def run_walk_forward(df: pd.DataFrame, min_train_seasons: int = 5):
    splits    = walk_forward_splits(df, min_train_seasons)
    all_preds = []

    for i, (train, val, test) in enumerate(splits):
        print(f'  Fold {i+1}/{len(splits)}...', end=' ', flush=True)
        model = train_model(train, val)
        test  = test.copy()
        test['prob_model'] = model.predict_proba(test[FEATURE_COLS])[:, 1]
        all_preds.append(test)
        print('OK')

    df_wf = pd.concat(all_preds, ignore_index=True)
    auc   = roc_auc_score(df_wf[TARGET].values, df_wf['prob_model'].values)
    acc   = ((df_wf['prob_model'] > 0.5) == df_wf[TARGET]).mean()
    print(f'\n  Walk-forward : {len(df_wf)} matchs | AUC={auc:.4f} | Acc={acc:.1%}')
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
    print(f'Modele sauvegarde -> {path}')


def load_model(path: str = 'model.pkl'):
    return joblib.load(path)
