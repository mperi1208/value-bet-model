"""
model.py
--------
Walk-forward XGBoost calibré pour la prédiction under 2.5 :
  - TARGET fixé à under_25
  - prob_bookie absent des features (indépendance du modèle)
  - scale_pos_weight pour rééquilibrer under/over
  - Calibration sigmoid sur le fold de validation (moins sujette à l'overfitting
    que l'isotonique sur de petits jeux de calibration)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
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
    # ── Sofascore (optionnel — NaN ignorés par XGBoost si absent) ────────────
    # xG rolling (remplace le proxy SOT quand disponible)
    'h_sf_expected_goals_avg5', 'a_sf_expected_goals_avg5',
    'h_sf_expected_goals_avg10', 'a_sf_expected_goals_avg10',
    'combined_sf_xg_avg5', 'combined_sf_xg_avg10',
    'sf_xg_delta_avg5',
    # Qualité de création
    'h_sf_big_chances_avg5', 'a_sf_big_chances_avg5',
    # GK performance
    'h_sf_goals_prevented_avg5', 'a_sf_goals_prevented_avg5',
    # Arbitre Sofascore (carrière — toujours dispo pré-match)
    'ref_sf_yc_per_game', 'ref_sf_rc_per_game',
    # Stade + journée
    'round',
    # Cotes no-vig dérivées
    'odds_implied_h', 'odds_implied_d', 'odds_implied_a', 'odds_implied_delta',
    # Suspension proxy (rouge dans les N derniers matchs — toujours dispo FD)
    'h_susp_proxy', 'a_susp_proxy', 'combined_susp_proxy',
    # Blessures Transfermarkt (nb joueurs blessés + valeur manquante)
    'h_inj_n', 'a_inj_n', 'combined_inj_n',
    'h_inj_value_ratio', 'a_inj_value_ratio',
    # Suspensions exactes (incidents Sofascore — dispo après scraping complet)
    'h_suspended_n', 'a_suspended_n', 'combined_susp_n',
    # Valeur marchande des joueurs suspendus / valeur totale squad
    'h_susp_value_ratio', 'a_susp_value_ratio',
]

TARGET = 'under_25'

# Features de base (obligatoires — dropna strict)
# Les 53 premières = features fd originales (jusqu'à a_days_rest inclus)
BASE_FEATURE_COLS = FEATURE_COLS[:53]

# Features SF optionnelles — XGBoost les utilise quand non-NaN, les ignore sinon
SF_FEATURE_COLS = FEATURE_COLS[53:]

# Colonnes à passer au modèle : base + SF présentes dans le df
def get_active_features(df: pd.DataFrame) -> list[str]:
    """Retourne les features disponibles dans df (SF incluses si présentes)."""
    present_sf = [c for c in SF_FEATURE_COLS if c in df.columns]
    return BASE_FEATURE_COLS + present_sf


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════════

def walk_forward_splits(df: pd.DataFrame, min_train_seasons: int = 5):
    """
    Saison 1 = plus récente, N = plus ancienne.
    Splits : train (≥ train_s) | val (= val_s) | test (= test_s)
    dropna strict sur BASE_FEATURE_COLS seulement ; les SF cols peuvent rester NaN.
    """
    n = df['season'].max()
    splits = []

    for test_s in range(1, n - min_train_seasons):
        val_s   = test_s + 1
        train_s = test_s + 2

        train = df[df['season'] >= train_s].dropna(subset=BASE_FEATURE_COLS + [TARGET]).copy()
        val   = df[df['season'] == val_s].dropna(subset=BASE_FEATURE_COLS + [TARGET]).copy()
        test  = df[df['season'] == test_s].dropna(subset=BASE_FEATURE_COLS + [TARGET]).copy()

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


def train_model(train: pd.DataFrame, val: pd.DataFrame,
                feat_cols: list[str] | None = None):
    cols = feat_cols if feat_cols is not None else get_active_features(train)
    # Assure que les deux splits ont les mêmes colonnes
    cols = [c for c in cols if c in train.columns and c in val.columns]

    X_train, y_train = train[cols], train[TARGET]
    X_val,   y_val   = val[cols],   val[TARGET]

    n_under = y_train.sum()
    n_over  = len(y_train) - n_under
    spw = n_over / n_under if n_under > 0 else 1.0

    xgb_model = _make_xgb(scale_pos_weight=spw)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibration sigmoid (Platt scaling) — plus régularisée qu'isotonique
    # sur de petits sets de calibration, réduit la surconfiance sélective.
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_model), method='sigmoid')
    calibrated.fit(X_val, y_val)
    # Mémorise les colonnes utilisées sur le modèle pour predict cohérent
    calibrated._feature_cols = cols
    return calibrated


def run_walk_forward(df: pd.DataFrame,
                     min_train_seasons: int = 5,
                     predict_tier: int = None):
    """
    Entraîne sur toutes les données disponibles dans df.
    Si predict_tier est fourni, ne retourne les prédictions que pour ce tier.
    """
    splits    = walk_forward_splits(df, min_train_seasons)
    all_preds = []
    feat_cols = get_active_features(df)   # déterminé une fois sur df complet

    for i, (train, val, test) in enumerate(splits):
        print(f'  Fold {i+1}/{len(splits)}...', end=' ', flush=True)
        model = train_model(train, val, feat_cols=feat_cols)
        cols  = model._feature_cols
        test  = test.copy()
        test['prob_model'] = model.predict_proba(test[cols])[:, 1]
        test['fold_id']    = i    # fold 0 = saison la plus récente

        if predict_tier is not None:
            test = test[test['tier'] == predict_tier]

        all_preds.append(test)
        print('OK')

    df_wf = pd.concat(all_preds, ignore_index=True)

    # Recalibration isotonique WALK-FORWARD : pour chaque fold, l'isotonique est
    # fittée uniquement sur les prédictions OOS des folds PLUS ANCIENS (season id
    # plus grand). L'ancienne version fittait une isotonique globale sur tout
    # l'OOS — y compris le futur de chaque pari — ce qui constituait un data
    # leakage et gonflait le ROI backtest d'environ +2 pts.
    df_wf['prob_model_raw'] = df_wf['prob_model']
    calibrated_parts = []
    for i in sorted(df_wf['fold_id'].unique()):
        cur  = df_wf[df_wf['fold_id'] == i].copy()
        past = df_wf[df_wf['fold_id'] > i]          # folds plus anciens uniquement
        if len(past) >= 1000:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(past['prob_model_raw'].values, past[TARGET].values)
            cur['prob_model'] = iso.predict(cur['prob_model_raw'].values)
        # sinon : on garde la calibration sigmoid par fold (déjà sans leakage)
        calibrated_parts.append(cur)
    df_wf = pd.concat(calibrated_parts, ignore_index=True)

    auc   = roc_auc_score(df_wf[TARGET].values, df_wf['prob_model'].values)
    auc_raw = roc_auc_score(df_wf[TARGET].values, df_wf['prob_model_raw'].values)
    acc   = ((df_wf['prob_model'] > 0.5) == df_wf[TARGET]).mean()
    tier_info = f' | tier={predict_tier}' if predict_tier else ''
    print(f'\n  Walk-forward : {len(df_wf)} matchs{tier_info} | AUC={auc:.4f} (raw={auc_raw:.4f}) | Acc={acc:.1%}')

    # Calibration check par bucket
    bins = np.arange(0.30, 0.85, 0.05)
    df_wf['_bucket'] = pd.cut(df_wf['prob_model'], bins=bins)
    cal = df_wf.groupby('_bucket', observed=True).agg(
        n=(TARGET, 'count'),
        actual=(TARGET, 'mean'),
        pred=('prob_model', 'mean'),
    ).assign(bias=lambda x: x['actual'] - x['pred'])
    print('  Calibration post-iso :')
    for _, r in cal.iterrows():
        bar = '█' * int(abs(r['bias']) * 100 / 2)
        sign = '+' if r['bias'] > 0 else '-'
        print(f'    {r.name}: n={int(r["n"]):>4} actual={r["actual"]:.2f} pred={r["pred"]:.2f} bias={sign}{abs(r["bias"]):.3f} {bar}')
    df_wf = df_wf.drop(columns=['_bucket'])

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

    feat_names = getattr(model, '_feature_cols', None) or getattr(base, 'feature_names_in_', None) or FEATURE_COLS[:len(base.feature_importances_)]
    imp = pd.Series(base.feature_importances_, index=feat_names).sort_values(ascending=False)
    top5 = imp.head(5)
    print('  Top features : ' + ' | '.join(f'{f}={s:.3f}' for f, s in top5.items()))
    return imp


def save_model(model, path: str = 'model.pkl'):
    joblib.dump(model, path)
    print(f'Modèle sauvegardé -> {path}')


def load_model(path: str = 'model.pkl'):
    return joblib.load(path)
