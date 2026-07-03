"""
main.py — Pipeline value bet walk-forward sur Div2 — Under 2.5
Usage:
    python main.py
    python main.py --edge 0.05
    python main.py --download
"""

import argparse
import os

import numpy as np
import pandas as pd

from download import download_all
from load import load_all
from features import build_features
from model import run_walk_forward, feature_importance, save_model
from backtest import (
    detect_value_bets, simulate_roi, print_summary,
    test_significance, plot_results, export_value_bets,
    optimize_edge_threshold,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CSV        = os.path.join(_HERE, 'csv')
# fix : le fichier est à la racine du repo, pas dans src/ (avant ce fix,
# 0 rivalité chargée → composante rivalité de match_importance toujours nulle)
DEFAULT_RIVALS     = os.path.join(os.path.dirname(_HERE), 'teams_by_country.csv')
DEFAULT_MODEL      = os.path.join(_HERE, 'model.pkl')
DEFAULT_EDGE       = 0.05
DEFAULT_SF_DIR     = os.path.join(os.path.dirname(_HERE), 'data', 'sofascore')

# ⚠ L'ancienne exclusion {'D2','SP2','I2'} était choisie d'après le ROI observé
# sur le test set lui-même = sélection a posteriori (overfitting du backtest,
# ~+2 pts de ROI artificiels). Le périmètre honnête = toutes les ligues.
EXCLUDED_DIVS: set = set()


def run_full(csv_root, rivals_csv, edge_min, model_out,
             sofascore_dir=None, download=False, update_only=False):

    print('\n' + '='*55)
    print(f'  VALUE BET — Div2 Walk-forward — UNDER 2.5')
    print('='*55)

    if download:
        download_all(csv_root=csv_root, n_seasons=25, update_only=update_only)

    target   = 'under_25'
    odds_col = 'Avg<2.5'

    # ── 1. Chargement
    df_all = load_all(csv_root)

    # ── 2. Features sur toutes les divisions (plus de données d'entraînement)
    df_feat = build_features(df_all, rivals_csv=rivals_csv, sofascore_dir=sofascore_dir)

    # Assigner prob_bookie = under ici, après build_features
    if 'prob_bookie_under' in df_feat.columns:
        df_feat['prob_bookie'] = df_feat['prob_bookie_under']
    else:
        df_feat['prob_bookie'] = 1.0 - df_feat['prob_bookie_over']

    if odds_col not in df_feat.columns:
        df_feat[odds_col] = 1.0 / df_feat['prob_bookie'].replace(0, np.nan)

    before = len(df_feat)
    df_feat = df_feat.dropna(subset=['prob_bookie', target, odds_col]).copy()
    dropped = before - len(df_feat)
    if dropped > 0:
        print(f'  ⚠ {dropped} matchs sans cotes exclus ({len(df_feat)} retenus)')

    # ── 3. Walk-forward : entraîne sur Div1+Div2, prédit sur Div2 uniquement
    df_wf, model = run_walk_forward(df_feat, predict_tier=2)
    feature_importance(model)
    save_model(model, model_out)

    # ── 4. Backtest
    if odds_col not in df_wf.columns:
        keys = ['Div', 'Date', 'HomeTeam', 'AwayTeam']
        src  = df_feat[keys + [odds_col]].drop_duplicates(subset=keys)
        df_wf = df_wf.merge(src, on=keys, how='left')

    if 'prob_bookie_under' in df_wf.columns:
        df_wf['prob_bookie'] = df_wf['prob_bookie_under']

    df_wf_clean = df_wf.dropna(subset=['prob_model', 'prob_bookie', target, odds_col]).copy()

    # Exclure les ligues à ROI systématiquement négatif
    if EXCLUDED_DIVS:
        before_excl = len(df_wf_clean)
        df_wf_clean = df_wf_clean[~df_wf_clean['Div'].isin(EXCLUDED_DIVS)].copy()
        print(f'  Ligues exclues {EXCLUDED_DIVS} : {before_excl - len(df_wf_clean)} matchs retirés')

    vb = detect_value_bets(df_wf_clean, edge_min=edge_min,
                           odds_col=odds_col, odds_min=1.65, odds_max=2.35)
    if len(vb) == 0:
        print('Aucun value bet. Baisse le seuil edge.')
        return

    vb = simulate_roi(vb, odds_col=odds_col, target_col=target)
    print_summary(vb, odds_col=odds_col, target_col=target)
    test_significance(vb)

    # ROI @Max<2.5 (meilleure cote du marché = line shopping) : ~+4 pts vs Avg
    if 'Max<2.5' in vb.columns and vb['Max<2.5'].notna().any():
        vb_max = simulate_roi(vb.dropna(subset=['Max<2.5']),
                              odds_col='Max<2.5', target_col=target)
        roi_max = vb_max['profit'].sum() / len(vb_max) * 100
        print(f'\n  ROI @Max<2.5 (line shopping, {len(vb_max)} paris) : {roi_max:+.1f}%')
    plot_results(vb, save_path='backtest_under.png')
    export_value_bets(vb, save_path='value_bets_under.csv',
                      odds_col=odds_col, target_col=target)
    # ⚠ optimize_edge_threshold retiré du pipeline : choisir le seuil d'edge
    # d'après le ROI du test set = overfitting du backtest (cf. AUDIT.md, P3).
    # La fonction reste dispo dans backtest.py pour exploration uniquement.

    print(f'\n✅ Terminé | modele: model.pkl | paris: value_bets_under.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv',          default=DEFAULT_CSV)
    p.add_argument('--rivals',       default=DEFAULT_RIVALS)
    p.add_argument('--edge',         default=DEFAULT_EDGE, type=float)
    p.add_argument('--model',        default=DEFAULT_MODEL)
    p.add_argument('--sofascore-dir', default=DEFAULT_SF_DIR)
    p.add_argument('--download',     action='store_true')
    p.add_argument('--update',       action='store_true')
    a = p.parse_args()
    run_full(a.csv, a.rivals, a.edge, a.model,
             sofascore_dir=a.sofascore_dir,
             download=a.download, update_only=a.update)
