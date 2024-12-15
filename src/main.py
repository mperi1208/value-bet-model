"""
main.py — Pipeline value bet walk-forward sur Div2 — Under 2.5
Usage:
    python main.py
    python main.py --edge 0.05
    python main.py --download
"""

import argparse, importlib.util as _ilu, importlib, os as _os
import pandas as pd

_HERE = _os.path.dirname(_os.path.abspath(__file__))

def _load_mod(name, path):
    m = importlib.util.module_from_spec(s := _ilu.spec_from_file_location(name, path))
    s.loader.exec_module(m); return m

_d = _load_mod('d', _os.path.join(_HERE, '00_download.py'))
_l = _load_mod('l', _os.path.join(_HERE, '01_load.py'))
_f = _load_mod('f', _os.path.join(_HERE, '02_features.py'))
_m = _load_mod('m', _os.path.join(_HERE, '03_model.py'))
_b = _load_mod('b', _os.path.join(_HERE, '04_backtest.py'))

DEFAULT_CSV    = _os.path.join(_HERE, 'csv')
DEFAULT_RIVALS = _os.path.join(_HERE, 'teams_by_country.csv')
DEFAULT_MODEL  = _os.path.join(_HERE, 'model.pkl')
DEFAULT_EDGE   = 0.05


def run_full(csv_root, rivals_csv, edge_min, model_out,
             download=False, update_only=False):

    print('\n' + '='*55)
    print(f'  VALUE BET — Div2 Walk-forward — UNDER 2.5')
    print('='*55)

    if download:
        _d.download_all(csv_root=csv_root, n_seasons=25, update_only=update_only)

    target   = 'under_25'
    odds_col = 'Avg<2.5'

    # ── 1. Chargement
    df_all  = _l.load_all(csv_root)
    df_div2 = df_all[df_all['tier'] == 2].copy().reset_index(drop=True)

    # ── 2. Features
    df_feat = _f.build_features(df_div2, rivals_csv=rivals_csv)

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

    # ── 3. Walk-forward
    df_wf, model = _m.run_walk_forward(df_feat)
    _m.feature_importance(model)
    _m.save_model(model, model_out)

    # ── 4. Backtest
    if odds_col not in df_wf.columns:
        keys = ['Div', 'Date', 'HomeTeam', 'AwayTeam']
        src  = df_feat[keys + [odds_col]].drop_duplicates(subset=keys)
        df_wf = df_wf.merge(src, on=keys, how='left')

    # S'assurer que prob_bookie dans df_wf est bien l'under
    if 'prob_bookie_under' in df_wf.columns:
        df_wf['prob_bookie'] = df_wf['prob_bookie_under']

    df_wf_clean = df_wf.dropna(subset=['prob_model', 'prob_bookie', target, odds_col]).copy()

    vb = _b.detect_value_bets(df_wf_clean, edge_min=edge_min)
    if len(vb) == 0:
        print('Aucun value bet. Baisse le seuil edge.')
        return

    vb = _b.simulate_roi(vb, odds_col=odds_col, target_col=target)
    _b.print_summary(vb, odds_col=odds_col, target_col=target)
    _b.test_significance(vb)
    _b.plot_results(vb, save_path='backtest_under.png')
    _b.export_value_bets(vb, save_path='value_bets_under.csv',
                         odds_col=odds_col, target_col=target)
    _b.optimize_edge_threshold(df_wf_clean, odds_col=odds_col, target_col=target)

    print(f'\n✅ Terminé | modele: model.pkl | paris: value_bets_under.csv')


if __name__ == '__main__':
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--csv',      default=DEFAULT_CSV)
    p.add_argument('--rivals',   default=DEFAULT_RIVALS)
    p.add_argument('--edge',     default=DEFAULT_EDGE, type=float)
    p.add_argument('--model',    default=DEFAULT_MODEL)
    p.add_argument('--download', action='store_true')
    p.add_argument('--update',   action='store_true')
    a = p.parse_args()
    run_full(a.csv, a.rivals, a.edge, a.model,
             download=a.download, update_only=a.update)
