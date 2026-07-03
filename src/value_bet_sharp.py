"""
value_bet_sharp.py — Stratégie « sharp anchor » multi-marchés (sans ML).
------------------------------------------------------------------------
Constat de l'audit 2026 (AUDIT.md) : le XGBoost n'apporte aucun signal au-delà
du marché. Le seul edge positif reproductible dans ces données est structurel :

    fair prob  = probabilité no-vig Pinnacle (dévig « power », qui corrige le
                 biais favori-outsider de la normalisation multiplicative)
    pari       = meilleure cote du marché (Max)
    condition  = EV = fair_prob × cote_max − 1 > seuil (défaut 0.02)

Marchés couverts : 1X2 (Pinnacle dispo depuis ~2012), Over/Under 2.5 et
Asian Handicap (depuis ~2019). Validation par CLV : EV recalculé contre la
closing line no-vig Pinnacle — un CLV moyen positif est la signature d'un
edge réel, indépendamment du bruit du ROI réalisé.

Résultats backtest (10 ligues, EV > 0.02, cotes @Max) :
    1X2 : n=17 890 | ROI +4.8% | IC95 [+2.2, +7.4] | CLV +3.2% (69% > 0)
          positif 10/13 saisons, 9/10 ligues, sur les 3 issues, sur les 2 ères
    O/U : n= 1 287 | ROI +7.6% | IC95 [+1.2, +14.1] | CLV +4.2%
    AH  : n= 1 346 | ROI +4.7% | IC95 [−0.2, +9.6]  | CLV +1.6%

Limites honnêtes :
  - capturer la cote Max suppose des comptes chez de nombreux bookmakers ;
    en B365-only le ROI 1X2 tombe à +1.6% (CLV +1.4%, toujours positif) ;
  - les bookmakers soft limitent les comptes gagnants — c'est LA contrainte
    opérationnelle réelle de cette stratégie ;
  - les cotes football-data sont des snapshots : l'exécution réelle diffère.

Usage :
    python value_bet_sharp.py                          # tous marchés, EV 0.02
    python value_bet_sharp.py --markets 1x2 --ev 0.05
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))

KEY_COLS = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
ODDS_COLS = [
    # 1X2 : Pinnacle pré-match + closing, Max marché
    'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA',
    'MaxH', 'MaxD', 'MaxA', 'BbMxH', 'BbMxD', 'BbMxA',
    # O/U 2.5
    'P>2.5', 'P<2.5', 'PC>2.5', 'PC<2.5',
    'Max>2.5', 'Max<2.5', 'BbMx>2.5', 'BbMx<2.5',
    # Asian Handicap
    'AHh', 'BbAHh', 'PAHH', 'PAHA', 'PCAHH', 'PCAHA',
    'MaxAHH', 'MaxAHA', 'BbMxAHH', 'BbMxAHA',
]
# Harmonisation ère Bb (fichiers antérieurs à 2019)
BB_FALLBACKS = [('MaxH', 'BbMxH'), ('MaxD', 'BbMxD'), ('MaxA', 'BbMxA'),
                ('Max>2.5', 'BbMx>2.5'), ('Max<2.5', 'BbMx<2.5'),
                ('AHh', 'BbAHh'), ('MaxAHH', 'BbMxAHH'), ('MaxAHA', 'BbMxAHA')]


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════════

def load_odds(csv_root: str) -> pd.DataFrame:
    frames = []
    for f in glob.glob(os.path.join(csv_root, '*', '*.csv')):
        try:
            d = pd.read_csv(f, low_memory=False, encoding='latin-1',
                            on_bad_lines='skip')
        except Exception:
            continue
        keep = [c for c in KEY_COLS + ODDS_COLS if c in d.columns]
        if 'FTR' not in keep:
            continue
        frames.append(d[keep])

    od = pd.concat(frames, ignore_index=True)
    od['Date'] = pd.to_datetime(od['Date'], format='mixed', dayfirst=True,
                                errors='coerce')
    for c in od.columns:
        if c not in ('Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTR'):
            od[c] = pd.to_numeric(od[c], errors='coerce')
    for new, old in BB_FALLBACKS:
        if old in od.columns:
            od[new] = od.get(new, np.nan)
            od[new] = od[new].fillna(od[old])

    od = od.dropna(subset=['Date', 'FTR'])
    od = od.drop_duplicates(subset=['Div', 'Date', 'HomeTeam', 'AwayTeam'])
    od['season_year'] = np.where(od['Date'].dt.month >= 7,
                                 od['Date'].dt.year, od['Date'].dt.year - 1)
    return od.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# DÉVIG « POWER »
# ═══════════════════════════════════════════════════════════════

def novig_power(odds_cols: list) -> np.ndarray:
    """Résout sum((1/o_i)^k) = 1. Contrairement à la normalisation
    multiplicative, ne surestime pas la fair prob des outsiders
    (biais favori-outsider). Retourne un array (n, n_issues)."""
    arr = np.column_stack([np.asarray(o, dtype=float) for o in odds_cols])
    arr[arr <= 1.0] = np.nan          # cotes invalides (0 ou ≤1) → ignorées
    inv = 1.0 / arr
    out = np.full_like(inv, np.nan)
    for i in range(len(inv)):
        r = inv[i]
        if np.any(~np.isfinite(r)) or r.sum() <= 1.0:
            if np.all(np.isfinite(r)):
                out[i] = r / r.sum()
            continue
        try:
            k = brentq(lambda k: (r ** k).sum() - 1, 0.5, 3.0)
            out[i] = r ** k
        except Exception:
            out[i] = r / r.sum()
    return out


# ═══════════════════════════════════════════════════════════════
# DÉTECTION PAR MARCHÉ
# ═══════════════════════════════════════════════════════════════

def _clv(fair_close_col, odds):
    return fair_close_col * odds - 1


def bets_1x2(od: pd.DataFrame, ev_min: float) -> pd.DataFrame:
    d = od.dropna(subset=['PSH', 'PSD', 'PSA', 'MaxH', 'MaxD', 'MaxA']).copy()
    fair = novig_power([d['PSH'], d['PSD'], d['PSA']])
    fairc = np.full_like(fair, np.nan)
    hc = d[['PSCH', 'PSCD', 'PSCA']].notna().all(axis=1).values
    if hc.any():
        fairc[np.where(hc)[0]] = novig_power(
            [d.loc[hc, 'PSCH'], d.loc[hc, 'PSCD'], d.loc[hc, 'PSCA']])
    bets = []
    for j, side in enumerate(['H', 'D', 'A']):
        oc = f'Max{side}'
        ev = fair[:, j] * d[oc].values - 1
        mask = ev > ev_min
        sel = d[mask].copy()
        sel['market'] = '1X2'
        sel['side'] = side
        sel['odds_bet'] = sel[oc]
        sel['ev_pre'] = ev[mask]
        sel['profit'] = np.where(sel['FTR'] == side, sel[oc] - 1, -1.0)
        sel['clv_ev'] = _clv(fairc[mask][:, j], sel[oc].values)
        bets.append(sel)
    return pd.concat(bets) if bets else pd.DataFrame()


def bets_ou(od: pd.DataFrame, ev_min: float) -> pd.DataFrame:
    d = od.dropna(subset=['P<2.5', 'P>2.5', 'Max<2.5', 'Max>2.5',
                          'FTHG', 'FTAG']).copy()
    fair = novig_power([d['P<2.5'], d['P>2.5']])
    fairc = np.full_like(fair, np.nan)
    hc = d[['PC<2.5', 'PC>2.5']].notna().all(axis=1).values
    if hc.any():
        fairc[np.where(hc)[0]] = novig_power(
            [d.loc[hc, 'PC<2.5'], d.loc[hc, 'PC>2.5']])
    under = (d['FTHG'] + d['FTAG'] < 2.5).values
    bets = []
    for j, (side, oc, win) in enumerate([('U', 'Max<2.5', True),
                                         ('O', 'Max>2.5', False)]):
        ev = fair[:, j] * d[oc].values - 1
        mask = ev > ev_min
        sel = d[mask].copy()
        sel['market'] = 'OU2.5'
        sel['side'] = side
        sel['odds_bet'] = sel[oc]
        sel['ev_pre'] = ev[mask]
        sel['profit'] = np.where(under[mask] == win, sel[oc] - 1, -1.0)
        sel['clv_ev'] = _clv(fairc[mask][:, j], sel[oc].values)
        bets.append(sel)
    return pd.concat(bets) if bets else pd.DataFrame()


def _settle_ah(diff: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Profit d'un pari AH pour une mise 1. diff = marge du côté parié,
    handicap inclus. Les quarts de ligne sont splittés en deux demi-mises."""
    def half(dd):
        return np.where(dd > 1e-9, odds - 1, np.where(np.abs(dd) < 1e-9, 0.0, -1.0))
    is_quarter = ~np.isclose(diff * 2, np.round(diff * 2))
    lo = np.where(is_quarter, diff - 0.25, diff)
    hi = np.where(is_quarter, diff + 0.25, diff)
    return 0.5 * half(lo) + 0.5 * half(hi)


def bets_ah(od: pd.DataFrame, ev_min: float) -> pd.DataFrame:
    d = od.dropna(subset=['AHh', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
                          'FTHG', 'FTAG']).copy()
    fair = novig_power([d['PAHH'], d['PAHA']])
    fairc = np.full_like(fair, np.nan)
    hc = d[['PCAHH', 'PCAHA']].notna().all(axis=1).values
    if hc.any():
        fairc[np.where(hc)[0]] = novig_power(
            [d.loc[hc, 'PCAHH'], d.loc[hc, 'PCAHA']])
    diff_home = (d['FTHG'] - d['FTAG'] + d['AHh']).values
    bets = []
    for j, (side, oc, sign) in enumerate([('AH_H', 'MaxAHH', 1),
                                          ('AH_A', 'MaxAHA', -1)]):
        ev = fair[:, j] * d[oc].values - 1
        mask = ev > ev_min
        sel = d[mask].copy()
        sel['market'] = 'AH'
        sel['side'] = side
        sel['odds_bet'] = sel[oc]
        sel['ev_pre'] = ev[mask]
        sel['profit'] = _settle_ah(diff_home[mask] * sign, sel[oc].values)
        sel['clv_ev'] = _clv(fairc[mask][:, j], sel[oc].values)
        bets.append(sel)
    return pd.concat(bets) if bets else pd.DataFrame()


MARKETS = {'1x2': bets_1x2, 'ou': bets_ou, 'ah': bets_ah}


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def report(vb: pd.DataFrame, n_bootstrap: int = 5000):
    p = vb['profit'].values
    rng = np.random.default_rng(42)
    bm = np.array([rng.choice(p, len(p)).mean()
                   for _ in range(n_bootstrap)]) * 100
    lo, hi = np.percentile(bm, [2.5, 97.5])
    clv = vb['clv_ev'].dropna()

    print(f'\n{"═" * 62}')
    print(f'  SHARP ANCHOR MULTI-MARCHÉS | {len(vb)} paris')
    print(f'  ROI : {p.mean() * 100:+.2f}% | IC95 bootstrap [{lo:+.1f}, {hi:+.1f}]')
    if len(clv):
        print(f'  CLV vs Pinnacle closing : {clv.mean() * 100:+.2f}% '
              f'({(clv > 0).mean():.0%} des paris battent la close)')
    print(f'{"═" * 62}')
    for name, key in [('Par marché', 'market'), ('Par saison', 'season_year'),
                      ('Par ligue', 'Div')]:
        print(f'  {name} :')
        g = vb.groupby(key).agg(n=('profit', 'count'),
                                roi=('profit', lambda x: x.mean() * 100))
        for k, r in g.iterrows():
            print(f'    {str(k):>6} : {int(r["n"]):>5} paris | ROI {r["roi"]:+.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_HERE, 'csv'))
    ap.add_argument('--ev', type=float, default=0.02)
    ap.add_argument('--markets', default='1x2,ou,ah',
                    help='parmi : 1x2, ou, ah (séparés par des virgules)')
    ap.add_argument('--out', default='value_bets_sharp.csv')
    a = ap.parse_args()

    od = load_odds(a.csv)
    print(f'  {len(od)} matchs | Pinnacle 1X2 : {od["PSH"].notna().mean():.0%} '
          f'| O/U : {od["P<2.5"].notna().mean():.0%} '
          f'| AH : {od["PAHH"].notna().mean():.0%}')

    parts = [MARKETS[mk.strip()](od, a.ev) for mk in a.markets.split(',')]
    vb = pd.concat(parts).sort_values('Date').reset_index(drop=True)
    report(vb)

    cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'season_year', 'market',
            'side', 'odds_bet', 'ev_pre', 'clv_ev', 'FTR', 'FTHG', 'FTAG',
            'profit']
    vb[[c for c in cols if c in vb.columns]].to_csv(a.out, index=False)
    print(f'\n  💾 Paris exportés → {a.out}')
