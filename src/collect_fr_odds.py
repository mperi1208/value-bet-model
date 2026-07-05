"""
collect_fr_odds.py — Mesure l'edge sharp anchor sur les bookmakers FRANÇAIS.
-----------------------------------------------------------------------------
Le backtest historique (AUDIT.md) est calculé sur les books internationaux de
football-data ; les cotes des opérateurs ANJ (Betclic, Winamax, Unibet.fr,
PMU, NetBet) n'existent pas en archive publique. Ce script mesure donc l'edge
français PROSPECTIVEMENT via The Odds API (région 'fr' + Pinnacle en 'eu').

À chaque exécution :
  1. récupère 1X2 + O/U 2.5 sur les 10 ligues du projet ;
  2. fair prob = Pinnacle no-vig (dévig power) ;
  3. logge chaque cote FR avec EV > --ev dans paper_trades_fr.csv ;
  4. logge aussi, pour CHAQUE pick, la meilleure cote FR (même sans value)
     → permet de mesurer l'écart books FR vs cote Max du backtest.

Setup :
  1. clé gratuite sur https://the-odds-api.com (500 crédits/mois)
  2. export ODDS_API_KEY=xxxx
  3. python collect_fr_odds.py            # chaque vendredi
  Coût : ~4 crédits/ligue/passage (2 régions × 2 marchés) ≈ 160/mois.

Évaluation : python value_bet_sharp.py --evaluate --log paper_trades_fr.csv
(après téléchargement des résultats via main.py --download --update)
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests

API = 'https://api.the-odds-api.com/v4'

SPORT_KEYS = {
    'E0': 'soccer_epl',
    'E1': 'soccer_efl_champ',
    'F1': 'soccer_france_ligue_one',
    'F2': 'soccer_france_ligue_two',
    'D1': 'soccer_germany_bundesliga',
    'D2': 'soccer_germany_bundesliga2',
    'I1': 'soccer_italy_serie_a',
    'I2': 'soccer_italy_serie_b',
    'SP1': 'soccer_spain_la_liga',
    'SP2': 'soccer_spain_segunda_division',
}
FR_BOOKS = {'betclic_fr', 'winamax_fr', 'unibet_fr', 'pmu_fr', 'netbet_fr'}
ANCHOR = 'pinnacle'


def novig_power(odds: list[float]) -> list[float]:
    from scipy.optimize import brentq
    inv = np.array([1.0 / o for o in odds])
    if inv.sum() <= 1.0:
        return list(inv / inv.sum())
    try:
        k = brentq(lambda k: (inv ** k).sum() - 1, 0.5, 3.0)
        return list(inv ** k)
    except Exception:
        return list(inv / inv.sum())


def fetch_league(div: str, key: str, api_key: str) -> list[dict]:
    r = requests.get(f'{API}/sports/{key}/odds', params={
        'apiKey': api_key,
        'regions': 'fr,eu',
        'markets': 'h2h,totals',
        'oddsFormat': 'decimal',
    }, timeout=30)
    r.raise_for_status()
    remaining = r.headers.get('x-requests-remaining')
    print(f'  {div}: {len(r.json())} matchs (crédits restants: {remaining})')
    rows = []
    for ev in r.json():
        books = {b['key']: b for b in ev.get('bookmakers', [])}
        if ANCHOR not in books:
            continue
        base = {'Div': div, 'commence': ev['commence_time'],
                'HomeTeam': ev['home_team'], 'AwayTeam': ev['away_team'],
                'collected_at': pd.Timestamp.now().isoformat(timespec='minutes')}

        # ── 1X2 ────────────────────────────────────────────────────
        anchor_h2h = _market(books[ANCHOR], 'h2h')
        if anchor_h2h:
            names = [ev['home_team'], 'Draw', ev['away_team']]
            p_odds = [_price(anchor_h2h, n) for n in names]
            if all(p_odds):
                fair = novig_power(p_odds)
                for i, (name, side) in enumerate(zip(names, ['H', 'D', 'A'])):
                    best_o, best_b = _best_fr(books, 'h2h', name)
                    if best_o is None:
                        continue
                    rows.append(base | {
                        'market': '1X2', 'side': side, 'book': best_b,
                        'odds_bet': best_o, 'fair_p': fair[i],
                        'pinnacle_odds': p_odds[i],
                        'ev_pre': fair[i] * best_o - 1,
                    })

        # ── O/U 2.5 ────────────────────────────────────────────────
        anchor_tot = _market(books[ANCHOR], 'totals', point=2.5)
        if anchor_tot:
            p_odds = [_price(anchor_tot, 'Under'), _price(anchor_tot, 'Over')]
            if all(p_odds):
                fair = novig_power(p_odds)
                for i, name_side in enumerate([('Under', 'U'), ('Over', 'O')]):
                    name, side = name_side
                    best_o, best_b = _best_fr(books, 'totals', name, point=2.5)
                    if best_o is None:
                        continue
                    rows.append(base | {
                        'market': 'OU2.5', 'side': side, 'book': best_b,
                        'odds_bet': best_o, 'fair_p': fair[i],
                        'pinnacle_odds': p_odds[i],
                        'ev_pre': fair[i] * best_o - 1,
                    })
    return rows


def _market(book: dict, key: str, point: float = None):
    for mk in book.get('markets', []):
        if mk['key'] != key:
            continue
        if point is None:
            return mk['outcomes']
        outs = [o for o in mk['outcomes'] if o.get('point') == point]
        if outs:
            return outs
    return None


def _price(outcomes: list, name: str):
    for o in outcomes:
        if o['name'] == name:
            return o['price']
    return None


def _best_fr(books: dict, market: str, name: str, point: float = None):
    best_o, best_b = None, None
    for bk, bdata in books.items():
        if bk not in FR_BOOKS:
            continue
        outs = _market(bdata, market, point)
        if not outs:
            continue
        o = _price(outs, name)
        if o and (best_o is None or o > best_o):
            best_o, best_b = o, bk
    return best_o, best_b


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ev', type=float, default=0.02)
    ap.add_argument('--log', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'paper_trades_fr.csv'))
    ap.add_argument('--all-log', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'fr_odds_snapshots.csv'))
    a = ap.parse_args()

    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        raise SystemExit('export ODDS_API_KEY=... (clé gratuite sur the-odds-api.com)')

    all_rows = []
    for div, key in SPORT_KEYS.items():
        try:
            all_rows += fetch_league(div, key, api_key)
        except Exception as e:
            print(f'  ⚠ {div}: {e}')

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        raise SystemExit('Aucune donnée (intersaison ?).')

    # snapshot complet (pour mesurer l'écart FR vs backtest, même hors value)
    header = not os.path.exists(a.all_log)
    df.to_csv(a.all_log, mode='a', header=header, index=False)

    picks = df[df['ev_pre'] > a.ev].copy()
    print(f'\n  {len(df)} cotes FR loggées | {len(picks)} value bets EV>{a.ev:.0%}')
    if len(picks):
        picks['stake_pct'] = np.minimum(
            0.25 * picks['ev_pre'] / (picks['odds_bet'] - 1), 0.02)
        keys = ['Div', 'commence', 'HomeTeam', 'AwayTeam', 'market', 'side']
        if os.path.exists(a.log):
            old = pd.read_csv(a.log)
            merged = pd.concat([old, picks]).drop_duplicates(subset=keys, keep='first')
        else:
            merged = picks
        merged.to_csv(a.log, index=False)
        print(picks[['Div', 'HomeTeam', 'AwayTeam', 'market', 'side', 'book',
                     'odds_bet', 'ev_pre', 'stake_pct']].to_string(index=False))
