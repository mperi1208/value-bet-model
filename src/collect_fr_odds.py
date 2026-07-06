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

# Ligues foot du backtest (clé -> code Div historique)
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
# Mode --all-sports : tout sport actif de ces groupes est collecté
# (le mécanisme sharp anchor est agnostique au sport — l'edge par sport
#  doit néanmoins être MESURÉE avant d'être pariée, cf. AUDIT.md)
ALL_SPORT_GROUPS = {'Soccer', 'Basketball', 'American Football', 'Baseball',
                    'Tennis', 'Ice Hockey', 'Mixed Martial Arts', 'Rugby League'}
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


def list_active_sports(api_key: str) -> dict:
    """Sports actifs des groupes couverts (coût : 0 crédit)."""
    r = requests.get(f'{API}/sports', params={'apiKey': api_key}, timeout=30)
    r.raise_for_status()
    return {s['key']: s['title'] for s in r.json()
            if s['group'] in ALL_SPORT_GROUPS
            and s['active'] and not s['has_outrights']}


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
    now_utc = pd.Timestamp.now(tz='UTC')
    for ev in r.json():
        # ⚠ filtre in-play : un match déjà commencé mélange cotes live
        # (Pinnacle) et cotes pré-match figées (books FR) → faux EV énormes.
        # Vu en réel : PMU affichait Bublik 5.1 vs Pinnacle live 4.1 alors
        # que le match était en cours — EV illusoire de +15%.
        if pd.Timestamp(ev['commence_time']) <= now_utc:
            continue
        books = {b['key']: b for b in ev.get('bookmakers', [])}
        if ANCHOR not in books:
            continue
        base = {'Div': div, 'commence': ev['commence_time'],
                'HomeTeam': ev['home_team'], 'AwayTeam': ev['away_team'],
                'collected_at': pd.Timestamp.now().isoformat(timespec='minutes')}

        # ── h2h : générique 2 issues (tennis, MMA...) ou 3 (foot...) ─
        # Les noms d'issues sont pris chez Pinnacle, pas hardcodés.
        anchor_h2h = _market(books[ANCHOR], 'h2h')
        if anchor_h2h:
            names = [o['name'] for o in anchor_h2h]
            p_odds = [o['price'] for o in anchor_h2h]
            if all(p_odds) and len(p_odds) in (2, 3):
                fair = novig_power(p_odds)
                for i, name in enumerate(names):
                    side = ('D' if name == 'Draw' else
                            'H' if name == ev['home_team'] else 'A')
                    best_o, best_b = _best_fr(books, 'h2h', name)
                    if best_o is None:
                        continue
                    rows.append(base | {
                        'market': 'h2h', 'side': side, 'book': best_b,
                        'odds_bet': best_o, 'fair_p': fair[i],
                        'pinnacle_odds': p_odds[i],
                        'ev_pre': fair[i] * best_o - 1,
                    })

        # ── totals : ligne principale de Pinnacle (2.5 au foot,
        #    variable en NBA/NFL/MLB...) — on matche le même point FR ─
        anchor_tot = _market(books[ANCHOR], 'totals')
        if anchor_tot:
            points = sorted({o.get('point') for o in anchor_tot
                             if o.get('point') is not None})
            main_pt = points[len(points) // 2] if points else None
            outs = [o for o in anchor_tot if o.get('point') == main_pt]
            p_under = _price(outs, 'Under')
            p_over = _price(outs, 'Over')
            if main_pt is not None and p_under and p_over:
                fair = novig_power([p_under, p_over])
                for i, (name, side) in enumerate([('Under', 'U'), ('Over', 'O')]):
                    best_o, best_b = _best_fr(books, 'totals', name, point=main_pt)
                    if best_o is None:
                        continue
                    rows.append(base | {
                        'market': f'OU{main_pt}', 'side': side, 'book': best_b,
                        'odds_bet': best_o, 'fair_p': fair[i],
                        'pinnacle_odds': p_under if side == 'U' else p_over,
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
    ap.add_argument('--all-sports', action='store_true',
                    help='collecte TOUS les sports actifs (basket, tennis, '
                    'NFL, MLB...) au lieu des 10 ligues foot. '
                    '~4 crédits/sport/passage — surveille ton budget.')
    a = ap.parse_args()

    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        raise SystemExit('export ODDS_API_KEY=... (clé gratuite sur the-odds-api.com)')

    if a.all_sports:
        targets = list_active_sports(api_key)
        print(f'  {len(targets)} sports actifs : {", ".join(sorted(targets))}')
    else:
        targets = {k: k for k in SPORT_KEYS.values()}

    div_of = {v: d for d, v in SPORT_KEYS.items()}
    all_rows = []
    for key in targets:
        div = div_of.get(key, key)   # code Div historique si ligue du backtest
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
