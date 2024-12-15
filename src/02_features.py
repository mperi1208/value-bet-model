"""
02_features.py
--------------
v3 — Améliorations :
  - Nouvelles features under-spécifiques : taux under rolling, variance buts,
    defensive solidity, low-scoring streak
  - Windows multiples : 3, 5, 10 matchs
  - prob_bookie N'EST PAS initialisé ici (assigné dans main.py selon le marché)
"""

import pandas as pd
import numpy as np
from typing import Optional

ROLLING_WINDOW = 5


# ═══════════════════════════════════════════════════════════════
# 1. STATS ROLLING PAR ÉQUIPE
# ═══════════════════════════════════════════════════════════════

def _compute_team_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    home = df[['Div', 'season', 'Date', 'HomeTeam',
               'FTHG', 'FTAG', 'FTR',
               'HS', 'HST', 'HC', 'HY', 'HR']].copy()
    home.columns = ['Div', 'season', 'Date', 'team',
                    'scored', 'conceded', 'FTR',
                    'shots', 'sot', 'corners', 'yellow', 'red']
    home['is_home'] = True
    home['orig_idx'] = df.index

    away = df[['Div', 'season', 'Date', 'AwayTeam',
               'FTAG', 'FTHG', 'FTR',
               'AS', 'AST', 'AC', 'AY', 'AR']].copy()
    away.columns = ['Div', 'season', 'Date', 'team',
                    'scored', 'conceded', 'FTR',
                    'shots', 'sot', 'corners', 'yellow', 'red']
    away['is_home'] = False
    away['orig_idx'] = df.index

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(['Div', 'season', 'team', 'Date']).reset_index(drop=True)

    long['result'] = 0
    long.loc[(long['is_home']) & (long['FTR'] == 'H'), 'result'] = 1
    long.loc[(~long['is_home']) & (long['FTR'] == 'A'), 'result'] = 1
    long.loc[(long['is_home']) & (long['FTR'] == 'A'), 'result'] = -1
    long.loc[(~long['is_home']) & (long['FTR'] == 'H'), 'result'] = -1
    long['is_null'] = (long['FTR'] == 'D').astype(float)

    # Total buts du match
    long['total_goals_match'] = long['scored'] + long['conceded']
    long['is_under'] = (long['total_goals_match'] < 2.5).astype(float)
    long['is_low_score'] = (long['total_goals_match'] <= 1).astype(float)

    grp = ['Div', 'season', 'team']
    stat_cols = ['scored', 'conceded', 'shots', 'sot', 'corners',
                 'yellow', 'red', 'is_null', 'is_under', 'is_low_score',
                 'total_goals_match']

    shifted = long.groupby(grp, group_keys=False)[stat_cols].shift(1)
    for col in stat_cols:
        long[f'avg_{col}'] = (
            shifted[col]
            .groupby([long['Div'], long['season'], long['team']])
            .expanding()
            .mean()
            .values
        )

    # Rolling form N matchs
    long['form'] = (
        long.groupby(grp, group_keys=False)['result']
        .shift(1)
        .groupby([long['Div'], long['season'], long['team']])
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).sum())
    )

    # Rolling under rate sur 3, 5, 10 matchs
    for w in [3, 5, 10]:
        long[f'under_rate_{w}'] = (
            long.groupby(grp, group_keys=False)['is_under']
            .shift(1)
            .groupby([long['Div'], long['season'], long['team']])
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

    # Rolling variance buts sur 5 matchs
    long['goals_var_5'] = (
        long.groupby(grp, group_keys=False)['total_goals_match']
        .shift(1)
        .groupby([long['Div'], long['season'], long['team']])
        .transform(lambda x: x.rolling(5, min_periods=2).std().fillna(0))
    )

    rename_map = {
        'avg_scored':            'avg_goals_scored',
        'avg_conceded':          'avg_goals_conceded',
        'avg_shots':             'avg_shots',
        'avg_sot':               'avg_sot',
        'avg_corners':           'avg_corners',
        'avg_yellow':            'avg_yellow',
        'avg_red':               'avg_red',
        'avg_is_null':           'null_pct',
        'avg_is_under':          'avg_under_rate',
        'avg_is_low_score':      'avg_low_score_rate',
        'avg_total_goals_match': 'avg_total_goals',
        'form':                  f'form_{ROLLING_WINDOW}',
        'under_rate_3':          'under_rate_3',
        'under_rate_5':          'under_rate_5',
        'under_rate_10':         'under_rate_10',
        'goals_var_5':           'goals_var_5',
    }

    keep = list(rename_map.keys()) + ['orig_idx', 'is_home']
    long_sub = long[keep].rename(columns=rename_map)

    home_stats = long_sub[long_sub['is_home']].drop(columns='is_home').set_index('orig_idx')
    away_stats = long_sub[~long_sub['is_home']].drop(columns='is_home').set_index('orig_idx')

    df = df.join(home_stats.add_prefix('h_'), how='left')
    df = df.join(away_stats.add_prefix('a_'), how='left')

    return df


# ═══════════════════════════════════════════════════════════════
# 2. CLASSEMENT DYNAMIQUE
# ═══════════════════════════════════════════════════════════════

def _compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h_ranks, a_ranks = {}, {}

    for (div, season), grp in df.groupby(['Div', 'season'], sort=False):
        grp = grp.sort_values('Date')
        teams = pd.concat([grp['HomeTeam'], grp['AwayTeam']]).unique()
        stats = {t: {'pts': 0, 'gd': 0} for t in teams}

        for idx, row in grp.iterrows():
            sorted_teams = sorted(stats.items(), key=lambda x: (-x[1]['pts'], -x[1]['gd']))
            rank_map = {t: i+1 for i, (t, _) in enumerate(sorted_teams)}
            h_ranks[idx] = rank_map.get(row['HomeTeam'])
            a_ranks[idx] = rank_map.get(row['AwayTeam'])

            h, a = row['HomeTeam'], row['AwayTeam']
            hg, ag = row['FTHG'], row['FTAG']
            if row['FTR'] == 'H':
                stats[h]['pts'] += 3
            elif row['FTR'] == 'A':
                stats[a]['pts'] += 3
            else:
                stats[h]['pts'] += 1
                stats[a]['pts'] += 1
            stats[h]['gd'] += (hg - ag)
            stats[a]['gd'] += (ag - hg)

    df['h_rank'] = pd.Series(h_ranks)
    df['a_rank'] = pd.Series(a_ranks)
    return df


# ═══════════════════════════════════════════════════════════════
# 3. MATCH IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def _build_rivalries(rivals_csv: Optional[str]) -> set:
    if rivals_csv is None or not __import__('os').path.exists(rivals_csv):
        return set()
    df = pd.read_csv(rivals_csv, sep=';', header=0)
    rivalries = set()
    if 'Team' in df.columns and 'Rivals' in df.columns:
        for _, row in df.iterrows():
            team = row.get('Team')
            rivals_raw = row.get('Rivals')
            if pd.isna(team) or pd.isna(rivals_raw):
                continue
            for rival in str(rivals_raw).split(','):
                rival = rival.strip()
                if rival:
                    rivalries.add(frozenset([team.strip(), rival]))
    return rivalries


def _compute_match_importance(df: pd.DataFrame, rivalries: set,
                               n_teams: int = 20) -> pd.DataFrame:
    df = df.copy()
    df['_date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['_season_year'] = df['_date_parsed'].apply(
        lambda d: d.year if pd.notna(d) and d.month >= 7 else (d.year - 1 if pd.notna(d) else np.nan)
    )
    df['_season_prog'] = df.groupby(['Div', '_season_year'])['_date_parsed'].rank(pct=True)

    relegation_zone = n_teams - 2
    title_zone = 4

    h_rank = df['h_rank'].copy()
    a_rank = df['a_rank'].copy()
    for div, grp_idx in df.groupby('Div').groups.items():
        h_rank.loc[grp_idx] = h_rank.loc[grp_idx].fillna(df.loc[grp_idx, 'h_rank'].median())
        a_rank.loc[grp_idx] = a_rank.loc[grp_idx].fillna(df.loc[grp_idx, 'a_rank'].median())

    scores = []
    for idx, row in df.iterrows():
        s = 0.0
        rh, ra = h_rank.loc[idx], a_rank.loc[idx]

        if rivalries and frozenset([row['HomeTeam'], row['AwayTeam']]) in rivalries:
            s += 0.25
        if rh <= title_zone or ra <= title_zone:
            s += 0.10
        if rh <= title_zone and ra <= title_zone:
            s += 0.10
        if rh >= relegation_zone or ra >= relegation_zone:
            s += 0.10
        if rh >= relegation_zone and ra >= relegation_zone:
            s += 0.15
        prog = row.get('_season_prog', np.nan)
        if pd.notna(prog):
            if prog >= 0.85:
                s += 0.25
            elif prog >= 0.70:
                s += 0.15
        gap = abs(rh - ra)
        if gap <= 2:
            s += 0.15
        elif gap <= 5:
            s += 0.08
        scores.append(min(s, 1.0))

    df['match_importance'] = scores
    df = df.drop(columns=['_date_parsed', '_season_year', '_season_prog'], errors='ignore')
    return df


# ═══════════════════════════════════════════════════════════════
# 4. PROBABILITÉS BOOKMAKER NO-VIG
# ═══════════════════════════════════════════════════════════════

def _compute_novig_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_over  = 'Avg>2.5' in df.columns
    has_under = 'Avg<2.5' in df.columns

    if has_over and has_under:
        raw_over  = 1.0 / df['Avg>2.5'].replace(0, np.nan)
        raw_under = 1.0 / df['Avg<2.5'].replace(0, np.nan)
        margin    = raw_over + raw_under
        df['prob_bookie_over']  = raw_over  / margin
        df['prob_bookie_under'] = raw_under / margin
    elif has_over:
        print('  ⚠ Avg<2.5 absent — fallback approximation (marge 5%)')
        raw_over = 1.0 / df['Avg>2.5'].replace(0, np.nan)
        df['prob_bookie_over']  = raw_over / 1.05
        df['prob_bookie_under'] = (1.0 - raw_over) / 1.05
    else:
        df['prob_bookie_over']  = np.nan
        df['prob_bookie_under'] = np.nan

    if not has_under:
        df['Avg<2.5'] = 1.0 / df['prob_bookie_under'].replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def build_features(df_raw: pd.DataFrame,
                   rivals_csv: Optional[str] = None,
                   n_teams: int = 20) -> pd.DataFrame:
    df = _compute_team_rolling(df_raw)
    df = _compute_rankings(df)

    rivalries = _build_rivalries(rivals_csv)
    print(f'   → {len(rivalries)} paires de rivaux chargées')
    df = _compute_match_importance(df, rivalries, n_teams=n_teams)

    df['total_goals'] = df['FTHG'] + df['FTAG']
    df['over_25']     = (df['total_goals'] > 2.5).astype(int)
    df['under_25']    = (df['total_goals'] < 2.5).astype(int)

    df = _compute_novig_probs(df)
    # prob_bookie assigné dans main.py selon le marché (over/under)

    df['delta_form']             = df['h_form_5']              - df['a_form_5']
    df['delta_rank']             = df['a_rank']                - df['h_rank']
    df['delta_goals_scored']     = df['h_avg_goals_scored']    - df['a_avg_goals_conceded']
    df['delta_goals_against']    = df['a_avg_goals_scored']    - df['h_avg_goals_conceded']
    df['delta_sot']              = df['h_avg_sot']             - df['a_avg_sot']
    df['delta_under_rate']       = df['h_under_rate_5']        + df['a_under_rate_5']   # somme intentionnelle
    df['combined_under_rate_5']  = (df['h_under_rate_5']  + df['a_under_rate_5'])  / 2
    df['combined_under_rate_10'] = (df['h_under_rate_10'] + df['a_under_rate_10']) / 2
    df['combined_goals_var']     = (df['h_goals_var_5']   + df['a_goals_var_5'])   / 2
    df['combined_avg_goals']     = (df['h_avg_total_goals'] + df['a_avg_total_goals']) / 2
    df['combined_low_score']     = (df['h_avg_low_score_rate'] + df['a_avg_low_score_rate']) / 2

    print(f'  Features : {df.shape[1]} colonnes | {len(df)} matchs | '
          f'over={df["over_25"].mean():.1%} under={df["under_25"].mean():.1%}')

    return df
