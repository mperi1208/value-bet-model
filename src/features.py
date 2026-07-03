"""
features.py
-----------
Construit toutes les features pour le modèle under 2.5 :
  - Stats rolling par équipe (buts, tirs, under-rate, variance)
  - Classement dynamique intra-saison
  - Match importance (rivalités, enjeux, avancement saison)
  - Probabilités bookmaker no-vig
  - prob_bookie N'EST PAS initialisé ici (assigné dans main.py selon le marché)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
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

    long['total_goals_match'] = long['scored'] + long['conceded']
    long['is_under'] = (long['total_goals_match'] < 2.5).astype(float)
    long['is_low_score'] = (long['total_goals_match'] <= 1).astype(float)

    grp = ['Div', 'season', 'team']
    stat_cols = ['scored', 'conceded', 'shots', 'sot', 'corners',
                 'yellow', 'red', 'is_null', 'is_under', 'is_low_score',
                 'total_goals_match']

    # Expanding mean sur les valeurs passées (shift(1) pour éviter le data leakage)
    for col in stat_cols:
        long[f'avg_{col}'] = (
            long.groupby(grp, group_keys=False)[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    long['form'] = (
        long.groupby(grp, group_keys=False)['result']
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum())
    )

    for w in [3, 5, 10]:
        long[f'under_rate_{w}'] = (
            long.groupby(grp, group_keys=False)['is_under']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    long['goals_var_5'] = (
        long.groupby(grp, group_keys=False)['total_goals_match']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std().fillna(0))
    )

    # Proxy suspension : rouges sur les N derniers matchs
    # Rouge = ban 1-3 matchs → red_last1 capture le match précédent, red_last3 la fenêtre
    for w in [1, 3]:
        long[f'red_last{w}'] = (
            long.groupby(grp, group_keys=False)['red']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
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
        'red_last1':             'red_last1',
        'red_last3':             'red_last3',
    }

    keep = list(rename_map.keys()) + ['orig_idx', 'is_home']
    long_sub = long[keep].rename(columns=rename_map)

    home_stats = long_sub[long_sub['is_home']].drop(columns='is_home').set_index('orig_idx')
    away_stats = long_sub[~long_sub['is_home']].drop(columns='is_home').set_index('orig_idx')

    df = df.join(home_stats.add_prefix('h_'), how='left')
    df = df.join(away_stats.add_prefix('a_'), how='left')

    return df


# ═══════════════════════════════════════════════════════════════
# 2. FEATURES ARBITRE
# ═══════════════════════════════════════════════════════════════

def _compute_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, calcule le taux under 2.5 historique de l'arbitre
    et sa moyenne de buts, en utilisant uniquement ses matchs passés.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)

    df['total_goals'] = df['FTHG'] + df['FTAG']
    df['is_under']    = (df['total_goals'] < 2.5).astype(float)

    grp = 'Referee'
    df['ref_under_rate'] = (
        df.groupby(grp, group_keys=False)['is_under']
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )
    df['ref_avg_goals'] = (
        df.groupby(grp, group_keys=False)['total_goals']
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )

    df = df.drop(columns=['total_goals', 'is_under'], errors='ignore')
    return df


# ═══════════════════════════════════════════════════════════════
# 3. HEAD-TO-HEAD UNDER RATE
# ═══════════════════════════════════════════════════════════════

def _compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, calcule le taux under 2.5 historique entre les deux équipes
    en utilisant uniquement leurs confrontations passées (shift + expanding).
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)

    df['_goals_h2h'] = df['FTHG'] + df['FTAG']
    df['_under_h2h'] = (df['_goals_h2h'] < 2.5).astype(float)

    # Clé canonique : ordre alphabétique pour que A-B == B-A
    df['_h2h_key'] = df.apply(
        lambda r: '__'.join(sorted([r['HomeTeam'], r['AwayTeam']])), axis=1
    )

    # Moyenne cumulée glissante (passé uniquement, pas de leakage)
    df['h2h_under_rate'] = (
        df.groupby('_h2h_key', group_keys=False)['_under_h2h']
        .transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
    )
    # Fillna avec la moyenne cumulée globale sur données passées uniquement
    cumulative_mean = df['_under_h2h'].expanding().mean().shift(1)
    df['h2h_under_rate'] = df['h2h_under_rate'].fillna(cumulative_mean)

    df = df.drop(columns=['_goals_h2h', '_under_h2h', '_h2h_key'], errors='ignore')
    return df


# ═══════════════════════════════════════════════════════════════
# 3b. FIXTURE CONGESTION
# ═══════════════════════════════════════════════════════════════

def _compute_fixture_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, calcule le nombre de jours depuis le dernier match
    de l'équipe à domicile et de l'équipe extérieure.
    Default = 7 jours (milieu de saison normal) quand aucune donnée passée.
    """
    df = df.copy()
    df['_orig_idx'] = range(len(df))

    home = df[['_orig_idx', 'Date', 'HomeTeam']].rename(
        columns={'HomeTeam': 'team'})
    home['side'] = 'h'

    away = df[['_orig_idx', 'Date', 'AwayTeam']].rename(
        columns={'AwayTeam': 'team'})
    away['side'] = 'a'

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(['team', 'Date']).reset_index(drop=True)

    long['prev_date'] = long.groupby('team')['Date'].shift(1)
    long['days_rest'] = (long['Date'] - long['prev_date']).dt.days
    long['days_rest'] = long['days_rest'].clip(upper=30).fillna(7)

    for side in ['h', 'a']:
        side_data = (long[long['side'] == side]
                     .set_index('_orig_idx')['days_rest'])
        df[f'{side}_days_rest'] = df['_orig_idx'].map(side_data)

    df = df.drop(columns=['_orig_idx'], errors='ignore')
    return df


# ═══════════════════════════════════════════════════════════════
# 4. CLASSEMENT DYNAMIQUE
# ═══════════════════════════════════════════════════════════════

def _compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Classement live avant chaque match. Séquentiel par nature — optimisé avec itertuples."""
    df = df.copy()
    h_ranks: dict = {}
    a_ranks: dict = {}

    for (div, season), grp in df.groupby(['Div', 'season'], sort=False):
        grp = grp.sort_values('Date')
        teams = pd.concat([grp['HomeTeam'], grp['AwayTeam']]).unique()
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}

        for row in grp.itertuples():
            sorted_teams = sorted(teams, key=lambda t: (-pts[t], -gd[t]))
            rank_map = {t: i + 1 for i, t in enumerate(sorted_teams)}
            h_ranks[row.Index] = rank_map[row.HomeTeam]
            a_ranks[row.Index] = rank_map[row.AwayTeam]

            hg, ag = row.FTHG, row.FTAG
            if row.FTR == 'H':
                pts[row.HomeTeam] += 3
            elif row.FTR == 'A':
                pts[row.AwayTeam] += 3
            else:
                pts[row.HomeTeam] += 1
                pts[row.AwayTeam] += 1
            gd[row.HomeTeam] += hg - ag
            gd[row.AwayTeam] += ag - hg

    df['h_rank'] = pd.Series(h_ranks)
    df['a_rank'] = pd.Series(a_ranks)
    return df


# ═══════════════════════════════════════════════════════════════
# 5. MATCH IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def _build_rivalries(rivals_csv: Optional[str]) -> set:
    if rivals_csv is None or not os.path.exists(rivals_csv):
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
    """Calcul vectorisé du score d'importance de chaque match."""
    df = df.copy()

    date_col = pd.to_datetime(df['Date'], errors='coerce')
    df['_season_year'] = np.where(
        date_col.dt.month >= 7, date_col.dt.year, date_col.dt.year - 1
    )
    df['_season_prog'] = df.groupby(['Div', '_season_year'])['Date'].rank(pct=True)

    relegation_zone = n_teams - 2
    title_zone = 4

    h_rank = df['h_rank'].copy()
    a_rank = df['a_rank'].copy()
    for div, grp_idx in df.groupby('Div').groups.items():
        median_h = h_rank.loc[grp_idx].median()
        median_a = a_rank.loc[grp_idx].median()
        h_rank.loc[grp_idx] = h_rank.loc[grp_idx].fillna(median_h)
        a_rank.loc[grp_idx] = a_rank.loc[grp_idx].fillna(median_a)

    s = pd.Series(0.0, index=df.index)

    if rivalries:
        s += df.apply(
            lambda r: frozenset([r['HomeTeam'], r['AwayTeam']]) in rivalries, axis=1
        ).astype(float) * 0.25

    h_title = h_rank <= title_zone
    a_title = a_rank <= title_zone
    s += (h_title | a_title).astype(float) * 0.10
    s += (h_title & a_title).astype(float) * 0.10

    h_relg = h_rank >= relegation_zone
    a_relg = a_rank >= relegation_zone
    s += (h_relg | a_relg).astype(float) * 0.10
    s += (h_relg & a_relg).astype(float) * 0.15

    prog = df['_season_prog']
    s += (prog >= 0.85).astype(float) * 0.25
    s += ((prog >= 0.70) & (prog < 0.85)).astype(float) * 0.15

    gap = (h_rank - a_rank).abs()
    s += (gap <= 2).astype(float) * 0.15
    s += ((gap > 2) & (gap <= 5)).astype(float) * 0.08

    df['match_importance'] = s.clip(upper=1.0)
    df = df.drop(columns=['_season_year', '_season_prog'], errors='ignore')
    return df


# ═══════════════════════════════════════════════════════════════
# 6. PROBABILITÉS BOOKMAKER NO-VIG
# ═══════════════════════════════════════════════════════════════

def _compute_novig_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise les colonnes BbAv (anciennes saisons FD) vers Avg (nouvelles saisons)
    # Après concat, les deux colonnes coexistent avec NaN → on remplit les trous
    for side in ['<', '>']:
        if f'BbAv{side}2.5' in df.columns:
            if f'Avg{side}2.5' not in df.columns:
                df[f'Avg{side}2.5'] = df[f'BbAv{side}2.5']
            else:
                df[f'Avg{side}2.5'] = df[f'Avg{side}2.5'].fillna(df[f'BbAv{side}2.5'])
        if f'BbMx{side}2.5' in df.columns:
            if f'Max{side}2.5' not in df.columns:
                df[f'Max{side}2.5'] = df[f'BbMx{side}2.5']
            else:
                df[f'Max{side}2.5'] = df[f'Max{side}2.5'].fillna(df[f'BbMx{side}2.5'])

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
# 7. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def build_features(df_raw: pd.DataFrame,
                   rivals_csv: Optional[str] = None,
                   n_teams: int = 20,
                   sofascore_dir: Optional[str] = None) -> pd.DataFrame:
    df = _compute_team_rolling(df_raw)
    df = _compute_h2h_features(df)
    df = _compute_fixture_congestion(df)
    df = _compute_rankings(df)

    # ── Enrichissement Sofascore (optionnel) ──────────────────────────────────
    if sofascore_dir is not None:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))
        league = df_raw["Div"].iloc[0] if "Div" in df_raw.columns else None

        try:
            from enrich_sofascore import enrich_with_sofascore, compute_sf_rolling
            # fix : avant, seule la première ligue du df (df["Div"].iloc[0]) était
            # enrichie — les 9 autres restaient à NaN. On boucle sur toutes.
            if league:
                for lg in df["Div"].unique():
                    print(f"  Enrichissement Sofascore [{lg}] depuis {sofascore_dir}...")
                    mask_lg = df["Div"] == lg
                    df_lg = enrich_with_sofascore(df[mask_lg].copy(),
                                                  sofascore_dir=sofascore_dir, league=lg)
                    for c in df_lg.columns:
                        if c not in df.columns:
                            df[c] = np.nan
                        df.loc[mask_lg, c] = df_lg[c].values
                df = compute_sf_rolling(df)
                sf_cols = [c for c in df.columns if c.startswith("h_sf_") or c.startswith("a_sf_")]
                cov = df[sf_cols].notna().mean().mean() if sf_cols else 0
                print(f"  Sofascore : {len(sf_cols)} colonnes rolling | couverture moy. {cov:.1%}")
        except Exception as e:
            print(f"  ⚠ Sofascore enrichissement ignoré : {e}")

        # ── Suspensions exactes (incidents TM/Sofascore) — par ligue ─────────
        try:
            from enrich_suspensions import enrich_with_suspensions
            parts = []
            for lg in df["Div"].unique():
                df_lg = df[df["Div"] == lg].copy()
                df_lg = enrich_with_suspensions(df_lg, sofascore_dir=sofascore_dir, league=lg)
                parts.append(df_lg)
            df = pd.concat(parts).sort_index()
        except Exception as e:
            print(f"  ⚠ Suspensions enrichissement ignoré : {e}")
            for col in ["h_suspended_n", "a_suspended_n", "h_suspended_names",
                        "a_suspended_names", "combined_susp_n"]:
                if col not in df.columns:
                    df[col] = np.nan

    # ── Blessures Transfermarkt — par ligue ──────────────────────────────────
    _tm_dir = str(Path(__file__).parent.parent / "data" / "tm")
    try:
        from enrich_transfermarkt import enrich_with_injuries
        inj_parts = []
        for lg in df["Div"].unique():
            df_lg = df[df["Div"] == lg].copy()
            df_lg = enrich_with_injuries(df_lg, tm_dir=_tm_dir, league=lg)
            inj_parts.append(df_lg)
        df = pd.concat(inj_parts).sort_index()
    except Exception as e:
        print(f"  ⚠ Blessures TM ignorées : {e}")
        for col in ["h_inj_n", "a_inj_n", "h_inj_value_out",
                    "a_inj_value_out", "combined_inj_n",
                    "h_inj_value_ratio", "a_inj_value_ratio"]:
            if col not in df.columns:
                df[col] = np.nan

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

    # Proxy suspension (rouge = joueur probablement suspendu)
    df['h_susp_proxy']  = df['h_red_last1']   # rouge dans le match précédent → ban quasi-certain
    df['a_susp_proxy']  = df['a_red_last1']
    df['h_susp_proxy3'] = df['h_red_last3']   # fenêtre 3 matchs → accumulation
    df['a_susp_proxy3'] = df['a_red_last3']
    df['combined_susp_proxy'] = df['h_susp_proxy'] + df['a_susp_proxy']

    # Qualité de tir : ratio SOT/shots (proxy xG sans données externes)
    df['h_shot_acc']        = df['h_avg_sot'] / df['h_avg_shots'].clip(lower=0.5)
    df['a_shot_acc']        = df['a_avg_sot'] / df['a_avg_shots'].clip(lower=0.5)
    df['combined_shot_acc'] = (df['h_shot_acc'] + df['a_shot_acc']) / 2

    # xG approximatif : SOT × taux de conversion moyen (~0.11 sur les divisions 1-2)
    df['h_xg_proxy']        = df['h_avg_sot'] * 0.11
    df['a_xg_proxy']        = df['a_avg_sot'] * 0.11
    df['combined_xg_proxy'] = df['h_xg_proxy'] + df['a_xg_proxy']

    # ── Features arbitre ──────────────────────────────────────────
    # Taux under historique de chaque arbitre (rolling, sans data leakage)
    if 'Referee' in df.columns:
        df = _compute_referee_features(df)
    else:
        df['ref_under_rate'] = np.nan
        df['ref_avg_goals']  = np.nan

    # Fill NaN arbitre avec la médiane globale (ligues sans données arbitre,
    # ou premiers matchs d'un arbitre avant d'avoir min_periods)
    df['ref_under_rate'] = df['ref_under_rate'].fillna(df['ref_under_rate'].median())
    df['ref_avg_goals']  = df['ref_avg_goals'].fillna(df['ref_avg_goals'].median())

    # ── Signal marché : écart Max/Avg ─────────────────────────────
    # Un écart élevé Max<2.5/Avg<2.5 indique que de l'argent "sharp"
    # a bougé la cote — signal d'information sur le résultat probable.
    if 'Max<2.5' in df.columns and 'Avg<2.5' in df.columns:
        df['odds_spread_under'] = (
            df['Max<2.5'] / df['Avg<2.5'].replace(0, np.nan)
        ).fillna(1.0)
    else:
        df['odds_spread_under'] = 1.0

    print(f'  Features : {df.shape[1]} colonnes | {len(df)} matchs | '
          f'over={df["over_25"].mean():.1%} under={df["under_25"].mean():.1%}')

    return df
