"""
enrich_sofascore.py
-------------------
Joint les données Sofascore (pré-match + stats) sur le DataFrame football-data.

Stratégie de merge :
  - Clé : (Date ±1 jour, HomeTeam normalisé, AwayTeam normalisé)
  - Le ±1 jour absorbe le décalage UTC → heure locale pour les matchs en soirée
  - On garde uniquement les colonnes utiles au modèle (pas les raw IDs)

Usage :
    from enrich_sofascore import enrich_with_sofascore
    df = enrich_with_sofascore(df_fd, sofascore_dir="./sofascore_data", league="E1")
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from team_name_map import normalize_team, build_fd_pool

# ─── Colonnes pré-match retenues pour le modèle ──────────────────────────────
PREMATCH_COLS = [
    # Absents
    "h_missing_total", "h_missing_confirmed", "h_missing_gk",
    "h_missing_def",   "h_missing_mid",       "h_missing_fwd",
    "a_missing_total", "a_missing_confirmed", "a_missing_gk",
    "a_missing_def",   "a_missing_mid",       "a_missing_fwd",
    # 1X2 cotes + mouvement
    "odds_h_curr", "odds_h_open", "odds_h_move",
    "odds_d_curr", "odds_d_open", "odds_d_move",
    "odds_a_curr", "odds_a_open", "odds_a_move",
    # Under 2.5 (le marché central)
    "under_25_odds", "under_25_open", "under_25_implied", "under_25_move",
    # Under 1.5 / 3.5 aussi utiles
    "under_15_odds", "under_15_implied",
    "under_35_odds", "under_35_implied",
    # BTTS
    "btts_yes_odds", "btts_implied_yes",
    # Asian handicap (proxy market efficiency)
    "ah_h_odds", "ah_h_open",
    "ah_a_odds", "ah_a_open",
    # Crowd
    "crowd_h_pct", "crowd_d_pct", "crowd_a_pct",
    "crowd_btts_yes_pct", "crowd_votes_n",
    # Arbitre (carrière)
    "ref_sf_yc_per_game", "ref_sf_rc_per_game", "ref_sf_games",
    # Stade
    "venue_capacity",
    # Round / journée
    "round",
    # Cotes dérivées (no-vig)
    "odds_implied_h", "odds_implied_d", "odds_implied_a", "odds_implied_delta",
    # Valeurs marchandes squad
    "h_squad_value", "h_squad_value_avail", "h_squad_value_ratio", "h_squad_avg_age",
    "a_squad_value", "a_squad_value_avail", "a_squad_value_ratio", "a_squad_avg_age",
    # H2H agrégat
    "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate", "h2h_n_matches",
    # Pregame form Sofascore
    "h_sf_avg_rating",  "a_sf_avg_rating",
    "h_sf_position",    "a_sf_position",
    "h_sf_pts",         "a_sf_pts",
    "h_sf_form_pts5",   "a_sf_form_pts5",
    "h_sf_wins5",       "a_sf_wins5",
]

# Colonnes stats post-match (pour rolling via Sofascore xG — meilleur que proxy SOT)
STATS_COLS = [
    "h_expected_goals", "a_expected_goals",
    "h_ball_possession", "a_ball_possession",
    "h_total_shots",     "a_total_shots",
    "h_shots_on_target", "a_shots_on_target",
    "h_big_chances",     "a_big_chances",
    "h_passes",          "a_passes",
    "h_accurate_passes", "a_accurate_passes",
    # Pressing / duels
    "h_tackles",         "a_tackles",
    "h_interceptions",   "a_interceptions",
    # Momentum
    "h_momentum_avg",    "a_momentum_avg",
    "momentum_std",      "momentum_late_75",
]


def _parse_season_str(fname: str) -> str | None:
    """'sofascore_prematch_E1_2324.csv' → '2324'"""
    m = re.search(r"_(\d{4})\.csv$", fname)
    return m.group(1) if m else None


def _load_sofascore_csv(path: Path, fd_pool: list[str]) -> pd.DataFrame:
    """
    Lit un CSV Sofascore, normalise les noms d'équipe et la date.
    """
    df = pd.read_csv(path, on_bad_lines="skip")
    if df.empty:
        return df

    # Normalise les noms → clé football-data
    df["home_fd"] = df["home_team"].apply(lambda x: normalize_team(str(x), fd_pool))
    df["away_fd"] = df["away_team"].apply(lambda x: normalize_team(str(x), fd_pool))

    # Date depuis timestamp UNIX (colonne start_timestamp dans prematch)
    if "start_timestamp" in df.columns:
        df["date_sf"] = pd.to_datetime(df["start_timestamp"], unit="s", utc=True).dt.date
    elif "date" in df.columns:
        df["date_sf"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        df["date_sf"] = None

    return df


def _merge_with_tolerance(df_fd: pd.DataFrame, df_sf: pd.DataFrame,
                           suffix: str = "_sf") -> pd.DataFrame:
    """
    Merge sur (HomeTeam, AwayTeam) avec tolérance ±1 jour sur la date.
    Retourne df_fd enrichi avec les colonnes Sofascore.
    """
    if df_sf.empty:
        return df_fd

    # Colonnes Sofascore disponibles (intersection avec ce qu'on veut garder)
    want = PREMATCH_COLS + STATS_COLS
    avail = [c for c in want if c in df_sf.columns]
    key_cols = ["home_fd", "away_fd", "date_sf"] + avail
    df_sf_sub = df_sf[[c for c in key_cols if c in df_sf.columns]].copy()

    # Pour chaque ligne fd, cherche la ligne sf correspondante
    # On fait un merge large puis on filtre par |date - date_sf| <= 1
    df_fd = df_fd.copy()
    df_fd["_date_dt"] = pd.to_datetime(df_fd["Date"]).dt.date

    # Merge + calcul de validité date (±1 jour)
    merged = df_fd.merge(
        df_sf_sub,
        left_on=["HomeTeam", "AwayTeam"],
        right_on=["home_fd", "away_fd"],
        how="left",
    ).reset_index(drop=True)

    date_sf_series = pd.to_datetime(merged["date_sf"], errors="coerce")
    date_fd_series = pd.to_datetime(merged["_date_dt"])
    diff_days = (date_fd_series - date_sf_series).dt.days.abs()
    merged["_valid"] = diff_days.le(1) & date_sf_series.notna()

    # Invalide les colonnes SF pour les lignes hors ±1j
    for col in avail:
        merged.loc[~merged["_valid"], col] = np.nan

    # Déduplique : pour chaque (Date, Home, Away), préférer la ligne valide
    # → trier _valid DESC avant drop_duplicates
    merged = (merged
              .sort_values("_valid", ascending=False)
              .drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first"))

    n_matched = int(merged["_valid"].sum())
    total = len(df_fd)

    merged = merged.drop(columns=["home_fd", "away_fd", "date_sf", "_date_dt", "_valid"],
                         errors="ignore")

    print(f"  Sofascore merge : {n_matched}/{total} matchs liés ({n_matched/total:.1%})")
    if n_matched < total * 0.7:
        no_match = merged[[c for c in avail if c in merged.columns][0]].isna()
        unmatched = merged[no_match][["Date", "HomeTeam", "AwayTeam"]].head(10)
        print("  ⚠ Exemples non liés :")
        print(unmatched.to_string(index=False))

    return merged


def enrich_with_sofascore(
    df_fd: pd.DataFrame,
    sofascore_dir: str | Path,
    league: str,
    mode: str = "both",    # "prematch" | "stats" | "both"
) -> pd.DataFrame:
    """
    Enrichit df_fd avec les données Sofascore disponibles dans sofascore_dir.

    Parameters
    ----------
    df_fd : DataFrame football-data (avec colonnes HomeTeam, AwayTeam, Date, season)
    sofascore_dir : dossier contenant les CSV sofascore_prematch_* et sofascore_stats_*
    league : code football-data (ex: "E1", "F2")
    mode : quelles données charger

    Returns
    -------
    df_fd enrichi — les colonnes SF sont NaN pour les matchs sans correspondance
    """
    sdir = Path(sofascore_dir)
    fd_pool = build_fd_pool(df_fd)

    dfs_pm: list[pd.DataFrame] = []
    dfs_st: list[pd.DataFrame] = []

    for fpath in sorted(sdir.glob(f"sofascore_prematch_{league}_*.csv")):
        df_pm = _load_sofascore_csv(fpath, fd_pool)
        if not df_pm.empty:
            dfs_pm.append(df_pm)

    for fpath in sorted(sdir.glob(f"sofascore_stats_{league}_*.csv")):
        df_st = _load_sofascore_csv(fpath, fd_pool)
        if not df_st.empty:
            dfs_st.append(df_st)

    df_out = df_fd.copy()

    if mode in ("prematch", "both") and dfs_pm:
        df_pm_all = pd.concat(dfs_pm, ignore_index=True)
        print(f"[{league}] Prematch : {len(df_pm_all)} lignes SF chargées")
        df_out = _merge_with_tolerance(df_out, df_pm_all)

    if mode in ("stats", "both") and dfs_st:
        df_st_all = pd.concat(dfs_st, ignore_index=True)
        print(f"[{league}] Stats : {len(df_st_all)} lignes SF chargées")
        df_out = _merge_with_tolerance(df_out, df_st_all)

    # ── Features dérivées post-merge ─────────────────────────────────────────
    df_out = _compute_derived_sf_features(df_out)

    return df_out


def _compute_derived_sf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features calculées à partir des données Sofascore brutes :
    - delta_missing_confirmed : déséquilibre d'absents entre les équipes
    - sf_under25_line_move : le marché a-t-il bougé vers l'under ?
    - xg_total_sf : xG total (remplace le proxy SOT quand dispo)
    - crowd_h_bias : biais domicile dans les votes crowd
    - sf_missing_gk_any : au moins 1 GK absent (très signal under)
    """
    df = df.copy()

    if "h_missing_confirmed" in df.columns and "a_missing_confirmed" in df.columns:
        df["delta_missing_confirmed"] = (
            df["h_missing_confirmed"].fillna(0) - df["a_missing_confirmed"].fillna(0)
        )
        df["total_missing_confirmed"] = (
            df["h_missing_confirmed"].fillna(0) + df["a_missing_confirmed"].fillna(0)
        )

    if "h_missing_gk" in df.columns and "a_missing_gk" in df.columns:
        df["sf_missing_gk_any"] = (
            (df["h_missing_gk"].fillna(0) > 0) | (df["a_missing_gk"].fillna(0) > 0)
        ).astype(float)

    if "under_25_move" in df.columns:
        # move < 1 → cote under a baissé (argent sur under = signal under)
        df["sf_under25_move_signal"] = df["under_25_move"].apply(
            lambda x: 1.0 if (pd.notna(x) and x < 0.97) else
                      (-1.0 if (pd.notna(x) and x > 1.03) else 0.0)
        )

    if "h_expected_goals" in df.columns and "a_expected_goals" in df.columns:
        df["xg_total_sf"]  = df["h_expected_goals"].fillna(np.nan) + df["a_expected_goals"].fillna(np.nan)
        df["xg_delta_sf"]  = df["h_expected_goals"].fillna(np.nan) - df["a_expected_goals"].fillna(np.nan)

    if "crowd_h_pct" in df.columns and "crowd_a_pct" in df.columns:
        df["crowd_h_bias"] = df["crowd_h_pct"].fillna(np.nan) - df["crowd_a_pct"].fillna(np.nan)

    # Proxy efficacité de tir Sofascore
    if "h_shots_on_target" in df.columns and "h_total_shots" in df.columns:
        df["h_sf_shot_acc"] = df["h_shots_on_target"] / df["h_total_shots"].clip(lower=0.5)
        df["a_sf_shot_acc"] = df["a_shots_on_target"] / df["a_total_shots"].clip(lower=0.5)

    # ── Features combinées forme récente ─────────────────────────────────────
    if "h_sf_under_rate5" in df.columns and "a_sf_under_rate5" in df.columns:
        df["combined_sf_under_rate5"] = (
            df["h_sf_under_rate5"].fillna(np.nan) + df["a_sf_under_rate5"].fillna(np.nan)
        ) / 2

    if "h_sf_position" in df.columns and "a_sf_position" in df.columns:
        # Positif = home pire classé que away (plus susceptible de jouer défensif)
        df["sf_position_delta"] = (
            df["h_sf_position"].fillna(np.nan) - df["a_sf_position"].fillna(np.nan)
        )

    if "h_sf_form_pts5" in df.columns and "a_sf_form_pts5" in df.columns:
        df["sf_form_delta"] = (
            df["h_sf_form_pts5"].fillna(np.nan) - df["a_sf_form_pts5"].fillna(np.nan)
        )

    # Rating delta
    if "h_sf_avg_rating" in df.columns and "a_sf_avg_rating" in df.columns:
        df["sf_avg_rating_delta"] = (
            df["h_sf_avg_rating"].fillna(np.nan) - df["a_sf_avg_rating"].fillna(np.nan)
        )

    # % pts possibles (position relative indépendante de l'avancement de saison)
    if "h_sf_pts" in df.columns and "round" in df.columns:
        max_pts = df["round"].fillna(1) * 3
        df["h_sf_pts_pct"] = df["h_sf_pts"].fillna(np.nan) / max_pts
        df["a_sf_pts_pct"] = df["a_sf_pts"].fillna(np.nan) / max_pts

    # Interaction sharp money × sentiment public
    if "under_25_move" in df.columns and "crowd_btts_yes_pct" in df.columns:
        # move < 1 (under raccourcit) ET crowd vote peu BTTS → signal under fort
        df["sharp_crowd_under_signal"] = (
            (2 - df["under_25_move"].fillna(1)) * (1 - df["crowd_btts_yes_pct"].fillna(0.5))
        )

    # Delta valeur marchande
    if "h_squad_value" in df.columns and "a_squad_value" in df.columns:
        df["squad_value_delta"] = (
            df["h_squad_value"].fillna(np.nan) - df["a_squad_value"].fillna(np.nan)
        )
        df["squad_value_ratio_delta"] = (
            df["h_squad_value_ratio"].fillna(np.nan) - df["a_squad_value_ratio"].fillna(np.nan)
        )

    return df


# ─── Rolling features Sofascore ──────────────────────────────────────────────

# Colonnes post-match à roller (xG, shots qualité, GK)
_SF_ROLL_COLS = [
    "expected_goals",     # → h_sf_xg_avg_5, a_sf_xg_avg_5
    "big_chances",        # → h_sf_chances_avg
    "shots_on_target",    # → h_sf_sot_avg (plus propre que fd)
    "total_shots",        # → h_sf_shots_avg
    "goals_prevented",    # → h_sf_gk_prevented_avg (GK performance)
    "goalkeeper_saves",   # → h_sf_saves_avg
    "ball_possession",    # possession en % string → needs clean
]

_ROLL_W = [5, 10]


def _parse_pct(val) -> float | None:
    """'54%' → 0.54, '54' → 0.54, NaN → NaN"""
    if pd.isna(val):
        return None
    s = str(val).replace("%", "").strip()
    try:
        f = float(s)
        return f / 100 if f > 1 else f
    except ValueError:
        return None


def compute_sf_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des moyennes rolling (sans data leakage) sur les stats Sofascore
    post-match. Nécessite que df contienne les colonnes h_/a_ des STATS_COLS.

    Produit pour chaque col C dans _SF_ROLL_COLS et chaque fenêtre W :
        h_sf_{C}_avg{W}, a_sf_{C}_avg{W}

    + Features combinées :
        combined_sf_xg_{W} = (h + a) / 2
        sf_xg_delta_{W}    = h - a
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Nettoyage possession
    for side in ("h", "a"):
        col = f"{side}_ball_possession"
        if col in df.columns:
            df[col] = df[col].apply(_parse_pct)

    # Reconstruit une vue long (home + away) pour roller par équipe
    home = df[["Date", "HomeTeam"]].copy()
    home["team"] = df["HomeTeam"]
    home["side"] = "h"
    home["_idx"] = df.index

    away = df[["Date", "AwayTeam"]].copy()
    away["team"] = df["AwayTeam"]
    away["side"] = "a"
    away["_idx"] = df.index

    for col in _SF_ROLL_COLS:
        h_col = f"h_{col}"
        a_col = f"a_{col}"
        if h_col in df.columns:
            home[col] = df[h_col].values
        if a_col in df.columns:
            away[col] = df[a_col].values

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "Date"]).reset_index(drop=True)

    # Expanding mean + rolling windows
    available_cols = [c for c in _SF_ROLL_COLS
                      if c in long.columns and long[c].notna().any()]

    for col in available_cols:
        # Expanding (toutes données passées)
        long[f"sf_{col}_avg"] = (
            long.groupby("team", group_keys=False)[col]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        )
        for w in _ROLL_W:
            long[f"sf_{col}_avg{w}"] = (
                long.groupby("team", group_keys=False)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            )

    # Pivote home/away → colonnes h_ / a_ sur le df original
    new_cols = [c for c in long.columns
                if c.startswith("sf_") or c == "_idx" or c == "side"]
    long_sub = long[new_cols]

    for col in [c for c in long_sub.columns if c.startswith("sf_")]:
        h_vals = long_sub[long_sub["side"] == "h"].set_index("_idx")[col]
        a_vals = long_sub[long_sub["side"] == "a"].set_index("_idx")[col]
        df[f"h_{col}"] = df.index.map(h_vals)
        df[f"a_{col}"] = df.index.map(a_vals)

    # ── Features combinées ────────────────────────────────────────────────────
    for w_suf in ["", "5", "10"]:
        sfx = f"_avg{w_suf}" if w_suf else "_avg"
        hx = f"h_sf_expected_goals{sfx}"
        ax = f"a_sf_expected_goals{sfx}"
        if hx in df.columns and ax in df.columns:
            df[f"combined_sf_xg{sfx}"]  = (df[hx].fillna(np.nan) + df[ax].fillna(np.nan)) / 2
            df[f"sf_xg_delta{sfx}"]     = df[hx].fillna(np.nan) - df[ax].fillna(np.nan)

    return df


# ─── CLI minimal pour tester ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--league",       default="E1")
    ap.add_argument("--fd-csv",       required=True, help="Chemin vers un CSV football-data")
    ap.add_argument("--sofascore-dir", default="./sofascore_data")
    args = ap.parse_args()

    df_raw = pd.read_csv(args.fd_csv, on_bad_lines="skip")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)

    df_enriched = enrich_with_sofascore(
        df_raw, args.sofascore_dir, args.league
    )

    sf_cols = [c for c in df_enriched.columns if c in PREMATCH_COLS + STATS_COLS]
    coverage = df_enriched[sf_cols].notna().mean()
    print("\nCouverture des colonnes Sofascore :")
    print(coverage[coverage > 0].sort_values(ascending=False).to_string())
    print(f"\nShape final : {df_enriched.shape}")
