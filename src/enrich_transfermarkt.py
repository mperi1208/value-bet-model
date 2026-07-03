"""
enrich_transfermarkt.py
-----------------------
Enrichit le DataFrame football-data avec des features de blessures TM.

Features produites par match :
    h_inj_n            : nb joueurs home blessés à la date du match
    a_inj_n            : nb joueurs away blessés à la date du match
    h_inj_value_out    : valeur marchande totale des blessés home (€)
    a_inj_value_out    : valeur marchande totale des blessés away (€)
    combined_inj_n     : h_inj_n + a_inj_n
    h_inj_value_ratio  : h_inj_value_out / total_squad_value_home
    a_inj_value_ratio  : a_inj_value_out / total_squad_value_away

Logique :
    - Un joueur est blessé pour le match si date_from <= match_date <= date_until
    - date_until NaT → considéré encore blessé (inclus)
    - Normalisation des noms TM → noms football-data via team_name_map
    - Déduplication des blessures sur (player_id, date_from, date_until)
      pour éviter le surcounting dû au scraping multi-saison
    - Valeur squad depuis squads_{league}.csv (mêmes unités € que player_value)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from team_name_map import normalize_team, build_fd_pool


# ─── Chargement blessures ─────────────────────────────────────────────────────

def load_injuries(tm_dir: str, league: str) -> pd.DataFrame:
    """Charge injuries_{league}.csv, déduplique et parse les dates."""
    path = Path(tm_dir) / f"injuries_{league}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date_from"]    = pd.to_datetime(df["date_from"],  errors="coerce")
    df["date_until"]   = pd.to_datetime(df["date_until"], errors="coerce")
    df["player_value"] = pd.to_numeric(df["player_value"], errors="coerce").fillna(0)

    # Bug #1 : déduplication — le scraper répète les blessures pour chaque
    # saison où le joueur figure au squad → facteur de gonflement 3-8x
    df = df.drop_duplicates(subset=["player_id", "date_from", "date_until"])

    return df


# ─── Chargement squads (valeur totale par équipe × saison) ───────────────────

def load_squads(tm_dir: str, league: str) -> pd.DataFrame:
    """Charge squads_{league}.csv et déduplique."""
    path = Path(tm_dir) / f"squads_{league}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    # total_value=0 est un artefact de scraping (TM page non chargée) → NaN
    df.loc[df["total_value"] == 0, "total_value"] = np.nan
    # Bug #6 : doublons dans squads
    df = df.drop_duplicates(subset=["team_name_tm", "season"])
    return df


# ─── Normalisation des noms d'équipe ─────────────────────────────────────────

def _normalize_injuries(inj: pd.DataFrame, fd_pool: list[str]) -> pd.DataFrame:
    """Ajoute une colonne team_fd normalisée et filtre les équipes hors fd_pool."""
    inj = inj.copy()
    inj["team_fd"] = inj["team_name_tm"].map(
        lambda t: normalize_team(str(t), fd_pool=fd_pool)
    )
    # Bug #7/#8 : supprime les équipes qui ne sont jamais dans cette division
    # (ex: Montpellier dans F2, Wrexham dans E1 avant leur montée)
    inj = inj[inj["team_fd"].isin(fd_pool)]
    return inj


def _normalize_squads(squads: pd.DataFrame, fd_pool: list[str]) -> pd.DataFrame:
    """Ajoute team_fd et filtre les équipes hors fd_pool."""
    squads = squads.copy()
    squads["team_fd"] = squads["team_name_tm"].map(
        lambda t: normalize_team(str(t), fd_pool=fd_pool)
    )
    squads = squads[squads["team_fd"].isin(fd_pool)]
    return squads


# ─── Jointure vectorisée par intervalle de dates ─────────────────────────────

def _count_active_injuries(df: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, compte les blessures actives de l'équipe home et away.

    Stratégie : merge schedule × injuries sur team, puis filtre date range.
    """
    if inj.empty:
        df = df.copy()
        for col in ["h_inj_n", "a_inj_n", "h_inj_value_out", "a_inj_value_out"]:
            df[col] = 0
        df["combined_inj_n"] = 0
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home_sched = (
        df[["Date", "HomeTeam"]]
        .assign(_idx=df.index, _side="h")
        .rename(columns={"HomeTeam": "team"})
    )
    away_sched = (
        df[["Date", "AwayTeam"]]
        .assign(_idx=df.index, _side="a")
        .rename(columns={"AwayTeam": "team"})
    )
    sched = pd.concat([home_sched, away_sched], ignore_index=True)

    inj_cols = inj[["team_fd", "player_name", "player_value", "date_from", "date_until"]]

    merged = sched.merge(inj_cols, left_on="team", right_on="team_fd", how="left")

    active_mask = (
        merged["date_from"].notna()
        & (merged["date_from"] <= merged["Date"])
        & (
            merged["date_until"].isna()
            | (merged["date_until"] >= merged["Date"])
        )
    )
    active = merged[active_mask]

    agg = (
        active.groupby(["_idx", "_side"])
        .agg(n_inj=("player_name", "count"), value_out=("player_value", "sum"))
        .reset_index()
    )

    agg_h = agg[agg["_side"] == "h"].set_index("_idx")[["n_inj", "value_out"]]
    agg_a = agg[agg["_side"] == "a"].set_index("_idx")[["n_inj", "value_out"]]

    df["h_inj_n"]         = agg_h["n_inj"].reindex(df.index).fillna(0).astype(int)
    df["a_inj_n"]         = agg_a["n_inj"].reindex(df.index).fillna(0).astype(int)
    df["h_inj_value_out"] = agg_h["value_out"].reindex(df.index).fillna(0).astype(float)
    df["a_inj_value_out"] = agg_a["value_out"].reindex(df.index).fillna(0).astype(float)
    df["combined_inj_n"]  = df["h_inj_n"] + df["a_inj_n"]

    return df


# ─── Ratio valeur manquante / valeur totale squad ─────────────────────────────

def _add_value_ratio(df: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    """
    Bug #2 fix : utilise squads_{league}.csv (total_value en €) comme dénominateur,
    au lieu de h_squad_value de Sofascore (en M€ et souvent absent/1.0).

    Ajoute h_inj_value_ratio et a_inj_value_ratio ∈ [0, 1].
    """
    if squads.empty:
        df["h_inj_value_ratio"] = np.nan
        df["a_inj_value_ratio"] = np.nan
        return df

    df = df.copy()
    df["_season_year"] = pd.to_datetime(df["Date"]).apply(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )

    squad_lookup = squads.set_index(["team_fd", "season"])["total_value"].to_dict()

    for side, team_col in (("h", "HomeTeam"), ("a", "AwayTeam")):
        squad_val = df.apply(
            lambda r: squad_lookup.get((r[team_col], r["_season_year"]), np.nan),
            axis=1,
        )
        inj_val = df[f"{side}_inj_value_out"]
        # ratio = valeur blessés / valeur totale squad, clipé à [0, 1]
        df[f"{side}_inj_value_ratio"] = (inj_val / squad_val.clip(lower=1)).clip(0, 1)

    df = df.drop(columns=["_season_year"], errors="ignore")
    return df


# ─── Point d'entrée public ────────────────────────────────────────────────────

def enrich_with_injuries(
    df: pd.DataFrame,
    tm_dir: str = "data/tm",
    league: str = "E1",
) -> pd.DataFrame:
    """
    Ajoute les features de blessures TM au DataFrame principal.

    Parameters
    ----------
    df      : DataFrame football-data (doit avoir Date, HomeTeam, AwayTeam)
    tm_dir  : répertoire contenant injuries_{league}.csv et squads_{league}.csv
    league  : 'E1' ou 'F2'
    """
    inj = load_injuries(tm_dir, league)

    if inj.empty:
        print(f"  ⚠ Pas de blessures TM trouvées pour {league} dans {tm_dir}")
        df = df.copy()
        for col in ["h_inj_n", "a_inj_n", "h_inj_value_out", "a_inj_value_out",
                    "combined_inj_n", "h_inj_value_ratio", "a_inj_value_ratio"]:
            df[col] = np.nan
        return df

    fd_pool = build_fd_pool(df)
    inj     = _normalize_injuries(inj, fd_pool)

    unmatched_raw = set(
        normalize_team(str(t), fd_pool=fd_pool)
        for t in pd.read_csv(Path(tm_dir) / f"injuries_{league}.csv")["team_name_tm"].unique()
    ) - set(fd_pool)
    if unmatched_raw:
        print(f"  ⚠ {league} injuries : {len(unmatched_raw)} équipes TM hors division ignorées")

    df = _count_active_injuries(df, inj)

    squads = load_squads(tm_dir, league)
    if not squads.empty:
        squads = _normalize_squads(squads, fd_pool)
    df = _add_value_ratio(df, squads)

    n_with = (df["combined_inj_n"] > 0).sum()
    print(f"  Blessures TM {league} : {n_with}/{len(df)} matchs avec ≥1 blessé "
          f"| moy home={df['h_inj_n'].mean():.2f} away={df['a_inj_n'].mean():.2f}")

    return df
