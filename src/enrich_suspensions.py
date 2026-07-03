"""
enrich_suspensions.py
---------------------
Calcule les features de suspension à partir des incidents Sofascore.
Si les fichiers incidents sont absents, fallback sur les colonnes HR/AR
du DataFrame principal (football-data.co.uk).

Logique Sofascore :
  - Rouge direct (red) → ban minimum 3 matchs en Championship / Ligue 2
  - Rouge après jaune (yellowRed) → ban minimum 1 match
  - rescinded=True → ignoré

Logique fallback HR/AR :
  - On somme les rouges reçus par chaque équipe sur ses 3 matchs précédents
    (proxy du nombre de joueurs sous suspension, ban supposé 3 matchs)

Output enrichi sur le DataFrame principal :
  h_suspended_n      : nb joueurs home suspendus
  a_suspended_n      : nb joueurs away suspendus
  h_suspended_names  : noms (debug, vide en mode fallback)
  a_suspended_names  : noms (debug, vide en mode fallback)
  combined_susp_n    : somme

Usage :
    from enrich_suspensions import enrich_with_suspensions
    df = enrich_with_suspensions(df, sofascore_dir="data/sofascore", league="E1")
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from team_name_map import normalize_team, build_fd_pool

# Ban lengths par défaut (matchs)
BAN_DIRECT_RED   = 3   # rouge direct → 3 matchs (règle Championship/L2)
BAN_YELLOW_RED   = 1   # 2e jaune → 1 match

# Fenêtre rolling pour le fallback HR/AR (matchs)
_FALLBACK_BAN_WINDOW = 3

# Mapping noms TM → noms football-data
_TM_NAME_MAP = {
    # E1 (Championship)
    "Burton Albion":    "Burton",
    "Hull City":        "Hull",
    "Peterborough":     "Peterboro",
    "Sheff Utd":        "Sheffield United",
    "Sheff Wed":        "Sheffield Weds",
    "Stoke City":       "Stoke",
    # F2 (Ligue 2)
    "AC Ajaccio":       "Ajaccio",
    "AJ Auxerre":       "Auxerre",
    "AS Béziers":       "Beziers",
    "AS Nancy":         "Nancy",
    "Amiens SC":        "Amiens",
    "Angers SCO":       "Angers",
    "Bourg-en-Bresse":  "Bourg Peronnas",
    "Chamois Niort":        "Niort",
    "Chamois Niortais FC":  "Niort",
    "FC Chambly Oise":      "Chambly",
    "Clermont Foot":    "Clermont",
    "FC Annecy":        "Annecy",
    "FC Chambly Oise":  "Chambly",
    "FC Lorient":       "Lorient",
    "FC Metz":          "Metz",
    "FC Sochaux":       "Sochaux",
    "G. Ajaccio":       "Ajaccio GFCO",
    "G. Bordeaux":      "Bordeaux",
    "LB Châteauroux":   "Chateauroux",
    "Le Havre AC":      "Le Havre",
    "Le Mans FC":       "Le Mans",
    "Nîmes Olympique":  "Nimes",
    "QRM":              "Quevilly Rouen",
    "R. Strasbourg":    "Strasbourg",
    "Red Star FC":      "Red Star",
    "Rodez AF":         "Rodez",
    "SC Bastia":        "Bastia",
    "SM Caen":          "Caen",
    "Saint-Étienne":    "St Etienne",
    "Stade Brestois":   "Brest",
    "Stade Lavallois":  "Laval",
    "Stade Reims":      "Reims",
    "Tours FC":         "Tours",
    "US Orléans":       "Orleans",
    "USL Dunkerque":    "Dunkerque",
    "Valenciennes FC":  "Valenciennes",
}


def load_incidents(sofascore_dir: str, league: str) -> pd.DataFrame:
    """Charge et concatène tous les fichiers incidents_{league}_*.csv."""
    d = Path(sofascore_dir)
    files = sorted(d.glob(f"sofascore_incidents_{league}_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["home_team"] = out["home_team"].replace(_TM_NAME_MAP)
    out["away_team"] = out["away_team"].replace(_TM_NAME_MAP)
    return out


def _build_suspension_index(incidents: pd.DataFrame,
                              team_schedule: pd.DataFrame) -> dict:
    """
    Construit un index : (team, match_date) → liste de noms suspendus.

    team_schedule : DataFrame avec colonnes [team, date] — tous les matchs
                    de chaque équipe dans l'ordre chronologique.
    """
    if incidents.empty:
        return {}

    inc = incidents[~incidents["rescinded"]].copy()
    inc["ban_len"] = inc["card_class"].map(
        {"red": BAN_DIRECT_RED, "yellowRed": BAN_YELLOW_RED}
    ).fillna(1).astype(int)

    # Pour chaque rouge, trouve les N matchs suivants de l'équipe
    susp_index: dict[tuple, list] = {}  # (team, date) → [player_name, ...]

    for _, row in inc.iterrows():
        # Détermine l'équipe concernée
        team = row["home_team"] if row["is_home"] else row["away_team"]
        red_date = row["date"]
        player   = row["player_name"] or "Unknown"
        ban_len  = int(row["ban_len"])

        # Cherche les matchs futurs de cette équipe
        future = team_schedule[
            (team_schedule["team"] == team) &
            (team_schedule["date"] > red_date)
        ].sort_values("date").head(ban_len)

        for _, m in future.iterrows():
            key = (team, m["date"])
            susp_index.setdefault(key, [])
            if player not in susp_index[key]:
                susp_index[key].append(player)

    return susp_index


def enrich_with_suspensions_fallback(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Fallback quand les fichiers incidents Sofascore sont absents.
    Utilise les colonnes HR/AR (football-data.co.uk) déjà présentes dans df.

    Pour chaque équipe dans chaque match M, somme les rouges reçus dans ses
    _FALLBACK_BAN_WINDOW matchs précédents → proxy du nb de joueurs suspendus.
    """
    df = df.copy()
    df["h_suspended_n"]     = 0.0
    df["a_suspended_n"]     = 0.0
    df["h_suspended_names"] = ""
    df["a_suspended_names"] = ""

    if "HR" not in df.columns or "AR" not in df.columns:
        print(f"  ⚠ Colonnes HR/AR absentes pour {league} — suspension=0")
        df["combined_susp_n"] = 0
        return df

    df["Date"] = pd.to_datetime(df["Date"])

    # Long format : une ligne par (équipe, match, rouges reçus)
    home_side = df[["Date", "HomeTeam", "HR"]].rename(
        columns={"HomeTeam": "team", "HR": "reds"}
    )
    away_side = df[["Date", "AwayTeam", "AR"]].rename(
        columns={"AwayTeam": "team", "AR": "reds"}
    )
    schedule = (
        pd.concat([home_side, away_side], ignore_index=True)
        .drop_duplicates(subset=["Date", "team"])
        .sort_values(["team", "Date"])
    )
    schedule["reds"] = pd.to_numeric(schedule["reds"], errors="coerce").fillna(0)

    # Rolling sum sur les N matchs PRÉCÉDENTS (shift=1 pour exclure le match courant)
    schedule["susp_proxy"] = (
        schedule.groupby("team")["reds"]
        .transform(lambda s: s.shift(1).rolling(_FALLBACK_BAN_WINDOW, min_periods=1).sum())
        .fillna(0)
    )

    lookup = schedule.set_index(["team", "Date"])["susp_proxy"].to_dict()

    df["h_suspended_n"] = df.apply(
        lambda r: lookup.get((r["HomeTeam"], r["Date"]), 0), axis=1
    )
    df["a_suspended_n"] = df.apply(
        lambda r: lookup.get((r["AwayTeam"], r["Date"]), 0), axis=1
    )
    df["combined_susp_n"] = df["h_suspended_n"] + df["a_suspended_n"]

    n_with = (df["combined_susp_n"] > 0).sum()
    print(f"  Fallback HR/AR {league} : {n_with}/{len(df)} matchs avec ≥1 suspendu estimé")
    return df


def _add_suspension_value_ratio(df: pd.DataFrame, league: str,
                                 tm_dir: str = "data/tm") -> pd.DataFrame:
    """
    Ajoute h_susp_value_ratio et a_susp_value_ratio au DataFrame.

    Pour chaque match, somme la valeur marchande des joueurs suspendus
    (lookup via injuries_{league}.csv) et divise par la valeur totale du squad
    (squads_{league}.csv), clippée à [0, 1].

    Si les fichiers TM sont absents → NaN.
    Si la liste des suspendus est vide → ratio = 0.
    """
    inj_path = Path(tm_dir) / f"injuries_{league}.csv"
    sq_path  = Path(tm_dir) / f"squads_{league}.csv"

    if not inj_path.exists() or not sq_path.exists():
        df["h_susp_value_ratio"] = np.nan
        df["a_susp_value_ratio"] = np.nan
        return df

    inj = pd.read_csv(inj_path)
    inj["player_value"] = pd.to_numeric(inj["player_value"], errors="coerce").fillna(0)
    # Valeur max par joueur (pour gérer les doublons multi-saison)
    player_values = inj.groupby("player_name")["player_value"].max().to_dict()

    squads = pd.read_csv(sq_path)
    squads["total_value"] = pd.to_numeric(squads["total_value"], errors="coerce")
    squads.loc[squads["total_value"] == 0, "total_value"] = np.nan

    fd_pool = build_fd_pool(df)
    squads["team_fd"] = squads["team_name_tm"].map(
        lambda t: normalize_team(str(t), fd_pool=fd_pool)
    )
    squads = squads[squads["team_fd"].isin(fd_pool)]
    squad_lookup = squads.set_index(["team_fd", "season"])["total_value"].to_dict()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["_season_year"] = df["Date"].apply(
        lambda d: d.year if d.month >= 7 else d.year - 1
    )

    for side, team_col, names_col in (
        ("h", "HomeTeam", "h_suspended_names"),
        ("a", "AwayTeam", "a_suspended_names"),
    ):
        def _compute_ratio(row, tc=team_col, nc=names_col):
            names_str = row.get(nc, "")
            if not names_str:
                return 0.0
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            if not names:
                return 0.0
            susp_value = sum(player_values.get(n, 0) for n in names)
            squad_val  = squad_lookup.get((row[tc], row["_season_year"]), np.nan)
            if pd.isna(squad_val) or squad_val <= 0:
                return np.nan
            return float(np.clip(susp_value / squad_val, 0, 1))

        df[f"{side}_susp_value_ratio"] = df.apply(_compute_ratio, axis=1)

    df = df.drop(columns=["_season_year"], errors="ignore")
    return df


def enrich_with_suspensions(df: pd.DataFrame,
                             sofascore_dir: str,
                             league: str) -> pd.DataFrame:
    """
    Ajoute h_suspended_n, a_suspended_n, h_suspended_names, a_suspended_names
    au DataFrame principal.
    Si les fichiers incidents Sofascore sont absents, bascule sur
    enrich_with_suspensions_fallback() (HR/AR football-data).
    """
    incidents = load_incidents(sofascore_dir, league)
    if incidents.empty:
        print(f"  ⚠ Pas d'incidents Sofascore pour {league} — fallback HR/AR")
        return enrich_with_suspensions_fallback(df, league)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Construit le schedule par équipe (home + away → long format)
    home_sched = df[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "team", "Date": "date"})
    away_sched = df[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "team", "Date": "date"})
    schedule   = pd.concat([home_sched, away_sched], ignore_index=True).drop_duplicates()

    # Normalise les noms d'équipe dans les incidents pour matcher le DataFrame
    # (incidents utilisent les noms Sofascore, df utilise les noms football-data)
    # → on fait confiance au matching par (team_name, date) avec join fuzzy si besoin
    susp_index = _build_suspension_index(incidents, schedule)

    print(f"  Suspensions {league} : {len(susp_index)} (équipe × match) avec ≥1 suspendu")

    def _get_suspended(team, date):
        return susp_index.get((team, date), [])

    df["h_suspended_names"] = df.apply(
        lambda r: ",".join(_get_suspended(r["HomeTeam"], r["Date"])), axis=1)
    df["a_suspended_names"] = df.apply(
        lambda r: ",".join(_get_suspended(r["AwayTeam"], r["Date"])), axis=1)
    df["h_suspended_n"] = df["h_suspended_names"].apply(
        lambda x: len(x.split(",")) if x else 0)
    df["a_suspended_n"] = df["a_suspended_names"].apply(
        lambda x: len(x.split(",")) if x else 0)
    df["combined_susp_n"] = df["h_suspended_n"] + df["a_suspended_n"]

    n_matches_with_susp = (df["combined_susp_n"] > 0).sum()
    print(f"  → {n_matches_with_susp}/{len(df)} matchs avec ≥1 suspendu")

    tm_dir = str(Path(__file__).parent.parent / "data" / "tm")
    df = _add_suspension_value_ratio(df, league, tm_dir=tm_dir)

    return df
