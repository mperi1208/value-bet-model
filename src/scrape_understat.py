#!/usr/bin/env python3
"""
scrape_understat.py — Scraper xG Understat via Selenium
========================================================
Scrape les xG par match depuis Understat pour les top 5 ligues européennes.

Output : understat_xg.csv avec colonnes :
    date, home, away, xG_home, xG_away, goals_home, goals_away,
    league_us, league_fd, season

Usage :
    python3 scrape_understat.py --seasons 2017 2025
    python3 scrape_understat.py --seasons 2014 2025 --leagues EPL Bundesliga
    python3 scrape_understat.py --seasons 2023 2025   # test rapide
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

# Understat league name → football-data.co.uk code
LEAGUE_MAP = {
    "EPL":        "E0",
    "La_Liga":    "SP1",
    "Bundesliga": "D1",
    "Serie_A":    "I1",
    "Ligue_1":    "F1",
}

# Mapping noms d'équipe Understat → football-data.co.uk
# Understat utilise des noms anglais, football-data aussi mais différents
TEAM_MAP = {
    # EPL
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham": "Tottenham",
    "Newcastle United": "Newcastle",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester": "Leicester",
    "Leeds": "Leeds",
    "Norwich": "Norwich",
    "Sheffield United": "Sheffield United",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nott'm Forest",
    "Ipswich": "Ipswich",
    "Luton": "Luton",
    "West Bromwich Albion": "West Brom",
    "Swansea": "Swansea",
    "Stoke": "Stoke",
    "Hull": "Hull",
    "Cardiff": "Cardiff",
    "Huddersfield": "Huddersfield",
    "Burnley": "Burnley",
    "Watford": "Watford",
    "Fulham": "Fulham",
    "Southampton": "Southampton",
    # La Liga
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis",
    "Alaves": "Alaves",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Rayo Vallecano": "Vallecano",
    "Cadiz": "Cadiz",
    "Almeria": "Almeria",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "Sporting Gijon": "Sp Gijon",
    "Deportivo La Coruna": "La Coruna",
    # Bundesliga
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Bayern Munich": "Bayern Munich",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "Werder Bremen": "Werder Bremen",
    "Mainz 05": "Mainz",
    "Greuther Fuerth": "Greuther Furth",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Nuernberg": "Nurnberg",
    "Hannover 96": "Hannover",
    "Hamburger SV": "Hamburg",
    "SC Paderborn 07": "Paderborn",
    "Arminia Bielefeld": "Bielefeld",
    "Holstein Kiel": "Holstein Kiel",
    "FC St. Pauli": "St Pauli",
    "Heidenheim": "Heidenheim",
    "Darmstadt 98": "Darmstadt",
    # Serie A
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Napoli": "Napoli",
    "Juventus": "Juventus",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Fiorentina": "Fiorentina",
    "Atalanta": "Atalanta",
    "Torino": "Torino",
    "Bologna": "Bologna",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Hellas Verona": "Verona",
    "Sampdoria": "Sampdoria",
    "Genoa": "Genoa",
    "Cagliari": "Cagliari",
    "Empoli": "Empoli",
    "Lecce": "Lecce",
    "Parma Calcio 1913": "Parma",
    "Salernitana": "Salernitana",
    "Frosinone": "Frosinone",
    "Venezia": "Venezia",
    "Spezia": "Spezia",
    "Benevento": "Benevento",
    "Cremonese": "Cremonese",
    "Monza": "Monza",
    "Como": "Como",
    "SPAL 2013": "Spal",
    "ChievoVerona": "Chievo",
    "Crotone": "Crotone",
    "Brescia": "Brescia",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Nice": "Nice",
    "Lille": "Lille",
    "Lens": "Lens",
    "Rennes": "Rennes",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Brest": "Brest",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Reims": "Reims",
    "Lorient": "Lorient",
    "Clermont Foot": "Clermont Foot",
    "Metz": "Metz",
    "Auxerre": "Auxerre",
    "Angers": "Angers",
    "Saint-Etienne": "St Etienne",
    "Caen": "Caen",
    "Dijon": "Dijon",
    "Amiens": "Amiens",
    "Nimes": "Nimes",
    "Troyes": "Troyes",
    "Guingamp": "Guingamp",
    "Bordeaux": "Bordeaux",
    "Le Havre": "Le Havre",
}

WAIT_SECONDS = 5  # temps d'attente après chargement de page


# ═══════════════════════════════════════════════════════════════
# SCRAPING
# ═══════════════════════════════════════════════════════════════

def init_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def scrape_season(driver: webdriver.Chrome,
                  league_us: str,
                  season: int) -> pd.DataFrame:
    """
    Scrape une saison d'une ligue depuis Understat.
    season = année de début (ex: 2024 pour 2024-25)
    """
    url = f"https://understat.com/league/{league_us}/{season}"
    league_fd = LEAGUE_MAP[league_us]

    driver.get(url)
    time.sleep(WAIT_SECONDS)

    # Extraire datesData depuis le JS
    data = driver.execute_script(
        "return typeof datesData !== 'undefined' ? datesData : null"
    )

    if not data:
        print(f"    ⚠ Pas de données pour {league_us} {season}")
        return pd.DataFrame()

    records = []
    for match in data:
        try:
            home_name = match["h"]["title"]
            away_name = match["a"]["title"]
            xg_home = float(match["xG"]["h"])
            xg_away = float(match["xG"]["a"])
            goals_home = int(match["goals"]["h"])
            goals_away = int(match["goals"]["a"])
            date_str = match["datetime"]
            is_result = match.get("isResult", True)

            # Ne garder que les matchs joués
            if not is_result:
                continue

            # Mapping noms
            home_fd = TEAM_MAP.get(home_name, home_name)
            away_fd = TEAM_MAP.get(away_name, away_name)

            records.append({
                "date": date_str[:10],  # YYYY-MM-DD
                "home_us": home_name,
                "away_us": away_name,
                "home_fd": home_fd,
                "away_fd": away_fd,
                "xG_home": round(xg_home, 4),
                "xG_away": round(xg_away, 4),
                "goals_home": goals_home,
                "goals_away": goals_away,
                "league_us": league_us,
                "league_fd": league_fd,
                "season": season,
            })
        except (KeyError, TypeError, ValueError):
            continue

    df = pd.DataFrame(records)
    print(f"    {league_us} {season}: {len(df)} matchs scrapés")
    return df


def scrape_all(leagues: list,
               season_start: int,
               season_end: int,
               output_path: str):
    """Scrape toutes les ligues et saisons."""
    driver = init_driver()
    all_frames = []

    total = len(leagues) * (season_end - season_start)
    done = 0

    for league in leagues:
        print(f"\n── {league} ({LEAGUE_MAP[league]}) ──")

        for season in range(season_start, season_end):
            done += 1
            print(f"  [{done}/{total}] Saison {season}-{str(season+1)[2:]}...", end=" ")
            df = scrape_season(driver, league, season)

            if len(df) > 0:
                all_frames.append(df)

                # Sauvegarde intermédiaire
                df_all = pd.concat(all_frames, ignore_index=True)
                df_all.to_csv(output_path, index=False)

            # Pause entre les requêtes
            time.sleep(2)

    driver.quit()

    if all_frames:
        df_final = pd.concat(all_frames, ignore_index=True)
        df_final.to_csv(output_path, index=False)

        print(f"\n{'='*55}")
        print(f"  RÉSUMÉ")
        print(f"{'='*55}")
        print(f"  Total: {len(df_final)} matchs")
        print(f"  Ligues: {df_final['league_fd'].nunique()}")
        print(f"  Saisons: {df_final['season'].nunique()} "
              f"({df_final['season'].min()}-{df_final['season'].max()})")
        print(f"  xG moyen: home={df_final['xG_home'].mean():.2f} "
              f"away={df_final['xG_away'].mean():.2f}")
        print(f"\n  💾 {output_path}")

        # Vérifier le mapping des noms
        unmapped = set()
        for col in ["home_us", "away_us"]:
            for name in df_final[col].unique():
                if name not in TEAM_MAP and name != TEAM_MAP.get(name, None):
                    fd_name = TEAM_MAP.get(name, name)
                    if fd_name == name:  # pas de mapping
                        unmapped.add(name)

        if unmapped:
            print(f"\n  ⚠ {len(unmapped)} équipes sans mapping explicite "
                  f"(nom Understat utilisé tel quel):")
            for name in sorted(unmapped):
                print(f"    - {name}")
    else:
        print("\n⚠ Aucune donnée scrapée")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Understat xG data"
    )
    parser.add_argument(
        "--leagues", nargs="+",
        default=list(LEAGUE_MAP.keys()),
        help=f"Leagues to scrape. Available: {list(LEAGUE_MAP.keys())}"
    )
    parser.add_argument(
        "--seasons", nargs=2, type=int,
        default=[2017, 2025],
        help="Season range [start, end) (e.g., 2017 2025)"
    )
    parser.add_argument(
        "--output", default="understat_xg.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--wait", type=float, default=5.0,
        help="Wait time after page load (seconds)"
    )
    args = parser.parse_args()

    global WAIT_SECONDS
    WAIT_SECONDS = args.wait

    print(f"\n{'='*55}")
    print(f"  UNDERSTAT xG SCRAPER")
    print(f"  Ligues: {', '.join(args.leagues)}")
    print(f"  Saisons: {args.seasons[0]}-{args.seasons[1]}")
    print(f"  Output: {args.output}")
    print(f"{'='*55}")

    scrape_all(
        leagues=args.leagues,
        season_start=args.seasons[0],
        season_end=args.seasons[1],
        output_path=args.output,
    )

    print(f"\n✅ Done")


if __name__ == "__main__":
    main()
