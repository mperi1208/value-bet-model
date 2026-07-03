"""
scrape_fbref.py — Scraper xG FBref (StatsBomb) via Selenium
=============================================================
Scrape les xG par match depuis FBref pour E1 (Championship) et F2 (Ligue 2).
Understat ne couvre pas ces ligues — FBref les a depuis 2017/18 (E1) et 2018/19 (F2).

Output : data/fbref/fbref_xg_{league}.csv avec colonnes :
    date, home_fd, away_fd, xg_home, xg_away, league, season

Usage :
    python3 scrape_fbref.py --leagues E1 F2 --seasons 2017 2024
    python3 scrape_fbref.py --leagues E1 --seasons 2023 2024
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

WAIT = 3.0

# FBref competition IDs
LEAGUE_CONFIG = {
    "E1": {"comp_id": 10, "name": "Championship",    "url_name": "Championship",    "fd_first_season": 2017},
    "F2": {"comp_id": 60, "name": "Ligue-2",         "url_name": "Ligue-2",         "fd_first_season": 2018},
    "E0": {"comp_id":  9, "name": "Premier-League",  "url_name": "Premier-League",  "fd_first_season": 2017},
    "F1": {"comp_id": 13, "name": "Ligue-1",         "url_name": "Ligue-1",         "fd_first_season": 2017},
    "D1": {"comp_id": 20, "name": "Bundesliga",      "url_name": "Bundesliga",      "fd_first_season": 2017},
    "SP1":{"comp_id": 12, "name": "La-Liga",         "url_name": "La-Liga",         "fd_first_season": 2017},
    "I1": {"comp_id": 11, "name": "Serie-A",         "url_name": "Serie-A",         "fd_first_season": 2017},
}

# Mapping noms FBref → football-data.co.uk
TEAM_MAP = {
    # E1
    "Nott'ham Forest":      "Nott'm Forest",
    "Nottingham Forest":    "Nott'm Forest",
    "Sheffield Weds":       "Sheffield Weds",
    "Sheffield Wednesday":  "Sheffield Weds",
    "Sheffield Utd":        "Sheffield United",
    "Peterborough Utd":     "Peterboro",
    "Huddersfield":         "Huddersfield",
    "Wigan Athletic":       "Wigan",
    "Rotherham Utd":        "Rotherham",
    "West Brom":            "West Brom",
    "Coventry City":        "Coventry",
    "Sunderland":           "Sunderland",
    "Queens Park Rangers":  "QPR",
    "Middlesbrough":        "Middlesbrough",
    "Plymouth Argyle":      "Plymouth",
    "Wycombe Wanderers":    "Wycombe",
    "Blackburn":            "Blackburn",
    "Birmingham City":      "Birmingham",
    "Swansea City":         "Swansea",
    "Cardiff City":         "Cardiff",
    "Hull City":            "Hull",
    "Stoke City":           "Stoke",
    "Preston NE":           "Preston",
    "Luton Town":           "Luton",
    "Millwall":             "Millwall",
    "Charlton Ath":         "Charlton",
    "Oxford Utd":           "Oxford",
    "Bristol City":         "Bristol City",
    # F2
    "Ajaccio":              "Ajaccio",
    "GFC Ajaccio":          "Ajaccio GFCO",
    "Le Havre":             "Le Havre",
    "Quevilly-Rouen":       "Quevilly Rouen",
    "Châteauroux":          "Chateauroux",
    "Bourg-en-Bresse":      "Bourg Peronnas",
    "Niort":                "Niort",
    "Sochaux":              "Sochaux",
    "Valenciennes":         "Valenciennes",
    "Nîmes":                "Nimes",
    "Dunkerque":            "Dunkerque",
    "Saint-Étienne":        "St Etienne",
    "Rodez":                "Rodez",
    "Auxerre":              "Auxerre",
    "Grenoble Foot":        "Grenoble",
    "Annecy":               "Annecy",
    "Pau":                  "Pau FC",
    "Concarneau":           "Concarneau",
    "Red Star":             "Red Star",
}


def _make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


def _season_str(season: int) -> str:
    return f"{season}-{season+1}"


def scrape_season(driver, league: str, season: int) -> pd.DataFrame:
    cfg = LEAGUE_CONFIG[league]
    comp_id = cfg["comp_id"]
    url_name = cfg["url_name"]
    s = _season_str(season)
    url = f"https://fbref.com/en/comps/{comp_id}/{s}/schedule/{s}-{url_name}-Scores-and-Fixtures"

    print(f"  {league} {s}... ", end="", flush=True)
    driver.get(url)
    time.sleep(WAIT)

    rows = []
    try:
        table = driver.find_element(By.ID, "sched_all")
        trs = table.find_elements(By.TAG_NAME, "tr")
    except Exception:
        print("⚠ table introuvable")
        return pd.DataFrame()

    for tr in trs:
        tds = tr.find_elements(By.TAG_NAME, "td")
        if len(tds) < 10:
            continue
        try:
            # Colonnes FBref : Wk, Day, Date, Time, Home, xG, Score, xG, Away, ...
            date_el  = tr.find_element(By.CSS_SELECTOR, "td[data-stat='date']")
            home_el  = tr.find_element(By.CSS_SELECTOR, "td[data-stat='home_team']")
            away_el  = tr.find_element(By.CSS_SELECTOR, "td[data-stat='away_team']")
            xgh_el   = tr.find_element(By.CSS_SELECTOR, "td[data-stat='home_xg']")
            xga_el   = tr.find_element(By.CSS_SELECTOR, "td[data-stat='away_xg']")
            score_el = tr.find_element(By.CSS_SELECTOR, "td[data-stat='score']")

            date  = date_el.text.strip()
            home  = home_el.text.strip()
            away  = away_el.text.strip()
            xg_h  = xgh_el.text.strip()
            xg_a  = xga_el.text.strip()
            score = score_el.text.strip()

            if not date or not home or not away or not xg_h or not xg_a:
                continue
            if not score or "–" not in score:
                continue

            rows.append({
                "date":     date,
                "home_raw": home,
                "away_raw": away,
                "home_fd":  TEAM_MAP.get(home, home),
                "away_fd":  TEAM_MAP.get(away, away),
                "xg_home":  float(xg_h),
                "xg_away":  float(xg_a),
                "league":   league,
                "season":   season,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"{len(df)} matchs")
    return df


def run(leagues: list, season_start: int, season_end: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = _make_driver()

    try:
        for league in leagues:
            cfg = LEAGUE_CONFIG.get(league)
            if cfg is None:
                print(f"  ⚠ Ligue inconnue : {league}")
                continue

            first = max(season_start, cfg["fd_first_season"])
            seasons = list(range(first, season_end + 1))

            all_dfs = []
            for s in seasons:
                df = scrape_season(driver, league, s)
                if not df.empty:
                    all_dfs.append(df)
                time.sleep(1.5)

            if not all_dfs:
                print(f"  ⚠ Aucune donnée pour {league}")
                continue

            out = pd.concat(all_dfs, ignore_index=True)

            # Merge avec existant si fichier déjà présent
            path = out_dir / f"fbref_xg_{league}.csv"
            if path.exists():
                old = pd.read_csv(path, parse_dates=["date"])
                out = pd.concat([old, out], ignore_index=True).drop_duplicates(
                    subset=["date", "home_fd", "away_fd"]
                )

            out = out.sort_values(["season", "date"]).reset_index(drop=True)
            out.to_csv(path, index=False)
            print(f"  → {league} : {len(out)} matchs sauvegardés → {path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FBref xG data")
    parser.add_argument("--leagues", nargs="+", default=["E1", "F2"])
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"),
                        default=[2017, 2024])
    parser.add_argument("--out", default="data/fbref")
    args = parser.parse_args()

    run(
        leagues=args.leagues,
        season_start=args.seasons[0],
        season_end=args.seasons[1],
        out_dir=Path(args.out),
    )
